"""A small client for a local Zetaris (Lightning) instance.

It logs in through the web UI's proxy with the admin user ID and password, or
uses an API key when ZETARIS_API_KEY and ZETARIS_ORG_ID are set, then runs one
Lightning SQL statement per call and returns the rows. Standard library only.

Credentials come from an env file (default: ~/.config/lfg/credentials/zetaris-local.env,
or the path in DALIL_ZETARIS_ENV). Variables already set in the process
environment win over the file. Nothing in this module prints or logs a secret.

Tested against Zetaris 2.4.3.1 (the free local bundle), October 2026.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_ENV_FILE = Path("~/.config/lfg/credentials/zetaris-local.env").expanduser()
DEFAULT_UI_URL = "http://127.0.0.1:13000"

# Where the loader stages files. This directory is inside the `customer-data`
# Docker volume, which the driver container mounts at /home/zetaris/customer-data
# and which the bundle names as local platform storage
# (ZETARIS_LOCAL_FILE_SYTEM_ROOT_PATH=/home/zetaris/customer-data/shared).
DRIVER_STAGE_ROOT = PurePosixPath("/home/zetaris/customer-data/shared/dalil")

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
# DESCRIBE BY accepts letters, digits, spaces, _ . - and , only (Zetaris docs).
_DESCRIPTION = re.compile(r"^[A-Za-z0-9 _.,-]{0,200}$")
_FORMATS = {"CSV", "JSON", "PARQUET", "ORC", "DELTA", "AVRO", "TEXT"}


class ZetarisError(RuntimeError):
    """Any failure talking to Zetaris. Messages never contain credentials."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


# --- configuration -----------------------------------------------------------


@dataclass(frozen=True)
class Config:
    ui_url: str = DEFAULT_UI_URL
    email: str | None = None
    password: str | None = field(default=None, repr=False)
    api_key: str | None = field(default=None, repr=False)
    org_id: str | None = None
    source: str = "environment"

    def __post_init__(self) -> None:
        # Checked first, and the URL is never echoed, because user:password@
        # in the URL would put a credential in an error message or a log line.
        if "@" in self.ui_url:
            raise ZetarisError(
                "ZETARIS_UI_URL must not carry a user name or password (user@host); "
                "set ZETARIS_ADMIN_EMAIL and ZETARIS_ADMIN_PASSWORD instead"
            )
        if not re.fullmatch(r"https?://[^/\s?#]+", self.ui_url):
            raise ZetarisError(
                "ZETARIS_UI_URL must be an http(s) origin with no path, "
                "for example http://127.0.0.1:13000"
            )
        if self.org_id is not None and not re.fullmatch(r"[0-9]+", self.org_id):
            raise ZetarisError("ZETARIS_ORG_ID must be the numeric organisation ID")
        if not self.api_key and not (self.email and self.password):
            raise ZetarisError(
                f"No Zetaris credentials in {self.source}: set ZETARIS_ADMIN_EMAIL and "
                "ZETARIS_ADMIN_PASSWORD, or ZETARIS_API_KEY and ZETARIS_ORG_ID"
            )
        if self.api_key and not self.org_id and not (self.email and self.password):
            raise ZetarisError("ZETARIS_API_KEY needs ZETARIS_ORG_ID as well")


def parse_env_file(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines. Ignores blanks and comments, strips matching quotes
    and an `export ` prefix. Does not expand variables."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load_config(
    env_file: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Config:
    """Build a Config from an env file plus the process environment.

    The file is optional when the environment already carries the credentials.
    A file named explicitly (argument or DALIL_ZETARIS_ENV) must exist.
    """
    environ = os.environ if environ is None else environ
    explicit = env_file or environ.get("DALIL_ZETARIS_ENV")
    path = Path(explicit).expanduser() if explicit else DEFAULT_ENV_FILE
    values: dict[str, str] = {}
    source = "the environment"
    if path.is_file():
        values = parse_env_file(path.read_text(encoding="utf-8"))
        source = str(path)
    elif explicit:
        raise ZetarisError(f"Zetaris env file not found: {path}")
    for key in (
        "ZETARIS_UI_URL",
        "ZETARIS_ADMIN_EMAIL",
        "ZETARIS_ADMIN_PASSWORD",
        "ZETARIS_API_KEY",
        "ZETARIS_ORG_ID",
    ):
        if environ.get(key):
            values[key] = environ[key]
    return Config(
        ui_url=values.get("ZETARIS_UI_URL", DEFAULT_UI_URL).rstrip("/"),
        email=values.get("ZETARIS_ADMIN_EMAIL") or None,
        password=values.get("ZETARIS_ADMIN_PASSWORD") or None,
        api_key=values.get("ZETARIS_API_KEY") or None,
        org_id=values.get("ZETARIS_ORG_ID") or None,
        source=source,
    )


# --- SQL text helpers ----------------------------------------------------------


def check_name(name: str) -> str:
    """A plain identifier for DDL (database, table). Zetaris stores names upper-case."""
    if not _NAME.match(name):
        raise ZetarisError(f"Not a plain SQL name: {name!r}")
    return name


def quote_ident(name: str) -> str:
    """Backtick-quote an identifier, doubling any backtick inside it."""
    return "`" + name.replace("`", "``") + "`"


def quote_string(value: str) -> str:
    """A single-quoted Spark SQL string literal. Spark treats backslash as an
    escape character inside literals, so escape it as well as the quote."""
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def strip_statement(sql: str) -> str:
    """One statement per call: trim whitespace and a trailing semicolon."""
    stmt = sql.strip()
    while stmt.endswith(";"):
        stmt = stmt[:-1].rstrip()
    if not stmt:
        raise ZetarisError("Empty SQL statement")
    return stmt


def create_database_sql(database: str, description: str) -> str:
    if not _DESCRIPTION.match(description):
        raise ZetarisError(
            "DESCRIBE BY text may hold only letters, digits, spaces and _ . - ,"
        )
    return f'CREATE LIGHTNING DATABASE {check_name(database)} DESCRIBE BY "{description}"'


def filestore_table_sql(
    database: str, table: str, path: str, fmt: str, options: Mapping[str, str] | None = None
) -> str:
    fmt = fmt.upper()
    if fmt not in _FORMATS:
        raise ZetarisError(f"Unsupported file format for Dalil: {fmt}")
    opts = {"path": path, **(options or {})}
    body = ", ".join(f"{check_name(k)} {quote_string(v)}" for k, v in opts.items())
    return (
        f"CREATE LIGHTNING FILESTORE TABLE {check_name(table)} FROM {check_name(database)} "
        f"FORMAT {fmt} OPTIONS ({body})"
    )


# --- results and client --------------------------------------------------------


@dataclass
class Result:
    headers: list[str]
    rows: list[list[Any]]

    def dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.headers, row)) for row in self.rows]

    def scalar(self) -> Any:
        if not self.rows or not self.rows[0]:
            raise ZetarisError("Query returned no value")
        return self.rows[0][0]


def _error_detail(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            text = str(parsed.get("msg") or parsed.get("message") or parsed.get("error") or text)
    except ValueError:
        if "<html" in text.lower():
            text = f"the server returned an HTML page; {_HTML_HINT}"
    text = " ".join(text.split())
    return text if len(text) <= 500 else text[:497] + "..."


_HTML_HINT = "check ZETARIS_UI_URL points at the Zetaris web UI"


class Client:
    """Runs Lightning SQL through the UI proxy route /api/proxy/sql-editor/sqls/run-query."""

    def __init__(self, config: Config | None = None, timeout: float = 600):
        self.config = config or load_config()
        self.timeout = timeout
        # An API key is used only with a known org ID; otherwise log in, which returns both.
        self._token: str | None = self.config.api_key if self.config.org_id else None
        self.org_id: str | None = self.config.org_id

    def _redact(self, text: str) -> str:
        """Blank out any credential a server might echo back in an error body."""
        for secret in (self.config.password, self.config.api_key, self._token):
            if secret and len(secret) >= 4:
                text = text.replace(secret, "[redacted]")
        return text

    # HTTP plumbing. The token only ever goes into a request header.
    def _post(self, path: str, body: Any, auth: bool = True) -> Any:
        headers = {"Content-Type": "application/json", "X-Request-ID": str(uuid.uuid4())}
        if auth:
            headers["Authorization"] = f"Bearer {self._token}"
            headers["X-Org-ID"] = str(self.org_id)
        req = Request(
            self.config.ui_url + path,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                status = resp.status
        except HTTPError as e:
            detail = self._redact(_error_detail(e.read()))
            raise ZetarisError(f"HTTP {e.code} from {path}: {detail}", e.code) from None
        except (URLError, OSError) as e:
            reason = self._redact(str(getattr(e, "reason", e)))
            raise ZetarisError(
                f"Cannot reach Zetaris at {self.config.ui_url}: {reason}. "
                "Is the stack up? (docker ps --filter name=zetaris)"
            ) from None
        try:
            return json.loads(raw.decode("utf-8") or "null")
        except ValueError:  # includes UnicodeDecodeError
            # The body is not echoed: it is not JSON, so it is not ours to quote.
            kind = "an HTML page" if b"<html" in raw[:4096].lower() else "a response that is not JSON"
            raise ZetarisError(f"HTTP {status} from {path} returned {kind}; {_HTML_HINT}", status) from None

    def login(self) -> None:
        if not (self.config.email and self.config.password):
            raise ZetarisError("Login needs ZETARIS_ADMIN_EMAIL and ZETARIS_ADMIN_PASSWORD")
        try:
            out = self._post(
                "/api/auth/login",
                {"username": self.config.email, "password": self.config.password},
                auth=False,
            )
        except ZetarisError as e:
            if e.status in (400, 401, 403):
                raise ZetarisError(f"Zetaris login refused (HTTP {e.status}); check the admin credentials", e.status) from None
            raise
        token = out.get("idToken") if isinstance(out, dict) else None
        if not isinstance(token, str) or not token:
            raise ZetarisError("Zetaris login returned no idToken")
        self._token = token
        if out.get("orgId") is not None:
            self.org_id = str(out["orgId"])
        if not self.org_id:
            raise ZetarisError("Zetaris login returned no orgId; set ZETARIS_ORG_ID")

    def sql(self, statement: str, limit: int = 1000) -> Result:
        """Run one Lightning SQL statement and return its headers and rows.

        REST returns every value as a string (or None); convert before comparing.
        """
        stmt = strip_statement(statement)
        if not self._token:
            self.login()
        body = {"sql": stmt, "limit": int(limit)}
        try:
            out = self._post("/api/proxy/sql-editor/sqls/run-query", body)
        except ZetarisError as e:
            if e.status != 401 or not (self.config.email and self.config.password):
                raise
            self.login()  # an expired login token: sign in again once
            out = self._post("/api/proxy/sql-editor/sqls/run-query", body)
        if not isinstance(out, dict):
            raise ZetarisError("Unexpected response shape from run-query")
        return Result(headers=list(out.get("headers") or []), rows=list(out.get("data") or []))

    # Convenience wrappers over SHOW statements.
    def lightning_databases(self) -> list[str]:
        return [str(r[0]).upper() for r in self.sql("SHOW LIGHTNING DATABASES").rows]

    def lightning_tables(self, database: str) -> dict[str, dict[str, Any]]:
        res = self.sql(f"SHOW LIGHTNING TABLES {check_name(database)}")
        return {str(d["name"]).lower(): d for d in res.dicts()}

    def table_columns(self, database: str, table: str) -> list[str]:
        res = self.sql(f"DESCRIBE DATASOURCE TABLE {check_name(database)}.{check_name(table)}")
        return [str(r[0]) for r in res.rows]


# --- staging files where the driver can read them ------------------------------


def find_driver_container() -> str:
    """Name of the running Zetaris driver container (compose project `zetaris`)."""
    try:
        out = subprocess.run(
            [
                "docker", "ps",
                "--filter", "label=com.docker.compose.project=zetaris",
                "--filter", "label=com.docker.compose.service=driver",
                "--format", "{{.Names}}",
            ],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError) as e:
        raise ZetarisError(f"Cannot list Docker containers: {e}") from None
    if len(out) != 1:
        raise ZetarisError(f"Expected one running Zetaris driver container, found {len(out)}")
    return out[0]


def stage_file(container: str, data: bytes, name: str, root: PurePosixPath = DRIVER_STAGE_ROOT) -> str:
    """Write bytes into the driver's storage volume and return the file:// URI
    Lightning should read. Written to a temp name then renamed, so a reader
    never sees a half-written file. Runs as the container's own user (zetaris)."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name.startswith("."):
        raise ZetarisError(f"Bad staged file name: {name!r}")
    dest = root / name
    script = 'umask 002; mkdir -p "$(dirname "$1")" && cat > "$1.tmp" && mv -f "$1.tmp" "$1"'
    try:
        subprocess.run(
            ["docker", "exec", "-i", container, "bash", "-c", script, "_", str(dest)],
            input=data, capture_output=True, check=True, timeout=300,
        )
    except subprocess.CalledProcessError as e:
        raise ZetarisError(f"Staging {name} in {container} failed: {e.stderr.decode(errors='replace')[:300]}") from None
    except (OSError, subprocess.SubprocessError) as e:
        raise ZetarisError(f"Staging {name} in {container} failed: {e}") from None
    return f"file://{dest}"


def json_array_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """Rows as one JSON array, one object per line.

    Lightning 2.4.3.1 reads FORMAT JSON as multi-line JSON: a JSON-lines file
    registers without error but shows only its first row. A top-level array
    is read as one row per element.
    """
    lines = [json.dumps(dict(r), ensure_ascii=False, sort_keys=True) for r in rows]
    return ("[\n" + ",\n".join(lines) + "\n]\n").encode("utf-8")


# --- citations -------------------------------------------------------------------


def page_lookup_sql(database: str, doc_id: str, page: int) -> str:
    """The exact SQL a reviewer reruns to see a cited page."""
    return (
        f"SELECT d.doc_id, d.title, d.source_url, d.sha256, p.page, p.text "
        f"FROM {check_name(database)}.documents d JOIN {check_name(database)}.pages p "
        f"ON d.doc_id = p.doc_id "
        f"WHERE p.doc_id = {quote_string(doc_id)} AND p.page = {int(page)}"
    )


def citation(database: str, doc_id: str, page: int, sha256: str | None = None,
             source_url: str | None = None, server_version: str | None = None) -> dict[str, Any]:
    """How Dalil records a Zetaris lookup so a reviewer can rerun it."""
    return {
        "engine": "zetaris-lightning" + (f" {server_version}" if server_version else ""),
        "database": database.upper(),
        "tables": [f"{database}.documents", f"{database}.pages"],
        "doc_id": doc_id,
        "page": int(page),
        "sha256": sha256,
        "source_url": source_url,
        "sql": page_lookup_sql(database, doc_id, page),
    }
