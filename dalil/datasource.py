"""Where Dalil's open-data tables come from: Zetaris, or a local database.

Both backends serve the same tables under the same names (dalil.open_*), take
the same fixed SELECT statements and return rows the same way, so the
pipeline's lookups and the checker's rerun do not know which one ran.

    zetaris  the Zetaris (Lightning) instance in dalil/zetaris.py, over HTTP
    local    SQLite from the Python standard library, in memory, loaded at
             start from the committed files in data/open/ (tables.csv names
             them), with the schema attached as `dalil` and every write refused

One setting picks the backend, DALIL_OPEN_DATA:

    auto     (the default) Zetaris when it is configured and answers SELECT 1,
             else the local database when the data files are present
    zetaris  Zetaris only
    local    the local database only
    off      no open data; lookups are skipped and the screen says so

DALIL_OPEN_DATA_DIR points the local database at another folder holding a
tables.csv, for example a knowledge-base profile's own data.

Why SQLite and not DuckDB: every fixed query in dalil/opendata.py runs on
SQLite unchanged, SQLite ships with Python, and it adds no dependency to the
offline copy. Values come back as strings, as Zetaris's REST interface returns
them, so a cited value compares the same way whichever backend ran.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OPEN_DATA_DIR = REPO_ROOT / "data" / "open"
SCHEMA = "dalil"
MODES = ("auto", "zetaris", "local", "off")

KIND_LOCAL = "local"
KIND_ZETARIS = "zetaris"

# What the screen shows. The header uses LABELS, a data citation SOURCE_LABELS.
LABELS = {
    KIND_LOCAL: "Open data: local database",
    KIND_ZETARIS: "Open data: Zetaris",
    None: "Open data: not configured",
}
SOURCE_LABELS = {
    KIND_LOCAL: "queried from the local database",
    KIND_ZETARIS: "queried through Zetaris",
}


class DataSourceError(RuntimeError):
    """The open-data source cannot be set up or cannot run a query."""


def source_label(kind: str | None) -> str | None:
    return SOURCE_LABELS.get(kind or "")


# --- settings -----------------------------------------------------------------


def open_data_mode(environ: Mapping[str, str] | None = None) -> str:
    environ = os.environ if environ is None else environ
    mode = (environ.get("DALIL_OPEN_DATA") or "auto").strip().lower()
    if mode not in MODES:
        raise DataSourceError(f"DALIL_OPEN_DATA must be one of {', '.join(MODES)}, not {mode!r}")
    return mode


def open_data_dir(environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    value = environ.get("DALIL_OPEN_DATA_DIR")
    return Path(value).expanduser() if value else DEFAULT_OPEN_DATA_DIR


# --- results --------------------------------------------------------------------


@dataclass
class Result:
    """Same shape as dalil.zetaris.Result: headers, and rows of strings or None."""

    headers: list[str]
    rows: list[list[Any]] = field(default_factory=list)

    def dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.headers, row)) for row in self.rows]

    def scalar(self) -> Any:
        if not self.rows or not self.rows[0]:
            raise DataSourceError("Query returned no value")
        return self.rows[0][0]


def as_text(value: Any) -> str | None:
    """A value as Zetaris's REST interface returns it: a string, or None."""
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value)) if abs(value) < 1e16 else repr(value)
    return str(value)


# --- the local database ----------------------------------------------------------


_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def _column_type(values: list[str]) -> tuple[str, Any]:
    """SQLite type and converter for a CSV column, inferred like Zetaris's
    inferSchema: whole numbers, then decimals, else text. Blanks are NULL."""
    present = [v for v in values if v != ""]
    for sql_type, convert in (("INTEGER", int), ("REAL", float)):
        try:
            for v in present:
                convert(v)
        except ValueError:
            continue
        return sql_type, convert
    return "TEXT", str


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class TableFile:
    table: str
    path: Path
    sha256: str | None  # from SOURCES.csv, when it lists the file


def table_files(data_dir: Path) -> list[TableFile]:
    """The tables named in data_dir/tables.csv, with the SHA-256 SOURCES.csv records."""
    index = data_dir / "tables.csv"
    if not index.is_file():
        raise DataSourceError(f"no open-data index at {index}")
    sums: dict[str, str] = {}
    sources = data_dir / "SOURCES.csv"
    if sources.is_file():
        with sources.open(newline="", encoding="utf-8") as fh:
            sums = {r["file"]: (r.get("sha256") or "").strip().lower() for r in csv.DictReader(fh)}
    out = []
    with index.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            table = (row.get("table") or "").strip()
            if not _TABLE_NAME.match(table):
                raise DataSourceError(f"{index}: not a plain table name: {table!r}")
            path = (data_dir / row["file"]).resolve()
            if data_dir.resolve() not in path.parents:
                raise DataSourceError(f"{index}: {row['file']!r} is outside {data_dir}")
            out.append(TableFile(table, path, sums.get(row["file"]) or None))
    if not out:
        raise DataSourceError(f"{index} names no tables")
    return out


def data_files_present(data_dir: Path | None = None) -> bool:
    try:
        return all(t.path.is_file() for t in table_files(data_dir or open_data_dir()))
    except DataSourceError:
        return False


# Only these SQLite actions may run once the tables are loaded: reading, and
# calling functions. Writes, ATTACH, PRAGMA and schema changes are refused.
_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
# Functions that read nothing outside the query. load_extension is never on this list.
_DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


def _authorize(action: int, arg1: Any, arg2: Any, db_name: Any, trigger: Any) -> int:
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and str(arg2 or "").lower() in _DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


class LocalSource:
    """The open-data tables in an in-memory SQLite database, read-only.

    Each table is loaded from its CSV in data_dir, checked against the row
    count and, where SOURCES.csv lists it, the SHA-256 recorded there, so the
    database serves exactly the committed files.
    """

    kind = KIND_LOCAL
    label = LABELS[KIND_LOCAL]
    source_label = SOURCE_LABELS[KIND_LOCAL]
    engine = f"sqlite {sqlite3.sqlite_version}"

    def __init__(self, data_dir: Path | str | None = None, schema: str = SCHEMA):
        self.data_dir = Path(data_dir) if data_dir else open_data_dir()
        self.schema = schema
        self._lock = threading.Lock()
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._db.execute(f"ATTACH DATABASE ':memory:' AS {schema}")
        self.row_counts: dict[str, int] = {}
        for tf in table_files(self.data_dir):
            self.row_counts[f"{schema}.{tf.table}"] = self._load(tf)
        self._db.commit()
        self._db.execute("PRAGMA query_only = ON")
        self._db.set_authorizer(_authorize)

    def _load(self, tf: TableFile) -> int:
        try:
            raw = tf.path.read_bytes()
        except OSError as e:
            raise DataSourceError(f"cannot read {tf.path.name}: {e.strerror}") from None
        if tf.sha256 and hashlib.sha256(raw).hexdigest() != tf.sha256:
            raise DataSourceError(f"{tf.path.name} does not match the SHA-256 in SOURCES.csv")
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
        if not rows or not rows[0]:
            raise DataSourceError(f"{tf.path.name} has no header row")
        header, body = rows[0], [r for r in rows[1:] if r]
        if len(set(h.lower() for h in header)) != len(header):
            raise DataSourceError(f"{tf.path.name}: column names repeat")
        if any(len(r) != len(header) for r in body):
            raise DataSourceError(f"{tf.path.name}: a row has a different number of fields from the header")
        types = [_column_type([r[i] for r in body]) for i in range(len(header))]
        cols = ", ".join(f"{_quote(h)} {t}" for h, (t, _) in zip(header, types))
        name = f"{self.schema}.{_quote(tf.table)}"
        self._db.execute(f"CREATE TABLE {name} ({cols})")
        self._db.executemany(
            f"INSERT INTO {name} VALUES ({', '.join('?' * len(header))})",
            [[None if v == "" else conv(v) for v, (_, conv) in zip(r, types)] for r in body],
        )
        got = self._db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        if got != len(body):
            raise DataSourceError(f"{tf.table} has {got} rows, expected {len(body)}")
        return got

    def sql(self, statement: str, limit: int = 1000) -> Result:
        """Run one read-only SELECT and return at most `limit` rows, as strings."""
        stmt = statement.strip().rstrip(";").strip()
        if not stmt:
            raise DataSourceError("Empty SQL statement")
        with self._lock:
            try:
                cur = self._db.execute(stmt)
                rows = cur.fetchmany(max(0, int(limit)))
            except sqlite3.Error as e:
                raise DataSourceError(f"local query failed: {e}") from None
        headers = [d[0] for d in cur.description or []]
        return Result(headers, [[as_text(v) for v in row] for row in rows])

    def tables(self) -> list[str]:
        return sorted(self.row_counts)

    def ping(self) -> None:
        self.sql("SELECT 1")


# --- Zetaris ------------------------------------------------------------------------


class ZetarisSource:
    """dalil.zetaris.Client under the same interface."""

    kind = KIND_ZETARIS
    label = LABELS[KIND_ZETARIS]
    source_label = SOURCE_LABELS[KIND_ZETARIS]
    engine = "zetaris-lightning"

    def __init__(self, client: Any = None, timeout: float = 30.0, schema: str = SCHEMA):
        if client is None:
            from dalil import zetaris as z

            client = z.Client(timeout=timeout)
        self.client = client
        self.schema = schema
        self._tables: list[str] | None = None

    def sql(self, statement: str, limit: int = 1000) -> Any:
        return self.client.sql(statement, limit=limit)

    def tables(self) -> list[str]:
        if self._tables is None:
            names = self.client.lightning_tables(self.schema)
            self._tables = sorted(f"{self.schema}.{n}" for n in names)
        return list(self._tables)

    def ping(self) -> None:
        self.client.sql("SELECT 1")


def zetaris_configured(environ: Mapping[str, str] | None = None) -> bool:
    """True when Zetaris credentials can be read. Says nothing about whether it answers."""
    from dalil import zetaris as z

    try:
        z.load_config(environ=environ)
    except z.ZetarisError:
        return False
    return True


# --- choosing one -------------------------------------------------------------------


def kind_of(source: Any) -> str:
    """The kind of any source; a bare Zetaris client (or a test stand-in) is Zetaris."""
    return getattr(source, "kind", None) or KIND_ZETARIS


def resolve(environ: Mapping[str, str] | None = None, zetaris_timeout: float = 30.0) -> tuple[Any, str]:
    """The open-data source DALIL_OPEN_DATA asks for, and "", or None and why not.

    In auto mode a configured Zetaris that does not answer falls back to the
    local database, and the reason is returned with it.
    """
    environ = os.environ if environ is None else environ
    try:
        mode = open_data_mode(environ)
    except DataSourceError as e:
        return None, str(e)
    if mode == "off":
        return None, "open data turned off (DALIL_OPEN_DATA=off); lookup skipped"
    why = ""
    if mode in ("auto", "zetaris") and (mode == "zetaris" or zetaris_configured(environ)):
        try:
            source = ZetarisSource(timeout=zetaris_timeout)
            source.ping()
            return source, ""
        except Exception as e:  # ZetarisError messages carry no credentials
            why = f"Zetaris unreachable ({type(e).__name__}: {str(e)[:160]})"
            if mode == "zetaris":
                return None, why + "; lookup skipped"
    if mode in ("auto", "local"):
        data_dir = open_data_dir(environ)
        if mode == "auto" and not data_files_present(data_dir):
            return None, (why + "; " if why else "") + "Open data not configured; lookup skipped"
        try:
            return LocalSource(data_dir), (why + "; using the local database" if why else "")
        except DataSourceError as e:
            return None, (why + "; " if why else "") + f"local database unavailable ({e}); lookup skipped"
    return None, why or "Open data not configured; lookup skipped"


def status(source: Any) -> dict[str, Any]:
    """The health endpoint's data_source block for a resolved source, or for none."""
    if source is None:
        return {"connected": False, "kind": None, "label": LABELS[None], "tables": []}
    kind = kind_of(source)
    try:
        if hasattr(source, "ping"):
            source.ping()
        else:
            source.sql("SELECT 1")
        tables = source.tables() if hasattr(source, "tables") else []
    except Exception:
        return {"connected": False, "kind": kind, "label": LABELS.get(kind, LABELS[None]) + " (not answering)",
                "tables": []}
    return {"connected": True, "kind": kind, "label": LABELS.get(kind, LABELS[None]), "tables": list(tables)}
