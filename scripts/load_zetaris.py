"""Load Dalil's documents into the local Zetaris, idempotently.

    uv run python scripts/load_zetaris.py                 # build/documents.jsonl
    uv run python scripts/load_zetaris.py --dry-run       # show the plan, send nothing
    uv run python scripts/load_zetaris.py --no-open       # documents and pages only
    uv run python scripts/load_zetaris.py --csv rates=data/open/rates.csv   # one more CSV

Creates the Lightning database `dalil` (if missing) with two tables:

    dalil.documents  one row per document: manifest metadata plus page_count
    dalil.pages      one row per page: doc_id, page, text, char_count

plus one table per row of data/open/tables.csv, the committed open data.
`--csv` adds tables to those; `--no-open` skips the committed ones.

Mechanism, and why: each table is a Lightning FILESTORE table over a file
staged inside the driver's `customer-data` volume, which is the bundle's own
local platform storage. The loader streams the file in with `docker exec`, so
nothing about the stack changes: no new service, no new port, no compose edit.
The alternatives were worse here. A Postgres datasource would mean either
writing into Zetaris's own metastore database or running a second database.
The file is a JSON array, which keeps the loader on the standard library and
carries long multi-line Arabic and English text without CSV quoting trouble.
Open-data CSV or Parquet files register the same way (data/open/tables.csv,
`--csv`, or `register_file_table(..., fmt="PARQUET")`).

Re-running replaces the staged files. A table whose columns still match is
kept and its cache cleared; one whose columns changed is dropped and
recreated, because Lightning fixes a file table's schema at registration.
Every table's row count is checked against what was written: the JSON rows
for documents and pages, and the CSV records parsed locally for open data.

Document text is private: it stays under build/ (git-ignored) and inside the
local Zetaris volume. If build/documents.jsonl does not exist yet, a stand-in
is built from data/MANIFEST.csv with one placeholder page per document, no
real text, and documents.corpus_source says so.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dalil import zetaris as z  # noqa: E402

DATABASE = "dalil"
DESCRIPTION = "Dalil corpus - UAE government documents, metadata and page text"
DOCUMENTS = ROOT / "build" / "documents.jsonl"
MANIFEST = ROOT / "data" / "MANIFEST.csv"
OUT_DIR = ROOT / "build" / "zetaris"

DOC_FIELDS = [
    "doc_id", "file", "title", "publisher", "source_url", "retrieved_utc",
    "sha256", "language", "topic", "kind",
]
STANDIN = "standin-from-manifest"


def read_documents(path: Path) -> list[dict[str, Any]]:
    docs = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    docs.append(json.loads(line))
                except ValueError as e:
                    raise z.ZetarisError(f"{path}:{n}: not JSON ({e})") from None
    return docs


def standin_documents(manifest: Path) -> list[dict[str, Any]]:
    """Metadata-only documents from MANIFEST.csv. doc_id is the file name
    without its extension (inferred to match the corpus builder; the real
    documents.jsonl decides)."""
    docs = []
    with manifest.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            stem, _, ext = row["file"].rpartition(".")
            doc = {k: row.get(k) for k in DOC_FIELDS if k in row}
            doc.update(doc_id=stem or row["file"], kind=ext.lower() or None)
            doc["pages"] = [{"page": 1, "text": f"[stand-in page, no extracted text] {row.get('title', '')}"}]
            docs.append(doc)
    return docs


def build_rows(docs: Iterable[Mapping[str, Any]], corpus_source: str, loaded_utc: str
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split documents into the two tables' rows, with checks."""
    documents, pages, seen = [], [], set()
    for doc in docs:
        doc_id = doc.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise z.ZetarisError(f"Document without a doc_id: {doc.get('file')!r}")
        if doc_id in seen:
            raise z.ZetarisError(f"Duplicate doc_id: {doc_id}")
        seen.add(doc_id)
        doc_pages = doc.get("pages") or []
        page_numbers = set()
        chars = 0
        for p in doc_pages:
            try:
                number = int(p["page"])
            except (KeyError, TypeError, ValueError):
                raise z.ZetarisError(f"{doc_id}: page without a whole-number 'page'") from None
            if number in page_numbers:
                raise z.ZetarisError(f"{doc_id}: page {number} appears twice")
            page_numbers.add(number)
            text = p.get("text") or ""
            chars += len(text)
            pages.append({"doc_id": doc_id, "page": number, "text": text, "char_count": len(text)})
        row = {k: (None if doc.get(k) in ("", None) else str(doc[k])) for k in DOC_FIELDS}
        # Other scalar fields from the corpus builder (sha256_ok, is_terms_page, ...) pass through.
        for k, v in doc.items():
            if k not in row and k != "pages" and z._NAME.match(k) and (v is None or isinstance(v, (str, int, float, bool))):
                row[k] = v
        row.update(page_count=len(doc_pages), char_count=chars,
                   corpus_source=corpus_source, loaded_utc=loaded_utc)
        documents.append(row)
    if not documents:
        raise z.ZetarisError("No documents to load")
    # Every row carries every column, so the inferred schema does not depend on row order.
    columns = sorted({k for row in documents for k in row})
    documents = [{k: row.get(k) for k in columns} for row in documents]
    return documents, pages


def ensure_database(client: z.Client, database: str, description: str) -> str:
    if database.upper() in client.lightning_databases():
        return "exists"
    client.sql(z.create_database_sql(database, description))
    return "created"


def register_file_table(client: z.Client, container: str, database: str, table: str,
                        data: bytes, fmt: str, expected_columns: set[str] | None,
                        options: Mapping[str, str] | None = None, file_name: str | None = None) -> str:
    """Stage `data` in the driver volume and make `database.table` read it.

    With expected_columns None (CSV/Parquet open data) the table is always
    recreated, since its schema is whatever the file holds."""
    file_name = file_name or f"{z.check_name(table)}.{fmt.lower()}"
    path = z.stage_file(container, data, file_name)
    existing = client.lightning_tables(database).get(table.lower())
    if existing is not None:
        same_path = existing.get("source_table") == path and str(existing.get("format", "")).upper() == fmt.upper()
        columns = set(client.table_columns(database, table))
        if same_path and expected_columns is not None and columns == expected_columns:
            client.sql(f"UNCACHE LIGHTNING FILESTORE TABLE {z.check_name(table)} FROM {z.check_name(database)}")
            return "refreshed"
        client.sql(f"DROP LIGHTNING DATABASE TABLE {z.check_name(database)}.{z.check_name(table)}")
        action = "recreated"
    else:
        action = "created"
    client.sql(z.filestore_table_sql(database, table, path, fmt, options))
    return action


OPEN_TABLES = ROOT / "data" / "open" / "tables.csv"
CORPUS_TABLES = ("documents", "pages")
# Used by both the dry-run plan and the live registration, so they cannot drift.
CSV_OPTIONS = {"header": "true", "inferSchema": "true", "multiLine": "true"}


def default_open_tables(path: Path = OPEN_TABLES) -> list[str]:
    """TABLE=PATH specs for the committed open-data tables, used when no --csv is given."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [f"{row['table']}={path.parent / row['file']}" for row in csv.DictReader(fh)]


def csv_record_count(path: Path) -> int:
    """Records in a CSV with a header row, counted the way a CSV reader sees them."""
    with path.open(newline="", encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def open_table_specs(cli_specs: list[str], include_defaults: bool,
                     defaults_path: Path = OPEN_TABLES) -> list[tuple[str, Path, str]]:
    """(table, path, origin) for every open-data table to register.

    Committed defaults come first, then --csv specs. Names are checked, must be
    unique and must not collide with the corpus tables.
    """
    specs = [(spec, str(defaults_path)) for spec in (default_open_tables(defaults_path) if include_defaults else [])]
    specs += [(spec, "--csv") for spec in cli_specs]
    out, seen = [], set(CORPUS_TABLES)
    for spec, origin in specs:
        table, sep, path = spec.partition("=")
        if not sep or not Path(path).is_file():
            raise z.ZetarisError(f"{origin}: wants TABLE=PATH to an existing file, got {spec!r}")
        table = z.check_name(table)
        if table.lower() in seen:
            raise z.ZetarisError(f"{origin}: table name {table!r} is already used")
        seen.add(table.lower())
        out.append((table, Path(path), origin))
    return out


def count(client: z.Client, database: str, table: str) -> int:
    return int(client.sql(f"SELECT COUNT(*) AS n FROM {z.check_name(database)}.{z.check_name(table)}").scalar())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--documents", type=Path, default=DOCUMENTS, help="documents.jsonl from the corpus builder")
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help="used for a stand-in when --documents is missing")
    ap.add_argument("--database", default=DATABASE)
    ap.add_argument("--env-file", help="Zetaris credentials file (default: DALIL_ZETARIS_ENV or ~/.config/lfg/credentials/zetaris-local.env)")
    ap.add_argument("--csv", action="append", default=[], metavar="TABLE=PATH",
                    help="register one more open-data CSV with a header row, added to data/open/tables.csv (repeatable)")
    ap.add_argument("--no-open", action="store_true", help="skip the committed open-data tables in data/open/tables.csv")
    ap.add_argument("--dry-run", action="store_true", help="build the files and print the plan; send nothing")
    args = ap.parse_args(argv)

    try:
        z.check_name(args.database)
        if args.documents.is_file():
            docs, source = read_documents(args.documents), args.documents.name
        else:
            print(f"warning: {args.documents} not found; loading a metadata-only stand-in from {args.manifest}",
                  file=sys.stderr)
            docs, source = standin_documents(args.manifest), STANDIN
        loaded = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        documents, pages = build_rows(docs, source, loaded)
        extra = open_table_specs(args.csv, include_defaults=not args.no_open)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        payloads = {"documents": z.json_array_bytes(documents), "pages": z.json_array_bytes(pages)}
        for name, data in payloads.items():
            (OUT_DIR / f"{name}.json").write_bytes(data)  # local copy under git-ignored build/
        print(f"source={source} documents={len(documents)} pages={len(pages)} files={OUT_DIR.relative_to(ROOT)}/")

        if args.dry_run:
            print(z.create_database_sql(args.database, DESCRIPTION) + "   (if missing)")
            for name in payloads:
                print(z.filestore_table_sql(args.database, name, f"file://{z.DRIVER_STAGE_ROOT / f'{name}.json'}", "JSON"))
            for table, path, _ in extra:
                print(z.filestore_table_sql(args.database, table, f"file://{z.DRIVER_STAGE_ROOT / f'{table}.csv'}", "CSV", CSV_OPTIONS))
            print("dry run: nothing sent")
            return 0

        client = z.Client(z.load_config(args.env_file))
        container = z.find_driver_container()
        print(f"database {args.database.upper()}: {ensure_database(client, args.database, DESCRIPTION)}")
        expected = {"documents": len(documents), "pages": len(pages)}
        columns = {"documents": set(documents[0]), "pages": {"doc_id", "page", "text", "char_count"}}
        for name, data in payloads.items():
            action = register_file_table(client, container, args.database, name, data, "JSON", columns[name],
                                         file_name=f"{name}.json")
            got = count(client, args.database, name)
            print(f"table {args.database}.{name}: {action}, rows={got}")
            if got != expected[name]:
                raise z.ZetarisError(f"{args.database}.{name} has {got} rows, expected {expected[name]}")
        failed = []
        for table, path, _ in extra:
            try:
                action = register_file_table(client, container, args.database, table, path.read_bytes(), "CSV", None,
                                             CSV_OPTIONS, file_name=f"{table}.csv")
                got, want = count(client, args.database, table), csv_record_count(path)
                print(f"table {args.database}.{table}: {action}, rows={got}")
                if got != want:
                    raise z.ZetarisError(f"{args.database}.{table} has {got} rows, expected {want} from {path.name}")
            except (z.ZetarisError, OSError) as e:
                print(f"error: {e}", file=sys.stderr)
                failed.append(table)
        if failed:
            print(f"error: open-data tables failed: {', '.join(failed)}; documents and pages are loaded", file=sys.stderr)
            return 1
    except (z.ZetarisError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
