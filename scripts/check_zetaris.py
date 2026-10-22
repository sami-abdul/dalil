"""Read-only check of Dalil's data in the local Zetaris.

    uv run python scripts/check_zetaris.py

Runs SELECT 1, lists the dalil tables, counts rows per table, checks that
every page belongs to a document, and runs one sample join of documents and
pages by doc_id. Prints counts and document metadata only, never page text
or credentials. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dalil import zetaris as z  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database", default="dalil")
    ap.add_argument("--env-file")
    args = ap.parse_args(argv)
    db = z.check_name(args.database)
    try:
        client = z.Client(z.load_config(args.env_file))
        print(f"select_1: {client.sql('SELECT 1 AS one').scalar()}")
        version = client.sql("SHOW SERVER VERSION").scalar()
        print(f"server_version: {version}  org_id: {client.org_id}")
        if db.upper() not in client.lightning_databases():
            raise z.ZetarisError(f"Lightning database {db.upper()} is missing; run scripts/load_zetaris.py")
        tables = client.lightning_tables(db)
        for name, t in sorted(tables.items()):
            n = client.sql(f"SELECT COUNT(*) AS n FROM {db}.{z.check_name(name)}").scalar()
            print(f"table {db}.{name}: rows={n} format={t.get('format')} path={t.get('source_table')}")
        for needed in ("documents", "pages"):
            if needed not in tables:
                raise z.ZetarisError(f"{db}.{needed} is missing")

        stats = client.sql(
            f"SELECT COUNT(DISTINCT p.doc_id) AS docs_with_pages, "
            f"SUM(CASE WHEN d.doc_id IS NULL THEN 1 ELSE 0 END) AS orphan_pages, "
            f"SUM(p.char_count) AS chars "
            f"FROM {db}.pages p LEFT JOIN {db}.documents d ON p.doc_id = d.doc_id"
        ).dicts()[0]
        print("pages: " + "  ".join(f"{k}={v}" for k, v in stats.items()))
        sources = client.sql(f"SELECT corpus_source, COUNT(*) AS n FROM {db}.documents GROUP BY corpus_source").rows
        print("corpus_source: " + ", ".join(f"{s}={n}" for s, n in sources))
        if int(stats["orphan_pages"] or 0):
            raise z.ZetarisError(f"{stats['orphan_pages']} pages have no matching document")

        join_sql = (
            f"SELECT d.doc_id, d.language, d.kind, d.page_count, p.page, p.char_count "
            f"FROM {db}.documents d JOIN {db}.pages p ON d.doc_id = p.doc_id "
            f"ORDER BY d.doc_id, p.page LIMIT 3"
        )
        print(f"sample_join_sql: {join_sql}")
        sample = client.sql(join_sql).dicts()
        for row in sample:
            print("  " + json.dumps(row, ensure_ascii=False))
        if not sample:
            raise z.ZetarisError("The sample join returned no rows")

        first = sample[0]
        meta = client.sql(
            f"SELECT sha256, source_url FROM {db}.documents WHERE doc_id = {z.quote_string(first['doc_id'])}"
        ).dicts()[0]
        cite = z.citation(db, first["doc_id"], int(first["page"]), meta["sha256"], meta["source_url"], version)
        rerun = client.sql(cite["sql"]).rows
        print(f"citation_rerun_rows: {len(rerun)}")
        print("citation: " + json.dumps(cite, ensure_ascii=False))
        if len(rerun) != 1:
            raise z.ZetarisError("The citation lookup did not return exactly one page")
    except z.ZetarisError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
