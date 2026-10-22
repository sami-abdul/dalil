"""Read-only check of Dalil's data in the local Zetaris.

    uv run python scripts/check_zetaris.py

Runs SELECT 1, lists the dalil tables, counts rows per table, checks that
every page belongs to a document, and runs one sample join of documents and
pages by doc_id. When dalil.passages exists it also proves the citations can
be rerun: inside Zetaris, every passage's text must equal
substring(page.text, start + 1, end - start) (SQL counts from 1, the offsets
from 0), and one sample passage's slice is hashed in SQL and compared with
the local build/passages.jsonl. Prints counts, ids, hashes and document
metadata only, never page text or credentials. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dalil import zetaris as z  # noqa: E402

PASSAGES = ROOT / "build" / "passages.jsonl"


# Page text from a passage's offsets. Offsets are 0-based and end-exclusive
# (Python slicing); Spark's substring is 1-based and takes a length.
SLICE_SQL = "substring(p.text, s.`start` + 1, s.`end` - s.`start`)"


def passage_proof_sql(db: str, passage_id: str) -> str:
    """The SQL a reviewer reruns to see that a cited passage is the page's own text."""
    db = z.check_name(db)
    return (
        f"SELECT s.passage_id, s.doc_id, s.page, s.`start`, s.`end`, s.char_count, "
        f"{SLICE_SQL} = s.text AS exact, sha2({SLICE_SQL}, 256) AS slice_sha256 "
        f"FROM {db}.passages s JOIN {db}.pages p ON s.doc_id = p.doc_id AND s.page = p.page "
        f"WHERE s.passage_id = {z.quote_string(passage_id)}"
    )


def check_passages(client: z.Client, db: str) -> None:
    stats = client.sql(
        f"SELECT COUNT(*) AS passages, COUNT(DISTINCT s.doc_id) AS docs, "
        f"SUM(CASE WHEN p.doc_id IS NULL THEN 1 ELSE 0 END) AS orphan_passages, "
        f"SUM(CASE WHEN p.doc_id IS NOT NULL AND NOT ({SLICE_SQL} <=> s.text) THEN 1 ELSE 0 END) AS slice_mismatches, "
        f"SUM(CASE WHEN length(s.text) <> s.char_count THEN 1 ELSE 0 END) AS length_mismatches "
        f"FROM {db}.passages s LEFT JOIN {db}.pages p ON s.doc_id = p.doc_id AND s.page = p.page"
    ).dicts()[0]
    print("passages: " + "  ".join(f"{k}={v}" for k, v in stats.items()))
    for key in ("orphan_passages", "slice_mismatches", "length_mismatches"):
        if int(stats[key] or 0):
            raise z.ZetarisError(f"{stats[key]} passages fail the {key.replace('_', ' ')} check")
    if not int(stats["passages"] or 0):
        raise z.ZetarisError("dalil.passages is empty")

    # A law article in the middle of a page, if there is one, else any passage.
    pick = client.sql(
        f"SELECT passage_id FROM {db}.passages WHERE heading LIKE 'Article (%' AND `start` > 0 "
        f"ORDER BY doc_id, page, `start` LIMIT 1"
    ).rows or client.sql(f"SELECT passage_id FROM {db}.passages ORDER BY passage_id LIMIT 1").rows
    pid = str(pick[0][0])
    proof_sql = passage_proof_sql(db, pid)
    print(f"passage_proof_sql: {proof_sql}")
    rows = client.sql(proof_sql).dicts()
    if len(rows) != 1:
        raise z.ZetarisError(f"The passage proof for {pid} returned {len(rows)} rows")
    row = rows[0]
    print("  " + json.dumps(row, ensure_ascii=False))
    if str(row["exact"]).lower() != "true":
        raise z.ZetarisError(f"{pid}: substring of the page text does not equal the passage text")
    if PASSAGES.is_file():
        with PASSAGES.open(encoding="utf-8") as fh:
            local = next((r for r in map(json.loads, filter(str.strip, fh)) if r["passage_id"] == pid), None)
        if local is None:
            raise z.ZetarisError(f"{pid} is not in {PASSAGES.name}; reload with scripts/load_zetaris.py")
        local_sha = hashlib.sha256(local["text"].encode("utf-8")).hexdigest()
        same = local_sha == row["slice_sha256"]
        print(f"passage_proof_local: {PASSAGES.name} sha256 {'matches' if same else 'DIFFERS'} "
              f"(start={local['start']} end={local['end']})")
        if not same:
            raise z.ZetarisError(f"{pid}: the Zetaris slice differs from {PASSAGES.name}")


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
        if "passages" in tables:
            check_passages(client, db)
        else:
            print("passages: table missing; run scripts/build_passages.py and scripts/load_zetaris.py")
    except z.ZetarisError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
