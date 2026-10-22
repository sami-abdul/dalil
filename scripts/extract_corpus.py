"""Extract per-page text from the private corpus into build/documents.jsonl.

    uv run python scripts/extract_corpus.py [--corpus-dir DIR] [--manifest CSV] [--out JSONL]

The corpus directory defaults to $DALIL_CORPUS_DIR, else ./corpus. The output
holds document text, so it goes under build/, which git ignores.

Exit status: 0 when every manifest row was extracted, even if some hashes no
longer match; 1 when a document is missing or cannot be read (the others are
still written); 2 when the manifest itself cannot be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dalil.corpus import DEFAULT_MANIFEST, CorpusError, iter_documents, load_manifest, resolve_corpus_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus-dir", type=Path, default=None, help="default: $DALIL_CORPUS_DIR, else ./corpus")
    parser.add_argument("--out", type=Path, default=Path("build/documents.jsonl"))
    args = parser.parse_args(argv)

    try:
        rows = load_manifest(args.manifest)
    except CorpusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    corpus_dir = resolve_corpus_dir(args.corpus_dir)
    if not corpus_dir.is_dir():
        print(f"error: corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    records, errors = [], []
    with tmp.open("w", encoding="utf-8") as fh:
        for row, record, error in iter_documents(rows, corpus_dir):
            if error is not None:
                errors.append((row["file"], error))
                continue
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
    tmp.replace(args.out)

    width = max([len(r["doc_id"]) for r in records] + [6])
    print(f"{'doc_id':<{width}}  kind  pages    chars  sha256_ok")
    for r in records:
        flag = "yes" if r["sha256_ok"] else "NO"
        print(f"{r['doc_id']:<{width}}  {r['kind']:<4}  {len(r['pages']):>5}  {r['char_count']:>7}  {flag}")

    mismatched = [r["doc_id"] for r in records if not r["sha256_ok"]]
    empty = [r["doc_id"] for r in records if r["char_count"] == 0]
    terms = sum(1 for r in records if r["is_terms_page"])
    print()
    print(f"documents: {len(records)} of {len(rows)} ({terms} terms pages)")
    print(f"pages: {sum(len(r['pages']) for r in records)}")
    print(f"chars: {sum(r['char_count'] for r in records)}")
    print(f"hash mismatches: {len(mismatched)}{': ' + ', '.join(mismatched) if mismatched else ''}")
    print(f"empty documents: {len(empty)}{': ' + ', '.join(empty) if empty else ''}")
    print(f"wrote {args.out}")
    for name, error in errors:
        print(f"error: {name}: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
