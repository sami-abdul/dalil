"""Split build/documents.jsonl into citable passages: build/passages.jsonl.

    uv run python scripts/build_passages.py
    uv run python scripts/build_passages.py --documents other.jsonl --out other-passages.jsonl

Terms and licence pages are left out. Every passage is checked against its
page before it is written: page_text[start:end] must equal the passage text.
Prints counts and lengths only, never document text. build/ is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dalil.chunk import chunk_document, read_jsonl, verify_passage  # noqa: E402

DOCUMENTS = ROOT / "build" / "documents.jsonl"
PASSAGES = ROOT / "build" / "passages.jsonl"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--documents", type=Path, default=DOCUMENTS)
    ap.add_argument("--out", type=Path, default=PASSAGES)
    args = ap.parse_args(argv)
    if not args.documents.is_file():
        print(f"error: {args.documents} not found; run scripts/extract_corpus.py first", file=sys.stderr)
        return 1
    try:
        docs = read_jsonl(args.documents)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    passages, empty, terms, seen = [], [], 0, set()
    for doc in docs:
        if doc.get("is_terms_page"):
            terms += 1
            continue
        pages = {int(p["page"]): p.get("text") or "" for p in doc.get("pages") or []}
        doc_passages = chunk_document(doc)
        for p in doc_passages:
            if not verify_passage(p, pages[p["page"]]):
                print(f"error: {p['passage_id']} does not match its page slice", file=sys.stderr)
                return 1
            if p["passage_id"] in seen:
                print(f"error: duplicate passage_id {p['passage_id']}", file=sys.stderr)
                return 1
            seen.add(p["passage_id"])
        if not doc_passages:
            empty.append(doc["doc_id"])
        passages.extend(doc_passages)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for p in passages:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    tmp.replace(args.out)

    lengths = [p["char_count"] for p in passages]
    by_kind = Counter(p["kind"] for p in passages)
    by_lang = Counter(p["language"] for p in passages)
    with_heading = sum(1 for p in passages if p["heading"])
    articles = sum(1 for p in passages if (p["heading"] or "").startswith(("Article (", "المادة (")))
    print(f"documents={len(docs)} terms_skipped={terms} passages={len(passages)} out={args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    print("per_kind: " + ", ".join(f"{k}={n}" for k, n in sorted(by_kind.items())))
    print("per_language: " + ", ".join(f"{k}={n}" for k, n in sorted(by_lang.items())))
    if lengths:
        print(f"length: mean={statistics.mean(lengths):.0f} median={statistics.median(lengths):.0f} "
              f"min={min(lengths)} max={max(lengths)} over_1200={sum(n > 1200 for n in lengths)} "
              f"under_300={sum(n < 300 for n in lengths)}")
    print(f"headings: with_heading={with_heading} article_headings={articles}")
    print(f"docs_with_zero_passages={len(empty)}" + (f" ({', '.join(empty)})" if empty else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
