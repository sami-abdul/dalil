"""Run eval/questions.jsonl through the retriever and report recall.

    uv run python scripts/eval_retrieval.py
    uv run python scripts/eval_retrieval.py --no-stem     # whole words, for comparison

For q* rows (questions the corpus should answer): whether any expected
document appears among the top 3 and top 8 passages, and recall@3 and
recall@8 over those rows. For e* rows (error scenarios): the top hits only.
Prints passage ids, documents and scores, never passage text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dalil.chunk import read_jsonl  # noqa: E402
from dalil.retrieve import HybridRetriever, Retriever  # noqa: E402

PASSAGES = ROOT / "build" / "passages.jsonl"
QUESTIONS = ROOT / "eval" / "questions.jsonl"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passages", type=Path, default=PASSAGES)
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--no-stem", action="store_true", help="index whole words, without stemming")
    ap.add_argument("--dense", action="store_true",
                    help="fuse BM25 with dense embeddings from DALIL_EMBED_URL (default 127.0.0.1:8791)")
    args = ap.parse_args(argv)
    if not args.passages.is_file():
        print(f"error: {args.passages} not found; run scripts/build_passages.py first", file=sys.stderr)
        return 1

    embedder = None
    if args.dense:
        from dalil.embed import HTTPEmbedder
        embedder = HTTPEmbedder()
        if not embedder.available():
            print("error: no embedding endpoint answers at " + embedder.url, file=sys.stderr)
            return 1
    searcher = HybridRetriever(Retriever.from_jsonl(args.passages, stem=not args.no_stem), embedder)
    questions = read_jsonl(args.questions)
    at3 = at8 = scored = 0
    misses3, misses8 = [], []
    for q in questions:
        hits = searcher.search(q["question"], k=max(args.k, 8))
        expected = set(q.get("expected_docs") or [])
        docs = [h.doc_id for h in hits]
        line = f"{q['id']}"
        if q["id"].startswith("q"):
            scored += 1
            in3 = bool(expected & set(docs[:3]))
            in8 = bool(expected & set(docs[:8]))
            at3 += in3
            at8 += in8
            if not in3:
                misses3.append(q["id"])
            if not in8:
                misses8.append(q["id"])
            first = next((r for r, d in enumerate(docs, 1) if d in expected), None)
            line += f"  top3={'hit' if in3 else 'miss'} top8={'hit' if in8 else 'miss'} first_expected_rank={first}"
        else:
            line += f"  expected_docs={sorted(expected) or '[]'}"
        print(line)
        for h in hits[:3]:
            mark = "*" if h.doc_id in expected else " "
            print(f"   {mark} {h.score:7.3f}  {h.passage_id}")
    if scored:
        print(f"summary: questions={scored} recall@3={at3}/{scored} ({at3 / scored:.0%}) "
              f"recall@8={at8}/{scored} ({at8 / scored:.0%}) "
              f"misses@3={','.join(misses3) or 'none'} misses@8={','.join(misses8) or 'none'} "
              f"dense={args.dense} stem={'off' if args.no_stem else 'on'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
