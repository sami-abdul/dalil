"""Measure the checker's cross-language test on labelled Arabic claims.

    DALIL_EMBED_URL=http://127.0.0.1:8792/v1 uv run python scripts/calibrate_cross_language.py

eval/cross-language-pairs.jsonl holds Arabic claims, each pointing at one
sentence of an English passage by passage id and sentence number, labelled
true when the claim says what the sentence says and false when it does not.
The file holds no document text: the sentence is read from build/passages.jsonl.
For each pair this prints the score the checker uses (the best similarity
between the claim and a clause of the sentence), then how many pairs each
threshold gets right, and the threshold in dalil/verify.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dalil.draft import units  # noqa: E402
from dalil.embed import HTTPEmbedder  # noqa: E402
from dalil.retrieve import HybridRetriever, Retriever  # noqa: E402
from dalil.verify import CROSS_LANGUAGE_MIN, RENDERING_MIN, clauses, load_passages  # noqa: E402


def main() -> int:
    passages = load_passages(ROOT / "build" / "passages.jsonl")
    embedder = HTTPEmbedder()
    if not embedder.available():
        print("the embedding server is not answering; set DALIL_EMBED_URL", file=sys.stderr)
        return 1
    sim = HybridRetriever(Retriever([]), embedder).similarities
    rows = [json.loads(line) for line in (ROOT / "eval" / "cross-language-pairs.jsonl").open(encoding="utf-8")]
    scored = []
    for r in rows:
        p = passages[r["passage_id"]]
        sentence = units(p["text"], p.get("kind") == "pdf")[r["unit"]]
        score = max(sim(r["claim"], clauses(sentence)) or [0.0])
        scored.append((score, r["match"]))
        print(f"{score:.2f}  {'match' if r['match'] else 'wrong'}  {r['passage_id']}#{r['unit']}")
    print()
    for t in [x / 100 for x in range(40, 86, 5)]:
        right = sum((s >= t) == m for s, m in scored)
        passed_wrong = sum(s >= t and not m for s, m in scored)
        cut_right = sum(s < t and m for s, m in scored)
        mark = "  <- dalil/verify.py" if abs(t - CROSS_LANGUAGE_MIN) < 1e-9 else ""
        print(f"threshold {t:.2f}: {right}/{len(scored)} right, {passed_wrong} wrong claims kept, "
              f"{cut_right} right claims cut{mark}")
    # The check used when the writer gives an English version of its claim:
    # the claim against that one English sentence. A right row carries a
    # faithful English version; a wrong row is paired with the right row's.
    english = {(r["passage_id"], r["unit"]): r["english"] for r in rows if r.get("english")}
    pairs = [(max(sim(r["claim"], [r.get("english") or english[(r["passage_id"], r["unit"])]]) or [0.0]), r["match"])
             for r in rows if r.get("english") or (r["passage_id"], r["unit"]) in english]
    print("\nclaim against its English version:")
    for t in [x / 100 for x in range(40, 86, 5)]:
        right = sum((s >= t) == m for s, m in pairs)
        kept_wrong = sum(s >= t and not m for s, m in pairs)
        cut_right = sum(s < t and m for s, m in pairs)
        mark = "  <- dalil/verify.py" if abs(t - RENDERING_MIN) < 1e-9 else ""
        print(f"threshold {t:.2f}: {right}/{len(pairs)} right, {kept_wrong} wrong claims kept, "
              f"{cut_right} right claims cut{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
