"""Measure the model drafter on the test questions: speed, and how much survives the checker.

    uv run python scripts/measure_drafter.py [--ids q01,q03] [--url http://127.0.0.1:8790/v1]

Prints counts and timings only, never document text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dalil.draft import LLMDrafter  # noqa: E402
from dalil.opendata import match_rules, run_rules  # noqa: E402
from dalil.pipeline import FIXED_UNCITED  # noqa: E402
from dalil.retrieve import make_retriever  # noqa: E402
from dalil.verify import Verifier, load_passages  # noqa: E402

PASSAGES = ROOT / "build" / "passages.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", default="q01,q02,q03,q04,q05,q06,q07,q08,q09,q10")
    ap.add_argument("--url")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    want = set(args.ids.split(","))
    rows = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text().splitlines() if line.strip()]
    retriever, passages = make_retriever(PASSAGES), load_passages(PASSAGES)
    drafter = LLMDrafter(url=args.url)
    totals = {"claims": 0, "kept": 0, "seconds": 0.0}
    for row in rows:
        if row["id"] not in want:
            continue
        hits = retriever.search(row["question"], args.k)
        data = run_rules(match_rules(row["question"]), None, "no data source in this measurement")
        t0 = time.monotonic()
        try:
            draft = drafter.draft(row["question"], hits, data)
        except Exception as e:
            print(f"{row['id']}  error {type(e).__name__}: {str(e)[:120]}")
            continue
        dt = time.monotonic() - t0
        check = Verifier(passages, allowed_sql=[d.sql for d in data], allowed_uncited=FIXED_UNCITED).check_answer(draft)
        claims = [s for s in check.sentences if s.kind == "claim"]
        kept = sum(s.kept for s in claims)
        cited = {c.doc_id for s in claims if s.kept for c in s.citations if c.doc_id}
        hit_expected = bool(cited & set(row.get("expected_docs") or []))
        u = drafter.last_usage
        reasons = sorted({r for s in claims for r in s.reasons})
        print(f"{row['id']}  {dt:5.1f}s  status={check.status:15s} claims={len(claims)} kept={kept}  "
              f"expected_doc_cited={hit_expected}  prompt={u.get('prompt_n')} tok @ {u.get('prompt_per_second', 0):.0f}/s  "
              f"gen={u.get('predicted_n')} tok @ {u.get('predicted_per_second', 0):.1f}/s  cut_reasons={reasons}")
        totals["claims"] += len(claims)
        totals["kept"] += kept
        totals["seconds"] += dt
    print(f"total: {totals['kept']}/{totals['claims']} claims kept, {totals['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
