"""After a reviewer rejects an answer, is the similar follow-up answered from other passages?

    DALIL_EMBED_URL=http://127.0.0.1:8792/v1 uv run python scripts/measure_rejection_followup.py [--runs 10] [--language en]

Each run starts with an empty review memory, asks scenario S7's first question,
rejects the answer with S7's note, then asks the follow-up. A run passes when
the follow-up goes out with at least one kept claim and none of its claims
cites a passage the rejected answer cited. A refusal counts as a miss here:
the passages that answer the question are not all gone, so declining is not
the right answer. Reports ids and counts only.

To replay the state of the 25 November recording, where the rejected answer
had cited four passages and the follow-up was declined:

    --reject uae-employment-contracts:p1:3 uae-employment-contracts:p1:4 \
             uae-terminating-employment-contracts:p1:4 uae-terminating-employment-contracts:p1:6
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_itu_scenarios import QUESTIONS, LocalMemory  # noqa: E402

from dalil.draft import ExtractiveDrafter, FallbackDrafter, LLMDrafter  # noqa: E402
from dalil.pipeline import Pipeline, cited_passages  # noqa: E402
from dalil.store import Store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--language", choices=sorted(QUESTIONS), default="en")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--reject", nargs="+", metavar="PASSAGE_ID",
                    help="reject exactly these passages instead of the ones the first answer cites, to replay a "
                         "recorded state; the first question is still asked and stored")
    args = ap.parse_args()
    q = QUESTIONS[args.language]
    drafter = FallbackDrafter(LLMDrafter(), ExtractiveDrafter())
    results = []
    for run in range(1, args.runs + 1):
        store = Store(Path(tempfile.mkdtemp()) / "s7.db")
        memory = LocalMemory()
        pipeline = Pipeline.from_build(store, drafter=drafter, memory=memory)
        first = pipeline.answer(q["S7.1"])
        store.review(first.id, "reject", "measure", q["S7.note"])
        rejected = args.reject or cited_passages(store.get(first.id)["check"])
        memory.remember("reject", first.id, first.question, "measure", rejected, note=q["S7.note"])
        second = pipeline.answer(q["S7.2"])
        cited = cited_passages(second.check)
        ok = bool(rejected) and bool(cited) and not set(cited) & set(rejected)
        results.append({"run": run, "first_cited": rejected, "second_cited": cited,
                        "second_status": second.check.get("status"), "pass": ok,
                        "second_kinds": [s["kind"] for s in second.check.get("sentences", []) if s["kept"]]})
        print(f"run {run}: {'PASS' if ok else 'MISS'} first cited {len(rejected)}, second cited {len(cited)} "
              f"status={second.check.get('status')} kept={results[-1]['second_kinds']}")
    passed = sum(r["pass"] for r in results)
    print(f"summary: {passed}/{len(results)} follow-ups answered from other passages")
    if args.out:
        args.out.write_text(json.dumps({"passed": passed, "runs": results}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
