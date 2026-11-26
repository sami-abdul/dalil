"""Where an answer's time goes: seconds per stage, and the model's own counts.

    DALIL_EMBED_URL=http://127.0.0.1:8792/v1 uv run python scripts/measure_speed.py [--runs 2] [--ids c01 c07]

Asks the given questions from eval/citations.jsonl through the full pipeline,
in order, and prints for each answer the seconds spent screening, recalling
past reviews, translating (Arabic only), searching, drafting and checking,
read from the audit trail, plus the model server's prompt tokens, output
tokens and their rates. Run it with nothing else using the model servers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dalil.draft import ExtractiveDrafter, FallbackDrafter, LLMDrafter  # noqa: E402
from dalil.pipeline import Pipeline  # noqa: E402
from dalil.store import Store  # noqa: E402

STAGES = ("screened", "recalled", "translated", "retrieved", "drafted", "verified")


def stage_seconds(audit: list[dict], started: datetime) -> dict[str, float]:
    """Seconds from the end of the previous stage to the end of each one."""
    out, last = {}, started
    for e in audit:
        at = datetime.fromisoformat(e["at"].replace("Z", "+00:00"))
        if e["kind"] in STAGES:
            out[e["kind"]] = round((at - last).total_seconds(), 2)
        last = at
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--ids", nargs="+", default=["c01", "c07", "c14", "c25", "c38"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = {json.loads(line)["id"]: json.loads(line) for line in (ROOT / "eval" / "citations.jsonl").open()}
    llm = LLMDrafter()
    store = Store(Path(tempfile.mkdtemp()) / "speed.db")
    pipeline = Pipeline.from_build(store, drafter=FallbackDrafter(llm, ExtractiveDrafter()), memory=None)
    results = []
    for run in range(1, args.runs + 1):
        for qid in args.ids:
            llm.last_usage = {}
            t0 = datetime.now().astimezone()
            rec = pipeline.answer(rows[qid]["question"])
            total = (datetime.now().astimezone() - t0).total_seconds()
            u = llm.last_usage
            res = {"id": qid, "run": run, "seconds": round(total, 2),
                   "stages": stage_seconds(store.get(rec.id)["audit"], t0),
                   "prompt_tokens": u.get("prompt_n"), "prompt_per_second": u.get("prompt_per_second"),
                   "output_tokens": u.get("predicted_n"), "output_per_second": u.get("predicted_per_second"),
                   "answer": rec.check.get("answer", "")}
            results.append(res)
            print(json.dumps({k: v for k, v in res.items() if k != "answer"}))

    def med(key, sub=None):
        vals = [(r[key].get(sub) if sub else r[key]) for r in results]
        vals = [v for v in vals if v is not None]
        return round(statistics.median(vals), 2) if vals else None
    summary = {"answers": len(results), "seconds": med("seconds"),
               **{f"{s}_seconds": med("stages", s) for s in STAGES},
               "prompt_tokens": med("prompt_tokens"), "prompt_per_second": med("prompt_per_second"),
               "output_tokens": med("output_tokens"), "output_per_second": med("output_per_second")}
    print(json.dumps(summary))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
