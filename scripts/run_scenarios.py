"""Run a scenario file through the real pipeline, each question N times.

    DALIL_EMBED_URL=http://127.0.0.1:8792/v1 uv run python scripts/run_scenarios.py eval/scenarios/open-data.json
    uv run python scripts/run_scenarios.py eval/scenarios/open-data.json --repeat 3 --extractive
    uv run python scripts/run_scenarios.py kb/uk/scenarios.json --kb uk

The file format and its checks are in dalil/scenarios.py. Each scenario holds
the question, the outcome it must show in words ("expect"), and the checks
that test exactly that, all written before the run. The ITU error scenarios
stay in scripts/run_itu_scenarios.py; this runner is for groups beside them,
such as the open-data numbers questions.

Each question goes through the same pipeline the reviewer screen uses: the
model drafting with the extractive drafter as fallback, retrieval, the
open-data source DALIL_OPEN_DATA picks, and the checker. Review memory is a
fresh in-process stand-in, so no live ledger is touched and no run sees an
earlier run's reviews. Results go to build/scenarios/<name>/ (git-ignored):
the store with every audit event, and report.json. The console prints the
status of each run and the sentences backed by open data, never passage text.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dalil import kb  # noqa: E402
from dalil import scenarios as sc  # noqa: E402
from dalil.draft import ExtractiveDrafter, FallbackDrafter, LLMDrafter  # noqa: E402
from dalil.pipeline import Pipeline  # noqa: E402
from dalil.store import Store  # noqa: E402


class LocalMemory:
    """An empty review memory with the H-MEM interface."""

    def recall(self, question: str, top: int = 50) -> dict:
        return {"memories": []}


def data_sentences(check: dict) -> list[str]:
    return [s["text"] for s in check.get("sentences", []) if s["kept"]
            and any(c.get("sql") for c in s.get("citations", []))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", type=Path, help="a scenario file, JSON")
    ap.add_argument("--repeat", type=int, help="runs per question (default: the file's own)")
    ap.add_argument("--extractive", action="store_true", help="draft without the model")
    ap.add_argument("--kb", help="knowledge-base profile under kb/ (default: DALIL_KB, else uae)")
    ap.add_argument("--only", action="append", default=[], metavar="ID", help="run only these scenario ids")
    args = ap.parse_args(argv)

    spec = sc.load(args.file)
    repeat = args.repeat or spec.repeat
    out = ROOT / "build" / "scenarios" / spec.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    store = Store(out / "scenarios.db")
    drafter = ExtractiveDrafter() if args.extractive else FallbackDrafter(LLMDrafter(), ExtractiveDrafter())
    profile = kb.load(args.kb or kb.active_id())
    if not args.extractive:
        drafter.primary.audience = profile.audience
    pipeline = Pipeline.from_profile(profile, store, drafter=drafter, memory=LocalMemory())
    source, why = pipeline.data_source()
    print(f"{spec.name}: knowledge base {profile.id} ({len(pipeline.passages)} passages); "
          f"{len(spec.scenarios)} scenarios x {repeat} runs; open data: "
          f"{getattr(source, 'label', None) or why}")

    report, summary = [], []
    for s in spec.scenarios:
        if args.only and s.id not in args.only:
            continue
        print(f"{s.id}  {s.title}\n    expect: {s.expect}")
        passed = 0
        for n in range(1, repeat + 1):
            started = time.monotonic()
            rec = pipeline.answer(s.question)
            took = time.monotonic() - started
            row = store.get(rec.id)
            results = sc.evaluate(s, rec, row, fixed=pipeline.screener.fixed_texts)
            ok = all(results.values())
            passed += ok
            drafted = next((e["detail"] for e in row["audit"] if e["kind"] == "drafted"), {})
            failed = [k for k, v in results.items() if not v]
            print(f"    run {n:2d}: {'PASS' if ok else 'FAIL'}  {took:5.1f}s status={rec.status} "
                  f"check={rec.check.get('status')} drafter={drafted.get('drafter')}"
                  + (f" failed={failed}" if failed else ""))
            for text in data_sentences(rec.check):
                print(f"            data: {text}")
            report.append({"scenario": s.id, "run": n, "pass": ok, "checks": results, "question_id": rec.id,
                           "status": rec.status, "check_status": rec.check.get("status"),
                           "drafter": drafted.get("drafter"), "fallback_reason": drafted.get("fallback_reason"),
                           "seconds": round(took, 1), "data_sentences": data_sentences(rec.check),
                           "cut_reasons": sorted({r for x in rec.check.get("sentences", []) if not x["kept"]
                                                  for r in x["reasons"]})})
        summary.append({"scenario": s.id, "title": s.title, "expect": s.expect, "passed": passed, "runs": repeat})
        print(f"    {s.id}: {passed} of {repeat}")
    (out / "report.json").write_text(json.dumps({"file": str(args.file), "summary": summary, "runs": report},
                                                indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    total = sum(x["passed"] for x in summary)
    runs = sum(x["runs"] for x in summary)
    print("summary: " + ", ".join(f"{x['scenario']} {x['passed']}/{x['runs']}" for x in summary)
          + f"; {total}/{runs} runs behaved as expected; evidence in {out}")
    return 0 if total == runs else 1


if __name__ == "__main__":
    sys.exit(main())
