"""Does each answer cite the right document? A rerunnable check over many questions.

    DALIL_EMBED_URL=http://127.0.0.1:8792/v1 uv run python scripts/eval_citations.py [--runs 1] [--out FILE]

eval/citations.jsonl lists questions across the covered policies, each with
the documents a correct answer may cite. Every question goes through the full
pipeline (Nemotron drafting unless --extractive, dense retrieval, the checker)
into a throwaway store, with no review memory.

Per answer it records which documents each kept claim cites, and from that:

* claim precision: the share of kept claims whose every cited document is an
  expected one. A claim citing another document, such as a Dubai law in an
  answer about a federal law, counts as wrong.
* clean answers: answers whose kept claims are all right.
* found: answers with at least one kept claim citing an expected document.
* answered: answers with at least one kept claim (not refused, not held).

The report holds ids, statuses and document ids only, no document text or
drafted wording, so it can be kept in the repository.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dalil.draft import ExtractiveDrafter, FallbackDrafter, LLMDrafter  # noqa: E402
from dalil.pipeline import Pipeline  # noqa: E402
from dalil.store import Store  # noqa: E402

QUESTIONS = ROOT / "eval" / "citations.jsonl"


def score(record, expected: set[str], must_not_say: list[str] = ()) -> dict:
    kept = [s for s in record.check.get("sentences", []) if s["kept"] and s["kind"] == "claim"]
    # A claim backed only by an open-data query cites no document; it is
    # counted apart, not as a wrong source.
    data_only = [s for s in kept if not any(c.get("doc_id") for c in s["citations"])]
    from dalil.retrieve import tokenize
    # A kept claim that says a word the question's sources never support,
    # such as "parents" for a Golden visa, is wrong whatever page it cites.
    forbidden = set(tokenize(" ".join(must_not_say)))
    said_wrong = [s for s in kept if set(tokenize(s["text"])) & forbidden]
    kept = [s for s in kept if s not in data_only]
    cited = [sorted({c["doc_id"] for c in s["citations"] if c.get("doc_id")}) for s in kept]
    right = [bool(docs) and set(docs) <= expected and s not in said_wrong for s, docs in zip(kept, cited)]
    return {
        "status": record.status, "check": record.check.get("status"),
        "kept": len(kept), "right": sum(right), "cited_docs": cited, "data_only": len(data_only),
        "said_what_sources_do_not": len(said_wrong),
        "found": any(set(d) & expected for d in cited), "clean": bool(kept) and all(right),
    }


def _median(values) -> float | None:
    got = [v for v in values if v is not None]
    return statistics.median(got) if got else None


def summarise(results: list[dict], questions: int, runs: int) -> dict:
    kept = sum(x["kept"] for x in results)
    right = sum(x["right"] for x in results)
    return {
        "questions": questions, "runs": runs, "answers": len(results),
        "answered": sum(bool(x["kept"]) for x in results),
        "found": sum(x["found"] for x in results),
        "clean": sum(x["clean"] for x in results),
        "claims_kept": kept, "claims_right": right,
        "data_only_claims": sum(x.get("data_only", 0) for x in results),
        "claim_precision": round(right / kept, 3) if kept else None,
        "seconds_median": statistics.median(x["seconds"] for x in results) if results else None,
        "prompt_tokens_median": _median(x.get("prompt_tokens") for x in results),
        "output_tokens_median": _median(x.get("output_tokens") for x in results),
        "drafters": sorted({x["drafter"] for x in results}),
    }


def rescore(path: Path, rows: list[dict]) -> dict:
    """Score a saved run again against the expected documents in eval/citations.jsonl."""
    expected = {r["id"]: set(r["expected_docs"]) for r in rows}
    saved = json.loads(path.read_text(encoding="utf-8"))
    for x in saved["results"]:
        want = expected[x["id"]]
        right = [bool(d) and set(d) <= want for d in x["cited_docs"]]
        x.update(right=sum(right), found=any(set(d) & want for d in x["cited_docs"]),
                 clean=bool(x["cited_docs"]) and all(right))
    saved["summary"] = summarise(saved["results"], saved["summary"]["questions"], saved["summary"]["runs"])
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=1, help="ask every question this many times")
    ap.add_argument("--extractive", action="store_true", help="draft without the model")
    ap.add_argument("--only", nargs="*", help="question ids to run")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "eval-citations.json")
    ap.add_argument("--rescore", type=Path, help="score a saved report again instead of asking anything")
    args = ap.parse_args()
    rows = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8") if line.strip()]
    if args.rescore:
        saved = rescore(args.rescore, rows)
        args.rescore.write_text(json.dumps(saved, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(saved["summary"]))
        return 0
    if args.only:
        rows = [r for r in rows if r["id"] in set(args.only)]
    drafter = ExtractiveDrafter() if args.extractive else FallbackDrafter(LLMDrafter(), ExtractiveDrafter())
    store = Store(Path(tempfile.mkdtemp()) / "eval.db")
    pipeline = Pipeline.from_build(store, drafter=drafter, memory=None)
    results = []
    for run in range(1, args.runs + 1):
        for r in rows:
            t0 = time.perf_counter()
            rec = pipeline.answer(r["question"])
            seconds = time.perf_counter() - t0
            usage = getattr(getattr(drafter, "primary", None), "last_usage", {}) or {}
            res = {"id": r["id"], "run": run, "seconds": round(seconds, 1),
                   "drafter": getattr(drafter, "last_used", "") or type(drafter).__name__,
                   # The model server's own counts for this answer's draft, when it drafted.
                   "prompt_tokens": usage.get("prompt_n"), "output_tokens": usage.get("predicted_n"),
                   "prompt_ms": usage.get("prompt_ms"), "output_ms": usage.get("predicted_ms"),
                   "fallback_reason": getattr(drafter, "last_error", "") or "",
                   **score(rec, set(r["expected_docs"]), r.get("must_not_say", []))}
            if getattr(drafter, "primary", None) is not None:
                drafter.primary.last_usage = {}
            results.append(res)
            wrong = [d for d in res["cited_docs"] if not set(d) <= set(r["expected_docs"])]
            print(f"{r['id']} run {run}: {res['right']}/{res['kept']} right, found={res['found']} "
                  f"status={res['check']} {seconds:.1f}s{'  wrong: ' + str(wrong) if wrong else ''}")
    summary = summarise(results, len(rows), args.runs)
    print(json.dumps(summary))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
