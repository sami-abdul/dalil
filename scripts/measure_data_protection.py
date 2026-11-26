"""Ask the data-protection question N times and count how often Article 2 is cited.

    DALIL_EMBED_URL=http://127.0.0.1:8792/v1 uv run python scripts/measure_data_protection.py --runs 10 --out FILE "question"

The question, in English or Arabic, asks whether the federal data protection
law (Federal Decree-Law 45 of 2021) covers a government entity. The law's own
answer is Article 2(2): it "shall not apply" to government data or to
government entities. Per run this records whether a kept claim cites Article
2, whether it cites the 2(2) exclusion, and whether a kept claim quotes the
Dubai Data Law without naming it. The report holds the kept claims' wording,
so keep it out of the repository.
"""
import argparse, json, re, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dalil.draft import ExtractiveDrafter, FallbackDrafter, LLMDrafter
from dalil.pipeline import Pipeline
from dalil.store import Store

EN = "uaelegislation-federal-decree-law-45-2021-personal-data-protection-en"
AR = "uae-pdpl-fdl-45-2021-ar"
EXCL = re.compile(r"shall not apply|Governmental entities|Government Data|لا تسري|الجهات الحكومية|البيانات الحكومية")

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--runs", type=int, default=10)
ap.add_argument("--out", required=True)
a = ap.parse_args()

tmp = Path(tempfile.mkdtemp())
p = Pipeline.from_build(Store(tmp / "m.db"), drafter=FallbackDrafter(LLMDrafter(), ExtractiveDrafter()), memory=None)
runs = []
for i in range(a.runs):
    t = time.time()
    r = p.answer(a.question)
    secs = time.time() - t
    kept = [s for s in r.check["sentences"] if s["kept"] and s["kind"] == "claim"]
    cut = [s for s in r.check["sentences"] if not s["kept"] and s["kind"] == "claim"]
    cits = [c for s in kept for c in s["citations"] if c.get("passage_id")]
    art2 = [c for c in cits if c["doc_id"] in (EN, AR) and (r.retrieval and any(
        h["passage_id"] == c["passage_id"] and (h.get("heading") or "").endswith("(2)") for h in r.retrieval))]
    excl = [c for c in art2 if EXCL.search(c.get("quote") or "")]
    dubai = [s for s in kept if any(c.get("doc_id") == "dubai-data-law-26-2015" for c in s["citations"])]
    retrieved = [h["passage_id"] for h in r.retrieval]
    runs.append({
        "run": i + 1, "seconds": round(secs, 1), "status": r.status, "check": r.check["status"],
        "drafter": p.drafter.last_used, "kept": len(kept), "cut": len(cut),
        "cut_reasons": [x for s in cut for x in s["reasons"]],
        "cites_article_2": bool(art2), "cites_exclusion_2_2": bool(excl),
        "dubai_claims_kept": len(dubai),
        "dubai_claims_naming_dubai": sum(bool(re.search(r"Dubai|دبي", s["text"])) for s in dubai),
        "exclusion_retrieved_rank": next((n + 1 for n, pid in enumerate(retrieved) if pid in (f"{EN}:p6:2", f"{AR}:p7:1")), None),
        "cited_passages": sorted({c["passage_id"] for c in cits}),
        "claims": [s["text"] for s in kept],
    })
    print(json.dumps({k: v for k, v in runs[-1].items() if k != "claims"}, ensure_ascii=False))
summary = {
    "question": a.question, "runs": len(runs),
    "cites_article_2": sum(r["cites_article_2"] for r in runs),
    "cites_exclusion_2_2": sum(r["cites_exclusion_2_2"] for r in runs),
    "runs_with_unnamed_dubai_claim": sum(r["dubai_claims_kept"] > r["dubai_claims_naming_dubai"] for r in runs),
    "drafters": sorted({r["drafter"] for r in runs}),
    "seconds_median": sorted(r["seconds"] for r in runs)[len(runs) // 2],
}
print(json.dumps(summary, ensure_ascii=False))
Path(a.out).write_text(json.dumps({"summary": summary, "runs": runs}, ensure_ascii=False, indent=1), encoding="utf-8")
