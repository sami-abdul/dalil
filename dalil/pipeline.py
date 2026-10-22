"""One question, start to finish: screen, retrieve, look up open data, draft,
verify, queue for a person. Every step lands in the audit store.

    from dalil.pipeline import answer
    record = answer("What do I need to renew my Emirates ID?")

The status is "held" when the verifier holds the draft, else
"awaiting_review". Nothing is ever delivered here; a reviewer approves it
through the store (or the HTTP app).
"""

from __future__ import annotations

import sys

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from dalil import safety
from dalil.draft import FEE_CAVEAT, LAW_CAVEAT, WEAK_REFUSAL, Drafter, ExtractiveDrafter, caveat, refusal
from dalil.opendata import DataResult, match_rules, run_rules
from dalil.retrieve import Hit, Searcher, make_retriever
from dalil.store import Store
from dalil.hmem import Guidance, guidance
from dalil.verify import Verifier, load_passages, zetaris_lookup

def document_titles() -> dict[str, str]:
    """doc_id -> title from the manifest, or nothing if it cannot be read."""
    from dalil.corpus import doc_id, load_manifest
    try:
        return {doc_id(r["file"]): r.get("title", "") for r in load_manifest()}
    except Exception as e:
        print(f"warning: no document titles for the checker ({type(e).__name__})", file=sys.stderr)
        return {}


# The only caveat and refusal sentences that may go out without a citation.
FIXED_UNCITED = frozenset({
    WEAK_REFUSAL, FEE_CAVEAT, LAW_CAVEAT,
    safety.REFUSAL_TEXT, safety.REFUSAL_FOLLOW_UP, safety.PERSONAL_DATA_CAVEAT, safety.ASK_FOR_DETAIL,
})

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PASSAGES = REPO_ROOT / "build" / "passages.jsonl"
K = 8
ZETARIS_TIMEOUT = 30.0


@dataclass
class Record:
    id: str
    question: str  # redacted
    flags: list[str]
    status: str
    retrieval: list[dict[str, Any]] = field(default_factory=list)
    data: list[dict[str, Any]] = field(default_factory=list)
    draft: list[dict[str, Any]] = field(default_factory=list)
    check: dict[str, Any] = field(default_factory=dict)


def default_zetaris() -> Any:
    """A Zetaris client, or raise. Imported lazily so tests never touch it."""
    from dalil import zetaris as z

    return z.Client(timeout=ZETARIS_TIMEOUT)


class Pipeline:
    def __init__(self, store: Store, retriever: Searcher, passages: dict[str, dict[str, Any]],
                 drafter: Drafter | None = None,
                 zetaris_factory: Callable[[], Any] | None = default_zetaris, k: int = K,
                 memory: Any = None):
        self.store = store
        self.retriever = retriever
        self.passages = passages
        self.drafter = drafter or ExtractiveDrafter()
        self.zetaris_factory = zetaris_factory
        self.k = k
        self._zetaris: Any = None
        # Review memory (dalil.hmem.HMem or a stand-in with recall()); None turns it off.
        self.memory = memory
        self.titles = document_titles()

    @classmethod
    def from_build(cls, store: Store | None = None, passages_path: Path | str = DEFAULT_PASSAGES,
                   **kwargs: Any) -> "Pipeline":
        return cls(store or Store(), make_retriever(passages_path), load_passages(passages_path), **kwargs)

    def zetaris(self) -> tuple[Any, str]:
        """The Zetaris client and "", or None and why it is unavailable."""
        if self._zetaris is not None:
            return self._zetaris, ""
        if self.zetaris_factory is None:
            return None, "Zetaris not configured; lookup skipped"
        try:
            client = self.zetaris_factory()
            client.sql("SELECT 1")
        except Exception as e:  # ZetarisError messages carry no credentials
            return None, f"Zetaris unreachable; lookup skipped ({type(e).__name__}: {str(e)[:160]})"
        self._zetaris = client
        return client, ""

    def recall(self, qid: str, question: str) -> Guidance:
        """Reviews of similar questions, from the review memory, and log what was used."""
        if self.memory is None:
            return Guidance()
        recalled = self.memory.recall(question)
        if recalled is None:
            self.store.event(qid, "recalled", {"available": False})
            return Guidance()
        g = guidance(question, recalled.get("memories") or [])
        self.store.event(qid, "recalled", {
            "available": True, "similar_reviews": g.similar,
            "passages_left_out": sorted(g.avoid), "notes": len(g.notes),
        })
        return g

    def search(self, question: str, avoid: set[str]) -> list[Hit]:
        """Top k hits, leaving out `avoid` and filling up from further down the ranking."""
        hits = self.retriever.search(question, self.k + len(avoid))
        return [h for h in hits if h.passage_id not in avoid][: self.k]

    def answer(self, question: str) -> Record:
        screened = safety.screen(question)
        # From here on only the redacted text exists.
        qid = self.store.create_question(screened.redacted_question, screened.flags)
        self.store.event(qid, "screened", {"flags": screened.flags})
        try:
            return self._answer(qid, screened)
        except Exception as e:
            self.store.event(qid, "failed", {"error": type(e).__name__, "message": str(e)[:300]})
            self.store.save_record(qid, [], [], [], {"status": "held", "answer": "", "sentences": []}, "held")
            self.store.event(qid, "queued", {"status": "held"})
            return Record(qid, screened.redacted_question, screened.flags, "held")

    def _answer(self, qid: str, screened: safety.ScreenResult) -> Record:
        q = screened.redacted_question
        hits: list[Hit] = []
        data: list[DataResult] = []
        client = None
        if screened.is_rule_evasion:
            self.store.event(qid, "retrieved", {"skipped": "rule_evasion", "hits": 0})
            draft = [refusal(safety.REFUSAL_TEXT), refusal(safety.REFUSAL_FOLLOW_UP)]
        elif screened.is_vague:
            # Too few key words to know the subject: ask, rather than guess.
            self.store.event(qid, "retrieved", {"skipped": "vague_question", "hits": 0})
            draft = [refusal(safety.ASK_FOR_DETAIL)]
        else:
            past = self.recall(qid, q)
            hits = self.search(q, past.avoid)
            self.store.event(qid, "retrieved", {
                "k": self.k, "hits": len(hits),
                "top": [{"passage_id": h.passage_id, "score": round(h.score, 3)} for h in hits[:3]],
            })
            rules = match_rules(q)
            if rules:
                client, why = self.zetaris()
                run = (lambda sql: client.sql(sql, limit=200).dicts()) if client else None
                data = run_rules(rules, run, why)
                self.store.event(qid, "data_looked_up", {
                    "tables": [d.table for d in data],
                    "rows": [len(d.rows) for d in data],
                    "notes": [d.note for d in data if d.note],
                })
            draft = self.drafter.draft(q, hits, data, notes=past.notes)
            if screened.has_personal_data:
                draft = [*draft, caveat(safety.PERSONAL_DATA_CAVEAT)]
        ran = "fixed refusal" if (screened.is_rule_evasion or screened.is_vague) else (
            getattr(self.drafter, "last_used", "") or type(self.drafter).__name__)
        error = "" if (screened.is_rule_evasion or screened.is_vague) else getattr(self.drafter, "last_error", "")
        self.store.event(qid, "drafted", {
            "drafter": ran, **({"fallback_reason": error} if error else {}),
            "sentences": len(draft),
            "kinds": [s.get("kind", "claim") for s in draft],
        })
        if client is None and any("sql" in c for s in draft for c in s.get("citations") or []):
            client, _ = self.zetaris()
        verifier = Verifier(self.passages, zetaris_lookup(client) if client else None,
                            allowed_sql=[d.sql for d in data], allowed_uncited=FIXED_UNCITED,
                            allowed_passages=[h.passage_id for h in hits], titles=self.titles)
        check = verifier.check_answer(draft).to_dict()
        kept = sum(s["kept"] for s in check["sentences"])
        self.store.event(qid, "verified", {"status": check["status"], "kept": kept,
                                           "cut": len(check["sentences"]) - kept})
        status = "held" if check["status"] == "held" else "awaiting_review"
        retrieval = [asdict(h) for h in hits]
        data_dicts = [d.to_dict() for d in data]
        self.store.save_record(qid, retrieval, data_dicts, draft, check, status)
        self.store.event(qid, "queued", {"status": status})
        return Record(qid, q, screened.flags, status, retrieval, data_dicts, draft, check)


_default: Pipeline | None = None


def answer(question: str) -> Record:
    """Answer through the default pipeline over build/passages.jsonl and build/dalil.db."""
    global _default
    if _default is None:
        _default = Pipeline.from_build()
    return _default.answer(question)
