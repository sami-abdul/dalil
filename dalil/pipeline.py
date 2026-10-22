"""One question, start to finish: screen, retrieve, look up open data, draft,
verify, queue for a person. Every step lands in the audit store.

    from dalil.pipeline import answer
    record = answer("What do I need to renew my Emirates ID?")

The status is "held" when the verifier holds the draft, else
"awaiting_review". Nothing is ever delivered here; a reviewer approves it
through the store (or the HTTP app).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from dalil import safety
from dalil.draft import FEE_CAVEAT, LAW_CAVEAT, WEAK_REFUSAL, Drafter, ExtractiveDrafter, caveat, refusal
from dalil.opendata import DataResult, match_rules, run_rules
from dalil.retrieve import Hit, Searcher, make_retriever
from dalil.store import Store
from dalil.verify import Verifier, load_passages, zetaris_lookup

# The only caveat and refusal sentences that may go out without a citation.
FIXED_UNCITED = frozenset({
    WEAK_REFUSAL, FEE_CAVEAT, LAW_CAVEAT,
    safety.REFUSAL_TEXT, safety.REFUSAL_FOLLOW_UP, safety.PERSONAL_DATA_CAVEAT,
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

    def apply_memory(self, qid: str, question: str, hits: list[Hit]) -> list[Hit]:
        """Leave out passages that reviewers rejected for similar questions."""
        if self.memory is None:
            return hits
        recalled = self.memory.recall(question)
        if recalled is None:
            self.store.event(qid, "recalled", {"available": False})
            return hits
        avoid = set(recalled.get("avoid_passage_ids") or [])
        kept = [h for h in hits if h.passage_id not in avoid]
        self.store.event(qid, "recalled", {
            "available": True,
            "memories": [m.get("id") for m in recalled.get("memories") or []],
            "rejections": len(recalled.get("reviewer_notes") or []),
            "passages_left_out": [h.passage_id for h in hits if h.passage_id in avoid],
        })
        return kept

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
        else:
            hits = self.retriever.search(q, self.k)
            hits = self.apply_memory(qid, q, hits)
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
            draft = self.drafter.draft(q, hits, data)
            if screened.has_personal_data:
                draft = [*draft, caveat(safety.PERSONAL_DATA_CAVEAT)]
        self.store.event(qid, "drafted", {
            "drafter": type(self.drafter).__name__, "sentences": len(draft),
            "kinds": [s.get("kind", "claim") for s in draft],
        })
        if client is None and any("sql" in c for s in draft for c in s.get("citations") or []):
            client, _ = self.zetaris()
        verifier = Verifier(self.passages, zetaris_lookup(client) if client else None,
                            allowed_sql=[d.sql for d in data], allowed_uncited=FIXED_UNCITED)
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
