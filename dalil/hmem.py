"""Client for Dalil's H-MEM bridge (hmem/bridge.ts), the review memory.

Approved answers and reviewers' rejections are written to H-MEM, which keeps
each with its provenance in an append-only trust ledger. Before drafting,
Dalil recalls rejections for similar questions and leaves out the passages
those answers cited.

The bridge is optional. If it is not running, every call returns None and the
pipeline records that in its own audit trail instead of failing.
"""

from __future__ import annotations

import http.client
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_URL = "http://127.0.0.1:8788"
TIMEOUT = 2.0


class HMem:
    def __init__(self, url: str | None = None, timeout: float = TIMEOUT):
        self.url = (url or os.environ.get("DALIL_HMEM_URL") or DEFAULT_URL).rstrip("/")
        self.timeout = timeout

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any] | None:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = Request(f"{self.url}{path}", data=data, method=method,
                      headers={"content-type": "application/json"} if data else {})
        try:
            with urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read().decode("utf-8"))
                return out if isinstance(out, dict) else None
        except (URLError, OSError, ValueError, http.client.HTTPException):
            return None

    def health(self) -> dict[str, Any] | None:
        return self._call("GET", "/health")

    def recall(self, question: str, top: int = 50) -> dict[str, Any] | None:
        return self._call("GET", "/recall?" + urlencode({"q": question, "top": top}))

    def remember(self, decision: str, question_id: str, question: str, reviewer: str,
                 passage_ids: list[str], answer: str = "", note: str = "") -> dict[str, Any] | None:
        """Returns {"memory": {"id", "kind", ...}}, or None if the bridge is down
        or answered with something else."""
        kind = "approved" if decision == "approve" else "rejected"
        out = self._call("POST", f"/{kind}", {
            "question_id": question_id, "question": question, "reviewer": reviewer,
            "passage_ids": passage_ids, "answer": answer, "note": note,
        })
        memory = (out or {}).get("memory")
        if not isinstance(memory, dict) or not memory.get("id"):
            return None
        return out


# Share of the shorter question's content words that the other one also uses.
# Rewordings of one question scored 0.6 to 1.0 on the test set, and unrelated
# questions 0.0; "notice period" against "notice during probation" scored 0.4.
SIMILAR = 0.5


def similarity(a: str, b: str) -> float:
    from dalil.retrieve import tokenize
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


@dataclass
class Guidance:
    """What past reviews of similar questions say about this one."""
    avoid: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    similar: list[str] = field(default_factory=list)  # memory ids


def guidance(question: str, memories: Iterable[dict[str, Any]], threshold: float = SIMILAR) -> Guidance:
    """Passages to leave out and notes to pass on, from reviews of similar questions.

    A passage is left out if a rejected answer to a similar question cited it,
    unless an approved answer to a similar question also cited it.
    """
    g = Guidance()
    rejected: set[str] = set()
    approved: set[str] = set()
    for m in memories:
        if m.get("kind") not in ("approved", "rejected"):
            continue
        if similarity(question, m.get("question") or "") < threshold:
            continue
        g.similar.append(str(m.get("id")))
        ids = {p for p in m.get("passage_ids") or [] if isinstance(p, str)}
        if m["kind"] == "rejected":
            rejected |= ids
            note = (m.get("note") or "").strip()
            if note and note != "(no note)":
                g.notes.append(note)
        else:
            approved |= ids
    g.avoid = rejected - approved
    return g
