"""Client for Dalil's H-MEM bridge (hmem/bridge.ts), the review memory.

Approved answers and reviewers' rejections are written to H-MEM, which keeps
each with its provenance in an append-only trust ledger. Before drafting,
Dalil recalls rejections for similar questions and leaves out the passages
those answers cited.

The bridge is optional. If it is not running, every call returns None and the
pipeline records that in its own audit trail instead of failing.
"""

from __future__ import annotations

import json
import os
from typing import Any
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
        except (URLError, OSError, ValueError):
            return None

    def health(self) -> dict[str, Any] | None:
        return self._call("GET", "/health")

    def recall(self, question: str, top: int = 5) -> dict[str, Any] | None:
        return self._call("GET", "/recall?" + urlencode({"q": question, "top": top}))

    def remember(self, decision: str, question_id: str, question: str, reviewer: str,
                 passage_ids: list[str], answer: str = "", note: str = "") -> dict[str, Any] | None:
        kind = "approved" if decision == "approve" else "rejected"
        return self._call("POST", f"/{kind}", {
            "question_id": question_id, "question": question, "reviewer": reviewer,
            "passage_ids": passage_ids, "answer": answer, "note": note,
        })

    def ledger(self, limit: int = 50) -> dict[str, Any] | None:
        return self._call("GET", f"/ledger?limit={int(limit)}")
