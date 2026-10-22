"""Shared synthetic fixtures. Every passage here is made-up text."""

from __future__ import annotations

from typing import Any

import pytest

from dalil.retrieve import Hit, Retriever

PASSAGES: list[dict[str, Any]] = [
    {"passage_id": "doc-a:p1:1", "doc_id": "doc-a", "page": 1, "start": 0, "heading": "Renewing the card",
     "text": "Renewing the card\nThe resident card can be renewed up to six months before it expires.\n"
             "Late renewal costs AED 20 a day,\nup to AED 1,000 in total.\nA new photo is needed for every renewal request."},
    {"passage_id": "doc-a:p1:2", "doc_id": "doc-a", "page": 1, "start": 0, "heading": None,
     "text": "Renewal requests are made through the service centres or the official website of the authority."},
    {"passage_id": "sample-law-1-2020:p2:1", "doc_id": "sample-law-1-2020", "page": 2, "start": 40,
     "heading": "Article (5)",
     "text": "Article (5)\nA worker who resigns must give the employer written notice of at least thirty days."},
]
for _p in PASSAGES:
    _p["end"] = _p["start"] + len(_p["text"])

PAGES = {
    ("doc-a", 1): PASSAGES[0]["text"] + "\n" + PASSAGES[1]["text"],
    ("sample-law-1-2020", 2): "x" * 40 + PASSAGES[2]["text"],
}


class FakeZetaris:
    """Answers the fixed open-data queries with a canned row; counts calls."""

    def __init__(self, rows: list[dict[str, Any]] | None = None, fail: bool = False):
        self.rows = rows if rows is not None else [{"year": "2025", "transactions": "4242"}]
        self.fail = fail
        self.calls: list[str] = []

    def sql(self, statement: str, limit: int = 1000):
        if self.fail:
            raise ConnectionError("down")
        self.calls.append(statement)
        rows = [{"1": "1"}] if statement == "SELECT 1" else self.rows

        class R:
            def dicts(self_inner):
                return [dict(r) for r in rows]
        return R()


class FakeRetriever:
    """Returns fixed hits with fixed scores, whatever the query."""

    def __init__(self, hits: list[Hit]):
        self.hits = hits
        self.queries: list[str] = []

    def search(self, query: str, k: int = 8) -> list[Hit]:
        self.queries.append(query)
        return self.hits[:k]


def hit(p: dict[str, Any], score: float) -> Hit:
    return Hit(p["passage_id"], p["doc_id"], p["page"], score, p["heading"], p["text"], p["start"], p["end"])


@pytest.fixture
def passages() -> dict[str, dict[str, Any]]:
    return {p["passage_id"]: dict(p) for p in PASSAGES}


@pytest.fixture
def bm25() -> Retriever:
    return Retriever(PASSAGES)
