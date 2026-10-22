"""Tests for dalil.retrieve, on synthetic passages only."""

from __future__ import annotations

import json

import pytest

from dalil import retrieve as r


def passage(pid: str, text: str, heading: str | None = None, doc: str | None = None) -> dict:
    doc = doc or pid.split(":")[0]
    return {"passage_id": pid, "doc_id": doc, "page": 1, "start": 0, "end": len(text),
            "heading": heading, "text": text, "char_count": len(text)}


PASSAGES = [
    passage("visa:p1:1", "A tourist visa lets visitors stay for thirty days. Visitors apply online."),
    passage("gratuity:p1:1", "End of service gratuity is paid to a worker after one year of continuous "
            "service. The gratuity is twenty-one days of basic wage for each year.", "Gratuity"),
    passage("licence:p1:1", "Drivers can exchange a foreign driving licence at a licensing centre."),
    passage("leave:p1:1", "Annual leave is thirty days for each year of service."),
    passage("ar:p1:1", "يحق للعامل الحصول على إجازة سنوية مدفوعة الأجر."),
    passage("ar:p1:2", "تُصدر الهيئة بطاقة الهوية للمواطنين والمقيمين."),
]


def test_tokenize_english():
    assert r.tokenize("The Visas, cancelled!") == ["visa", "cancel"]
    assert r.tokenize("Driving licences") == r.tokenize("drive licence")
    assert r.tokenize("Visas", stem=False) == ["visas"]
    assert r.tokenize("don't") == []


def test_normalise_arabic():
    assert r.normalise_arabic("مُوَظَّف") == "موظف"  # diacritics stripped
    assert r.normalise_arabic("أحمد إلى آخر") == r.normalise_arabic("احمد الي اخر")  # alef, maqsura
    assert r.normalise_arabic("الهيئة") == "الهيئه"  # ta marbuta
    assert r.normalise_arabic("المادة ٥١") == "الماده 51"
    assert r.normalise_arabic("بطاقـــة") == "بطاقه"  # tatweel


def test_tokenize_arabic_strips_definite_article():
    assert r.tokenize("الإجازة") == r.tokenize("إجازة") == ["اجازه"]
    assert r.tokenize("والهوية") == ["هويه"]


def test_bm25_ranks_the_relevant_passage_first():
    ret = r.Retriever(PASSAGES)
    hits = ret.search("How is my end-of-service gratuity worked out?", k=3)
    assert hits[0].passage_id == "gratuity:p1:1"
    assert hits[0].score > (hits[1].score if len(hits) > 1 else 0)
    assert ret.search("exchange my foreign driving license", k=1)[0].passage_id == "licence:p1:1"


def test_bm25_arabic_query_matches_across_spelling_forms():
    ret = r.Retriever(PASSAGES)
    # Diacritics, a different alef and no definite article in the query.
    assert ret.search("اجازة سَنويّة", k=1)[0].passage_id == "ar:p1:1"
    assert ret.search("بطاقة هوية", k=1)[0].passage_id == "ar:p1:2"


def test_hit_fields_k_and_empty_queries():
    ret = r.Retriever(PASSAGES)
    hits = ret.search("thirty days", k=2)
    assert len(hits) == 2
    h = hits[0]
    assert isinstance(h, r.Hit)
    src = next(p for p in PASSAGES if p["passage_id"] == h.passage_id)
    assert (h.doc_id, h.page, h.text, h.start, h.end) == (src["doc_id"], 1, src["text"], 0, len(src["text"]))
    assert ret.search("", k=5) == []
    assert ret.search("the of and", k=5) == []
    assert ret.search("zzzz unknown", k=5) == []


def test_ties_are_stable():
    same = [passage(f"d{i}:p1:1", "identical words here") for i in range(4)]
    ids = [h.passage_id for h in r.Retriever(same).search("identical", k=4)]
    assert ids == ["d0:p1:1", "d1:p1:1", "d2:p1:1", "d3:p1:1"]


def test_heading_is_indexed_when_not_in_text():
    ret = r.Retriever([passage("a:p1:1", "Thirty days of notice.", heading="Resignation"),
                       passage("b:p1:1", "Thirty days of leave.")])
    assert ret.search("resignation", k=1)[0].passage_id == "a:p1:1"


def test_from_jsonl(tmp_path):
    path = tmp_path / "passages.jsonl"
    path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in PASSAGES), encoding="utf-8")
    ret = r.Retriever.from_jsonl(path)
    assert len(ret) == len(PASSAGES)
    assert ret.search("tourist visa", k=1)[0].passage_id == "visa:p1:1"
    assert isinstance(r.make_retriever(path), r.Searcher)


# --- dense hook --------------------------------------------------------------------


class LetterEmbedder:
    """A toy embedder: counts of a few letters. Stands in for a real model."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[t.count(ch) for ch in "aeiouy"] for t in texts]


def test_hybrid_without_embedder_is_bm25():
    lex = r.Retriever(PASSAGES)
    hyb = r.HybridRetriever(lex)
    q = "gratuity for each year of service"
    assert hyb.search(q, k=4) == lex.search(q, k=4)


def test_hybrid_with_embedder_fuses_rankings():
    emb = LetterEmbedder()
    assert isinstance(emb, r.Embedder)
    hyb = r.HybridRetriever(r.Retriever(PASSAGES), emb, batch_size=2)
    hits = hyb.search("gratuity", k=len(PASSAGES))
    assert hits[0].passage_id == "gratuity:p1:1"  # first in both rankings
    # Passages BM25 cannot see still come back through the dense ranking.
    assert len(hits) == len(PASSAGES)
    assert all(0 < h.score <= 2 / 61 for h in hits)
    calls = emb.calls
    hyb.search("visa", k=2)
    assert emb.calls == calls + 1  # passage vectors are computed once


def test_hybrid_rejects_wrong_vector_counts():
    with pytest.raises(ValueError):
        r.HybridRetriever(r.Retriever(PASSAGES), LetterEmbedder(), passage_vectors=[[1.0]])

    class Short:
        def embed(self, texts):
            return [[1.0]]

    with pytest.raises(ValueError):
        r.HybridRetriever(r.Retriever(PASSAGES), Short()).search("visa")
