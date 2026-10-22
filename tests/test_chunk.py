"""Tests for dalil.chunk, on synthetic text only (no corpus text in the repo)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from dalil import chunk as c

ROOT = Path(__file__).resolve().parent.parent


def sentence(i: int, words: int = 14) -> str:
    return " ".join(["Applicants"] + [f"word{i}x{j}" for j in range(words)]) + "."


def paragraph(i: int, sentences: int = 4) -> str:
    return " ".join(sentence(i * 10 + j) for j in range(sentences))


def html_doc(text: str, doc_id: str = "synthetic-page", **extra) -> dict:
    return {"doc_id": doc_id, "kind": "html", "language": "en", "is_terms_page": False,
            "pages": [{"page": 1, "text": text}], **extra}


def assert_exact_cover(text: str, passages: list[dict]) -> None:
    """Each passage is its page slice, passages do not overlap, and only
    whitespace falls between them."""
    pos = 0
    for p in passages:
        assert text[p["start"] : p["end"]] == p["text"]
        assert p["char_count"] == len(p["text"]) == p["end"] - p["start"]
        assert p["start"] >= pos
        assert text[pos : p["start"]].strip() == ""
        assert p["text"] == p["text"].strip()
        pos = p["end"]
    assert text[pos:].strip() == ""


# --- offsets and ids --------------------------------------------------------------


def test_offsets_exact_and_cover_the_page():
    text = "\n".join(
        ["Renewing a permit"]
        + [paragraph(i) for i in range(3)]
        + ["Required documents"]
        + [paragraph(i) for i in range(3, 7)]
        + ["Updated on: 01 Jan 2026"]
    )
    passages = c.chunk_document(html_doc(text))
    assert len(passages) > 2
    assert_exact_cover(text, passages)
    for p in passages:
        assert c.verify_passage(p, text)


def test_ids_are_stable_and_numbered_per_page():
    pages = [{"page": n, "text": "\n".join(paragraph(n * 10 + i) for i in range(4))} for n in (1, 2)]
    doc = {"doc_id": "law-x", "kind": "pdf", "language": "en", "pages": pages}
    first = c.chunk_document(doc)
    assert first == c.chunk_document(doc)
    for page in (1, 2):
        ids = [p["passage_id"] for p in first if p["page"] == page]
        assert ids == [f"law-x:p{page}:{n}" for n in range(1, len(ids) + 1)]
    assert list(first[0]) == list(c.PASSAGE_FIELDS)


def test_sizes_and_sentence_ends():
    text = "\n".join(paragraph(i, sentences=3) for i in range(12))
    passages = c.chunk_document(html_doc(text))
    assert_exact_cover(text, passages)
    assert all(p["char_count"] <= c.HARD_CHARS for p in passages)
    assert all(p["char_count"] >= c.MIN_CHARS for p in passages)
    assert all(p["text"].endswith(".") for p in passages)  # never mid-sentence
    assert sum(c.TARGET_CHARS <= p["char_count"] <= c.MAX_CHARS for p in passages) >= len(passages) - 1


def test_pdf_wrapped_lines_are_not_cut_mid_sentence():
    # A PDF line break inside a sentence is not a boundary.
    words = [f"term{i}" for i in range(400)]
    lines = [" ".join(words[i : i + 12]) for i in range(0, 400, 12)]
    text = "\n".join(lines) + ". Last sentence here."
    spans = c.split_page(text, "pdf")
    assert len(spans) > 1  # one 2,800-character sentence, so it must be cut
    assert all(x.end - x.start <= c.MAX_CHARS for x in spans)
    short = "First wrapped\nsentence goes on here. Second one\nalso wraps."
    found, _ = c.boundaries(short, "pdf")
    assert sorted(found) == [short.index("Second")]  # the sentence end, not the line breaks
    assert [(x.start, x.end) for x in c.split_page(short, "pdf")] == [(0, len(short))]


def test_abbreviations_and_list_markers_are_not_sentence_ends():
    text = "Bring documents, e.g. a passport. See Art. 5 here.\n1. First item.\nb. Second item."
    found, _ = c.boundaries(text, "pdf")
    cut_after = {text[:pos].rstrip()[-6:] for pos in found}
    assert not any(t.endswith("e.g.") or t.endswith("Art.") for t in cut_after)
    assert text.index("a passport") not in found
    assert text.index("First item") not in found and text.index("Second item") not in found


def test_long_sentence_without_punctuation_is_cut_at_spaces():
    text = " ".join(f"w{i:04d}" for i in range(700))  # ~3,500 chars, no full stop
    spans = c.split_page(text, "html")
    assert len(spans) >= 3
    for s in spans:
        piece = text[s.start : s.end]
        assert len(piece) <= c.MAX_CHARS
        assert re.fullmatch(r"w\d{4}( w\d{4})*", piece)  # whole words only


# --- headings ----------------------------------------------------------------------


@pytest.mark.parametrize("line,label", [
    ("Article (51)", "Article (51)"),
    ("Article )2 (", "Article (2)"),
    ("Article )4(", "Article (4)"),
    ("ARTICLE (7)", "Article (7)"),
    (")1( المادة", "المادة (1)"),
    (") 12( المادة", "المادة (12)"),
    ("المادة (٣)", "المادة (3)"),
    ("controls stipulated in Article (5). These", None),
    ("Article (5) of this law applies", None),
])
def test_article_heading(line, label):
    assert c.article_heading(line) == label


def law_page(numbers: list[int], paras: int = 3) -> str:
    parts = []
    for n in numbers:
        parts += [f"Article ({n})", f"Title of article {n}"]
        parts += [paragraph(n * 10 + i, sentences=2) for i in range(paras)]
    return "\n".join(parts)


def test_law_splits_at_article_boundaries_and_carries_heading():
    page1 = law_page([50, 51])
    page2 = "\n".join(paragraph(900 + i, sentences=2) for i in range(2)) + "\n" + law_page([52], paras=1)
    doc = {"doc_id": "law", "kind": "pdf", "language": "en",
           "pages": [{"page": 1, "text": page1}, {"page": 2, "text": page2}]}
    passages = c.chunk_document(doc)
    for page, text in ((1, page1), (2, page2)):
        assert_exact_cover(text, [p for p in passages if p["page"] == page])
    p1 = [p for p in passages if p["page"] == 1]
    starts = [p["text"].split("\n", 1)[0] for p in p1]
    assert "Article (51)" in starts  # a new passage opens at the article
    assert {p["heading"] for p in p1} == {"Article (50)", "Article (51)"}
    for p in p1:  # no passage straddles two articles
        assert len(re.findall(r"^Article \(\d+\)$", p["text"], re.M)) <= 1
    p2 = [p for p in passages if p["page"] == 2]
    assert p2[0]["heading"] == "Article (51)"  # carried over the page break
    assert p2[-1]["heading"] == "Article (52)"


def test_html_heading_lines():
    text = "\n".join(["Eligibility for the permit", paragraph(1), paragraph(2),
                      "How to apply?", paragraph(3), paragraph(4)])
    passages = c.chunk_document(html_doc(text))
    assert passages[0]["heading"] == "Eligibility for the permit"
    assert passages[-1]["heading"] == "How to apply?"
    assert not c.is_heading_line("a list item without a capital")
    assert not c.is_heading_line("Fee | AED 100")
    assert not c.is_heading_line("This line ends a sentence.")


# --- Arabic ---------------------------------------------------------------------


def test_arabic_sentence_boundaries():
    q = "هل يمكن تجديد بطاقة الهوية عبر الإنترنت؟ "
    s = "نعم، يمكن ذلك من خلال الموقع الرسمي للهيئة. "
    text = (q + s) * 30
    found, _ = c.boundaries(text.strip(), "html")
    t = text.strip()
    for pos in found:
        assert t[pos - 2] in "؟."  # each cut follows an Arabic question mark or full stop
    assert t.index("نعم") in found  # the question mark itself ends a sentence
    spans = c.split_page(t, "html")
    assert len(spans) > 1
    for sp in spans:
        assert t[sp.end - 1] in "؟."


def test_long_arabic_sentence_cut_after_arabic_comma():
    clause = "يلتزم صاحب العمل بدفع الأجر في موعده المحدد وفق العقد المبرم بين الطرفين، "
    text = (clause * 40).strip() + "."
    spans = c.split_page(text, "pdf")
    assert len(spans) >= 2
    for sp in spans[:-1]:
        assert text[sp.end - 1] == "،"
        assert sp.end - sp.start <= c.MAX_CHARS


def test_arabic_article_heading_in_page():
    body = " ".join(["تسري أحكام هذا المرسوم بقانون على معالجة البيانات الشخصية."] * 12)
    text = f")1( المادة\n{body}\n) 2( المادة\n{body}"
    passages = c.chunk_document({"doc_id": "ar-law", "kind": "pdf", "language": "ar",
                                 "pages": [{"page": 1, "text": text}]})
    assert_exact_cover(text, passages)
    assert [p["heading"] for p in passages][0] == "المادة (1)"
    assert passages[-1]["heading"] == "المادة (2)"


# --- terms pages and the loader -------------------------------------------------------


def test_terms_pages_give_no_passages():
    doc = html_doc(paragraph(1), is_terms_page=True)
    assert c.chunk_document(doc) == []
    assert list(c.iter_passages([doc, html_doc(paragraph(2), doc_id="other")]))[0]["doc_id"] == "other"


def test_empty_page_gives_no_passages():
    assert c.split_page("", "pdf") == []
    assert c.split_page("  \n ", "html") == []


def _loader():
    spec = importlib.util.spec_from_file_location("load_zetaris", ROOT / "scripts" / "load_zetaris.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loader_passage_rows_check_the_slice():
    loader = _loader()
    text = "\n".join(paragraph(i) for i in range(3))
    pages = [{"doc_id": "synthetic-page", "page": 1, "text": text}]
    passages = c.chunk_document(html_doc(text))
    rows = loader.build_passage_rows(passages, pages)
    assert [list(r) for r in rows] == [loader.PASSAGE_COLUMNS] * len(rows)
    bad = dict(passages[0], start=passages[0]["start"] + 1)
    with pytest.raises(loader.z.ZetarisError, match="page_text"):
        loader.build_passage_rows([bad], pages)
    with pytest.raises(loader.z.ZetarisError, match="no such page"):
        loader.build_passage_rows([dict(passages[0], page=9)], pages)
    with pytest.raises(loader.z.ZetarisError, match="unique"):
        loader.build_passage_rows([passages[0], passages[0]], pages)
