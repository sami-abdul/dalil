"""Tests for the citation checker. All text here is synthetic."""

from __future__ import annotations

import pytest

from dalil.verify import Verifier, fold, numbers

PAGE = (
    "Article (51)\n"
    "The worker is entitled to an end of service gratuity of 21 days' basic wage\n"
    "for each year of the first five years of service.\n"
    "Late renewal costs AED 20 a day, up to AED 1,000."
)
START = 100  # the passage begins 100 characters into its page


@pytest.fixture
def v():
    passages = {
        "law:p3:2": {"passage_id": "law:p3:2", "doc_id": "law", "page": 3, "start": START,
                     "end": START + len(PAGE), "heading": "Article (51)", "text": PAGE},
    }
    return Verifier(passages)


def claim(text, *cits):
    return {"text": text, "citations": list(cits)}


def q(quote, pid="law:p3:2"):
    return {"passage_id": pid, "quote": quote}


def test_fold_collapses_whitespace_and_maps_back():
    folded, index = fold("a  b\n“c”")
    assert folded == 'a b "c"'
    assert [index[i] for i in range(len(folded))] == [0, 1, 3, 4, 5, 6, 7]


def test_numbers_ignore_grouping_and_read_arabic_digits():
    assert numbers("AED 1,000 and ٢١ days, 2.5%") == {"1000", "21", "2.5"}


def test_quote_across_a_line_break_is_found_and_mapped_to_the_page(v):
    s = v.check_sentence(claim("You get 21 days of basic wage a year for the first five years.",
                               q("21 days' basic wage for each year of the first five years")))
    assert s.kept, s.reasons
    c = s.citations[0]
    assert (c.doc_id, c.page) == ("law", 3)
    assert PAGE[c.page_start - START:c.page_end - START].startswith("21 days")
    assert PAGE[c.page_start - START:c.page_end - START].endswith("five years")


def test_curly_quote_in_the_draft_still_matches(v):
    assert v.check_sentence(claim("Gratuity is 21 days a year.", q("21 days’ basic wage for each"))).kept


def test_changed_word_fails(v):
    s = v.check_sentence(claim("Gratuity is 30 days a year.", q("30 days' basic wage for each year")))
    assert not s.kept
    assert "word for word" in s.reasons[0]


def test_short_quote_fails(v):
    s = v.check_sentence(claim("It is a gratuity.", q("gratuity of")))
    assert not s.kept and "shorter" in s.reasons[0]


def test_number_not_in_the_quote_cuts_the_sentence(v):
    s = v.check_sentence(claim("Late renewal costs AED 20 a day, up to AED 1,500.",
                               q("Late renewal costs AED 20 a day")))
    assert not s.kept
    assert "1500" in s.reasons[0]


def test_article_number_is_backed_by_the_heading(v):
    s = v.check_sentence(claim("Under Article 51, gratuity is 21 days a year.",
                               q("21 days' basic wage for each year")))
    assert s.kept, s.reasons


def test_uncited_claim_and_unknown_passage_fail(v):
    assert v.check_sentence(claim("Gratuity is generous.")).reasons == ["no citation"]
    assert "unknown passage" in v.check_sentence(claim("x", q("a b c d e", pid="nope:p1:1"))).reasons[0]


def test_refusal_needs_no_citation_but_may_not_state_numbers(v):
    assert v.check_sentence({"text": "The sources don't give this year's fee.", "kind": "refusal"}).kept
    s = v.check_sentence({"text": "The fee is probably AED 500.", "kind": "refusal"})
    assert not s.kept


def test_data_citation_is_rerun():
    rows = {"SELECT n": [{"year": "2025", "transactions": "5856085"}]}
    v = Verifier({}, data_lookup=lambda sql: rows[sql])
    ok = v.check_sentence(claim("ICP renewed 5,856,085 ID cards in 2025.",
                                {"table": "t", "sql": "SELECT n", "value": "5856085"},
                                {"table": "t", "sql": "SELECT n", "value": "2025"}))
    assert ok.kept, ok.reasons
    bad = v.check_sentence(claim("ICP renewed 6,000,000 ID cards.",
                                 {"table": "t", "sql": "SELECT n", "value": "6000000"}))
    assert not bad.kept and "did not return" in bad.reasons[0]


def test_data_citation_without_a_lookup_fails():
    s = Verifier({}).check_sentence(claim("x", {"table": "t", "sql": "SELECT 1", "value": "1"}))
    assert not s.kept and "no data source" in s.reasons[0]


def test_answer_status(v):
    good = claim("Gratuity is 21 days a year.", q("21 days' basic wage for each year"))
    bad = claim("Gratuity is 30 days a year.", q("30 days' basic wage for each year"))
    assert v.check_answer([good, good]).status == "pass"
    mixed = v.check_answer([good, bad])
    assert mixed.status == "pass_with_cuts"
    assert mixed.answer_text() == "Gratuity is 21 days a year."
    assert v.check_answer([good, bad, bad]).status == "held"
    assert v.check_answer([]).status == "held"
    assert v.check_answer([{"text": "The sources don't cover this.", "kind": "refusal"}]).status == "pass"


@pytest.mark.parametrize("sql", [
    "SELECT * FROM dalil.open_icp_transactions WHERE year = 2025",
    "select t.year from dalil.pages p join dalil.open_icp_transactions t on 1=1;",
])
def test_safe_select_allows_read_only_dalil_queries(sql):
    from dalil.verify import safe_select
    assert safe_select(sql)


@pytest.mark.parametrize("sql", [
    "DROP TABLE dalil.pages",
    "SELECT 1; DROP TABLE dalil.pages",
    "SELECT * FROM metastore.users",
    "SELECT * FROM dalil.pages -- sneaky",
    "SELECT * FROM dalil.pages WHERE 1=1 UNION SELECT * FROM other.secrets",
    "SELECT 1",
    "WITH x AS (SELECT year FROM dalil.open_icp_transactions) SELECT * FROM x",
])
def test_safe_select_refuses_everything_else(sql):
    from dalil.verify import safe_select
    with pytest.raises(ValueError):
        safe_select(sql)
