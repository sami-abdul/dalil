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
    assert v.check_sentence(claim("The gratuity is 21 days of basic wage for each year.", q("gratuity of 21 days’ basic wage for each year"))).kept


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
    s = v.check_sentence(claim("Under Article 51, the gratuity is 21 days of basic wage for each year.",
                               q("21 days' basic wage for each year")))
    assert s.kept, s.reasons


def test_uncited_claim_and_unknown_passage_fail(v):
    assert v.check_sentence(claim("Gratuity is generous.")).reasons == ["no citation"]
    assert "unknown passage" in v.check_sentence(claim("x", q("a b c d e", pid="nope:p1:1"))).reasons[0]


def test_refusal_needs_no_citation_but_may_not_state_numbers(v):
    assert v.check_sentence({"text": "The sources don't give this year's fee.", "kind": "refusal"}).kept
    s = v.check_sentence({"text": "The fee is probably AED 500.", "kind": "refusal"})
    assert not s.kept


SQL = "SELECT year, transactions FROM dalil.t WHERE year = 2025"


def data_cit(value, column="transactions", sql=SQL, table="dalil.t"):
    return {"table": table, "sql": sql, "column": column, "value": value}


def test_data_citation_is_rerun_and_read_from_its_column():
    rows = {SQL: [{"year": "2025", "transactions": "5856085"}]}
    v = Verifier({}, data_lookup=lambda sql: rows[sql], allowed_sql=[SQL])
    ok = v.check_sentence(claim("ICP renewed 5,856,085 ID cards in 2025.", data_cit("5856085"), data_cit("2025", "year")))
    assert ok.kept, ok.reasons
    bad = v.check_sentence(claim("ICP renewed 6,000,000 ID cards.", data_cit("6000000")))
    assert not bad.kept and "did not return" in bad.reasons[0]
    # The year 2025 is in the row, but not in the transactions column.
    wrong_col = v.check_sentence(claim("ICP renewed 2,025 ID cards.", data_cit("2025")))
    assert not wrong_col.kept


def test_data_citation_must_use_a_query_dalil_ran():
    literal = "SELECT 1500 AS fee FROM dalil.t"
    v = Verifier({}, data_lookup=lambda sql: [{"fee": "1500"}], allowed_sql=[SQL])
    s = v.check_sentence(claim("Renewal costs AED 1500.", data_cit("1500", "fee", sql=literal)))
    assert not s.kept and "not one Dalil ran" in s.reasons[0]


def test_data_citation_table_must_be_the_one_queried():
    v = Verifier({}, data_lookup=lambda sql: [{"transactions": "5"}], allowed_sql=[SQL])
    s = v.check_sentence(claim("There were 5.", data_cit("5", table="dalil.other")))
    assert not s.kept and "table" in s.reasons[0]


def test_data_citation_without_a_lookup_fails():
    s = Verifier({}).check_sentence(claim("x", data_cit("1")))
    assert not s.kept and "no data source" in s.reasons[0]


def test_quote_may_not_stop_inside_a_number_or_word(v):
    s = v.check_sentence(claim("Late renewal costs up to AED 1.", q("a day, up to AED 1")))
    assert not s.kept and "word for word" in s.reasons[0]
    s = v.check_sentence(claim("It is a gratuity.", q("worker is entitled to an end of serv")))
    assert not s.kept
    assert v.check_sentence(claim("Late renewal costs AED 20 a day, up to AED 1,000.", q("Late renewal costs AED 20 a day, up to AED 1,000"))).kept


def test_model_written_caveat_is_not_trusted():
    fixed = "The sources I have don't state this fee."
    v = Verifier({}, allowed_uncited=[fixed])
    assert v.check_sentence({"text": fixed, "kind": "caveat"}).kept
    s = v.check_sentence({"text": "Renewal is free for retirees.", "kind": "caveat"})
    assert not s.kept and "needs a citation" in s.reasons[0]


def test_answer_status(v):
    good = claim("The gratuity is 21 days of basic wage for each year.", q("gratuity of 21 days' basic wage for each year"))
    bad = claim("Gratuity is 30 days a year.", q("30 days' basic wage for each year"))
    assert v.check_answer([good, good]).status == "pass"
    mixed = v.check_answer([good, bad])
    assert mixed.status == "pass_with_cuts"
    assert mixed.answer_text() == "The gratuity is 21 days of basic wage for each year."
    assert v.check_answer([good, bad, bad]).status == "held"
    assert v.check_answer([]).status == "held"
    assert v.check_answer([{"text": "The sources don't cover this.", "kind": "refusal"}]).status == "pass"
    # A kept caveat cannot carry an answer whose only claim was cut.
    note = {"text": "Don't share ID numbers here.", "kind": "caveat"}
    assert v.check_answer([bad, note]).status == "held"
    assert v.check_answer([good, bad, note]).status == "pass_with_cuts"


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
    "SELECT * FROM dalil.open_icp_transactions, secret.users",
    "SELECT java_method('java.lang.System','getenv','HOME') FROM dalil.open_icp_transactions",
    "SELECT * FROM `other`.`t`",
    "WITH x AS (SELECT year FROM dalil.open_icp_transactions) SELECT * FROM x",
])
def test_safe_select_refuses_everything_else(sql):
    from dalil.verify import safe_select
    with pytest.raises(ValueError):
        safe_select(sql)


def test_spelled_out_numbers_are_checked(v):
    s = v.check_sentence(claim("Late renewal costs fifty dirhams a day.", q("Late renewal costs AED 20 a day")))
    assert not s.kept and "50" in s.reasons[0]
    assert v.check_sentence(claim("Late renewal costs twenty dirhams a day.", q("Late renewal costs AED 20 a day"))).kept


def test_claim_must_be_supported_by_its_quote(v):
    s = v.check_sentence(claim("Retirees renew for free at any police station.", q("Late renewal costs AED 20 a day")))
    assert not s.kept and "not supported" in s.reasons[0]


def test_claim_may_not_add_a_negation(v):
    s = v.check_sentence(claim("Late renewal does not cost AED 20 a day.", q("Late renewal costs AED 20 a day")))
    assert not s.kept and "negation" in s.reasons[0]


def test_data_values_must_share_a_row():
    rows = [{"year": "2024", "transactions": "4242"}, {"year": "2025", "transactions": "3100"}]
    v = Verifier({}, data_lookup=lambda sql: rows, allowed_sql=[SQL])
    s = v.check_sentence(claim("In 2025 there were 4242 transactions.", data_cit("4242"), data_cit("2025", "year")))
    assert not s.kept and "different rows" in s.reasons[0]
    assert v.check_sentence(claim("In 2025 there were 3100 transactions.", data_cit("3100"), data_cit("2025", "year"))).kept


def test_citations_are_limited_to_the_sources_given():
    passages = {"p1": {"passage_id": "p1", "doc_id": "d", "page": 1, "start": 0, "heading": None,
                       "text": "Late renewal costs AED 20 a day, up to AED 1,000."}}
    v = Verifier(passages, allowed_passages=["p2"])
    s = v.check_sentence(claim("Late renewal costs AED 20 a day.", {"passage_id": "p1", "quote": "Late renewal costs AED 20 a day"}))
    assert not s.kept and "not among the sources" in s.reasons[0]
