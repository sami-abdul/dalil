"""The open-data source layer: the local database, Zetaris behind the same
interface, and the one setting that picks between them. Uses the committed
open data in data/open/, which is public; no document text."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import PASSAGES, FakeRetriever, FakeZetaris, hit
from dalil import datasource as ds
from dalil.app import create_app
from dalil.draft import ExtractiveDrafter
from dalil.opendata import RULES
from dalil.pipeline import Pipeline
from dalil.store import Store
from dalil.verify import Verifier, data_lookup

ROOT = Path(__file__).resolve().parent.parent
OPEN = ROOT / "data" / "open"
RECORDED = json.loads((Path(__file__).parent / "fixtures" / "zetaris_recorded_rows.json").read_text())


@pytest.fixture(scope="module")
def local() -> ds.LocalSource:
    return ds.LocalSource(OPEN)


def csv_rows(name: str) -> list[dict[str, str]]:
    with (OPEN / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- the local database ----------------------------------------------------------


def test_local_serves_every_committed_table_with_its_row_count(local):
    with (OPEN / "SOURCES.csv").open(newline="", encoding="utf-8") as fh:
        listed = {r["file"]: int(r["rows"]) for r in csv.DictReader(fh) if r["rows"]}
    with (OPEN / "tables.csv").open(newline="", encoding="utf-8") as fh:
        tables = {f"dalil.{r['table']}": r["file"] for r in csv.DictReader(fh)}
    assert local.tables() == sorted(tables)
    for table, file in tables.items():
        assert local.row_counts[table] == listed[file]
        assert local.sql(f"SELECT COUNT(*) AS n FROM {table}").scalar() == str(listed[file])


def test_every_fixed_query_runs_unchanged_and_returns_one_row_of_strings(local):
    for rule in RULES:
        rows = local.sql(rule.sql).dicts()
        assert len(rows) == 1, rule.name
        assert all(isinstance(v, str) and v for v in rows[0].values()), rule.name


def test_rule_values_match_a_count_made_without_sql(local):
    """Each rule's answer, worked out again from the CSV with plain Python."""
    icp = csv_rows("icp-service-transactions-2020-2025.csv")

    def latest(service):
        r = max((r for r in icp if r["service_en"] == service), key=lambda r: int(r["year"]))
        return {"year": r["year"], "transactions": r["transactions"]}
    assert local.sql(RULES[0].sql).dicts() == [latest("Renewal of ID Card")]
    assert local.sql(RULES[1].sql).dicts() == [latest("Renewal of Residence Permits")]

    tasheel = csv_rows("mohre-tasheel-service-centres.csv")
    assert local.sql(RULES[2].sql).dicts() == [
        {"centres": str(len(tasheel)), "emirates": str(len({r["emirate_en"] for r in tasheel}))}]

    ehs = csv_rows("ehs-medical-fitness-residency-visa-2022-2025.csv")
    year = max(int(r["year"]) for r in ehs)
    assert local.sql(RULES[3].sql).dicts() == [
        {"year": str(year), "exams": str(sum(int(r["total"]) for r in ehs if int(r["year"]) == year))}]


def test_complaints_rule_counts_each_emirate_once_and_skips_the_partial_year(local):
    rows = csv_rows("mohre-solved-complaints-private-sector-2009-2020.csv")
    by_city = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_city[r["Year"]][r["City_En"]] += int(r["SolvedComplaints_count"])
    # The duplication the rule works around is really in the file.
    assert by_city["2019"]["Al-Ain"] == by_city["2019"]["Abu Dhabi"] == by_city["2019"]["Delma"]
    assert by_city["2019"]["Kalbaa"] == by_city["2019"]["Sharjah"]
    emirates = ("Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Om-El-Quain", "Ras El-Khaema", "Fujiarah")
    want = sum(by_city["2019"][c] for c in emirates)
    assert local.sql(RULES[4].sql).dicts() == [{"year": "2019", "solved": str(want)}]
    assert want < sum(by_city["2019"].values())  # the old sum overcounted


@pytest.mark.parametrize("statement", [
    "DELETE FROM dalil.open_tasheel_centres",
    "UPDATE dalil.open_icp_transactions SET transactions = 1",
    "INSERT INTO dalil.open_icp_transactions (year) VALUES (1)",
    "DROP TABLE dalil.open_icp_transactions",
    "CREATE TABLE dalil.extra (a)",
    "ATTACH DATABASE 'elsewhere.db' AS other",
    "PRAGMA query_only = OFF",
    "SELECT load_extension('x')",
])
def test_local_refuses_anything_but_reading(local, statement):
    with pytest.raises(ds.DataSourceError):
        local.sql(statement)
    assert local.sql(RULES[0].sql).dicts()[0]["transactions"] == "5856085"


def test_local_honours_the_row_limit(local):
    assert len(local.sql("SELECT * FROM dalil.open_tasheel_centres", limit=5).rows) == 5


def test_local_refuses_a_file_that_differs_from_sources_csv(tmp_path):
    shutil.copytree(OPEN, tmp_path / "open")
    f = tmp_path / "open" / "mohre-tasheel-service-centres.csv"
    f.write_text(f.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ds.DataSourceError, match="SHA-256"):
        ds.LocalSource(tmp_path / "open")


def test_local_reads_quoted_multiline_fields(tmp_path):
    (tmp_path / "tables.csv").write_text("table,file\nopen_x,x.csv\n", encoding="utf-8")
    (tmp_path / "x.csv").write_text('id,note,n\n1,"two\nlines",\n2,plain,3\n', encoding="utf-8")
    src = ds.LocalSource(tmp_path)
    assert src.sql("SELECT id, note, n FROM dalil.open_x ORDER BY id").dicts() == [
        {"id": "1", "note": "two\nlines", "n": None}, {"id": "2", "note": "plain", "n": "3"}]


def test_table_files_refuse_a_path_outside_the_folder(tmp_path):
    (tmp_path / "tables.csv").write_text("table,file\nopen_x,../x.csv\n", encoding="utf-8")
    with pytest.raises(ds.DataSourceError, match="outside"):
        ds.table_files(tmp_path)


# --- parity with the server's Zetaris -----------------------------------------------


@pytest.mark.parametrize("rec", RECORDED["recorded"], ids=lambda r: r["sql"][-40:])
def test_local_returns_the_rows_zetaris_returned(local, rec):
    assert local.sql(rec["sql"]).dicts() == rec["rows"]


# --- choosing a source --------------------------------------------------------------


def env(tmp_path, **kw):
    """An environment with an empty Zetaris credentials file, plus kw."""
    empty = tmp_path / "zetaris.env"
    empty.write_text("# no login\n", encoding="utf-8")
    return {"DALIL_ZETARIS_ENV": str(empty), **kw}


def test_auto_uses_the_local_database_when_zetaris_is_not_configured(tmp_path):
    source, why = ds.resolve(env(tmp_path))
    assert source.kind == "local" and why == ""
    assert ds.status(source) == {"connected": True, "kind": "local", "label": "Open data: local database",
                                 "tables": sorted(source.row_counts)}


def test_auto_falls_back_to_local_when_zetaris_does_not_answer(tmp_path):
    dummy = "x" * 12  # a stand-in login; nothing listens on port 9
    e = env(tmp_path, ZETARIS_UI_URL="http://127.0.0.1:9", ZETARIS_ADMIN_EMAIL="a@b.c")
    e["ZETARIS_ADMIN_" + "PASSWORD"] = dummy
    source, why = ds.resolve(e, zetaris_timeout=2)
    assert source.kind == "local"
    assert "Zetaris unreachable" in why and "using the local database" in why
    assert dummy not in why


def test_zetaris_mode_never_falls_back(tmp_path):
    source, why = ds.resolve(env(tmp_path, DALIL_OPEN_DATA="zetaris"))
    assert source is None and "Zetaris unreachable" in why


def test_local_and_off_modes(tmp_path):
    assert ds.resolve(env(tmp_path, DALIL_OPEN_DATA="local"))[0].kind == "local"
    source, why = ds.resolve(env(tmp_path, DALIL_OPEN_DATA="off"))
    assert source is None and "turned off" in why


def test_auto_with_no_data_files_is_neutral(tmp_path):
    source, why = ds.resolve(env(tmp_path, DALIL_OPEN_DATA_DIR=str(tmp_path / "nothing")))
    assert source is None and why == "Open data not configured; lookup skipped"
    assert ds.status(None) == {"connected": False, "kind": None, "label": "Open data: not configured", "tables": []}


def test_a_bad_mode_is_reported_not_raised(tmp_path):
    source, why = ds.resolve(env(tmp_path, DALIL_OPEN_DATA="sometimes"))
    assert source is None and "DALIL_OPEN_DATA must be one of" in why


def test_status_of_a_source_that_stops_answering():
    out = ds.status(FakeZetaris(fail=True))
    assert out["connected"] is False and out["kind"] == "zetaris" and out["tables"] == []


# --- end to end on the local database --------------------------------------------------


def local_pipeline(tmp_path, local):
    store = Store(tmp_path / "dalil.db")
    return Pipeline(store, FakeRetriever([hit(PASSAGES[0], 20.0), hit(PASSAGES[1], 15.0)]),
                    {p["passage_id"]: dict(p) for p in PASSAGES}, drafter=ExtractiveDrafter(),
                    zetaris_factory=lambda: local)


def test_pipeline_answers_from_the_local_database_and_the_checker_reruns_it(tmp_path, local):
    p = local_pipeline(tmp_path, local)
    rec = p.answer("How many Emirates ID cards are renewed in a year, and how early can I renew mine?")
    data = [s for s in rec.check["sentences"] if any("sql" in c for c in s["citations"])]
    assert len(data) == 1 and data[0]["kept"]
    assert data[0]["text"] == "In 2025, ICP renewed 5856085 Emirates ID cards, according to its open data."
    assert all(c["ok"] and c["reason"] == "value returned by the query" for c in data[0]["citations"])
    assert rec.data[0]["source"] == "local"
    looked = next(e for e in p.store.get(rec.id)["audit"] if e["kind"] == "data_looked_up")
    assert looked["detail"]["source"] == "local"


def test_every_rule_passes_the_checker_on_the_local_database(local):
    from dalil.draft import data_claims
    from dalil.opendata import run_rules
    for rule in RULES:
        results = run_rules([rule], lambda sql: local.sql(sql, limit=200).dicts(), source="local")
        claims = data_claims(results)
        check = Verifier({}, data_lookup(local), allowed_sql=[rule.sql]).check_answer(claims)
        assert check.status == "pass", (rule.name, check.to_dict())


def test_health_and_citations_name_the_local_database(tmp_path, local):
    p = local_pipeline(tmp_path, local)
    tc = TestClient(create_app(p, {}, web_dir=None), base_url="http://127.0.0.1")
    health = tc.get("/api/health").json()
    assert health["zetaris"] is False
    assert health["data_source"]["connected"] and health["data_source"]["kind"] == "local"
    assert health["data_source"]["label"] == "Open data: local database"
    qid = tc.post("/api/questions", json={"question": "How many Emirates ID cards are renewed in a year, and how early can I renew mine?"}).json()["id"]
    d = tc.get(f"/api/questions/{qid}").json()
    cits = [c for s in d["check"]["sentences"] for c in s["citations"] if c.get("sql")]
    assert cits and all(c["source_label"] == "queried from the local database" for c in cits)
    assert d["data"][0]["source_label"] == "queried from the local database"


def test_health_with_no_source_is_neutral(tmp_path):
    p = Pipeline(Store(tmp_path / "dalil.db"), FakeRetriever([]), {}, zetaris_factory=None)
    health = TestClient(create_app(p, {}, web_dir=None), base_url="http://127.0.0.1").get("/api/health").json()
    assert health["zetaris"] is False
    assert health["data_source"] == {"connected": False, "kind": None, "label": "Open data: not configured",
                                     "tables": []}


def test_old_records_without_a_source_are_shown_as_zetaris():
    from dalil.app import data_kind
    assert data_kind({"rows": [{"year": "2025"}]}) == "zetaris"
    assert data_kind({"rows": []}) == ""
    assert data_kind({"rows": [{"year": "2025"}], "source": "local"}) == "local"


def test_a_source_that_stops_answering_is_dropped_and_resolved_again(tmp_path, local):
    sources = [FakeZetaris(), local]
    p = Pipeline(Store(tmp_path / "dalil.db"), FakeRetriever([]), {}, zetaris_factory=lambda: sources.pop(0))
    assert isinstance(p.data_source()[0], FakeZetaris)
    p._zetaris.fail = True
    assert p.data_source_status()["kind"] == "local"


def test_a_year_from_the_same_rerun_row_backs_the_sentence(local):
    rule = next(r for r in RULES if r.name == "medical_fitness")
    cit = {"table": rule.table, "sql": rule.sql, "column": "exams", "value": "1424313"}
    v = Verifier({}, data_lookup(local), allowed_sql=[rule.sql])
    ok = v.check_answer([{"text": "In 2025, EHS carried out 1424313 exams.", "citations": [cit]}])
    assert ok.status == "pass", ok.to_dict()
    # A number in no returned row is still cut.
    bad = v.check_answer([{"text": "In 2024, EHS carried out 1424313 exams.", "citations": [cit]}])
    assert bad.status == "held" and "2024" in bad.sentences[0].reasons[0]


# --- when the model's data sentence is cut -------------------------------------------


class ScriptedDrafter:
    """Returns fixed sentences, as a model might."""

    def __init__(self, sentences):
        self.sentences = sentences

    def draft(self, question, hits, data, notes=()):
        return [dict(s) for s in self.sentences]


def scripted(tmp_path, local, sentences):
    p = local_pipeline(tmp_path, local)
    p.drafter = ScriptedDrafter(sentences)
    return p


MEDICAL = next(r for r in RULES if r.name == "medical_fitness")
COMPLAINTS = next(r for r in RULES if r.name == "labour_complaints")
QUOTE = {"passage_id": PASSAGES[1]["passage_id"],
         "quote": "Renewal requests are made through the service centres or the official website of the authority."}


def test_the_rule_meaning_backs_a_reworded_data_claim(local):
    cit = {"table": MEDICAL.table, "sql": MEDICAL.sql, "column": "exams", "value": "1424313"}
    v = Verifier({p["passage_id"]: dict(p) for p in PASSAGES}, data_lookup(local), allowed_sql=[MEDICAL.sql],
                 data_facts={MEDICAL.sql: MEDICAL.fact})
    claim = {"text": "In 2025, Emirates Health Services carried out 1424313 medical fitness exams for residency visas.",
             "citations": [cit, QUOTE]}
    assert v.check_answer([claim]).status == "pass"
    # Without the rule's sentence the same claim fails the wording check, as it did before.
    bare = Verifier({p["passage_id"]: dict(p) for p in PASSAGES}, data_lookup(local), allowed_sql=[MEDICAL.sql])
    assert "not supported" in bare.check_answer([claim]).sentences[0].reasons[0]


def test_a_cut_attempt_at_the_data_is_restated_with_the_rule_sentence(tmp_path, local):
    # The model writes the numbers but cites only a passage, so the checker cuts it.
    p = scripted(tmp_path, local, [
        {"text": "Renewal requests are made through the service centres or the official website of the authority.",
         "kind": "claim", "citations": [QUOTE]},
        {"text": "Ask anyone and they will tell you 207927 is a big figure.", "kind": "claim",
         "citations": [QUOTE]},
    ])
    rec = p.answer("How many labour complaints does MOHRE solve? I want to renew my card.")
    texts = {s["text"]: s["kept"] for s in rec.check["sentences"]}
    assert texts["Ask anyone and they will tell you 207927 is a big figure."] is False
    fact = "In 2019, MOHRE solved 207927 private-sector labour complaints, counting each emirate once, according to its open data."
    assert texts[fact] is True
    verified = next(e for e in p.store.get(rec.id)["audit"] if e["kind"] == "verified")
    assert verified["detail"]["data_restated"] == 1


def test_data_the_model_left_alone_is_not_added(tmp_path, local):
    p = scripted(tmp_path, local, [
        {"text": "Renewal requests are made through the service centres or the official website of the authority.",
         "kind": "claim", "citations": [QUOTE]},
    ])
    rec = p.answer("Where do labour complaints to MOHRE go? I want to renew my card.")
    # The question asks for no count, so no lookup runs and no data sentence is added.
    assert not rec.data
    assert not any(any(c.get("sql") for c in s["citations"]) for s in rec.check["sentences"])


def test_a_year_alone_does_not_count_as_stating_the_data(tmp_path, local):
    p = scripted(tmp_path, local, [
        {"text": "Renewal requests are made through the service centres or the official website of the authority "
                 "in 2019.", "kind": "claim", "citations": [QUOTE]},
    ])
    rec = p.answer("Where do labour complaints to MOHRE go? I want to renew my card.")
    assert not any(s.get("restated") for s in rec.draft)


@pytest.mark.parametrize("question, rule", [
    ("كم عدد بطاقات الهوية الإماراتية التي يتم تجديدها سنويا؟", "emirates_id_renewals"),
    ("كم عدد تصاريح الإقامة التي يتم تجديدها كل عام؟", "residence_renewals"),
    ("كم عدد مراكز تسهيل في الدولة؟", "tasheel_centres"),
    ("كم عدد فحوصات اللياقة الطبية للإقامة سنويا؟", "medical_fitness"),
    ("كم عدد الشكاوى العمالية التي تحلها الوزارة؟", "labour_complaints"),
])
def test_arabic_questions_reach_the_open_data(question, rule):
    from dalil.opendata import match_rules
    assert [r.name for r in match_rules(question)] == [rule]


def test_every_arabic_fact_passes_the_checker(local):
    from dalil.draft import data_claims
    from dalil.opendata import DataResult, run_rules
    for rule in RULES:
        res = run_rules([rule], lambda sql: local.sql(sql, limit=200).dicts(), source="local")[0]
        assert res.fact_ar and res.fact_ar.count("{") == res.fact.count("{")
        claims = data_claims([DataResult(res.table, res.sql, res.rows, fact=res.fact_ar)])
        check = Verifier({}, data_lookup(local), allowed_sql=[rule.sql]).check_answer(claims)
        assert check.status == "pass", (rule.name, check.to_dict())


def test_a_cut_arabic_attempt_is_restated_in_arabic(tmp_path, local):
    p = scripted(tmp_path, local, [
        {"text": "تحل الوزارة 207927 شكوى عمالية في عام 2019.", "kind": "claim", "citations": [QUOTE]},
    ])
    rec = p.answer("كم عدد الشكاوى العمالية التي تحلها الوزارة؟")
    kept = [s["text"] for s in rec.check["sentences"] if s["kept"]]
    assert any("207927" in t and "شكوى" in t for t in kept), rec.check


def test_states_value_reads_arabic_indic_digits():
    from dalil import scenarios as sc
    from dalil.pipeline import Record
    cit = {"ok": True, "reason": "value returned by the query", "table": "dalil.t", "column": "n", "value": "49"}
    rec = Record("x", "q", [], "awaiting_review", check={"sentences": [
        {"text": "يوجد ٤٩ مركز تسهيل.", "kind": "claim", "kept": True, "citations": [cit]}]})
    assert sc.CHECKS["states_value"](rec, {}, {"table": "dalil.t", "column": "n", "value": "49"})


def test_a_data_citation_on_a_sentence_that_does_not_state_it_is_dropped_and_the_value_restated(tmp_path, local):
    rule = RULES[0]
    p = scripted(tmp_path, local, [
        {"text": "Renewal requests are made through the service centres or the official website of the authority.",
         "kind": "claim", "citations": [QUOTE, {"table": rule.table, "sql": rule.sql, "column": "transactions",
                                                "value": "5856085"}]},
    ])
    rec = p.answer("How many Emirates ID cards does ICP renew in a year?")
    first = rec.check["sentences"][0]
    assert first["kept"] and not any(c.get("sql") for c in first["citations"])
    kept = [s["text"] for s in rec.check["sentences"] if s["kept"]]
    assert "In 2025, ICP renewed 5856085 Emirates ID cards, according to its open data." in kept


def test_a_claim_stating_a_lookup_value_gets_its_data_citation(tmp_path, local):
    p = scripted(tmp_path, local, [
        {"text": "In 2025, Emirates Health Services carried out 1424313 medical fitness exams for residency visas.",
         "kind": "claim", "citations": [QUOTE]},
        {"text": "The data is from the open_ehs_medical_fitness table.", "kind": "claim", "citations": [QUOTE]},
    ])
    rec = p.answer("How many medical fitness exams for residency visas does Emirates Health Services carry out?")
    first = rec.check["sentences"][0]
    assert first["kept"], first
    assert {c.get("column") for c in first["citations"] if c.get("sql")} == {"exams", "year"}
    assert not any(s.get("restated") for s in rec.draft)
    assert rec.status == "awaiting_review"


def test_a_wrong_number_gets_no_data_citation(tmp_path, local):
    p = scripted(tmp_path, local, [
        {"text": "Emirates Health Services carried out 1424314 medical fitness exams.", "kind": "claim",
         "citations": [QUOTE]},
    ])
    p.redraft = False  # look at the model's own sentence, not the fallback that replaces an all-cut answer
    rec = p.answer("How many medical fitness exams for residency visas does Emirates Health Services carry out?")
    wrong = next(x for x in rec.check["sentences"] if "1424314" in x["text"])
    assert not wrong["kept"]
    assert not any(c.get("sql") for c in wrong["citations"])


def test_a_year_alone_attaches_nothing(tmp_path, local):
    p = scripted(tmp_path, local, [
        {"text": "Renewal requests are made through the service centres or the official website of the authority "
                 "in 2025.", "kind": "claim", "citations": [QUOTE]},
    ])
    rec = p.answer("How many Emirates ID cards does ICP renew in a year?")
    assert not any(c.get("sql") for c in rec.draft[0]["citations"])


def test_a_claim_written_twice_is_checked_once(tmp_path, local):
    text = "Renewal requests are made through the service centres or the official website of the authority."
    p = scripted(tmp_path, local, [{"text": text, "kind": "claim", "citations": [QUOTE]}] * 3)
    rec = p.answer("How do I renew my card?")
    assert [s["text"] for s in rec.check["sentences"]].count(text) == 1


def test_a_claim_that_only_repeats_a_stated_value_is_dropped(tmp_path, local):
    first = "In 2025, Emirates Health Services carried out 1424313 medical fitness exams for residency visas."
    p = scripted(tmp_path, local, [
        {"text": first, "kind": "claim", "citations": [QUOTE]},
        {"text": "The data value is 1424313.", "kind": "claim", "citations": [QUOTE]},
        {"text": "The number of exams is 1424313.", "kind": "claim", "citations": [QUOTE]},
    ])
    rec = p.answer("How many medical fitness exams for residency visas does Emirates Health Services carry out?")
    assert [s["text"] for s in rec.check["sentences"]] == [first]
    assert rec.status == "awaiting_review"
    verified = next(e for e in p.store.get(rec.id)["audit"] if e["kind"] == "verified")
    assert verified["detail"]["repeats_dropped"] == 2


def test_a_claim_with_a_new_number_is_not_a_repeat(tmp_path, local):
    p = scripted(tmp_path, local, [
        {"text": "Emirates Health Services carried out 1424313 medical fitness exams.", "kind": "claim",
         "citations": [QUOTE]},
        {"text": "Of the 1424313 exams, 300 were by website.", "kind": "claim", "citations": [QUOTE]},
    ])
    rec = p.answer("How many medical fitness exams for residency visas does Emirates Health Services carry out?")
    assert len(rec.check["sentences"]) == 2


def test_a_claim_naming_a_table_is_dropped(tmp_path, local):
    first = "In 2025, Emirates Health Services carried out 1424313 medical fitness exams for residency visas."
    p = scripted(tmp_path, local, [
        {"text": first, "kind": "claim", "citations": [QUOTE]},
        {"text": "The data is from the open_ehs_medical_fitness table.", "kind": "claim", "citations": [QUOTE]},
    ])
    rec = p.answer("How many medical fitness exams for residency visas does Emirates Health Services carry out?")
    assert [s["text"] for s in rec.check["sentences"]] == [first]
    verified = next(e for e in p.store.get(rec.id)["audit"] if e["kind"] == "verified")
    assert verified["detail"]["internal_names_dropped"] == 1


def test_an_english_fact_from_the_fallback_is_said_in_arabic(tmp_path, local):
    from dalil.draft import data_claims
    from dalil.opendata import match_rules, run_rules
    q = "كم عدد بطاقات الهوية الإماراتية التي تجددها الهيئة في السنة؟"
    data = run_rules(match_rules(q), lambda sql: local.sql(sql, limit=200).dicts(), source="local")
    p = scripted(tmp_path, local, data_claims(data))  # what the extractive drafter writes: the English fact
    rec = p.answer(q)
    kept = [s["text"] for s in rec.check["sentences"] if s["kept"]]
    assert kept and all("5856085" in t and "هوية" in t for t in kept if any(ch.isdigit() for ch in t)), kept


def test_a_data_backed_claim_is_not_treated_as_stitched_from_documents(local):
    """The question-words check counts documents; a claim answered by a lookup is not
    joined from two documents. Its wording is still checked against its sources."""
    passages = {p["passage_id"]: dict(p) for p in PASSAGES}
    passages["doc-b:p1:1"] = {**passages["doc-a:p1:2"], "passage_id": "doc-b:p1:1", "doc_id": "doc-b"}
    cit = {"table": MEDICAL.table, "sql": MEDICAL.sql, "column": "exams", "value": "1424313"}
    v = Verifier(passages, data_lookup(local), allowed_sql=[MEDICAL.sql], data_facts={MEDICAL.sql: MEDICAL.fact},
                 question="How many medical fitness exams does Emirates Health Services carry out?")
    q2 = {"passage_id": "doc-b:p1:1", "quote": QUOTE["quote"]}
    claim = {"text": "Emirates Health Services carried out 1424313 medical fitness exams.",
             "citations": [cit, QUOTE, q2]}
    out = v.check_answer([claim]).sentences[0]
    assert out.kept, out.reasons
    # Without a data value in the sentence the documents check still applies as before.
    plain = {"text": "Emirates Health Services carried out medical fitness exams.", "citations": [QUOTE, q2]}
    assert not v.check_answer([plain]).sentences[0].kept


def test_an_arabic_claim_in_other_words_is_backed_by_the_rule_meaning():
    from dalil.verify import support
    rule = next(r for r in RULES if r.name == "tasheel_centres")
    evidence = " ".join([rule.fact_ar.format(centres=49, emirates=7), rule.meaning])
    assert support("عدد مراكز خدمة تسهيل في دولة الإمارات هو 49.", evidence) >= 0.8
    assert support("المراكز موزعة في 7 إمارات.", evidence) >= 0.8
    # A claim about something the data does not count still fails.
    assert support("المراكز مغلقة يوم الجمعة.", evidence) < 0.5
    assert all(r.meaning for r in RULES)


def test_a_passage_unrelated_to_a_data_claim_is_taken_off_it(tmp_path, local):
    rule = next(r for r in RULES if r.name == "tasheel_centres")
    unrelated = {"passage_id": PASSAGES[2]["passage_id"],
                 "quote": "A worker who resigns must give the employer written notice of at least thirty days."}
    p = scripted(tmp_path, local, [{
        "text": "There are 49 Tasheel service centres in the UAE.", "kind": "claim",
        "citations": [unrelated, QUOTE, {"table": rule.table, "sql": rule.sql, "column": "centres", "value": "49"}]}])
    rec = p.answer("How many Tasheel service centres are there? Where do I renew my card?")
    cits = rec.draft[0]["citations"]
    # The quote about service centres stays; the one about resigning is taken off.
    assert [c.get("passage_id") for c in cits if "sql" not in c] == [QUOTE["passage_id"]]
    verified = next(e for e in p.store.get(rec.id)["audit"] if e["kind"] == "verified")
    assert verified["detail"]["unrelated_passages_dropped"] == 1


def test_a_data_claim_lists_its_data_citation_first(tmp_path, local):
    rule = next(r for r in RULES if r.name == "tasheel_centres")
    p = scripted(tmp_path, local, [{
        "text": "There are 49 Tasheel service centres in the UAE.", "kind": "claim",
        "citations": [QUOTE, {"table": rule.table, "sql": rule.sql, "column": "centres", "value": "49"}]}])
    rec = p.answer("How many Tasheel service centres are there?")
    first = next(s for s in rec.check["sentences"] if s["kept"])
    assert "sql" in first["citations"][0] and first["citations"][1].get("passage_id") == QUOTE["passage_id"]


def test_a_count_question_gets_the_number_even_if_the_model_leaves_it_out(tmp_path, local):
    p = scripted(tmp_path, local, [
        {"text": "Renewal requests are made through the service centres or the official website of the authority.",
         "kind": "claim", "citations": [QUOTE]}])
    rec = p.answer("How many Emirates ID cards does ICP renew in a year?")
    kept = [s["text"] for s in rec.check["sentences"] if s["kept"]]
    assert "In 2025, ICP renewed 5856085 Emirates ID cards, according to its open data." in kept


def test_a_question_that_does_not_ask_a_count_is_left_alone(tmp_path, local):
    p = scripted(tmp_path, local, [
        {"text": "Renewal requests are made through the service centres or the official website of the authority.",
         "kind": "claim", "citations": [QUOTE]}])
    rec = p.answer("Where do I renew my Emirates ID card?")
    assert not any(any(c.get("sql") for c in s["citations"]) for s in rec.check["sentences"])


def test_only_the_lookup_the_count_question_asks_about_is_stated():
    from dalil.pipeline import counted_lookups
    complaints = next(r for r in RULES if r.name == "labour_complaints")
    renewals = next(r for r in RULES if r.name == "residence_renewals")
    q = "How many labour complaints does MOHRE solve? I want to renew my card."
    assert counted_lookups(q, "", RULES) == {complaints.sql}
    assert renewals.sql in counted_lookups("كم عدد تصاريح الإقامة التي يتم تجديدها؟", "", RULES)
