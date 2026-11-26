"""Scenario files and their checks. The checks must be able to fail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dalil import scenarios as sc
from dalil.pipeline import Record

ROOT = Path(__file__).resolve().parent.parent
FILES = sorted((ROOT / "eval" / "scenarios").glob("*.json")) + sorted(ROOT.glob("kb/*/scenarios.json"))


def record(sentences=(), flags=(), question="q", data=(), retrieval=(), status="awaiting_review"):
    return Record(id="x", question=question, flags=list(flags), status=status, retrieval=list(retrieval),
                  data=list(data), check={"status": "pass", "sentences": list(sentences)})


def claim(text, citations=(), kept=True, kind="claim"):
    return {"text": text, "kind": kind, "kept": kept, "reasons": [], "citations": list(citations)}


DATA_CIT = {"ok": True, "reason": "value returned by the query", "table": "dalil.t", "column": "n", "value": "5856085",
            "sql": "SELECT n FROM dalil.t"}
WANT = {"table": "dalil.t", "column": "n", "value": "5856085"}


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_committed_scenario_files_load_and_fail_on_an_empty_answer(path):
    spec = sc.load(path)
    assert spec.repeat >= 1
    for s in spec.scenarios:
        assert s.expect.strip()
        assert not all(sc.evaluate(s, record(status="held"), {"audit": []}).values()), s.id


def test_states_value_accepts_thousands_separators_and_needs_the_rerun():
    ok = record([claim("ICP renewed 5,856,085 cards in 2025.", [DATA_CIT])])
    assert sc.CHECKS["states_value"](ok, {}, WANT)
    wrong_value = record([claim("ICP renewed 5,856,086 cards.", [DATA_CIT])])
    assert not sc.CHECKS["states_value"](wrong_value, {}, WANT)
    rounded = record([claim("ICP renewed about 5.9 million cards.", [DATA_CIT])])
    assert not sc.CHECKS["states_value"](rounded, {}, WANT)
    failed_rerun = record([claim("ICP renewed 5856085 cards.", [{**DATA_CIT, "ok": False, "reason": "query failed"}])])
    assert not sc.CHECKS["states_value"](failed_rerun, {}, WANT)
    cut = record([claim("ICP renewed 5856085 cards.", [DATA_CIT], kept=False)])
    assert not sc.CHECKS["states_value"](cut, {}, WANT)
    longer = record([claim("ICP renewed 15856085 cards.", [DATA_CIT])])
    assert not sc.CHECKS["states_value"](longer, {}, WANT)


def test_source_check_needs_a_lookup_that_returned_rows():
    assert sc.CHECKS["source"](record(data=[{"rows": [{"n": "1"}], "source": "local"}]), {}, "local")
    assert not sc.CHECKS["source"](record(data=[{"rows": [{"n": "1"}], "source": "zetaris"}]), {}, "local")
    assert not sc.CHECKS["source"](record(data=[{"rows": [], "source": ""}]), {}, "local")


def test_no_claims_needs_something_kept():
    from dalil.draft import WEAK_REFUSAL
    assert sc.CHECKS["no_claims"](record([claim(WEAK_REFUSAL, kind="refusal")]), {}, True)
    assert not sc.CHECKS["no_claims"](record([]), {}, True)
    assert not sc.CHECKS["no_claims"](record([claim("A claim.")]), {}, True)


def test_not_stored_reads_the_whole_row():
    assert not sc.CHECKS["not_stored"](record(), {"question": "AB123456C"}, ["AB123456C"])
    assert sc.CHECKS["not_stored"](record(), {"question": "[NATIONAL INSURANCE]"}, ["AB123456C"])


def test_unknown_check_names_are_refused(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"name": "x", "scenarios": [
        {"id": "A", "title": "t", "question": "q", "expect": "e", "checks": {"stats": "held"}}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        sc.load(f)
