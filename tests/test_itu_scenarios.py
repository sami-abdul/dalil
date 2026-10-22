"""The ITU scenario checks must be able to fail. Synthetic records only."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from dalil.pipeline import Record

ROOT = Path(__file__).resolve().parent.parent


def load():
    spec = importlib.util.spec_from_file_location("run_itu_scenarios", ROOT / "scripts" / "run_itu_scenarios.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


def record(sentences, question="q", flags=(), retrieval=()):
    return Record(id="x", question=question, flags=list(flags), status="awaiting_review",
                  retrieval=list(retrieval), check={"status": "pass", "sentences": sentences})


def sentence(text, kind="claim", kept=True, citations=()):
    return {"text": text, "kind": kind, "kept": kept, "reasons": [], "citations": list(citations)}


def steps():
    return {sid: steps for sid, _, steps in load().scenarios()}


def test_every_scenario_check_fails_on_an_empty_answer():
    empty = record([])
    for sid, ss in steps().items():
        for step in ss:
            assert not step.check(empty, {}, {"audit": []}), f"{sid}: '{step.expect}' passed an empty answer"


def test_fee_check_fails_on_a_price_in_any_form():
    s1 = steps()["S1"][0]
    from dalil.draft import FEE_CAVEAT
    for price in ("The fee is AED 500.", "It costs 500 dirhams.", "Pay Dh 500."):
        assert not s1.check(record([sentence(price), sentence(FEE_CAVEAT, kind="caveat")]), {}, {})


def test_personal_data_check_fails_if_the_raw_number_is_stored():
    from dalil import safety
    s2 = steps()["S2"][0]
    r = record([sentence(safety.PERSONAL_DATA_CAVEAT, kind="caveat")], flags=["personal_data:emirates_id"])
    assert not s2.check(r, {}, {"question": "My Emirates ID is 784-1990-1234567-1"})
