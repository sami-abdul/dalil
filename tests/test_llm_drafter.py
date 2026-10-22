"""Tests for the model drafter, with the model call faked. Synthetic text only."""

from __future__ import annotations

import json

import pytest

from dalil.draft import DraftError, ExtractiveDrafter, FallbackDrafter, LLMDrafter
from dalil.opendata import DataResult
from dalil.retrieve import Hit


def hit(pid, text="Late renewal costs AED 20 a day, up to AED 1,000.", doc="doc-a"):
    return Hit(passage_id=pid, doc_id=doc, page=1, score=10.0, heading=None, text=text, start=0, end=len(text))


class Canned(LLMDrafter):
    def __init__(self, reply):
        super().__init__(url="http://127.0.0.1:1/v1")
        self.reply, self.seen = reply, {}

    def complete(self, messages, schema):
        self.seen = {"messages": messages, "schema": schema}
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


def test_sentences_are_numbered_and_short_ones_left_out():
    n = LLMDrafter.numbered([hit("p1", "Late renewal costs AED 20 a day. Ok then. Up to AED 1,000 in all."), hit("p2")])
    assert n["S1"] == ("p1", "Late renewal costs AED 20 a day.")
    assert ("p1", "Ok then.") not in n.values()
    assert n["S2"] == ("p1", "Up to AED 1,000 in all.")
    assert n["S3"][0] == "p2"


def test_schema_admits_only_the_given_sentences_and_queries():
    data = [DataResult("dalil.t", "SELECT a FROM dalil.t", [{"a": "1"}])]
    numbered = LLMDrafter.numbered([hit("p1"), hit("p2")])
    s = LLMDrafter.schema(numbered, data)
    cite = s["properties"]["sentences"]["items"]["properties"]["citations"]["items"]["anyOf"]
    assert cite[0]["properties"]["sentence"]["enum"] == list(numbered)
    assert cite[1]["properties"]["sql"]["enum"] == ["SELECT a FROM dalil.t"]


def test_reply_becomes_claims_and_prompt_carries_notes():
    d = Canned({"sentences": [{"text": "Late renewal costs AED 20 a day.", "citations": [{"sentence": "S1"}]},
                              {"text": "Made up.", "citations": [{"sentence": "S99"}]}]})
    out = d.draft("How much is the late renewal fine?", [hit("p1")], [], notes=["cite the fee page"])
    assert out[0]["citations"] == [{"passage_id": "p1", "quote": "Late renewal costs AED 20 a day, up to AED 1,000."}]
    assert out[1]["citations"][0]["passage_id"] == "unknown:S99"  # the checker cuts it
    assert "cite the fee page" in d.seen["messages"][1]["content"]
    assert "[S1]" in d.seen["messages"][1]["content"]


def test_empty_reply_is_a_fixed_refusal():
    out = Canned({"sentences": []}).draft("q", [hit("p1")], [])
    assert out[0]["kind"] == "refusal"


def test_non_json_reply_raises():
    with pytest.raises(DraftError):
        Canned("not json").draft("q", [hit("p1")], [])


def test_fallback_uses_extractive_drafter_and_says_so():
    f = FallbackDrafter(Canned("not json"), ExtractiveDrafter())
    f.draft("How much is the late renewal fine?", [hit("p1")], [])
    assert f.last_used == "ExtractiveDrafter" and f.last_error == "DraftError"
    ok = FallbackDrafter(Canned({"sentences": []}), ExtractiveDrafter())
    ok.draft("q", [hit("p1")], [])
    assert ok.last_used == "Canned" and ok.last_error == ""
