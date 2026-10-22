"""Tests for the review-memory guidance. Synthetic questions only."""

from __future__ import annotations

from dalil.hmem import guidance, similarity

Q = "How much notice do I have to give to resign, and what happens if I do not?"


def mem(i, kind, question, passages, note=""):
    return {"id": f"mem-{i}", "kind": kind, "question": question, "passage_ids": passages, "note": note}


def test_similarity_separates_rewordings_from_other_topics():
    assert similarity(Q, "If I resign, how many days notice must I give my employer?") >= 0.5
    assert similarity(Q, "What is the fine for overstaying a tourist visa?") == 0
    assert similarity(Q, "") == 0


def test_only_similar_rejections_leave_passages_out():
    g = guidance(Q, [
        mem(1, "rejected", "How much notice do I give to resign?", ["a", "b"], "about probation"),
        mem(2, "rejected", "How do I renew my Emirates ID?", ["c"], "wrong"),
    ])
    assert g.avoid == {"a", "b"} and g.notes == ["about probation"] and g.similar == ["mem-1"]


def test_an_approval_of_a_similar_question_keeps_its_passages():
    g = guidance(Q, [
        mem(1, "rejected", "How much notice do I give to resign?", ["a", "b"], "stiff wording"),
        mem(2, "approved", "What notice must I give to resign?", ["b"]),
    ])
    assert g.avoid == {"a"}


def test_no_note_placeholder_is_not_passed_on():
    g = guidance(Q, [mem(1, "rejected", "How much notice do I give to resign?", ["a"], "(no note)")])
    assert g.notes == []
