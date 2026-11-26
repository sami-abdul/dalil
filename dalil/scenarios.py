"""Scenario files: questions with the outcome each must show, written down
before the run, checked the same way every time.

A scenario file is JSON:

    {"name": "open-data", "description": "...", "repeat": 10,
     "scenarios": [{"id": "D1", "title": "...", "question": "...",
                    "expect": "what should happen, in words",
                    "checks": {"states_value": {"table": "...", "column": "...", "value": "..."}}}]}

Every check under "checks" must hold for the run to pass. The checks:

    status            the record's status, "awaiting_review" or "held"
    states_value      a kept claim states the value (thousands separators
                      allowed) and carries a data citation on that table and
                      column whose rerun returned it
    cites_table       a kept claim carries a passing data citation on the table
    source            every lookup that returned rows ran on this source kind
    cites_passage     true: a kept claim carries a passing passage citation
    cites_doc         a kept claim carries a passing citation to this doc_id
    no_claims         true: nothing but fixed refusals and caveats
    nothing_retrieved true: retrieval was skipped
    flags_include     each listed flag, or flag prefix ending in ":", is set
    question_contains each string is in the stored, redacted question
    kept_includes     each named fixed sentence is kept: WEAK_REFUSAL,
                      FEE_CAVEAT, LAW_CAVEAT, PERSONAL_DATA_CAVEAT,
                      REFUSAL_TEXT, REFUSAL_FOLLOW_UP, ASK_FOR_DETAIL
    kept_any_of       at least one of the named fixed sentences is kept
    not_stored        none of these strings appears anywhere in the stored row
    no_money          true: no kept sentence states a price
    claims_language   "en" or "ar": every kept claim is written in that language
    no_lookup         true: no open-data query ran

The fixed sentences are the knowledge base's own (dalil/kb.py), so a UK
scenario's REFUSAL_TEXT is the UK refusal.

A check that is not on this list fails the run, so a typo cannot pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from dalil import safety
from dalil.draft import _MONEY, FEE_CAVEAT, LAW_CAVEAT, WEAK_REFUSAL

FIXED = {
    "WEAK_REFUSAL": WEAK_REFUSAL, "FEE_CAVEAT": FEE_CAVEAT, "LAW_CAVEAT": LAW_CAVEAT,
    "PERSONAL_DATA_CAVEAT": safety.PERSONAL_DATA_CAVEAT, "REFUSAL_TEXT": safety.REFUSAL_TEXT,
    "REFUSAL_FOLLOW_UP": safety.REFUSAL_FOLLOW_UP, "ASK_FOR_DETAIL": safety.ASK_FOR_DETAIL,
}


@dataclass
class Scenario:
    id: str
    title: str
    question: str
    expect: str
    checks: dict[str, Any]


@dataclass
class ScenarioFile:
    name: str
    description: str
    repeat: int
    scenarios: list[Scenario] = field(default_factory=list)


def load(path: Path | str) -> ScenarioFile:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = [Scenario(s["id"], s["title"], s["question"], s["expect"], dict(s["checks"]))
             for s in raw["scenarios"]]
    if not items or len({s.id for s in items}) != len(items):
        raise ValueError(f"{path}: scenarios need unique ids")
    for s in items:
        unknown = set(s.checks) - set(CHECKS)
        if unknown or not s.checks:
            raise ValueError(f"{path}: {s.id} has unknown or no checks: {sorted(unknown)}")
    return ScenarioFile(raw["name"], raw.get("description", ""), int(raw.get("repeat", 1)), items)


def _kept(check: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [s for s in check.get("sentences", []) if s.get("kept")]


def _claims(check: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [s for s in _kept(check) if s.get("kind", "claim") == "claim"]


_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
                                "01234567890123456789")


def _digits(text: str) -> str:
    """Arabic-Indic digits as ASCII, without thousands separators."""
    return re.sub(r"(?<=\d)[,٬](?=\d{3})", "", text.translate(_ARABIC_DIGITS))


def _states_value(rec: Any, row: Any, want: Mapping[str, str]) -> bool:
    value = str(want["value"])
    for s in _claims(rec.check):
        if not re.search(rf"(?<![\d.]){re.escape(value)}(?![\d]|\.\d)", _digits(s["text"])):
            continue
        for c in s.get("citations", []):
            if (c.get("ok") and c.get("table") == want["table"] and c.get("column") == want["column"]
                    and str(c.get("value")) == value and c.get("reason") == "value returned by the query"):
                return True
    return False


def _cites_table(rec: Any, row: Any, table: str) -> bool:
    return any(c.get("ok") and c.get("table") == table for s in _claims(rec.check) for c in s["citations"])


def _source(rec: Any, row: Any, kind: str) -> bool:
    ran = [d for d in rec.data if d.get("rows")]
    return bool(ran) and all(d.get("source") == kind for d in ran)


def _cites_passage(rec: Any, row: Any, want: bool) -> bool:
    has = any(c.get("ok") and c.get("passage_id") for s in _claims(rec.check) for c in s["citations"])
    return has == bool(want)


def _cites_doc(rec: Any, row: Any, doc: str) -> bool:
    return any(c.get("ok") and c.get("doc_id") == doc for s in _claims(rec.check) for c in s["citations"])


def _no_claims(rec: Any, row: Any, want: bool) -> bool:
    return (not _claims(rec.check)) == bool(want) and bool(_kept(rec.check))


def _nothing_retrieved(rec: Any, row: Any, want: bool) -> bool:
    return (not rec.retrieval) == bool(want)


def _flags_include(rec: Any, row: Any, flags: list[str]) -> bool:
    return all(any(f == want or (want.endswith(":") and f.startswith(want)) for f in rec.flags) for want in flags)


def _question_contains(rec: Any, row: Any, parts: list[str]) -> bool:
    return all(p in rec.question for p in parts)


def _kept_text(rec: Any) -> str:
    return " ".join(s["text"] for s in _kept(rec.check))


def _kept_includes(rec: Any, row: Any, names: list[str], fixed: Mapping[str, str] = FIXED) -> bool:
    return all(fixed[n] in _kept_text(rec) for n in names)


def _kept_any_of(rec: Any, row: Any, names: list[str], fixed: Mapping[str, str] = FIXED) -> bool:
    return any(fixed[n] in _kept_text(rec) for n in names)


def _no_lookup(rec: Any, row: Any, want: bool) -> bool:
    return (not rec.data) == bool(want)


def _not_stored(rec: Any, row: Any, raw: list[str]) -> bool:
    dumped = json.dumps(row, ensure_ascii=False)
    return not any(x in dumped for x in raw)


def _no_money(rec: Any, row: Any, want: bool) -> bool:
    return (not _MONEY.search(_kept_text(rec))) == bool(want)


def _claims_language(rec: Any, row: Any, want: str) -> bool:
    from dalil.lang import language
    claims = _claims(rec.check)
    return bool(claims) and all(language(s["text"]) == want for s in claims)


def _status(rec: Any, row: Any, want: str) -> bool:
    return rec.status == want


CHECKS = {
    "status": _status, "states_value": _states_value, "cites_table": _cites_table, "source": _source,
    "cites_passage": _cites_passage, "cites_doc": _cites_doc, "no_claims": _no_claims,
    "nothing_retrieved": _nothing_retrieved, "flags_include": _flags_include,
    "question_contains": _question_contains, "kept_includes": _kept_includes, "kept_any_of": _kept_any_of,
    "not_stored": _not_stored, "no_money": _no_money, "claims_language": _claims_language,
    "no_lookup": _no_lookup,
}
_USES_FIXED = {"kept_includes", "kept_any_of"}


def evaluate(scenario: Scenario, rec: Any, row: Any, fixed: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Each check's result for one run. The run passes when all are True.
    `fixed` overrides the named fixed sentences, for another knowledge base."""
    names = {**FIXED, **(fixed or {})}
    return {name: bool(CHECKS[name](rec, row, arg, names) if name in _USES_FIXED else CHECKS[name](rec, row, arg))
            for name, arg in scenario.checks.items()}
