"""Open-data lookups that can back a sentence with a Zetaris query.

A small rule-based mapper: certain words in the question pick one fixed,
parameter-free, read-only SELECT over a dalil.open_* table. Nothing from the
question reaches the SQL. Each rule returns one row, and its `fact` template
names the columns a drafter may state, so every number in the sentence is a
value the verifier can rerun.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from dalil.verify import safe_select


@dataclass(frozen=True)
class DataRule:
    name: str
    pattern: re.Pattern[str]
    table: str
    sql: str
    # A sentence with {column} slots; each slot becomes a data citation.
    fact: str


@dataclass
class DataResult:
    table: str
    sql: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""
    fact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"table": self.table, "sql": self.sql, "rows": self.rows, "note": self.note, "fact": self.fact}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DataResult":
        return cls(d["table"], d["sql"], list(d.get("rows") or []), d.get("note") or "", d.get("fact"))


RULES: tuple[DataRule, ...] = (
    DataRule(
        "emirates_id_renewals", re.compile(r"\bemirates\s+id\b|\bid\s+card\b", re.I),
        "dalil.open_icp_transactions",
        "SELECT year, transactions FROM dalil.open_icp_transactions "
        "WHERE service_en = 'Renewal of ID Card' ORDER BY year DESC LIMIT 1",
        "ICP open data counts {transactions} Emirates ID card renewals in {year}.",
    ),
    DataRule(
        "residence_renewals", re.compile(r"\brenew(?:al|als|ing|ed)?\b", re.I),
        "dalil.open_icp_transactions",
        "SELECT year, transactions FROM dalil.open_icp_transactions "
        "WHERE service_en = 'Renewal of Residence Permits' ORDER BY year DESC LIMIT 1",
        "ICP open data counts {transactions} residence permit renewals in {year}.",
    ),
    DataRule(
        "tasheel_centres", re.compile(r"\btasheel\b", re.I),
        "dalil.open_tasheel_centres",
        "SELECT COUNT(*) AS centres, COUNT(DISTINCT emirate_en) AS emirates FROM dalil.open_tasheel_centres",
        "MOHRE open data lists {centres} Tasheel service centres across {emirates} emirates.",
    ),
    DataRule(
        "medical_fitness", re.compile(r"\bmedical\s+(?:fitness|test|exam)", re.I),
        "dalil.open_ehs_medical_fitness",
        "SELECT year, CAST(SUM(CAST(total AS BIGINT)) AS BIGINT) AS exams "
        "FROM dalil.open_ehs_medical_fitness GROUP BY year ORDER BY year DESC LIMIT 1",
        "Emirates Health Services open data counts {exams} residency medical fitness exams in {year}.",
    ),
    DataRule(
        "labour_complaints", re.compile(r"\bcomplain(?:t|ts|ing)?\b", re.I),
        "dalil.open_mohre_solved_complaints",
        "SELECT Year AS year, CAST(SUM(CAST(SolvedComplaints_count AS BIGINT)) AS BIGINT) AS solved "
        "FROM dalil.open_mohre_solved_complaints GROUP BY Year ORDER BY Year DESC LIMIT 1",
        "MOHRE open data records {solved} solved private-sector labour complaints in {year}.",
    ),
)

# Every fixed query must pass the same guard the verifier applies.
for _rule in RULES:
    safe_select(_rule.sql)


def match_rules(question: str, rules: Iterable[DataRule] = RULES) -> list[DataRule]:
    """Rules the question triggers. The Emirates ID rule wins over the general
    renewal rule, so one question runs at most one ICP query."""
    hits = [r for r in rules if r.pattern.search(question)]
    if any(r.name == "emirates_id_renewals" for r in hits):
        hits = [r for r in hits if r.name != "residence_renewals"]
    return hits


SqlRunner = Callable[[str], Iterable[Mapping[str, Any]]]


def run_rules(rules: Iterable[DataRule], run: SqlRunner | None, unavailable: str = "") -> list[DataResult]:
    """Run each rule's query. With no runner, or on a failure, the result has
    no rows and a note saying why; the pipeline carries on without it."""
    out = []
    for rule in rules:
        res = DataResult(rule.table, rule.sql, fact=rule.fact)
        if run is None:
            res.note = unavailable or "Zetaris unavailable; lookup skipped"
        else:
            try:
                res.rows = [dict(r) for r in run(rule.sql)]
                if not res.rows:
                    res.note = "query returned no rows"
            except Exception as e:  # ZetarisError messages carry no credentials
                res.note = f"lookup failed: {type(e).__name__}: {str(e)[:200]}"
        out.append(res)
    return out
