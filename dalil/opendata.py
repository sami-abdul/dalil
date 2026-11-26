"""Open-data lookups that can back a sentence with a fixed query.

A small rule-based mapper: certain words in the question pick one fixed,
parameter-free, read-only SELECT over a dalil.open_* table. Nothing from the
question reaches the SQL. Each rule returns one row, and its `fact` template
names the columns a drafter may state, so every number in the sentence is a
value the verifier can rerun.

The queries run on whichever source dalil.datasource picked, Zetaris or the
local database; each result records which one, so the screen can say so.
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
    # The same sentence in Arabic, for a question asked in Arabic.
    fact_ar: str = ""
    # What the value counts, in plain words, in both languages. The checker
    # reads it with the fact as the meaning of a data citation, so a claim that
    # says the same thing in other words is not cut for its wording.
    meaning: str = ""


@dataclass
class DataResult:
    table: str
    sql: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""
    fact: str | None = None
    # "local" or "zetaris": the source that ran the query. Empty when none did.
    source: str = ""
    fact_ar: str | None = None
    meaning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"table": self.table, "sql": self.sql, "rows": self.rows, "note": self.note, "fact": self.fact,
                "source": self.source, "fact_ar": self.fact_ar, "meaning": self.meaning}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DataResult":
        return cls(d["table"], d["sql"], list(d.get("rows") or []), d.get("note") or "", d.get("fact"),
                   d.get("source") or "", d.get("fact_ar"), d.get("meaning"))


RULES: tuple[DataRule, ...] = (
    DataRule(
        "emirates_id_renewals",
        re.compile(r"\bemirates\s+id\b|\bid\s+card\b|الهوية\s+ال[إا]ماراتية|بطاقة\s+الهوية", re.I),
        "dalil.open_icp_transactions",
        "SELECT year, transactions FROM dalil.open_icp_transactions "
        "WHERE service_en = 'Renewal of ID Card' ORDER BY year DESC LIMIT 1",
        "In {year}, ICP renewed {transactions} Emirates ID cards, according to its open data.",
        "جددت الهيئة الاتحادية للهوية والجنسية {transactions} بطاقة هوية إماراتية في عام {year}، "
        "بحسب بياناتها المفتوحة.",
        "The number of Emirates ID cards the Federal Authority for Identity and Citizenship (ICP) renews in a "
        "year. عدد بطاقات الهوية الإماراتية التي تجددها الهيئة الاتحادية للهوية والجنسية في السنة، أي عدد "
        "عمليات تجديد الهوية.",
    ),
    DataRule(
        "residence_renewals", re.compile(r"\brenew(?:al|als|ing|ed)?\b|تجديد|يجدد|تجدد", re.I),
        "dalil.open_icp_transactions",
        "SELECT year, transactions FROM dalil.open_icp_transactions "
        "WHERE service_en = 'Renewal of Residence Permits' ORDER BY year DESC LIMIT 1",
        "In {year}, ICP renewed {transactions} residence permits, according to its open data.",
        "جددت الهيئة الاتحادية للهوية والجنسية {transactions} تصريح إقامة في عام {year}، بحسب بياناتها المفتوحة.",
        "The number of residence permits (residence visas) ICP renews in a year. عدد تصاريح الإقامة وتأشيرات "
        "الإقامة التي تجددها الهيئة الاتحادية للهوية والجنسية في السنة.",
    ),
    DataRule(
        "tasheel_centres", re.compile(r"\btasheel\b|تسهيل", re.I),
        "dalil.open_tasheel_centres",
        "SELECT COUNT(*) AS centres, COUNT(DISTINCT emirate_en) AS emirates FROM dalil.open_tasheel_centres",
        "There are {centres} Tasheel service centres across {emirates} emirates, according to MOHRE's open data.",
        "يوجد {centres} مركز خدمة تسهيل في {emirates} إمارات، بحسب البيانات المفتوحة لوزارة الموارد البشرية "
        "والتوطين.",
        "The number of Tasheel service centres of the Ministry of Human Resources and Emiratisation (MOHRE) in "
        "the UAE, and the number of emirates they are spread across. عدد مراكز خدمة تسهيل التابعة لوزارة "
        "الموارد البشرية والتوطين في دولة الإمارات، والمراكز موزعة على عدد من الإمارات.",
    ),
    DataRule(
        "medical_fitness",
        re.compile(r"\bmedical\s+(?:fitness|test|exam)|الفحص\s+الطبي|فحص\s+طبي|فحوصات\s+طبية|اللياقة\s+الطبية", re.I),
        "dalil.open_ehs_medical_fitness",
        "SELECT year, CAST(SUM(CAST(total AS BIGINT)) AS BIGINT) AS exams "
        "FROM dalil.open_ehs_medical_fitness GROUP BY year ORDER BY year DESC LIMIT 1",
        "In {year}, Emirates Health Services carried out {exams} medical fitness exams for residency, according "
        "to its open data.",
        "أجرت مؤسسة الإمارات للخدمات الصحية {exams} فحص لياقة طبية للإقامة في عام {year}، بحسب بياناتها المفتوحة.",
        "The number of medical fitness exams for residency visas that Emirates Health Services carries out in a "
        "year. عدد فحوصات اللياقة الطبية لتأشيرات الإقامة التي تجريها مؤسسة الإمارات للخدمات الصحية في السنة.",
    ),
    DataRule(
        # The file repeats Abu Dhabi's figures under Al-Ain, Delma and Madina
        # Zayed, Sharjah's under Kalbaa and Khorfkan, and one free-zone figure
        # under five names, and 2020 looks partial (data/open/README.md). So
        # the sum counts each emirate's own row once and stops at 2019.
        "labour_complaints", re.compile(r"\bcomplain(?:t|ts|ing)?\b|شكوى|شكاوى|شكاوي", re.I),
        "dalil.open_mohre_solved_complaints",
        "SELECT Year AS year, CAST(SUM(CAST(SolvedComplaints_count AS BIGINT)) AS BIGINT) AS solved "
        "FROM dalil.open_mohre_solved_complaints "
        "WHERE City_En IN ('Abu Dhabi', 'Dubai', 'Sharjah', 'Ajman', 'Om-El-Quain', 'Ras El-Khaema', 'Fujiarah') "
        "AND Year <= 2019 GROUP BY Year ORDER BY Year DESC LIMIT 1",
        "In {year}, MOHRE solved {solved} private-sector labour complaints, counting each emirate once, "
        "according to its open data.",
        "حلت وزارة الموارد البشرية والتوطين {solved} شكوى عمالية في القطاع الخاص في عام {year}، مع احتساب كل "
        "إمارة دون تكرار، بحسب بياناتها المفتوحة.",
        "The number of private-sector labour complaints that MOHRE solves in a year. عدد الشكاوى العمالية في "
        "القطاع الخاص التي تحلها وزارة الموارد البشرية والتوطين في السنة.",
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


def run_rules(rules: Iterable[DataRule], run: SqlRunner | None, unavailable: str = "",
              source: str = "") -> list[DataResult]:
    """Run each rule's query. With no runner, or on a failure, the result has
    no rows and a note saying why; the pipeline carries on without it.
    `source` names the backend that ran the queries ("local" or "zetaris")."""
    out = []
    for rule in rules:
        res = DataResult(rule.table, rule.sql, fact=rule.fact, source=source if run is not None else "",
                         fact_ar=rule.fact_ar or None, meaning=rule.meaning or None)
        if run is None:
            res.note = unavailable or "Open data not configured; lookup skipped"
        else:
            try:
                res.rows = [dict(r) for r in run(rule.sql)]
                if not res.rows:
                    res.note = "query returned no rows"
            except Exception as e:  # neither source puts credentials in its messages
                res.note = f"lookup failed: {type(e).__name__}: {str(e)[:200]}"
        out.append(res)
    return out
