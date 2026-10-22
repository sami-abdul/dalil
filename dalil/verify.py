"""Check every sentence of a drafted answer against the text it cites.

A draft is a list of sentences. Each sentence carries citations:

    {"text": "...", "citations": [{"passage_id": "doc:p1:3", "quote": "..."}]}
    {"text": "...", "citations": [{"table": "dalil.open_icp_transactions",
                                   "sql": "SELECT ...", "value": "5856085"}]}

A sentence is kept only if all of these hold:

1. It has at least one citation.
2. Every passage citation's quote is found word for word in that passage,
   is at least MIN_QUOTE_WORDS words long, so "the" cannot prop up a claim,
   and does not start or end partway through a word or a number.
3. Every data citation names a query the pipeline ran, the table it reads and
   a column, and rerunning it through `data_lookup` returns the value in
   that column.
4. Every number in the sentence appears in one of its verified quotes, the
   heading of a cited passage, or a data value. A model that writes "AED 1,500" over a quote saying
   "AED 1,000" loses the sentence.

"Word for word" forgives only what a copy-paste or a line break changes:
runs of whitespace, curly against straight quotes, and dash variants. Case,
spelling and punctuation must match.

The answer passes when every sentence is kept. It passes with cuts when some
are cut but at most MAX_CUT_SHARE of them, and at least one is kept. Otherwise
it is held for a person, and nothing goes out without one anyway.

Sentences marked "kind": "refusal" or "kind": "caveat" are statements about
the answer itself, such as "The sources don't give this year's fee." They
need no citation, but they may not contain numbers.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

MIN_QUOTE_WORDS = 4
MAX_CUT_SHARE = 0.5
UNCITED_KINDS = {"refusal", "caveat"}

_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
})
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# A number: digits, optionally grouped with commas, optionally with a decimal part.
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def fold(text: str) -> tuple[str, list[int]]:
    """Normalise for matching, and map each output character to its input index.

    NFKC, fold quotes and dashes, and collapse every whitespace run to one
    space. Returns the folded string and, for each of its characters, the
    index in `text` it came from, so a match can be mapped back to the page.
    """
    out: list[str] = []
    index: list[int] = []
    in_space = False
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch).translate(_FOLD):
            if c.isspace():
                if not in_space and out:
                    out.append(" ")
                    index.append(i)
                in_space = True
                continue
            in_space = False
            out.append(c)
            index.append(i)
    if out and out[-1] == " ":
        out.pop()
        index.pop()
    return "".join(out), index


def numbers(text: str) -> set[str]:
    """Numbers in `text`, with grouping commas removed, so 1,000 matches 1000."""
    text = unicodedata.normalize("NFKC", text).translate(_ARABIC_DIGITS)
    return {m.group(0).replace(",", "") for m in _NUMBER.finditer(text)}


@dataclass
class CitationCheck:
    ok: bool
    reason: str
    citation: dict[str, Any]
    # For a verified passage quote: where it sits in the page text.
    doc_id: str | None = None
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    evidence: str = ""  # the verified quote or data value, used for the number check


@dataclass
class SentenceCheck:
    text: str
    kind: str
    kept: bool
    reasons: list[str] = field(default_factory=list)
    citations: list[CitationCheck] = field(default_factory=list)


@dataclass
class AnswerCheck:
    status: str  # "pass", "pass_with_cuts" or "held"
    sentences: list[SentenceCheck]

    @property
    def kept(self) -> list[SentenceCheck]:
        return [s for s in self.sentences if s.kept]

    @property
    def cut(self) -> list[SentenceCheck]:
        return [s for s in self.sentences if not s.kept]

    def answer_text(self) -> str:
        return " ".join(s.text for s in self.kept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "answer": self.answer_text(),
            "sentences": [
                {
                    "text": s.text, "kind": s.kind, "kept": s.kept, "reasons": s.reasons,
                    "citations": [
                        {"ok": c.ok, "reason": c.reason, "doc_id": c.doc_id, "page": c.page,
                         "page_start": c.page_start, "page_end": c.page_end, **c.citation}
                        for c in s.citations
                    ],
                }
                for s in self.sentences
            ],
        }


def _find_whole(text: str, quote: str) -> int:
    """Index of `quote` in `text` where it neither starts nor ends inside a word
    or a number, or -1. "AED 1" is not found in "AED 1,000"."""
    start = 0
    while (at := text.find(quote, start)) >= 0:
        end = at + len(quote)
        before = text[at - 1] if at else ""
        after = text[end] if end < len(text) else ""
        after2 = text[end + 1] if end + 1 < len(text) else ""
        breaks_word = (before.isalnum() and quote[:1].isalnum()) or (after.isalnum() and quote[-1:].isalnum())
        # A number continues through a grouping comma or decimal point: 1,000 and 2.5.
        breaks_number = (quote[-1:].isdigit() and after in ",." and after2.isdigit()) or \
                        (quote[:1].isdigit() and before in ",." and at >= 2 and text[at - 2].isdigit())
        if not (breaks_word or breaks_number):
            return at
        start = at + 1
    return -1


def load_passages(path: Path | str) -> dict[str, dict[str, Any]]:
    """passage_id -> passage record, from build/passages.jsonl."""
    out: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["passage_id"]] = rec
    return out


DataLookup = Callable[[str], Iterable[Mapping[str, Any]]]


class Verifier:
    def __init__(self, passages: Mapping[str, Mapping[str, Any]], data_lookup: DataLookup | None = None,
                 allowed_sql: Iterable[str] | None = None, allowed_uncited: Iterable[str] | None = None):
        """`allowed_sql`: the queries the pipeline itself ran. A data citation must
        name one of them exactly, so a drafter cannot write its own SQL, such as
        `SELECT 1500 AS fee`, to back a number. `allowed_uncited`: the fixed
        caveat and refusal texts the pipeline adds. A drafter cannot label its own
        claim a caveat to skip the check. None leaves either rule off, for tests
        and the self-test only."""
        self.passages = passages
        self.data_lookup = data_lookup
        self.allowed_sql = None if allowed_sql is None else {fold(q)[0] for q in allowed_sql}
        self.allowed_uncited = None if allowed_uncited is None else {fold(t)[0] for t in allowed_uncited}

    def check_quote(self, citation: Mapping[str, Any]) -> CitationCheck:
        cit = dict(citation)
        pid, quote = cit.get("passage_id"), cit.get("quote") or ""
        passage = self.passages.get(pid) if pid else None
        if passage is None:
            return CitationCheck(False, f"unknown passage {pid!r}", cit)
        folded_quote, _ = fold(quote)
        if len(folded_quote.split()) < MIN_QUOTE_WORDS:
            return CitationCheck(False, f"quote shorter than {MIN_QUOTE_WORDS} words", cit)
        folded_text, index = fold(passage["text"])
        at = _find_whole(folded_text, folded_quote)
        if at < 0:
            return CitationCheck(False, "quote not found word for word in the passage", cit)
        start = passage["start"] + index[at]
        end = passage["start"] + index[at + len(folded_quote) - 1] + 1
        # The heading counts as evidence for numbers, so "Under Article 51" is
        # backed by a quote from the body of Article (51).
        evidence = f"{quote} {passage.get('heading') or ''}"
        return CitationCheck(True, "quote found", cit, passage["doc_id"], passage["page"], start, end,
                             evidence=evidence)

    def check_data(self, citation: Mapping[str, Any]) -> CitationCheck:
        cit = dict(citation)
        sql, value, column, table = cit.get("sql"), cit.get("value"), cit.get("column"), cit.get("table")
        if not sql or not column or value in (None, ""):
            return CitationCheck(False, "data citation needs sql, column and value", cit)
        if self.allowed_sql is not None and fold(sql)[0] not in self.allowed_sql:
            return CitationCheck(False, "query is not one Dalil ran", cit)
        read = {t.lower() for t in _TABLE_REF.findall(sql)}
        if not table or table.lower() not in read:
            return CitationCheck(False, "cited table is not the table the query reads", cit)
        if self.data_lookup is None:
            return CitationCheck(False, "no data source to rerun the query", cit)
        try:
            rows = list(self.data_lookup(sql))
        except Exception as e:  # the reviewer sees why the rerun failed
            return CitationCheck(False, f"query failed: {type(e).__name__}", cit)
        want = fold(str(value))[0]
        if any(row.get(column) is not None and want == fold(str(row[column]))[0] for row in rows):
            return CitationCheck(True, "value returned by the query", cit, evidence=str(value))
        return CitationCheck(False, f"query did not return the cited value in column {column!r}", cit)

    def check_sentence(self, sentence: Mapping[str, Any]) -> SentenceCheck:
        text = (sentence.get("text") or "").strip()
        kind = sentence.get("kind") or "claim"
        result = SentenceCheck(text, kind, kept=True)
        citations = sentence.get("citations") or []
        if kind in UNCITED_KINDS:
            if self.allowed_uncited is not None and fold(text)[0] not in self.allowed_uncited:
                result.kept = False
                result.reasons.append(f"not one of Dalil's fixed {kind} sentences, so it needs a citation")
            elif numbers(text):
                result.kept = False
                result.reasons.append(f"a {kind} may not state numbers")
            return result
        if not citations:
            result.kept = False
            result.reasons.append("no citation")
            return result
        for cit in citations:
            check = self.check_data(cit) if "sql" in cit or "table" in cit else self.check_quote(cit)
            result.citations.append(check)
            if not check.ok:
                result.kept = False
                result.reasons.append(check.reason)
        if result.kept:
            backed = set().union(*(numbers(c.evidence) for c in result.citations))
            missing = sorted(numbers(text) - backed)
            if missing:
                result.kept = False
                result.reasons.append("numbers not in any cited source: " + ", ".join(missing))
        return result

    def check_answer(self, sentences: Iterable[Mapping[str, Any]]) -> AnswerCheck:
        """Status from the claims. Caveats and refusals the pipeline adds do not
        count toward what was kept, so they cannot carry an answer whose claims
        were all cut. An answer that is only refusals, and all of them kept,
        passes: refusing is the right answer then."""
        checks = [self.check_sentence(s) for s in sentences]
        claims = [s for s in checks if s.kind not in UNCITED_KINDS]
        cut_claims = sum(not s.kept for s in claims)
        cut_other = sum(not s.kept for s in checks if s.kind in UNCITED_KINDS)
        if not checks:
            status = "held"
        elif not claims:
            status = "pass" if cut_other == 0 else "held"
        elif cut_claims == len(claims):
            status = "held"
        elif cut_claims == 0 and cut_other == 0:
            status = "pass"
        elif cut_claims / len(claims) <= MAX_CUT_SHARE:
            status = "pass_with_cuts"
        else:
            status = "held"
        return AnswerCheck(status, checks)


# --- rerunning data citations through Zetaris --------------------------------

_WRITE_WORDS = re.compile(
    r"\b(insert|update|delete|merge|drop|create|alter|truncate|grant|revoke|cache|uncache|"
    r"refresh|msck|load|set|use|call|exec|execute|describe|show)\b", re.IGNORECASE)
_CALL = re.compile(r"\b(?:java_method|reflect|input_file_name|current_user|session_user|user)\s*\(", re.IGNORECASE)
_TABLE_REF = re.compile(r"\b(?:from|join)\s+([A-Za-z_][\w.]*)", re.IGNORECASE)


def safe_select(sql: str, allowed_schema: str = "dalil") -> str:
    """Return `sql` if it is one read-only SELECT over `allowed_schema` tables.

    A drafted answer's data citation carries SQL the model wrote, so the
    checker refuses anything else before it reaches Zetaris.
    """
    s = sql.strip().rstrip(";").strip()
    if any(ch in s for ch in "`\""):
        raise ValueError("no quoted identifiers in a citation query")
    if not re.match(r"(?is)^(select|with)\b", s):
        raise ValueError("only SELECT queries can back a citation")
    if ";" in s or "--" in s or "/*" in s:
        raise ValueError("one statement, no comments")
    if _WRITE_WORDS.search(s):
        raise ValueError("query uses a word that is not allowed in a citation")
    # A comma join, FROM a, b, would hide the second table from the check below.
    for clause in re.findall(r"(?is)\bfrom\b(.*?)(?=\bwhere\b|\bgroup\b|\border\b|\blimit\b|\bjoin\b|\bon\b|$)", s):
        if "," in clause:
            raise ValueError("no comma joins in a citation query")
    if _CALL.search(s):
        raise ValueError("query calls a function that is not allowed in a citation")
    tables = _TABLE_REF.findall(s)
    if not tables or any(t.split(".")[0].lower() != allowed_schema for t in tables):
        raise ValueError(f"queries may read only {allowed_schema}.* tables")
    return s


def zetaris_lookup(client: Any, allowed_schema: str = "dalil") -> DataLookup:
    """A DataLookup that reruns a citation's SQL on Zetaris, read-only."""
    def lookup(sql: str) -> list[dict[str, Any]]:
        return client.sql(safe_select(sql, allowed_schema), limit=200).dicts()
    return lookup
