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
   heading of a cited passage, or a data value: the cited value or another
   value in the same row the rerun returned. A model that writes "AED 1,500" over a quote saying
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
import string
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


_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i >= 2}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
_NUMBER_WORD = re.compile(
    rf"\b(?:({'|'.join(_TENS)})[\s-]+({'|'.join(list(_UNITS)[1:10])})|({'|'.join([*_UNITS, *_TENS, *_SCALES])}))\b",
    re.IGNORECASE)


# Arabic number words, in the normalised form tokenize() makes (alef forms
# unified, ta marbuta as ha, alef maqsura as ya), with their values. A word
# may carry a prefix such as و or ب; tokenize() strips the article.
_AR_NUMBER_WORDS: dict[str, int] = {}
for _value, _forms in {
    1: "واحد واحده", 2: "اثنان اثنين اثنتان اثنتين يومين شهرين عامين سنتين", 3: "ثلاث ثلاثه", 4: "اربع اربعه",
    5: "خمس خمسه", 6: "ست سته", 7: "سبع سبعه", 8: "ثماني ثمانيه ثمان", 9: "تسع تسعه", 10: "عشر عشره",
    20: "عشرون عشرين", 30: "ثلاثون ثلاثين", 40: "اربعون اربعين", 50: "خمسون خمسين", 60: "ستون ستين",
    70: "سبعون سبعين", 80: "ثمانون ثمانين", 90: "تسعون تسعين", 100: "مائه مئه", 200: "مائتان مائتين مئتان مئتين",
    300: "ثلاثمائه ثلاثمئه", 400: "اربعمائه اربعمئه", 500: "خمسمائه خمسمئه", 600: "ستمائه ستمئه",
    700: "سبعمائه سبعمئه", 800: "ثمانمائه ثمانمئه", 900: "تسعمائه تسعمئه",
    1000: "الف", 2000: "الفان الفين الفا الفي", 1000000: "مليون", 1000000000: "مليار",
}.items():
    for _form in _forms.split():
        _AR_NUMBER_WORDS[_form] = _value


# Dual nouns of time: "أسبوعين" is two weeks.
for _form in "اسبوعين ساعتين دقيقتين ليلتين".split():
    _AR_NUMBER_WORDS[_form] = 2
# Any other word built on these is a number the table does not know, such as
# "آلاف" (thousands) after a count. It is kept as itself, so the check fails
# closed: the claim is cut unless its source uses the same word.
# Whole words only, after a prefix and the article are taken off: "مخالفه"
# and "فتره" contain "الف" and are not numbers.
_AR_LARGE = re.compile(r"^(?:\S*مائه|\S*مئه|الاف|الوف|مليون|ملايين|مليار|مليارات)$")


def _arabic_number_words(text: str) -> set[str]:
    from dalil.retrieve import tokenize
    out = set()
    for token in tokenize(text, stem=False):  # whole words: the stemmer would cut "آلاف" to "اف"
        bare = token[1:] if len(token) > 3 and token[0] in "وبلف" else token
        forms = [token, bare, *(w[2:] for w in (token, bare) if w.startswith("ال") and len(w) > 4)]
        if token.startswith("لل") and len(token) > 4:  # "للثلاثين": لِ + ال
            forms.append(token[2:])
        known = next((w for w in forms if w in _AR_NUMBER_WORDS), None)
        if known:
            out.add(str(_AR_NUMBER_WORDS[known]))
        elif any(_AR_LARGE.match(w) for w in forms):
            out.add(token)
    return out


def numbers(text: str) -> set[str]:
    """Numbers in `text`, with grouping commas removed, so 1,000 matches 1000.

    Numbers written as words count too: "fifty" is 50 and "twenty-one" is 21,
    in Arabic as in English ("ثلاثون" is 30), so a model cannot get a number
    past the check by spelling it out.
    """
    text = unicodedata.normalize("NFKC", text).translate(_ARABIC_DIGITS)
    found = {m.group(0).replace(",", "") for m in _NUMBER.finditer(text)}
    found |= _arabic_number_words(text)
    for m in _NUMBER_WORD.finditer(text):
        if m.group(1):
            found.add(str(_TENS[m.group(1).lower()] + _UNITS[m.group(2).lower()]))
        else:
            w = m.group(3).lower()
            found.add(str(_UNITS.get(w, _TENS.get(w, _SCALES.get(w)))))
    return found


_LAW_NUMBER = re.compile(
    r"\b(?:No\.?|Number|Law|Decree)\s*\(?(\d+)\)?|\b(?:of|in)\s+((?:19|20)\d\d)\b"
    r"|رقم\s*\(?\s*(\d+)\s*\)?|(?:لسنة|لعام|عام|سنة)\s*\(?\s*((?:19|20)\d\d)", re.IGNORECASE)


def law_numbers(text: str) -> set[str]:
    """Numbers a claim uses to name a law: "No. 45", "Law 33", "of 2021"."""
    text = unicodedata.normalize("NFKC", text).translate(_ARABIC_DIGITS)
    return {next(g for g in groups if g) for groups in _LAW_NUMBER.findall(text)}


# Words that flip a claim's meaning. A claim may only use one its evidence uses.
# Arabic ones count as the same class, so an Arabic claim's "لا" is backed by
# an English source's "not".
_NEGATIONS = frozenset("not no never cannot can't don't doesn't isn't aren't won't wasn't weren't "
                       "shouldn't mustn't without nor none neither unable "
                       "لا ولا لم ولم لن ولن ليس ليست ليسوا لست وليس غير دون بدون عدم بعدم".split())
# Share of a claim's key words that must appear in its evidence.
SUPPORT_MIN = 0.6
# For a claim in another language than its sources, where words cannot be
# compared: the least cosine similarity between the claim and its quotes by
# the multilingual embedding model. Set on translated and mismatched pairs;
# see docs/evaluation.md.
CROSS_LANGUAGE_MIN = 0.55
# For a claim with the writer's English version of it: the least similarity
# between the two sentences. On the 25 labelled pairs in
# eval/cross-language-pairs.jsonl, faithful renderings scored 0.67 to 0.83
# (one three-word fragment 0.47) and wrong ones at most 0.54.
RENDERING_MIN = 0.60
# How much of the passage before a quote counts as its context, so a list item
# such as "b. Governmental entities" keeps the lead-in that gives it meaning.
CONTEXT_CHARS = 200


def _words(text: str) -> set[str]:
    from dalil.retrieve import normalise_arabic
    # Arabic vowel marks are dropped first, so "لَا" is read as "لا".
    plain = normalise_arabic(unicodedata.normalize("NFKC", text).lower().translate(_FOLD))
    return set(re.findall(r"[a-z']+|[\u0621-\u064a]+", plain))


def clauses(text: str, size: int = 30) -> list[str]:
    """A quote cut at sentence and clause marks, with long pieces cut again into
    windows of `size` words that overlap by half, plus the whole quote."""
    out = [text]
    for piece in re.split(r"(?<=[.;:\u061b\u060c])\s+|,\s+(?=(?:and|or|provided|unless|if|which|who)\b)", text):
        words = piece.split()
        if len(words) < 4:
            continue
        if len(words) <= size + size // 2:
            out.append(piece)
        else:
            out += [" ".join(words[i:i + size]) for i in range(0, len(words) - size // 2, size // 2)]
    return list(dict.fromkeys(out))


# Arabic words that negate only sometimes: "غير الحكومية" is "non-governmental"
# and "دون سن" is "under the age of".
_AR_SOFT_NEGATIONS = frozenset({"غير", "دون"})
# "غير" backed by English "non"; "دون" before an age or a number ("دون سن
# 18") backed by "under", "below" or "less". Nothing else backs them.
_SOFT_BACKING = {"غير": frozenset({"non"}), "دون": frozenset({"under", "below", "less"})}
_DUN_AMOUNT = re.compile(r"دون\s+(?:سن|عمر|[\d\u0660-\u0669])")


# Phrases built on a negation word that do not negate: "ما لم" (unless),
# "لا سيما" (especially), "لا بد" (must), "لا يزال" and "ما زال" (still), and bounds, which say the same as
# "at least" or "at most": "لا يقل عن", "not less than", "no more than".
_AR_IDIOMS = re.compile(
    r"ما\s+لم|لا\s*سي[ّ]?ما|لا\s+بد|لابد|[لم]ا\s+[يت]زال|[لأ]ل?ا\s+[يت](?:قل|زيد|تجاوز)\w*|[لأ]ل?ا\s+تتجاوز"
    # "لا يمكن أن تكون أقل من" and "لا يجوز أن تزيد على": a bound said longer.
    r"|[لأ]ل?ا\s+(?:يمكن|يجوز)\s+(?:[أا]ن\s+)?(?:[يت]كون\s+)?(?:[أا]قل|[أا]كثر|[يت]قل|[يت]زيد|[يت]تجاوز)\w*"
    r"|\b(?:not|no)\s+(?:be\s+)?(?:less|more|fewer|later|earlier|longer|shorter)\s+than\b"
    r"|\bnot\s+(?:be\s+)?(?:exceed|exceeding)\b",
    re.IGNORECASE)
# A leading "According to Article 43 of the Labour Law," or "وفقاً للمادة 43
# من قانون العمل،": who says it, not what is said.
_ATTRIBUTION = re.compile(
    r"^\s*(?:(?:according\s+to|under|as\s+per|per)\b[^,]{0,120},|(?:وفق[اًا]*|طبق[اًا]*|بموجب|حسب)\s[^،,]{0,120}[،,]"
    # or a whole leading sentence that only names a law: "قانون رقم (26) من
    # عام 2007 ينظم العلاقة بين المالك والمستأجر في دبي." / "Law No. 26 of 2007 ..."
    r"|(?:ال)?(?:قانون|مرسوم)[^.]{0,40}?رقم[^.]{0,160}?\.|law\s+no\.?[^.]{0,160}?\.(?=\s))\s*",
    re.IGNORECASE)
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")
# Scripts no answer here is written in: Cyrillic, Greek, CJK, Hangul.
_FOREIGN_SCRIPT = re.compile("[\u0370-\u03ff\u0400-\u04ff\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]")


def _without_idioms(text: str) -> str:
    return _AR_IDIOMS.sub(" ", text)


def _negations(words: set[str]) -> set[str]:
    """Negation words, also with one attached و, ف or ب: "فلا", "وبدون"."""
    found = set()
    for w in words:
        for form in (w, w[1:] if len(w) > 2 and w[0] in "وفب" else w):
            if form in _NEGATIONS:
                found.add(form)
                break
    return found


# Units that must agree with the source, in English and Arabic. Matched on
# text with Arabic marks and alef forms normalised.
_CURRENCIES = {
    "AED": r"\baed\b|\bdhs?\b|\bdirhams?\b|درهم\w*|دراهم|د\.ا",
    "SAR": r"\bsar\b|\briyals?\b|ريال\w*",
    "USD": r"\busd\b|\bdollars?\b|\$|دولار\w*",
    "EUR": r"\beur\b|\beuros?\b|€|يورو",
    "GBP": r"\bgbp\b|\bpounds?\b|£|جنيه\w*",
}
_TIME_UNITS = {
    "hour": r"hours?|ساعه|ساعات|ساعتين",
    "day": r"days?|يوم\w*|ايام",
    "week": r"weeks?|اسبوع\w*|اسابيع",
    "month": r"months?|شهر\w*|اشهر",
    "year": r"years?|سنه|سنوات|سنين|عام|اعوام",
}
_TIME_AFTER_NUMBER = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:-\s*)?(?:working\s+|calendar\s+|full\s+|consecutive\s+)?("
    + "|".join(f"(?P<{k}>{v})" for k, v in _TIME_UNITS.items()) + r")(?![a-z])", re.IGNORECASE)


def _unit_text(text: str) -> str:
    from dalil.retrieve import normalise_arabic
    return normalise_arabic(unicodedata.normalize("NFKC", text).casefold()).translate(_ARABIC_DIGITS)


def currencies(text: str) -> set[str]:
    """Currencies a text names: "AED 20" and "20 درهماً" are both AED."""
    plain = _unit_text(text)
    return {c for c, pattern in _CURRENCIES.items() if re.search(pattern, plain)}


def timed_numbers(text: str) -> dict[str, set[str]]:
    """Number -> the time units written right after it: "30 days" -> {"30": {"day"}}."""
    out: dict[str, set[str]] = {}
    for m in _TIME_AFTER_NUMBER.finditer(_unit_text(text)):
        unit = next(k for k in _TIME_UNITS if m.group(k))
        out.setdefault(m.group(1).replace(",", ""), set()).add(unit)
    return out


def unit_mismatches(claim: str, evidence: str) -> list[str]:
    """Units the claim gives that its evidence does not: a currency its sources
    never name ("20 ريالاً" over "AED 20"), or a number with another time unit
    than the sources give it ("14 weeks" over "14 days")."""
    problems = [f"names {c}, which its sources do not" for c in sorted(currencies(claim) - currencies(evidence))]
    have = timed_numbers(evidence)
    for n, units in timed_numbers(claim).items():
        if have.get(n) and not units & have[n]:
            problems.append(f"gives {n} in {'/'.join(sorted(units))}s where its sources give "
                            f"{'/'.join(sorted(have[n]))}s")
    return problems


# Words for the person who holds or asks, not for what is asked about.
_ROLE_WORDS = frozenset({"holder", "owner", "applic", "resid", "person", "peopl", "individu"})


def _script(text: str) -> str:
    """"ar" or "en": which script most of the letters in `text` are in."""
    from dalil.lang import language
    return language(text)


def support(claim: str, evidence: str) -> float:
    """Share of the claim's key words, stemmed and without stopwords, found in the evidence."""
    from dalil.retrieve import tokenize
    words = {t for t in tokenize(claim) if not t.isdigit()}
    if not words:
        return 1.0
    return len(words & set(tokenize(evidence))) / len(words)


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
    rows: frozenset[int] = frozenset()  # for a data citation: the rerun rows that hold the value
    # For a passage quote: the text just before it in the passage, such as the
    # lead-in of a list item. Used for the wording and negation checks only.
    context: str = ""
    # For a passage quote: its document's title, which may back only a law's own
    # number and year when the claim names them, as in "Decree-Law No. 45 of 2021".
    title: str = ""


@dataclass
class SentenceCheck:
    text: str
    kind: str
    kept: bool
    reasons: list[str] = field(default_factory=list)
    citations: list[CitationCheck] = field(default_factory=list)
    # For a claim in another language than its sources: the writer's English
    # version of it, checked against the sources like an English claim.
    text_en: str = ""


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
                    **({"text_en": s.text_en} if s.text_en else {}),
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
                 allowed_sql: Iterable[str] | None = None, allowed_uncited: Iterable[str] | None = None,
                 allowed_passages: Iterable[str] | None = None, titles: Mapping[str, str] | None = None,
                 laws: Mapping[str, Any] | None = None,
                 similarity: Callable[[str, list[str]], list[float] | None] | None = None,
                 question: str = "", subjects: Mapping[str, str] | None = None, language: str = "",
                 data_facts: Mapping[str, str] | None = None):
        """`allowed_sql`: the queries the pipeline itself ran. A data citation must
        name one of them exactly, so a drafter cannot write its own SQL, such as
        `SELECT 1500 AS fee`, to back a number. `allowed_uncited`: the fixed
        caveat and refusal texts the pipeline adds. A drafter cannot label its own
        claim a caveat to skip the check. `allowed_passages`: the passages the
        drafter was given, so a citation cannot reach anything else in the corpus,
        such as a passage a reviewer rejected. None leaves a rule off, for tests
        and the self-test only."""
        self.passages = passages
        self.data_lookup = data_lookup
        self.allowed_sql = None if allowed_sql is None else {fold(q)[0] for q in allowed_sql}
        self.allowed_uncited = None if allowed_uncited is None else {fold(t)[0] for t in allowed_uncited}
        self.allowed_passages = None if allowed_passages is None else set(allowed_passages)
        # query -> the fixed sentence that says what its row means (dalil/opendata.py),
        # such as "... counts {exams} residency medical fitness exams in {year}."
        self.data_facts = {fold(q)[0]: f for q, f in (data_facts or {}).items() if f}
        # doc_id -> document title. A title such as "Federal Decree by Law No.
        # (45) of 2021" backs the law's own number and year in a claim.
        self.titles = dict(titles or {})
        # doc_id -> dalil.laws.LawRef, for the check that a claim from one law
        # is not presented as another law's. None or {} leaves it off.
        self.laws = dict(laws or {})
        # Meaning similarity for a claim in another language than its sources.
        # Without it such a claim cannot be checked and is cut.
        self.similarity = similarity
        # The (redacted) question, and doc_id -> title of the documents it is
        # about (dalil.subjects). Empty leaves those rules off.
        self.question = question
        self.subjects = dict(subjects or {})
        # The language the answer goes out in, "en" or "ar"; "" leaves it unchecked.
        self.language = language

    def check_quote(self, citation: Mapping[str, Any]) -> CitationCheck:
        cit = dict(citation)
        pid, quote = cit.get("passage_id"), cit.get("quote") or ""
        passage = self.passages.get(pid) if pid else None
        if passage is None:
            return CitationCheck(False, f"unknown passage {pid!r}", cit)
        if self.allowed_passages is not None and pid not in self.allowed_passages:
            return CitationCheck(False, "passage was not among the sources the drafter was given", cit)
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
        before = passage["text"][max(0, index[at] - CONTEXT_CHARS):index[at]]
        return CitationCheck(True, "quote found", cit, passage["doc_id"], passage["page"], start, end,
                             evidence=evidence, context=before, title=self.titles.get(passage["doc_id"], ""))

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
        matched = frozenset(i for i, row in enumerate(rows)
                            if row.get(column) is not None and want == fold(str(row[column]))[0])
        if matched:
            # The rerun returned the whole row, so its other values are verified
            # too: "1424313 exams in 2025" may cite the count alone when the year
            # sits in the same returned row. Only rows holding the cited value count.
            same_row = [str(rows[i][k]) for i in sorted(matched) for k in rows[i]
                        if k != column and rows[i][k] is not None]
            # The rule's own sentence, filled from the rerun row, says what the
            # value means, so a claim in other words can be checked against it.
            meaning = _fill(self.data_facts.get(fold(sql)[0]), rows[min(matched)])
            return CitationCheck(True, "value returned by the query", cit,
                                 evidence=" ".join([str(value), *same_row, meaning]), rows=matched)
        return CitationCheck(False, f"query did not return the cited value in column {column!r}", cit)

    def check_sentence(self, sentence: Mapping[str, Any]) -> SentenceCheck:
        text = (sentence.get("text") or "").strip()
        kind = sentence.get("kind") or "claim"
        result = SentenceCheck(text, kind, kept=True, text_en=(sentence.get("text_en") or "").strip())
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
        if self.language and text and _script(text) != self.language:
            result.kept = False
            result.reasons.append("not written in the language of the question")
            return result
        if _FOREIGN_SCRIPT.search(text):
            result.kept = False
            result.reasons.append("has words in a script that is neither the answer's nor its sources': "
                                  + "".join(sorted(set(_FOREIGN_SCRIPT.findall(text))))[:20])
            return result
        for cit in citations:
            check = self.check_data(cit) if "sql" in cit or "table" in cit else self.check_quote(cit)
            result.citations.append(check)
            if not check.ok:
                result.kept = False
                result.reasons.append(check.reason)
        if result.kept:
            backed = set().union(*(numbers(c.evidence) for c in result.citations))
            titled = set().union(*(numbers(c.title) for c in result.citations)) & law_numbers(text)
            missing = sorted(numbers(text) - backed - titled, key=lambda n: (len(n), n))
            if missing:
                result.kept = False
                result.reasons.append("numbers not in any cited source: " + ", ".join(missing))
        if result.kept:
            units = unit_mismatches(text, " ".join(f"{c.evidence} {c.title}" for c in result.citations))
            if units:
                result.kept = False
                result.reasons.append("units do not match its sources: " + "; ".join(units))
        if result.kept:
            # Values cited from one query must come from one row: "4242 in 2025"
            # cannot pair 2025 from one row with 4242 from another.
            by_sql: dict[str, list[CitationCheck]] = {}
            for c in result.citations:
                if c.rows:
                    by_sql.setdefault(fold(c.citation.get("sql", ""))[0], []).append(c)
            if any(not frozenset.intersection(*(c.rows for c in group)) for group in by_sql.values()):
                result.kept = False
                result.reasons.append("data values come from different rows")
        quotes = [c.evidence for c in result.citations if c.ok and c.doc_id]
        backing = [c.evidence for c in result.citations if c.ok and c.rows]
        if result.kept and backing and not quotes and self.data_facts:
            # Backed by open data alone: the claim's key words must mostly be in
            # what the lookup counts (the rule's fact and meaning), as a reworded
            # passage claim must be in its quotes.
            share = support(text, " ".join(backing))
            if share < SUPPORT_MIN:
                result.kept = False
                result.reasons.append(f"claim is not supported by its data (key words found: {share:.0%})")
        if result.kept and quotes:
            # A drafter may reword its sources. The claim's key words must mostly
            # be in them, and it may not add a negation they lack.
            # A cited document's title counts too: the writer is told to name
            # the law it quotes, and naming it adds no claim.
            evidence = " ".join(f"{c.title} {c.context} {c.evidence}" for c in result.citations)
            # A claim citing at least one passage in its own language is matched
            # word for word, as any same-language claim is: an Arabic quote of
            # an Arabic article stays checkable even when it also cites the
            # English version of that article.
            cross = _script(text) != _script(" ".join(quotes)) \
                and not any(_script(q) == _script(text) for q in quotes)
            if cross and result.text_en and _script(result.text_en) == _script(" ".join(quotes)):
                self.check_rendering(result, evidence, quotes)
            elif cross:
                # Words cannot be matched across languages; compare meaning,
                # against each clause of the quotes, since a claim usually
                # renders one clause of a long legal sentence.
                windows = [w for q in quotes for w in clauses(q)]
                sims = (self.similarity(text, windows) or []) if self.similarity else []
                if not sims:
                    result.kept = False
                    result.reasons.append("claim is in another language than its source, and no model is "
                                          "available to compare them")
                elif max(sims) < CROSS_LANGUAGE_MIN:
                    result.kept = False
                    result.reasons.append(f"claim does not match its source in meaning (similarity {max(sims):.2f}, "
                                          f"needs {CROSS_LANGUAGE_MIN:.2f})")
            else:
                share = support(text, evidence)
                if share < SUPPORT_MIN:
                    result.kept = False
                    result.reasons.append(f"claim is not supported by its sources (key words found: {share:.0%})")
                elif self.question:
                    missing = self.stitched(text, result.citations)
                    if missing:
                        result.kept = False
                        result.reasons.append("no one source says what the claim answers; missing from every "
                                              "single document: " + ", ".join(missing))
                    elif (absent := self.asked_but_absent(text, evidence)):
                        result.kept = False
                        result.reasons.append("names what the question asks about but its sources do not say: "
                                              + ", ".join(absent))
            # Negations count as one class: "cannot" is backed by a source's "will not".
            # "No." in "Law No. 45" is a law reference, not a negation.
            added = sorted(_negations(_words(_without_idioms(_LAW_NUMBER.sub(" ", text)))))
            backing = _negations(_words(evidence))
            if not backing and added and set(added) <= _AR_SOFT_NEGATIONS:
                # "غير" can be "non-" and "دون" "under"; English that says so backs them.
                words = _words(evidence)
                if all(words & _SOFT_BACKING[w] and (w != "دون" or _DUN_AMOUNT.search(text)) for w in added):
                    backing = set(added)
            if result.kept and added and not backing:
                result.kept = False
                result.reasons.append("claim adds a negation its sources do not have: " + ", ".join(added))
        return result

    def check_rendering(self, result: SentenceCheck, evidence: str, quotes: list[str]) -> None:
        """A claim in another language than its sources, with the writer's
        English version of it. The English version is checked against the
        sources as an English claim is: key words, numbers, negations, and
        not stitched from two documents. Then the claim must say what its
        English version says: the same numbers, a negation in one only if in
        the other, and a meaning similarity of at least RENDERING_MIN between
        the two sentences, which is a much closer test than a sentence
        against a whole legal paragraph."""
        text, en = result.text, result.text_en
        backed = set().union(*(numbers(c.evidence) for c in result.citations))
        # Latin words in the Arabic may only be names and acronyms from the
        # sources or the English version, such as "ICP" or "MoHRE". A stray
        # "acordo", or English such as "provided" left in an Arabic sentence,
        # is a slip of the model.
        known = {w for w in _LATIN_WORD.findall(f"{en} {evidence}") if w[0].isupper()}
        stray = sorted({w for w in _LATIN_WORD.findall(text) if w not in known})
        titled = set().union(*(numbers(c.title) for c in result.citations)) & law_numbers(en)
        missing = sorted(numbers(en) - backed - titled, key=lambda n: (len(n), n))
        problems = []
        if stray:
            problems.append("it has words in another language that neither its sources nor its English version "
                            "have: " + ", ".join(stray))
        if missing:
            problems.append("its English version has numbers not in any cited source: " + ", ".join(missing))
        share = support(en, evidence)
        if share < SUPPORT_MIN:
            problems.append(f"its English version is not supported by its sources (key words found: {share:.0%})")
        en_neg = _negations(_words(_without_idioms(_LAW_NUMBER.sub(" ", en))))
        if en_neg and not _negations(_words(evidence)):
            problems.append("its English version adds a negation its sources do not have: " + ", ".join(sorted(en_neg)))
        # Every number of the English version must be in the claim too. One and
        # two are left out: Arabic often says them in the noun itself, "سنة"
        # (a year), "الطرفين" (both parties). The claim's own numbers were
        # checked against the sources above.
        # A cited law's number and year in the English version name the law;
        # the claim may name it another way, so they are not counted as
        # dropped. A number the English says outside a law's name still is.
        said = numbers(_LAW_NUMBER.sub(" ", unicodedata.normalize("NFKC", en).translate(_ARABIC_DIGITS))) \
            | (law_numbers(en) - titled)
        dropped = sorted(said - numbers(text) - {"1", "2"}, key=lambda n: (len(n), n))
        if dropped:
            problems.append("it leaves out numbers its English version has: " + ", ".join(dropped))
        ar_neg = {w for w in _negations(_words(_without_idioms(text)))
                  if not (w in _AR_SOFT_NEGATIONS and _words(en) & _SOFT_BACKING[w])}
        if bool(ar_neg) != bool(en_neg):
            problems.append("it negates where its English version does not, or the other way round")
        if not problems:
            sims = (self.similarity(_ATTRIBUTION.sub("", text), [_ATTRIBUTION.sub("", en)]) or []) \
                if self.similarity else []
            if not sims:
                problems.append("no model is available to compare it with its English version")
            elif sims[0] < RENDERING_MIN:
                problems.append(f"it does not say what its English version says (similarity {sims[0]:.2f}, "
                                f"needs {RENDERING_MIN:.2f})")
        if not problems and self.question:
            gaps = self.stitched(en, result.citations)
            if gaps:
                problems.append("no one source says what the claim answers; missing from every single document: "
                                + ", ".join(gaps))
            elif (absent := self.asked_but_absent(en, evidence)):
                problems.append("its English version names what the question asks about but its sources do not "
                                "say: " + ", ".join(absent))
        if problems:
            result.kept = False
            result.reasons.extend(problems)

    def asked_but_absent(self, claim: str, evidence: str) -> list[str]:
        """The claim's words that repeat the question and that its sources do
        not carry, such as "parents" in "Golden visa holders can sponsor their
        parents" over "the ability to sponsor family members, including
        spouses and children". A claim answers with what its sources say;
        echoing the asked word over a source that never says it is the
        commonest way a small model states the answer the resident hoped for.
        Words that only say what kind of answer is wanted ("need", "can",
        "how long") do not count."""
        from dalil.retrieve import tokenize
        from dalil.safety import subject_generic
        asked = ({t for t in tokenize(claim) if not t.isdigit()} & set(tokenize(self.question))) \
            - subject_generic() - _ROLE_WORDS
        return sorted(asked - set(tokenize(evidence)))

    def stitched(self, claim: str, citations: list[CitationCheck]) -> list[str]:
        """The claim's words that repeat the question and that no single cited
        document holds, or [] when one document holds them all.

        The words a claim shares with the question are what it answers. When
        they are found only across two documents, the claim joins two rules,
        as in "Golden visa holders can sponsor their parents", built from a
        Golden visa page that names spouses and children and a page on a UAE
        citizen's parent. Each document counts with its title and the text
        just before the quote."""
        from dalil.retrieve import tokenize
        asked = {t for t in tokenize(claim) if not t.isdigit()} & set(tokenize(self.question))
        if not asked:
            return []
        # A claim that states a value its data citation reran answers from that
        # lookup, not from documents joined together; its wording is still
        # checked against its sources and the rule's own sentence above.
        stated = numbers(claim)
        if any(c.ok and c.citation.get("sql") and numbers(str(c.citation.get("value") or "")) <= stated
               and numbers(str(c.citation.get("value") or "")) for c in citations):
            return []
        by_doc: dict[str, set[str]] = {}
        for c in citations:
            if c.ok and c.doc_id:
                by_doc.setdefault(c.doc_id, set()).update(tokenize(f"{c.title} {c.context} {c.evidence}"))
        if len(by_doc) < 2:
            return []  # one document: the support check above already applies
        best = min((asked - words for words in by_doc.values()), key=len)
        return sorted(best)

    def check_subject(self, checks: list[SentenceCheck]) -> None:
        """Cut a claim that applies another service's rule to the one asked about.

        When the question names a service or law that has its own document,
        a claim that cites only other documents is kept only if the text it
        quotes names that subject too."""
        from dalil.subjects import names_subject
        if not self.subjects:
            return
        names = [t.split("|")[0].strip() for t in self.subjects.values()]
        for s in checks:
            if not s.kept or s.kind in UNCITED_KINDS:
                continue
            docs = {c.doc_id for c in s.citations if c.ok and c.doc_id}
            # A law is the authority on its topic, not another service's page;
            # the law rules above check which law a claim presents.
            if not docs or docs & set(self.subjects) or docs & set(self.laws):
                continue
            if any(names_subject(f"{c.title} {c.context} {c.evidence}", names) for c in s.citations if c.ok):
                continue
            s.kept = False
            s.reasons.append(f"its source is about another service, not {' or '.join(sorted(set(names)))}, "
                             "which the question asks about")

    def check_laws(self, checks: list[SentenceCheck]) -> None:
        """Cut a claim that presents one law's text as another law's.

        Two rules, on the kept claims of one answer:

        1. A claim that names a law's jurisdiction ("the federal law", "the
           Dubai law", "the UAE data protection law") must cite that
           jurisdiction's law.
        2. When the answer also cites other documents, a claim quoting a law
           must name it, by its jurisdiction, number or short name. Otherwise
           "The provisions of this Law will apply to Federal Government
           Entities", from a Dubai law, reads as if it were about the federal
           law the other sentences describe.
        """
        from dalil.laws import jurisdictions_named
        kept = [s for s in checks if s.kept and s.kind not in UNCITED_KINDS]
        docs_of = {id(s): {c.doc_id for c in s.citations if c.ok and c.doc_id} for s in kept}
        for s in kept:
            # Documents the other claims cite: a claim that cites a law and a
            # page summarising it, alone in its answer, confuses no one.
            all_docs = set().union(*(d for k, d in docs_of.items() if k != id(s)), docs_of[id(s)])
            other_docs = set().union(*(d for k, d in docs_of.items() if k != id(s)))
            refs = [self.laws[d] for d in docs_of[id(s)] if d in self.laws]
            if not refs:
                continue
            named = jurisdictions_named(s.text)
            if named and not any(r.jurisdiction in named for r in refs):
                s.kept = False
                s.reasons.append(f"names {' and '.join(sorted(named))} law but quotes {refs[0].label()}")
                continue
            same_law = {d for d in all_docs if d in self.laws and self.laws[d].key in {r.key for r in refs}}
            if other_docs - same_law and not any(r.named_in(s.text) for r in refs):
                s.kept = False
                s.reasons.append(f"quotes {refs[0].label()} without naming it, next to other sources")

    def check_answer(self, sentences: Iterable[Mapping[str, Any]]) -> AnswerCheck:
        """Status from the claims. Caveats and refusals the pipeline adds do not
        count toward what was kept, so they cannot carry an answer whose claims
        were all cut. An answer that is only refusals, and all of them kept,
        passes: refusing is the right answer then."""
        checks = [self.check_sentence(s) for s in sentences]
        if self.laws:
            self.check_laws(checks)
        self.check_subject(checks)
        claims = [s for s in checks if s.kind not in UNCITED_KINDS]
        cut_claims = sum(not s.kept for s in claims)
        cut_other = sum(not s.kept for s in checks if s.kind in UNCITED_KINDS)
        if not checks:
            status = "held"
        elif not claims:
            status = "pass" if cut_other == 0 else "held"
        elif cut_claims == len(claims):
            # Every claim was cut. With Dalil's fixed refusal kept, the
            # refusal is the answer and the cuts are shown to the reviewer.
            status = "pass_with_cuts" if any(s.kept and s.kind == "refusal" for s in checks) else "held"
        elif cut_claims == 0 and cut_other == 0:
            status = "pass"
        elif cut_claims / len(claims) <= MAX_CUT_SHARE:
            status = "pass_with_cuts"
        else:
            status = "held"
        return AnswerCheck(status, checks)


# --- rerunning data citations on the open-data source --------------------------

_WRITE_WORDS = re.compile(
    r"\b(insert|update|delete|merge|drop|create|alter|truncate|grant|revoke|cache|uncache|"
    r"refresh|msck|load|set|use|call|exec|execute|describe|show)\b", re.IGNORECASE)
_CALL = re.compile(r"\b(?:java_method|reflect|input_file_name|current_user|session_user|user)\s*\(", re.IGNORECASE)
_TABLE_REF = re.compile(r"\b(?:from|join)\s+([A-Za-z_][\w.]*)", re.IGNORECASE)


def _fill(template: str | None, row: Mapping[str, Any]) -> str:
    """A fact template with its {column} slots filled from a row, or "" if any is missing."""
    if not template:
        return ""
    slots = [f for _, f, _, _ in string.Formatter().parse(template) if f]
    if any(row.get(s) in (None, "") for s in slots):
        return ""
    return template.format(**{s: row[s] for s in slots})


def safe_select(sql: str, allowed_schema: str = "dalil") -> str:
    """Return `sql` if it is one read-only SELECT over `allowed_schema` tables.

    A drafted answer's data citation carries SQL the model wrote, so the
    checker refuses anything else before it reaches the data source.
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


def data_lookup(source: Any, allowed_schema: str = "dalil") -> DataLookup:
    """A DataLookup that reruns a citation's SQL on the open-data source,
    Zetaris or the local database (dalil.datasource), read-only."""
    def lookup(sql: str) -> list[dict[str, Any]]:
        return source.sql(safe_select(sql, allowed_schema), limit=200).dicts()
    return lookup


# The old name: the checker's rerun was Zetaris-only when it was written.
zetaris_lookup = data_lookup
