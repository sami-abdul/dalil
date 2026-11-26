"""Which law a document is, and what a question about a law's reach needs.

Two problems this module serves, both general rather than tied to one law:

* A question such as "Does the data protection law cover government
  entities?" is answered by the law's own scope article: what it applies to,
  and what it says it does not apply to. Retrieval can rank that article low,
  because its words ("shall not apply to the following") share little with
  the question. ``is_scope_question`` spots such a question, and
  ``scope_passages`` finds the scope article of each law in the results, so
  the pipeline can hand it to the writer.
* A sentence quoted from one law can sit in an answer as if it were about
  another, as when a Dubai law's scope ("will apply to Federal Government
  Entities") appears next to the federal law. ``LawRef`` names each law in the
  manifest by its jurisdiction, number and year, so the checker can tell when
  a claim names one law and cites another, or cites a law without naming it.

Everything here reads the manifest's titles and topics, so a knowledge base
for another country or field gets the same behaviour from its own manifest.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Bidi controls that some PDF titles carry around Arabic digits.
_BIDI = dict.fromkeys(map(ord, "‎‏‪‫‬‭‮⁦⁧⁨⁩"), None)

_LAW_TITLE = re.compile(r"\b(?:law|decree|act|code)\b|قانون|مرسوم", re.IGNORECASE)
_NUMBER_EN = re.compile(r"\bNo\.?\s*\(?\s*(\d+)\s*\)?", re.IGNORECASE)
_NUMBER_AR = re.compile(r"رقم\s*[()\s]*(\d+)")
_YEAR = re.compile(r"(?:\bof\b|لسنة)\s*[()\s]*((?:19|20)\d\d)\b", re.IGNORECASE)

# Jurisdictions, each with the words that name it in English and Arabic.
JURISDICTIONS: dict[str, tuple[str, ...]] = {
    "federal": ("federal", "اتحادي", "الاتحادي", "اتحادية"),
    "dubai": ("dubai", "دبي"),
    "abu dhabi": ("abu dhabi", "أبوظبي", "ابوظبي", "أبو ظبي"),
    "sharjah": ("sharjah", "الشارقة"),
}


def _plain(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").translate(_BIDI)


@dataclass(frozen=True)
class LawRef:
    """One law, as its manifest title names it."""

    doc_id: str
    title: str
    jurisdiction: str  # "federal", "dubai", ... or "" when the title does not say
    number: str
    year: str
    aliases: tuple[str, ...] = field(default=())  # acronyms from the manifest topic, such as "PDPL"

    @property
    def key(self) -> tuple[str, str, str]:
        """The same law in two languages has one key."""
        return (self.jurisdiction, self.number, self.year)

    def label(self) -> str:
        where = f" ({self.jurisdiction.title()})" if self.jurisdiction else ""
        return f"Law No. {self.number} of {self.year}{where}"

    def named_in(self, text: str) -> bool:
        """Whether `text` names this law: its jurisdiction as a law's, its number, or an alias."""
        plain = _plain(text).casefold()
        if self.jurisdiction and self.jurisdiction in jurisdictions_named(text):
            return True
        from dalil.verify import law_numbers
        if self.number in law_numbers(plain.translate(_DIGITS)):  # "No. 26", "Law 26", "رقم 26"
            return True
        return any(re.search(rf"\b{re.escape(a.casefold())}\b", plain) for a in self.aliases)


_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def law_ref(doc_id: str, title: str, topic: str = "") -> LawRef | None:
    """The LawRef for a manifest row, or None when the title is not a numbered law."""
    plain = _plain(title).translate(_DIGITS)
    if not _LAW_TITLE.search(plain):
        return None
    number = _NUMBER_EN.search(plain) or _NUMBER_AR.search(plain)
    year = _YEAR.search(plain)
    if not number or not year:
        return None
    folded = plain.casefold()
    jurisdiction = next((j for j, words in JURISDICTIONS.items() if any(w in folded for w in words)), "")
    if not jurisdiction:
        # A topic such as "PDPL Federal Decree-Law 45 of 2021" can say what the title leaves out.
        topic_folded = _plain(topic).casefold()
        jurisdiction = next((j for j, words in JURISDICTIONS.items() if any(w in topic_folded for w in words)), "")
    # A short name the manifest topic gives the law itself: "PDPL Federal Decree-Law 45".
    aliases = tuple(sorted(set(re.findall(r"\b((?-i:[A-Z]{3,6}))\b(?=\s+(?:federal\s+)?(?:decree|law))",
                                          topic or "", flags=re.IGNORECASE)) - {"UAE", "AED", "ICP"}))
    return LawRef(doc_id, title, jurisdiction, number.group(1), year.group(1), aliases)


def law_refs(rows: Iterable[Mapping[str, str]]) -> dict[str, LawRef]:
    """doc_id -> LawRef for every numbered law in manifest rows."""
    from dalil.corpus import doc_id
    out = {}
    for r in rows:
        ref = law_ref(doc_id(r["file"]), r.get("title", ""), r.get("topic", ""))
        if ref:
            out[ref.doc_id] = ref
    # The same law in two languages shares its short names: the Arabic PDPL's
    # topic says "PDPL", and the English one is the same law.
    by_key: dict[tuple[str, str, str], set[str]] = {}
    for ref in out.values():
        by_key.setdefault(ref.key, set()).update(ref.aliases)
    from dataclasses import replace
    return {d: replace(ref, aliases=tuple(sorted(by_key[ref.key]))) for d, ref in out.items()}


_LAW_EN = r"(?:law|decree|decree-law|decree\s+by\s+law|act|legislation)"
_LAW_AR = r"(?:قانون|القانون|مرسوم|المرسوم|تشريع|التشريع)"


_NOT_A_LAW = re.compile(r"(?:الجهات|جهات|الحكومية|حكومية|الحكومة|حكومة|الهيئة|هيئة|المؤسسات|مؤسسات)$")


def _arabic_law_of(plain: str, word: str) -> bool:
    """An Arabic law noun qualified by a jurisdiction: "القانون الاتحادي",
    "مرسوم بقانون اتحادي", "قانون ... لإمارة دبي". Not "الجهات الحكومية
    الاتحادية", where the adjective qualifies the entities, not the law."""
    for m in re.finditer(rf"{_LAW_AR}((?:\s+\S+){{0,3}}?)\s+(?:ل|في\s+|لإمارة\s+|إمارة\s+)?{word}(?![\u0621-\u064a])", plain):
        between = m.group(1).split()
        if not between or not _NOT_A_LAW.search(between[-1]):
            return True
    return False


def jurisdictions_named(text: str) -> set[str]:
    """Jurisdictions a claim names as the source of a law: "the Dubai law",
    "federal decree-law", "law of the Emirate of Dubai", "the UAE data protection
    law", "القانون الاتحادي", "قانون ... لإمارة دبي". A jurisdiction word that
    does not qualify a law, as in "Federal Government Entities", does not count."""
    plain = _plain(text).casefold()
    out = set()
    for j, words in {**JURISDICTIONS, "federal": (*JURISDICTIONS["federal"], "uae", "united arab emirates",
                                                    "الإمارات")}.items():
        for w in words:
            w = re.escape(w)
            if re.search(rf"\b{w}\b(?:\s+[\w'-]+){{0,3}}?\s+{_LAW_EN}\b", plain) \
                    or re.search(rf"\b{_LAW_EN}\b(?:\s+\S+){{0,4}}?\s+(?:of|in)\s+(?:the\s+)?(?:emirate\s+of\s+)?{w}\b", plain) \
                    or _arabic_law_of(plain, w):
                out.add(j)
    return out


# --------------------------------------------------------------------------
# Questions about a law's reach

_SCOPE_WORDS = re.compile(
    r"\b(?:cover(?:s|ed|age)?|appl(?:y|ies|ied|icable)|scope|exempt(?:s|ed|ion)?|exclud(?:e|es|ed)|"
    r"subject\s+to|protect(?:s|ed)?\s+(?:me|my|us|our))\b"
    r"|يطبق|يُطبق|تطبق|تُطبق|ينطبق|تنطبق|يسري|تسري|يشمل|تشمل|نطاق|مستثنى|مستثناة|يستثني|تستثني|يخضع|تخضع",
    re.IGNORECASE)
_LAW_WORDS = re.compile(r"\b(?:laws?|decree(?:-law)?|act|legislation|regulations?|pdpl)\b|قانون|مرسوم|تشريع|لائحة",
                        re.IGNORECASE)


def is_scope_question(question: str) -> bool:
    """A question about what a law covers, applies to or leaves out."""
    return bool(_SCOPE_WORDS.search(question) and _LAW_WORDS.search(question))


# A scope article: "Scope of Application", "shall (not) apply to", "will apply
# to", and the Arabic "نطاق التطبيق", "تسري أحكام", "لا تسري".
_SCOPE_TEXT = re.compile(
    r"\bscope\s+of\s+application\b|\b(?:shall|will)\s+(?:not\s+)?apply\s+to\b|\bshall\s+be\s+applicable\b"
    r"|نطاق\s+(?:ال)?تطبيق|تسري\s+أحكام|لا\s+تسري",
    re.IGNORECASE)
# How far into a law its scope article can sit, in pages. Scope comes first,
# after the definitions; a later "shall not apply" is an exception elsewhere.
SCOPE_PAGES = 12


def is_scope_passage(passage: Mapping[str, Any]) -> bool:
    return int(passage.get("page") or 0) <= SCOPE_PAGES and bool(_SCOPE_TEXT.search(passage.get("text") or ""))


def scope_passages(doc_ids: Sequence[str], passages: Mapping[str, Mapping[str, Any]],
                   language: str | None = None, per_doc: int = 2) -> list[str]:
    """Passage ids of each law's scope article, in the order of `doc_ids`.

    `language` keeps only documents in that language, so an English question
    gets the English text of a law and an Arabic one the Arabic text.
    """
    by_doc: dict[str, list[Mapping[str, Any]]] = {}
    for p in passages.values():
        if p["doc_id"] in doc_ids and (language is None or p.get("language") == language) and is_scope_passage(p):
            by_doc.setdefault(p["doc_id"], []).append(p)
    out: list[str] = []
    for d in doc_ids:
        found = sorted(by_doc.get(d, []), key=lambda p: (int(p["page"]), int(p["n"])))
        out.extend(p["passage_id"] for p in found[:per_doc])
    return out
