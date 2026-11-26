"""Screen a question before anything else sees it.

Two checks, both plain rules with no model:

* Personal data. Emirates ID numbers, passport-like numbers, UAE phone numbers
  and email addresses are replaced with a placeholder ([EMIRATES ID],
  [PASSPORT], [PHONE], [EMAIL]). Only the redacted question is stored, shown,
  searched or drafted from.
* Requests to get around the rules, such as "ignore your rules" or "overstay
  without a fine". The pipeline answers those with a refusal and nothing else.

Flags name what was found: "personal_data:emirates_id", "personal_data:passport",
"personal_data:phone", "personal_data:email" and "rule_evasion".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from dalil.retrieve import tokenize

PERSONAL_DATA = "personal_data"
RULE_EVASION = "rule_evasion"

# Order matters: the Emirates ID goes first so the phone pattern cannot eat
# part of it, and email before phone so digits in an address stay with it.
_REDACTIONS: list[tuple[str, str, re.Pattern[str]]] = [
    # 784-YYYY-NNNNNNN-N, with dashes, dots, spaces or nothing between the groups.
    ("emirates_id", "[EMIRATES ID]", re.compile(r"(?<!\d)784[-\s.]?\d{4}[-\s.]?\d{7}[-\s.]?\d(?!\d)")),
    ("email", "[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    # +971 / 00971 then a mobile or landline number; or a local 05x mobile.
    ("phone", "[PHONE]", re.compile(
        r"(?:(?<![\w+])(?:\+|00)971[\s.-]?\(?0?\)?\d{1,2}[\s.-]?\d{3}[\s.-]?\d{4}(?!\d))"
        r"|(?:(?<!\d)05\d[\s.-]?\d{3}[\s.-]?\d{4}(?!\d))")),
    # One or two letters then six to nine digits (N1234567, AB1234567), or any
    # 6-9 character code written straight after "passport (number|no)".
    ("passport", "[PASSPORT]", re.compile(
        r"\b[A-Za-z]{1,2}\d{6,9}\b"
        r"|\bpassport(?:\s+(?:number|no\.?|#))?(?:\s+is)?\s*:?\s*(?P<pii>(?=[A-Za-z]*\d)[A-Za-z0-9]{6,9})\b",
        re.IGNORECASE)),
]

_EVASION = re.compile(
    r"\bignore\s+(?:all\s+|any\s+)?(?:your|the|previous|prior|these|those)?\s*(?:rules|instructions|guidelines|policy|policies)\b"
    r"|\b(?:disregard|forget|bypass)\s+(?:all\s+|any\s+)?(?:your|the|previous|prior)?\s*(?:rules|instructions|guidelines|restrictions)\b"
    # "If I overstay without renewing, what is the fine?" is a fair question;
    # "overstay without a fine" or "without getting caught" is not.
    r"|\boverstay(?:ing)?\b[^.?!]*\bwithout\s+(?:getting\s+|being\s+)?(?:caught|paying|a\s+fine|any\s+fines?|fines?|penalt(?:y|ies)|consequences)\b"
    # "avoid the fine" is left out on purpose: "how do I avoid the fine for late
    # renewal?" is a fair question, and ICP publishes fee exemptions.
    r"|\b(?:dodge|evade|get\s+out\s+of)\s+(?:paying\s+)?(?:the\s+|a\s+|any\s+|my\s+)?(?:fines?|penalt(?:y|ies)|ban)\b"
    r"|\bwithout\s+(?:paying\s+)?(?:a\s+|the\s+|any\s+)?(?:fines?|penalt(?:y|ies))\b[^.?!]*\b(?:overstay|stay|work)\b"
    r"|\bfake\s+(?:an?\s+)?(?:document|certificate|contract|visa|id|emirates\s+id|passport|salary|stamp|tenancy|ejari|degree|letter|medical)"
    r"|\bforg(?:e|ed|ing|ery)\b"
    r"|\bwithout\s+(?:getting\s+)?(?:caught|detected|noticed)\b"
    r"|\b(?:get|work)\s+around\s+(?:the\s+)?(?:rules|law|ban|system)\b"
    r"|\b(?:evade|evading|circumvent)\b"
    # The same requests in Arabic: "ignore your rules", "evade", "get around",
    # "forge", "stay or work without a fine", "without being caught".
    r"|تجاهل\s+(?:كل\s+|جميع\s+)?(?:قواعدك|تعليماتك|القواعد|التعليمات|القوانين|الإرشادات|السياسات)"
    r"|(?:التهرب|أتهرب|اتهرب|نتهرب|يتهرب|تهرب)\s+من"
    r"|(?:التحايل|أتحايل|اتحايل|نتحايل|يتحايل|تحايل|الالتفاف|ألتف)\s+على"
    # Forging needs a document: «أزور» alone is also "I visit".
    r"|(?:تزوير|مزور|مزورة|مزيف|مزيفة)\s+(?:\S+\s+){0,2}?(?:وثيقة|مستند|شهادة|تأشيرة|هوية|جواز|عقد|ختم|راتب)"
    # "أزور" is also "I visit": only with the document straight after it.
    r"|(?:أزوّر|أزور|ازور|نزور|زورت)\s+(?:وثيقة|مستند|شهادة|تأشيرة|هوية|جواز|عقد|ختم|راتب)"
    r"|(?:وثيقة|مستند|شهادة|تأشيرة|هوية|جواز|عقد|ختم)\s+(?:\S+\s+){0,2}?(?:مزور|مزورة|مزيف|مزيفة)"
    # Staying past the visa without a fine, as "overstay without a fine" in
    # English; "renew my residency without a fine" is a fair question.
    r"|(?:تجاوز\s+مدة|(?:البقاء|أبقى|ابقى|أعمل|اعمل)[^.؟?!]*بعد\s+انتهاء|بعد\s+انتهاء[^.؟?!]*(?:البقاء|أبقى|ابقى|أعمل|اعمل))"
    r"[^.؟?!]*(?:دون|بدون|من\s+غير)\s+(?:دفع\s+)?(?:أي\s+)?(?:غرام|عقوب|مخالف)"
    r"|(?:دون|بدون|من\s+غير)\s+(?:أن\s+)?(?:يكتشف|يُكتشف|ينكشف|يلاحظ|يمسك|يُمسك|أُكشف|اكتشاف)",
    re.IGNORECASE,
)

REFUSAL_TEXT = "I can't help with getting around UAE rules, fines or penalties."
REFUSAL_FOLLOW_UP = "For your real options, ask ICP or GDRFA through their official channels."
VAGUE = "vague_question"
ASK_FOR_DETAIL = "Please say which service or document you mean, so I can answer from the right official source."
# Words that say what someone wants to know but not what it is about. A
# question whose key words are all of this kind, such as "How long can I
# stay?" or "What does it cost?", has no subject. Stems, as tokenize() makes.
_GENERIC = frozenset("long much mani get need can take stay go work cost time day year do make use want "
                     "allow possibl requir happen start finish valid".split()) | frozenset(
    # The same words in Arabic, as tokenize() makes them: "how long can I stay?"
    # is «كم من الوقت يمكنني البقاء؟».
    tokenize("وقت الوقت مده مدة طويل طويلة يمكنني يمكن أستطيع استطيع أقدر احتاج أحتاج اريد أريد يجب البقاء "
             "أبقى ابقى بقاء ابقي تكلفه تكلفة يكلف سعر يوم يوما أيام ايام سنه سنة شهر اشهر أشهر مسموح يسمح "
             "أعمل اعمل العمل أذهب اذهب افعل أفعل اعرف أعرف صالح صالحة سارية ساري", stem=True))
_PLACEHOLDER = re.compile(r"\[[A-Z ]+\]")


def subject_generic() -> frozenset[str]:
    """Words that say what someone wants to know but not about what."""
    return _GENERIC


def subject_words(text: str) -> set[str]:
    """Key words that name a subject, without redaction placeholders."""
    return {t for t in tokenize(_PLACEHOLDER.sub(" ", text)) if t not in _GENERIC}
PERSONAL_DATA_CAVEAT = (
    "Don't share ID or passport numbers here; check your status on the official ICP or GDRFA channels."
)


@dataclass(frozen=True)
class ScreenResult:
    redacted_question: str
    flags: list[str] = field(default_factory=list)

    @property
    def is_vague(self) -> bool:
        return VAGUE in self.flags

    @property
    def has_personal_data(self) -> bool:
        return any(f.startswith(PERSONAL_DATA + ":") for f in self.flags)

    @property
    def is_rule_evasion(self) -> bool:
        return RULE_EVASION in self.flags


def _replace(match: re.Match[str], placeholder: str) -> str:
    """Replace the whole match, or only its `pii` group when the pattern has
    one (so "passport number is X1" keeps its words)."""
    if "pii" in match.re.groupindex and match.group("pii") is not None:
        whole, start = match.group(0), match.start("pii") - match.start()
        return whole[:start] + placeholder + whole[start + len(match.group("pii")):]
    return placeholder


# Arabic-Indic and Persian digits, mapped one to one so string positions hold.
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


Redaction = tuple[str, str, re.Pattern[str]]  # (flag name, placeholder, pattern)


def redact(text: str, redactions: Sequence[Redaction] | None = None) -> tuple[str, list[str]]:
    """Replace personal data with placeholders; return the text and what was found.

    Arabic-Indic digits are read as ASCII first, so an Emirates ID typed in
    Arabic is caught too. Other numbers in the question come back in ASCII.
    `redactions` replaces the UAE formats, for another knowledge base.
    """
    text = text.translate(_DIGITS)
    found: list[str] = []
    for name, placeholder, pattern in (_REDACTIONS if redactions is None else redactions):
        text, n = pattern.subn(lambda m, ph=placeholder: _replace(m, ph), text)
        if n:
            found.append(f"{PERSONAL_DATA}:{name}")
    return text, found


def find_personal_data(text: str, redactions: Sequence[Redaction] | None = None) -> list[str]:
    """The personal-data strings in `text` that redact() would replace."""
    found = []
    text = text.translate(_DIGITS)
    for _, _, pattern in (_REDACTIONS if redactions is None else redactions):
        for m in pattern.finditer(text):
            pii = m.group("pii") if "pii" in pattern.groupindex and m.group("pii") else m.group(0)
            found.append(pii)
    return found


def screen(question: str, redactions: Sequence[Redaction] | None = None) -> ScreenResult:
    """Redact personal data and flag requests to get around the rules."""
    redacted, flags = redact(question.strip(), redactions)
    if not subject_words(redacted):
        flags.append(VAGUE)
    if _EVASION.search(redacted):
        flags.append(RULE_EVASION)
    return ScreenResult(redacted, flags)


@dataclass(frozen=True)
class Screener:
    """The screening settings of one knowledge base: the ID formats removed
    from a question, and the fixed sentences that go out without a citation.
    The defaults are the UAE ones above; a knowledge-base profile
    (dalil/kb.py) can replace any of them without a code change."""

    redactions: tuple[Redaction, ...] = tuple(_REDACTIONS)
    refusal_text: str = REFUSAL_TEXT
    refusal_follow_up: str = REFUSAL_FOLLOW_UP
    personal_data_caveat: str = PERSONAL_DATA_CAVEAT
    ask_for_detail: str = ASK_FOR_DETAIL

    def screen(self, question: str) -> ScreenResult:
        return screen(question, self.redactions)

    def redact(self, text: str) -> tuple[str, list[str]]:
        return redact(text, self.redactions)

    def find_personal_data(self, text: str) -> list[str]:
        return find_personal_data(text, self.redactions)

    @property
    def fixed_texts(self) -> dict[str, str]:
        return {"REFUSAL_TEXT": self.refusal_text, "REFUSAL_FOLLOW_UP": self.refusal_follow_up,
                "PERSONAL_DATA_CAVEAT": self.personal_data_caveat, "ASK_FOR_DETAIL": self.ask_for_detail}


def compile_redactions(specs: Sequence[dict]) -> tuple[Redaction, ...]:
    """Redactions from a profile: [{"name", "placeholder", "pattern", "ignore_case"}].
    A pattern may use a named group `pii` to replace only that part."""
    out = []
    for spec in specs:
        name, placeholder = str(spec["name"]), str(spec["placeholder"])
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or not re.fullmatch(r"\[[A-Z ]+\]", placeholder):
            raise ValueError(f"redaction needs a lower-case name and a [PLACEHOLDER]: {name!r} {placeholder!r}")
        out.append((name, placeholder, re.compile(spec["pattern"], re.IGNORECASE if spec.get("ignore_case") else 0)))
    return tuple(out)


DEFAULT_SCREENER = Screener()
