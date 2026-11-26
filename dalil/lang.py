"""The language of a question, and Dalil's fixed sentences in each language.

Dalil answers in the language the resident asked in. Today that is English or
Arabic. The fixed refusals and caveats are written here once per language, so
the checker can still tell them from drafted text: only these exact sentences
may go out without a citation.

Adding a language means adding its sentences to FIXED below and its words to
the screens in dalil/safety.py and dalil/draft.py; nothing else keys on it.
"""

from __future__ import annotations

import re

ARABIC, ENGLISH = "ar", "en"
_ARABIC_LETTER = re.compile("[ء-يٱ-ۓ]")
_LATIN_LETTER = re.compile("[A-Za-z]")


def language(text: str) -> str:
    """"ar" when most letters in `text` are Arabic, else "en"."""
    ar = len(_ARABIC_LETTER.findall(text or ""))
    en = len(_LATIN_LETTER.findall(text or ""))
    return ARABIC if ar and ar >= en else ENGLISH


# English fixed sentence -> the same sentence in Arabic. The English ones are
# the constants in dalil/draft.py and dalil/safety.py. None of them may hold a
# number, since the checker refuses numbers in uncited sentences.
ARABIC_FIXED: dict[str, str] = {
    "The sources I have don't answer this clearly.":
        "المصادر المتوفرة لدي لا تجيب عن هذا السؤال بوضوح.",
    "The sources I have don't state this fee, so check it with the issuing authority.":
        "المصادر المتوفرة لدي لا تذكر هذا الرسم، لذا يرجى التحقق منه لدى الجهة المختصة.",
    "This law text may not include later amendments; check the current version with its publisher.":
        "قد لا يتضمن نص هذا القانون التعديلات اللاحقة؛ يرجى التحقق من النسخة السارية لدى الجهة الناشرة.",
    "I can't help with getting around UAE rules, fines or penalties.":
        "لا يمكنني المساعدة في التحايل على قوانين دولة الإمارات أو غراماتها أو عقوباتها.",
    "For your real options, ask ICP or GDRFA through their official channels.":
        "لمعرفة الخيارات المتاحة لك فعلاً، تواصل مع الهيئة الاتحادية للهوية والجنسية والجمارك وأمن المنافذ "
        "أو الإدارة العامة للإقامة وشؤون الأجانب عبر قنواتهما الرسمية.",
    "Don't share ID or passport numbers here; check your status on the official ICP or GDRFA channels.":
        "لا تشارك أرقام الهوية أو جواز السفر هنا؛ تحقق من وضعك عبر القنوات الرسمية للهيئة الاتحادية للهوية "
        "والجنسية والجمارك وأمن المنافذ أو الإدارة العامة للإقامة وشؤون الأجانب.",
    "Please say which service or document you mean, so I can answer from the right official source.":
        "يرجى تحديد الخدمة أو الوثيقة التي تقصدها، حتى أجيب من المصدر الرسمي الصحيح.",
}


def fixed(text: str, lang: str) -> str:
    """A fixed English sentence in `lang`; unknown sentences come back unchanged."""
    return ARABIC_FIXED.get(text, text) if lang == ARABIC else text


def all_fixed(english: set[str] | frozenset[str]) -> frozenset[str]:
    """The given fixed English sentences plus their translations."""
    return frozenset(english) | frozenset(ARABIC_FIXED[t] for t in english if t in ARABIC_FIXED)
