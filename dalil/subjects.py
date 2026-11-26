"""Which documents a question is about, by the services and laws it names.

A question such as "Can a Golden Visa holder sponsor their parents?" names a
service that has its own page, titled "Golden visa". A rule from another
page, such as the salary a regular employee needs to sponsor family, or what
a UAE citizen's parent can get, does not answer it, however close the words.
``subject_docs`` finds the documents whose title the question names in full,
so the writer can be told the question's subject and the checker can cut a
sentence that applies another service's rule to it.

Titles come from the manifest, so another knowledge base gets the same
behaviour from its own titles.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from dalil.retrieve import tokenize

# Words in a title that say what kind of page it is, not what it is about.
_TITLE_FILLER = frozenset(tokenize(
    "uae united arab official portal platform government page guide getting get check "
    "steps step general provisions information services service", stem=True))


def title_terms(title: str) -> frozenset[str]:
    """The stemmed words that name a title's subject: "Golden visa" -> {golden, visa}."""
    head = title.split("|")[0]
    return frozenset(t for t in tokenize(head) if t not in _TITLE_FILLER and not t.isdigit())


def subject_docs(question: str, titles: Mapping[str, str], skip: Iterable[str] = ()) -> set[str]:
    """Documents whose whole subject, two words or more, the question names.

    "Can a Golden Visa holder sponsor their parents?" names "Golden visa";
    it does not name "Residence visa for families of employees".
    """
    words = set(tokenize(question))
    skipped = set(skip)
    return {d for d, t in titles.items() if d not in skipped
            for terms in [title_terms(t)] if len(terms) >= 2 and terms <= words and not _in_passing(question, t)}


_POSSESSIVE = r"(?:my|your|his|her|our|their)"


def _in_passing(question: str, title: str) -> bool:
    """Whether the title is named only to share a personal detail: "My
    Emirates ID is [EMIRATES ID]", "My National Insurance number is QQ...".
    The question is then about something else. "Renew my Emirates ID" and
    "my residence visa is cancelled" are about the title, so they count."""
    head = [w for w in re.findall(r"[A-Za-z]+", title.split("|")[0]) if tokenize(w) and not
            set(tokenize(w)) & _TITLE_FILLER]
    if not head:
        return False
    phrase = r"\W+(?:\w+\W+){0,2}?".join(re.escape(w) for w in head[-2:])
    found = [m for m in re.finditer(phrase, question, re.IGNORECASE)
             if not re.search(r"\[[^\]]*$", question[:m.start()])]  # not inside a "[PLACEHOLDER]"
    sharing = rf"\s+(?:number\s+|no\.?\s+)?(?:is|:)\s*(?:\[[A-Z ]+\]|[A-Z0-9])"
    return bool(found) and all(
        re.search(rf"\b{_POSSESSIVE}\s+(?:\w+\s+){{0,2}}$", question[:m.start()], re.IGNORECASE)
        and re.match(sharing, question[m.end():])
        for m in found)


def names_subject(text: str, subject_titles: Iterable[str]) -> bool:
    """Whether `text` names one of the subjects, in full."""
    words = set(tokenize(text))
    return any(title_terms(t) <= words for t in subject_titles)
