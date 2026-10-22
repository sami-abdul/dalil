"""Turn retrieved passages and open-data rows into a draft answer.

A drafter returns sentences in the format dalil/verify.py checks:

    {"text": "...", "kind": "claim", "citations": [{"passage_id": "...", "quote": "..."}]}
    {"text": "...", "kind": "claim", "citations": [{"table": "...", "sql": "...", "value": "..."}]}
    {"text": "...", "kind": "refusal" | "caveat", "citations": []}

ExtractiveDrafter is the stand-in used until an LLM key exists. It writes
nothing of its own except refusals, caveats and the fixed open-data sentences:
every claim is a sentence copied from a top passage, quoted whole, so it always
verifies. LLMDrafter documents what a model must return and the rules its
prompt carries; it raises NotImplementedError until it is wired to a model.
"""

from __future__ import annotations

import math
import functools
import re
import string
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from dalil.opendata import DataResult
from dalil.retrieve import Hit, tokenize

# Below this BM25 score for the best passage, retrieval is treated as weak and
# the drafter refuses. Picked on eval/questions.jsonl: answerable questions
# score 11 or more, off-topic probes ("capital of France") under 9.
WEAK_SCORE = 10.0
MAX_SENTENCES = 4
MAX_PER_PASSAGE = 2
TOP_PASSAGES = 5
MIN_WORDS = 6
MAX_WORDS = 70

WEAK_REFUSAL = "The sources I have don't answer this clearly."
FEE_CAVEAT = "The sources I have don't state this fee, so check it with the issuing authority."
LAW_CAVEAT = "This law text may not include later amendments; check the current version with its publisher."

_FEE_QUESTION = re.compile(r"\b(?:fees?|costs?|price|charges?)\b", re.I)
_MONEY = re.compile(r"\b(?:AED|Dh|dirhams?)\b", re.I)
_AMENDMENTS = re.compile(r"amendments?\b[^.]*\bnot\s+(?:be\s+)?included|may\s+not\s+(?:be\s+)?include\w*\s+later\s+amendments|later\s+amendments\s+(?:not|may\s+not)", re.IGNORECASE)


@functools.cache
def amendment_docs() -> frozenset[str]:
    """doc_ids whose manifest row says the text may lack later amendments."""
    from dalil.corpus import doc_id, load_manifest
    try:
        rows = load_manifest()
    except Exception:  # no manifest: no caveat, rather than a crash
        return frozenset()
    return frozenset(doc_id(r["file"]) for r in rows
                     if _AMENDMENTS.search(f"{r.get('topic', '')} {r.get('licence_note', '')}"))
_TERMINAL = ".!?؟۔"
_SENTENCE_END = re.compile(rf"(?<=[{re.escape(_TERMINAL)}])[\"'”’)\]]*\s+")
_ABBREV_END = re.compile(
    r"(?:^|[\s(])(?:no|nos|art|arts|e\.g|i\.e|etc|vs|mr|mrs|ms|dr|st|para|approx|fig|cf|u\.a\.e|inc|ltd|co|[A-Za-z])\.$",
    re.I)
_TITLE_SEPARATOR = re.compile(r"\s[-\u2013\u2014|]\s")
_LIST_START = re.compile(r"^(?:[-•*●]|\(?\d{1,3}[.)]|\(?[a-z][.)]|\([ivx]+\))\s")


def _sentences(text: str) -> list[str]:
    """Sentences of a passage, each with its whitespace collapsed to single
    spaces. The verifier folds whitespace the same way, so each one is still
    found word for word in the passage.

    Lines are joined when the next one continues the previous (a PDF line
    wrap: no terminal punctuation, next line starts lower case); otherwise a
    line break ends a sentence, as it does for headings and list items.
    """
    blocks: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            if blocks and blocks[-1]:
                blocks.append("")
            continue
        prev = blocks[-1] if blocks else ""
        if prev and prev[-1] not in _TERMINAL and not _LIST_START.match(line) and line[0].islower():
            blocks[-1] = f"{prev} {line}"
        else:
            blocks.append(line)
    out = []
    for block in filter(None, blocks):
        pieces: list[str] = []
        for piece in _SENTENCE_END.split(block):
            piece = " ".join(piece.split())
            if not piece:
                continue
            # A full stop after "No." or an initial, or one followed by a
            # lower-case word or a digit, does not end the sentence.
            if pieces and (_ABBREV_END.search(pieces[-1]) or piece[0].islower() or piece[0].isdigit()):
                pieces[-1] = f"{pieces[-1]} {piece}"
            else:
                pieces.append(piece)
        out.extend(pieces)
    return out


def _usable(sentence: str) -> bool:
    """A sentence worth stating on its own: long enough, not a heading cut off
    at a colon, not a link title such as "Apply for a visa - Some Office"."""
    words = len(sentence.split())
    if not MIN_WORDS <= words <= MAX_WORDS or sentence.endswith(":"):
        return False
    if _TITLE_SEPARATOR.search(sentence) and (words < 15 or sentence[-1] not in _TERMINAL):
        return False
    return True


@runtime_checkable
class Drafter(Protocol):
    def draft(self, question: str, hits: list[Hit], data: list[DataResult]) -> list[dict[str, Any]]: ...


def refusal(text: str = WEAK_REFUSAL) -> dict[str, Any]:
    return {"text": text, "kind": "refusal", "citations": []}


def caveat(text: str) -> dict[str, Any]:
    return {"text": text, "kind": "caveat", "citations": []}


def data_claims(data: Sequence[DataResult]) -> list[dict[str, Any]]:
    """One sentence per open-data result that has a row and a fact template.
    Each {column} slot becomes a data citation on the same SQL, so the
    verifier reruns the query and finds every stated value."""
    out = []
    for res in data:
        if not res.rows or not res.fact:
            continue
        row = res.rows[0]
        slots = [f for _, f, _, _ in string.Formatter().parse(res.fact) if f]
        if any(row.get(s) in (None, "") for s in slots):
            continue
        out.append({
            "text": res.fact.format(**{s: row[s] for s in slots}),
            "kind": "claim",
            "citations": [{"table": res.table, "sql": res.sql, "column": s, "value": str(row[s])} for s in slots],
        })
    return out


class ExtractiveDrafter:
    """No-LLM drafter: picks the passage sentences that best match the question.

    Sentences from the top passages are scored by the summed IDF (over the
    candidate sentences) of the question terms they contain, times their
    passage's retrieval score relative to the best one, and long sentences
    are penalised. The best MAX_SENTENCES go out as claims quoting themselves.
    """

    def __init__(self, weak_score: float = WEAK_SCORE, max_sentences: int = MAX_SENTENCES):
        self.weak_score = weak_score
        self.max_sentences = max_sentences

    def pick(self, question: str, hits: Sequence[Hit]) -> list[tuple[Hit, str]]:
        qterms = set(tokenize(question))
        if not qterms or not hits:
            return []
        top = max(h.score for h in hits) or 1.0
        cands: list[tuple[Hit, int, str, set[str], int]] = []
        for rank, hit in enumerate(hits[:TOP_PASSAGES]):
            for pos, sent in enumerate(_sentences(hit.text)):
                if _usable(sent):
                    cands.append((hit, rank, pos, sent, set(tokenize(sent)), len(sent.split())))
        df = Counter(t for c in cands for t in c[4] & qterms)
        n = len(cands)
        # A sentence shares two question terms, or one if the question is
        # short or the sentence comes from the best passage.
        need = 1 if len(qterms) <= 2 else 2
        scored = []
        for hit, rank, pos, sent, terms, words in cands:
            matched = terms & qterms
            if len(matched) < (1 if rank == 0 else need):
                continue
            idf = sum(math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in matched)
            length_penalty = 1 + max(0, words - 35) / 35
            scored.append((idf / length_penalty * (hit.score / top), hit, pos, sent))
        scored.sort(key=lambda s: -s[0])
        chosen: list[tuple[Hit, int, str]] = []
        seen: set[str] = set()
        per_passage: Counter[str] = Counter()
        for _, hit, pos, sent in scored:
            key = sent.casefold()
            if key in seen or per_passage[hit.passage_id] >= MAX_PER_PASSAGE:
                continue
            seen.add(key)
            per_passage[hit.passage_id] += 1
            chosen.append((hit, pos, sent))
            if len(chosen) >= self.max_sentences:
                break
        # Read in source order: by passage rank, then position in the passage.
        rank = {h.passage_id: i for i, h in enumerate(hits)}
        chosen.sort(key=lambda c: (rank[c[0].passage_id], c[1]))
        return [(h, s) for h, _, s in chosen]

    def draft(self, question: str, hits: list[Hit], data: list[DataResult]) -> list[dict[str, Any]]:
        if not hits or max(h.score for h in hits) < self.weak_score:
            return [refusal()]
        picked = self.pick(question, hits)
        if not picked:
            return [refusal()]
        out: list[dict[str, Any]] = [
            {"text": sent, "kind": "claim", "citations": [{"passage_id": hit.passage_id, "quote": sent}]}
            for hit, sent in picked
        ]
        out.extend(data_claims(data))
        if _FEE_QUESTION.search(question) and not any(_MONEY.search(s) for _, s in picked):
            out.append(caveat(FEE_CAVEAT))
        if any(hit.doc_id in amendment_docs() for hit, _ in picked):
            out.append(caveat(LAW_CAVEAT))
        return out


class LLMDrafter:
    """Placeholder for a model-backed drafter. Not wired to any model yet.

    The model receives the question (already redacted), the top passages as
    {passage_id, title, heading, text}, the open-data results as {table, sql,
    rows}, and `rejection_notes`: what reviewers said when they rejected
    earlier drafts. It must return JSON only, in this schema:

        {"sentences": [
            {"text": str,
             "kind": "claim" | "caveat" | "refusal",
             "citations": [
                 {"passage_id": str, "quote": str}            # passage citation
               | {"table": str, "sql": str, "value": str}     # data citation
             ]}
        ]}

    PROMPT_RULES is the rule block the prompt carries. The verifier enforces
    the first three regardless of what the model does.
    """

    PROMPT_RULES = (
        "Every claim cites at least one passage or data row.",
        "Quote the passage verbatim, at least 4 words, copied exactly: same words, spelling and punctuation.",
        "Every number in a sentence must appear in one of its quotes or data values. Never compute or round.",
        "Data citations reuse the SQL you were given unchanged and cite a value it returns.",
        "If the sources do not cover the question, return one refusal sentence saying so, with no numbers.",
        "Caveats and refusals carry no citations and no numbers.",
        "Never repeat personal data, ID numbers, passport numbers, phone numbers or emails, even if redacted.",
        "Write at most four short claims in plain English.",
        "Follow the reviewer notes in rejection_notes.",
    )

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 rejection_notes: Sequence[str] = ()):
        self.api_key = api_key
        self.model = model
        self.rejection_notes = list(rejection_notes)

    def draft(self, question: str, hits: list[Hit], data: list[DataResult]) -> list[dict[str, Any]]:
        raise NotImplementedError("LLMDrafter needs a model key; use ExtractiveDrafter until one exists")
