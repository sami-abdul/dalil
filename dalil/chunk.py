"""Split document pages into passages that answers can cite.

Each passage is an exact slice of one page's text: for a passage p of page t,
``t[p.start:p.end] == p.text``. The citation checker relies on that, so this
module never rewrites, trims inside or joins text; it only chooses offsets.

Splitting works in two steps. First the page is cut into small segments at
every sentence end and paragraph break, each boundary carrying a strength.
Then segments are packed greedily into passages of roughly TARGET_CHARS to
MAX_CHARS characters:

* before a heading (``Article (N)`` in a law, or a heading line on a u.ae
  page) the passage is closed once it holds MIN_CHARS;
* at a paragraph break it is closed once it holds TARGET_CHARS;
* at a sentence end it is closed only when the next segment would push it
  past MAX_CHARS.

A single sentence longer than HARD_CHARS is the only thing cut mid-sentence:
at a clause mark (, ، ;) if there is one, else at a space.

PDF text keeps the printed line breaks, which fall mid-sentence, so in a PDF a
line break counts as a paragraph break only after sentence-ending punctuation
or before a list marker. In HTML text every line is a block.

Page text is untrusted input; it is only scanned with regular expressions.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MIN_CHARS = 300
TARGET_CHARS = 600
MAX_CHARS = 1200
HARD_CHARS = 2000
# A heading this close after a passage's start names the passage.
HEADING_LEAD = 150

# Field order of each record in build/passages.jsonl.
PASSAGE_FIELDS = (
    "passage_id",
    "doc_id",
    "page",
    "n",
    "start",
    "end",
    "heading",
    "kind",
    "language",
    "char_count",
    "text",
)

# Boundary strengths, weakest first.
SENTENCE, PARAGRAPH, HEADING = 1, 2, 3

_DIGITS = "0-9\u0660-\u0669\u06f0-\u06f9"
# "Article (51)", and the mirrored forms PDF extraction produces: "Article )2 (".
_ARTICLE_EN = re.compile(rf"^Article\s*[()]?\s*([{_DIGITS}]+)\s*[()]?$", re.IGNORECASE)
# "المادة (5)", and the extracted RTL forms ")1( المادة" and ") 2( المادة".
_ARTICLE_AR = re.compile(
    rf"^(?:المادة\s*[()]?\s*([{_DIGITS}]+)\s*[()]?|[()]?\s*([{_DIGITS}]+)\s*[()]?\s*المادة)$"
)
# Sentence-ending marks: . ! ? and the Arabic question mark and full stop.
_TERMINAL = ".!?\u061f\u06d4"
_CLOSERS = "\"'\u201d\u2019\u00bb)]"
_SENTENCE_END = re.compile(rf"[{re.escape(_TERMINAL)}][{re.escape(_CLOSERS)}]*(?=\s)")
# Clause marks for cutting an over-long sentence: comma, Arabic comma, semicolons.
_CLAUSE_END = re.compile(r"[,\u060c;\u061b](?=\s)")
# "1.", "1)", "(a)", "b.", "iv." at the start of a line.
_LIST_MARKER = re.compile(rf"^(?:\(?[{_DIGITS}]{{1,3}}[.)\-]|\(?[a-zA-Z][.)]|\((?:[ivx]+|[a-z])\)|[ivx]{{1,4}}[.)])\s")
_ABBREVIATIONS = frozenset(
    "no nos art arts e.g i.e eg ie etc vs mr mrs ms dr st para paras approx fig cf u.a.e "
    "jan feb mar apr jun jul aug sep sept oct nov dec inc ltd co".split()
)
_WORD_BEFORE = re.compile(r"([\w.]+)$")


@dataclass(frozen=True)
class Span:
    """One passage of a page: text[start:end], with the heading in force there."""

    start: int
    end: int
    heading: str | None


# --------------------------------------------------------------------------
# Headings


def _number(digits: str) -> int:
    return int("".join(str(unicodedata.digit(c)) for c in digits))


def article_heading(line: str) -> str | None:
    """The normalised heading for a law's article line, else None.

    ``Article )2 (`` gives ``Article (2)``; ``)1( المادة`` gives ``المادة (1)``.
    """
    line = line.strip()
    if len(line) > 40:
        return None
    m = _ARTICLE_EN.match(line)
    if m:
        return f"Article ({_number(m.group(1))})"
    m = _ARTICLE_AR.match(line)
    if m:
        return f"المادة ({_number(m.group(1) or m.group(2))})"
    return None


def is_heading_line(line: str) -> bool:
    """A cheap test for a heading line on an HTML page: short, capitalised,
    no sentence or clause punctuation at the end, not a list item or table row."""
    line = line.strip()
    if not 2 <= len(line) <= 90 or " | " in line:
        return False
    if line[-1] in ".:;,\u060c\u061b!" or _LIST_MARKER.match(line + " "):
        return False
    if len(line.split()) > 12:
        return False
    first = line[0]
    return first.isupper() or first.isdigit() or "\u0600" <= first <= "\u06ff"


# --------------------------------------------------------------------------
# Boundaries


def _lines(text: str) -> list[tuple[int, int]]:
    """(start, end) of each line, excluding the newline."""
    out, pos = [], 0
    for line in text.split("\n"):
        out.append((pos, pos + len(line)))
        pos += len(line) + 1
    return out


def _skip_space(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _is_sentence_end(text: str, match: re.Match[str], line_start: int) -> bool:
    """Reject a full stop that ends an abbreviation or a list marker, or one
    followed by a lower-case word (an abbreviation we do not list)."""
    if text[match.start()] != ".":
        return True
    word_match = _WORD_BEFORE.search(text, line_start, match.start())
    word = word_match.group(1) if word_match else ""
    if word.lower().rstrip(".") in _ABBREVIATIONS:
        return False
    if len(word) == 1 and word.isalpha():
        return False  # an initial or a list letter: "b."
    at_line_start = word_match is not None and word_match.start() == line_start
    if at_line_start and (word.isdigit() or re.fullmatch(r"[ivx]{1,4}", word)):
        return False  # "1." or "iv." as a list marker
    nxt = _skip_space(text, match.end())
    return not (nxt < len(text) and text[nxt].islower())


def boundaries(text: str, kind: str) -> tuple[dict[int, int], list[tuple[int, str]]]:
    """Segment starts with their strength, and (offset, heading) for each heading.

    Every boundary offset points at a non-space character.
    """
    found: dict[int, int] = {}
    headings: list[tuple[int, str]] = []

    def mark(pos: int, strength: int) -> None:
        pos = _skip_space(text, pos)
        if 0 < pos < len(text):
            found[pos] = max(found.get(pos, 0), strength)

    lines = _lines(text)
    prev_heading = False
    prev_line = ""
    for i, (ls, le) in enumerate(lines):
        line = text[ls:le]
        stripped = line.strip()
        if not stripped:
            continue
        label = article_heading(stripped)
        if label is None and kind == "html" and i + 1 < len(lines) and is_heading_line(stripped):
            label = stripped
        if label is not None:
            headings.append((_skip_space(text, ls), label))
            mark(ls, HEADING)
        elif kind == "html" or prev_heading:
            mark(ls, PARAGRAPH)
        else:
            tail = prev_line.rstrip().rstrip(_CLOSERS)
            if tail and tail[-1] in _TERMINAL + ":;" or _LIST_MARKER.match(stripped + " "):
                mark(ls, PARAGRAPH)
        prev_heading = label is not None
        prev_line = line
        for m in _SENTENCE_END.finditer(text, ls, le + 1 if le < len(text) else le):
            if _is_sentence_end(text, m, ls):
                mark(m.end(), SENTENCE)
    return found, headings


# --------------------------------------------------------------------------
# Packing


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_long(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Cut one over-long sentence into pieces of at most MAX_CHARS, at a
    clause mark where possible, else at a space, else (no space at all) hard."""
    pieces = []
    while end - start > MAX_CHARS:
        lo, hi = start + TARGET_CHARS, start + MAX_CHARS
        cut = None
        for m in _CLAUSE_END.finditer(text, lo, hi):
            cut = m.end()
        if cut is None:
            space = text.rfind(" ", lo, hi)
            if space == -1:
                space = max(text.rfind(c, lo, hi) for c in "\n\t")
            cut = space if space > start else hi
        s, e = _trim(text, start, cut)
        if e > s:
            pieces.append((s, e))
        start = _skip_space(text, cut)
    s, e = _trim(text, start, end)
    if e > s:
        pieces.append((s, e))
    return pieces


def _segments(text: str, found: Mapping[int, int]) -> list[tuple[int, int, int]]:
    """(start, end, strength of the boundary before it) for each segment."""
    first = _skip_space(text, 0)
    if first >= len(text):
        return []
    starts = sorted(p for p in found if p > first)
    edges = [first, *starts, len(text)]
    out = []
    for i in range(len(edges) - 1):
        s, e = _trim(text, edges[i], edges[i + 1])
        if e <= s:
            continue
        strength = found.get(edges[i], HEADING) if i else HEADING
        if e - s > HARD_CHARS:
            for j, (ps, pe) in enumerate(_split_long(text, s, e)):
                out.append((ps, pe, strength if j == 0 else 0))
        else:
            out.append((s, e, strength))
    return out


def _closes(size: int, strength: int, next_len: int) -> bool:
    """Whether to close a passage of `size` characters before a segment."""
    if size == 0:
        return False
    if strength == 0:
        return True  # pieces of one over-long sentence stay apart
    if strength >= HEADING and size >= MIN_CHARS:
        return True
    if strength >= PARAGRAPH and size >= TARGET_CHARS:
        return True
    grown = size + next_len
    return (grown > MAX_CHARS and size >= MIN_CHARS) or grown > HARD_CHARS


def split_page(text: str, kind: str = "html", carry_heading: str | None = None) -> list[Span]:
    """Passages of one page. carry_heading is the heading in force at the end
    of the previous page, so a law article that runs over a page keeps its name."""
    found, headings = boundaries(text, kind)
    spans: list[tuple[int, int]] = []
    cur_start = cur_end = None
    for s, e, strength in _segments(text, found):
        size = 0 if cur_start is None else cur_end - cur_start
        if _closes(size, strength, e - s):
            spans.append((cur_start, cur_end))
            cur_start = None
        if cur_start is None:
            cur_start = s
        cur_end = e
    if cur_start is not None:
        spans.append((cur_start, cur_end))
    # A short tail (a page's last lines, an "Updated on" line) joins the
    # passage before it rather than standing alone.
    if len(spans) >= 2:
        (ps, _), (ts, te) = spans[-2], spans[-1]
        if te - ts < MIN_CHARS and te - ps <= HARD_CHARS:
            spans[-2:] = [(ps, te)]

    out = []
    for s, e in spans:
        heading = carry_heading
        for pos, label in headings:
            if pos < s:
                heading = label
            elif pos < min(e, s + HEADING_LEAD):
                heading = label
                break
            else:
                break
        out.append(Span(s, e, heading))
    return out


def _last_heading(text: str, kind: str, carry: str | None) -> str | None:
    """The heading in force at the end of a page, carried to the next page.
    Only law article headings carry over; a u.ae heading is page-local."""
    _, headings = boundaries(text, kind)
    for _, label in reversed(headings):
        if article_heading(label) is not None:
            return label
    return carry


# --------------------------------------------------------------------------
# Documents


def passage_id(doc_id: str, page: int, n: int) -> str:
    return f"{doc_id}:p{int(page)}:{int(n)}"


def chunk_document(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Passages for one documents.jsonl record. Terms pages give none."""
    if doc.get("is_terms_page"):
        return []
    doc_id = doc["doc_id"]
    kind = doc.get("kind") or "html"
    out: list[dict[str, Any]] = []
    carry: str | None = None
    for page in doc.get("pages") or []:
        text = page.get("text") or ""
        number = int(page["page"])
        for n, span in enumerate(split_page(text, kind, carry), start=1):
            piece = text[span.start : span.end]
            record = {
                "passage_id": passage_id(doc_id, number, n),
                "doc_id": doc_id,
                "page": number,
                "n": n,
                "start": span.start,
                "end": span.end,
                "heading": span.heading,
                "kind": kind,
                "language": doc.get("language"),
                "char_count": len(piece),
                "text": piece,
            }
            out.append({k: record[k] for k in PASSAGE_FIELDS})
        carry = _last_heading(text, kind, carry)
    return out


def iter_passages(docs: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    for doc in docs:
        yield from chunk_document(doc)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as e:
                    raise ValueError(f"{path}:{n}: not JSON ({e})") from None
    return rows


def verify_passage(passage: Mapping[str, Any], page_text: str) -> bool:
    """The rerunnable citation check: the stored slice equals the passage text."""
    return page_text[passage["start"] : passage["end"]] == passage["text"]
