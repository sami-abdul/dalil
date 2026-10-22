"""Find the passages most likely to answer a question.

The baseline is Okapi BM25 over build/passages.jsonl, in pure Python with no
external service. Tokenising is deliberately simple:

* English: case-folded, split on anything that is not a letter or digit,
  common function words dropped, and stemmed with Snowball (Porter2) so that
  "visas" matches "visa" and "cancelled" matches "cancellation".
* Arabic: diacritics and tatweel stripped, the alef forms unified, final
  alef maqsura read as ya and ta marbuta as ha, Arabic-Indic digits read as
  ASCII, and a leading definite article (with و ب ف ك ل) removed.

Dense retrieval plugs in through the Embedder protocol. HybridRetriever takes
a lexical Retriever and an optional Embedder: with no embedder it returns the
BM25 hits unchanged, and with one it fuses BM25 and cosine rankings by
reciprocal rank. Callers depend only on ``.search(query, k) -> list[Hit]``,
so adding a model later is a change at the construction site, nowhere else.
See ``make_retriever``.
"""

from __future__ import annotations

import heapq
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import snowballstemmer

from dalil.chunk import read_jsonl

# --------------------------------------------------------------------------
# Tokenising

_AR_MARKS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
_AR_FOLD = str.maketrans(
    {
        "\u0623": "\u0627",  # alef with hamza above
        "\u0625": "\u0627",  # alef with hamza below
        "\u0622": "\u0627",  # alef with madda
        "\u0671": "\u0627",  # alef wasla
        "\u0649": "\u064a",  # alef maqsura -> ya
        "\u0629": "\u0647",  # ta marbuta -> ha
        **{chr(0x0660 + i): str(i) for i in range(10)},
        **{chr(0x06F0 + i): str(i) for i in range(10)},
    }
)
_AR_PREFIXES = ("وال", "بال", "فال", "كال", "لل", "ال")
_WORD = re.compile(r"[^\W_]+")

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers him his how i if in into is it its itself just
    me more most my no nor not now of off on once only or other our out over own same she should
    so some such than that the their them then there these they this those through to too under
    until up very was we were what when where which while who whom why will with would you your
    yours don t s""".split()
)


def normalise_arabic(text: str) -> str:
    """Strip diacritics and tatweel, unify alef forms, fold ya, ta marbuta and digits."""
    return _AR_MARKS.sub("", text).translate(_AR_FOLD)


_STEMMER = snowballstemmer.stemmer("english")


def stem_english(word: str) -> str:
    """Porter2 (Snowball) stem of an ASCII word; other words pass through.
    Applied to queries and passages alike, so "visas" meets "visa" and
    "cancelled" meets "cancellation"."""
    if not word.isascii() or not word.isalpha():
        return word
    return _STEMMER.stemWord(word)


def _arabic_stem(word: str) -> str:
    for prefix in _AR_PREFIXES:
        if word.startswith(prefix) and len(word) - len(prefix) >= 2:
            return word[len(prefix) :]
    return word


def _is_arabic(word: str) -> bool:
    return "؀" <= word[0] <= "ۿ"


def tokenize(text: str, stem: bool = True) -> list[str]:
    """Search terms of a text. stem=False keeps whole words (normalised only)."""
    text = normalise_arabic(unicodedata.normalize("NFKC", text).casefold())
    out = []
    for word in _WORD.findall(text):
        if word in STOPWORDS:
            continue
        if not stem:
            out.append(word)
        elif _is_arabic(word):
            out.append(_arabic_stem(word))
        else:
            out.append(stem_english(word))
    return out


# --------------------------------------------------------------------------
# Hits and the lexical retriever


@dataclass(frozen=True)
class Hit:
    passage_id: str
    doc_id: str
    page: int
    score: float
    heading: str | None
    text: str
    start: int
    end: int


@runtime_checkable
class Searcher(Protocol):
    """What callers depend on: any retriever, lexical, dense or hybrid."""

    def search(self, query: str, k: int = 8) -> list[Hit]: ...


class Retriever:
    """Okapi BM25 over passages. k1 and b are the usual defaults."""

    def __init__(self, passages: Sequence[Mapping[str, Any]], k1: float = 1.5, b: float = 0.75,
                 stem: bool = True):
        self.passages = [dict(p) for p in passages]
        self.k1, self.b, self.stem = k1, b, stem
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths: list[int] = []
        for i, p in enumerate(self.passages):
            counts = Counter(tokenize(self.index_text(p), self.stem))
            self._lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self._postings[term].append((i, tf))
        n = len(self.passages)
        self._avg_len = (sum(self._lengths) / n) if n else 0.0
        self._idf = {
            term: math.log(1 + (n - len(post) + 0.5) / (len(post) + 0.5))
            for term, post in self._postings.items()
        }

    @staticmethod
    def index_text(passage: Mapping[str, Any]) -> str:
        """The text a passage is indexed under: its own text, plus its heading
        when the heading line is not already inside the passage."""
        text = passage.get("text") or ""
        heading = passage.get("heading") or ""
        return text if not heading or heading in text else f"{heading}\n{text}"

    @classmethod
    def from_jsonl(cls, path: Path | str, **kwargs: Any) -> "Retriever":
        return cls(read_jsonl(path), **kwargs)

    def __len__(self) -> int:
        return len(self.passages)

    def scores(self, query: str) -> dict[int, float]:
        """BM25 score per passage index, for passages sharing a term with the query."""
        out: dict[int, float] = defaultdict(float)
        avg = self._avg_len or 1.0
        for term, qtf in Counter(tokenize(query, self.stem)).items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf in self._postings[term]:
                norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self._lengths[i] / avg))
                out[i] += qtf * idf * norm
        return out

    def hit(self, i: int, score: float) -> Hit:
        p = self.passages[i]
        return Hit(
            passage_id=p["passage_id"], doc_id=p["doc_id"], page=int(p["page"]), score=float(score),
            heading=p.get("heading"), text=p["text"], start=int(p["start"]), end=int(p["end"]),
        )

    def search(self, query: str, k: int = 8) -> list[Hit]:
        ranked = _top(self.scores(query).items(), k)
        return [self.hit(i, s) for i, s in ranked]


def _top(items: Iterable[tuple[int, float]], k: int) -> list[tuple[int, float]]:
    """Highest score first; ties go to the earlier passage, so results are stable."""
    return heapq.nsmallest(k, items, key=lambda item: (-item[1], item[0]))


# --------------------------------------------------------------------------
# Dense hook


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into vectors. Implement this for a real model (an NVIDIA
    NIM embedding endpoint, a local model); nothing here depends on which."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

# An embedder may also define embed_query(text) and embed_passages(texts), for
# models that want different prefixes for queries and documents. The hybrid
# retriever uses them when they exist.


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class HybridRetriever:
    """BM25, optionally fused with dense similarity by reciprocal rank fusion.

    With embedder=None this is exactly the lexical retriever. With an embedder,
    passage vectors are computed once on first search (or passed in as
    `passage_vectors`, for a precomputed index), each ranking keeps its top
    `pool`, and a passage scores sum(1 / (rrf_k + rank)) over the rankings it
    appears in. Hits come back in fused order, but Hit.score stays the BM25
    score (0 for a passage only the dense side found), so thresholds on it,
    such as the drafters' weak-retrieval refusal, keep their meaning.
    """

    def __init__(self, lexical: Retriever, embedder: Embedder | None = None, *,
                 pool: int = 50, rrf_k: int = 60, batch_size: int = 64,
                 passage_vectors: Sequence[Sequence[float]] | None = None):
        self.lexical = lexical
        self.embedder = embedder
        self.pool, self.rrf_k, self.batch_size = pool, rrf_k, batch_size
        if passage_vectors is not None and len(passage_vectors) != len(lexical):
            raise ValueError("passage_vectors must hold one vector per passage")
        self._vectors = list(passage_vectors) if passage_vectors is not None else None

    def _passage_vectors(self) -> list[Sequence[float]]:
        if self._vectors is None:
            assert self.embedder is not None
            texts = [Retriever.index_text(p) for p in self.lexical.passages]
            vectors: list[Sequence[float]] = []
            embed = getattr(self.embedder, "embed_passages", None) or self.embedder.embed
            for i in range(0, len(texts), self.batch_size):
                vectors.extend(embed(texts[i : i + self.batch_size]))
            if len(vectors) != len(texts):
                raise ValueError("embedder returned the wrong number of vectors")
            self._vectors = vectors
        return self._vectors

    def search(self, query: str, k: int = 8) -> list[Hit]:
        if self.embedder is None:
            return self.lexical.search(query, k)
        bm25 = self.lexical.scores(query)
        lexical = _top(bm25.items(), self.pool)
        embed_query = getattr(self.embedder, "embed_query", None)
        qvec = embed_query(query) if embed_query else self.embedder.embed([query])[0]
        dense = _top(((i, _cosine(qvec, v)) for i, v in enumerate(self._passage_vectors())), self.pool)
        fused: dict[int, float] = defaultdict(float)
        for ranking in (lexical, dense):
            for rank, (i, _) in enumerate(ranking, start=1):
                fused[i] += 1.0 / (self.rrf_k + rank)
        return [self.lexical.hit(i, bm25.get(i, 0.0)) for i, _ in _top(fused.items(), k)]


def make_retriever(path: Path | str, embedder: Embedder | None = None) -> Searcher:
    """The one place callers get a retriever from.

    Dense stand-in point: today no embedder is configured, so this returns BM25
    through HybridRetriever. When an embedding model is available, construct
    an Embedder for it and pass it here (or build it here from configuration);
    callers keep calling ``.search(query, k)`` unchanged.
    """
    if embedder is None and os.environ.get("DALIL_EMBED_URL"):
        from dalil.embed import HTTPEmbedder
        candidate = HTTPEmbedder()
        embedder = candidate if candidate.available() else None
    return HybridRetriever(Retriever.from_jsonl(path), embedder)
