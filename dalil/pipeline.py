"""One question, start to finish: screen, retrieve, look up open data, draft,
verify, queue for a person. Every step lands in the audit store.

    from dalil.pipeline import answer
    record = answer("What do I need to renew my Emirates ID?")

The status is "held" when the verifier holds the draft, else
"awaiting_review". Nothing is ever delivered here; a reviewer approves it
through the store (or the HTTP app).
"""

from __future__ import annotations

import os
import re
import sys

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from dalil import lang, safety
from dalil.draft import (FEE_CAVEAT, LAW_CAVEAT, WEAK_REFUSAL, Drafter, ExtractiveDrafter, caveat, data_claims,
                         refusal)
from dalil.opendata import RULES, DataResult, match_rules, run_rules
from dalil.retrieve import Hit, Searcher, make_retriever, tokenize
from dalil.store import Store
from dalil.hmem import Guidance, guidance
from dalil.laws import is_scope_question, law_refs, scope_passages
from dalil.subjects import names_subject, subject_docs, title_terms

# Words for the person who holds a service, not for what is asked about it.
_ROLE_WORDS = frozenset(tokenize("holder holders owner owners applicant applicants resident residents person people"))
from dalil.verify import Verifier, data_lookup, load_passages

def cited_passages(check: dict[str, Any] | None) -> list[str]:
    """Passage ids the kept sentences of a checked answer cite."""
    return sorted({c["passage_id"] for s in (check or {}).get("sentences", []) if s.get("kept")
                   for c in s.get("citations", []) if c.get("passage_id")})


def document_titles(manifest: Path | str | None = None) -> dict[str, str]:
    """doc_id -> title from the manifest, or nothing if it cannot be read."""
    from dalil.corpus import doc_id, load_manifest
    try:
        rows = load_manifest(manifest) if manifest else load_manifest()
        return {doc_id(r["file"]): r.get("title", "") for r in rows}
    except Exception as e:
        print(f"warning: no document titles for the checker ({type(e).__name__})", file=sys.stderr)
        return {}


def manifest_laws(manifest: Path | str | None = None) -> dict[str, Any]:
    """doc_id -> LawRef for every numbered law in the manifest, or nothing."""
    from dalil.corpus import load_manifest
    try:
        return law_refs(load_manifest(manifest) if manifest else load_manifest())
    except Exception as e:
        print(f"warning: no law list for the checker ({type(e).__name__})", file=sys.stderr)
        return {}


# The only caveat and refusal sentences that may go out without a citation,
# in every language Dalil answers in.
FIXED_UNCITED = lang.all_fixed({
    WEAK_REFUSAL, FEE_CAVEAT, LAW_CAVEAT,
    safety.REFUSAL_TEXT, safety.REFUSAL_FOLLOW_UP, safety.PERSONAL_DATA_CAVEAT, safety.ASK_FOR_DETAIL,
})
# How many of the top hits are looked at for laws whose scope article a
# question about a law's reach needs, and how many scope passages are added.
SCOPE_FROM = 8
SCOPE_ADD = 3
_COUNT = re.compile(r"\bhow\s+(?:many|much)\b|\bnumber\s+of\b|\bhow\s+often\b|\bcount\b|\btotal\b"
                    r"|كم\s+(?:عدد|عددا|مرة)|^\s*كم\s+\S+|عدد\s", re.IGNORECASE)


def asks_for_a_count(question: str) -> bool:
    """Whether the question asks how many or how much, which open data answers."""
    return bool(_COUNT.search(question))


# How high a named subject's own page must rank for the search to keep to it.
SUBJECT_RANK = 3
_PLACEHOLDER = re.compile(r"\[[A-Z][A-Z ]+\]")
# Past reviews whose questions are translated for matching, per request, and
# how many translations are kept across requests.
MEMORY_TRANSLATIONS = 5
ENGLISH_CACHE_MAX = 2000

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PASSAGES = REPO_ROOT / "build" / "passages.jsonl"
K = 8
ZETARIS_TIMEOUT = 30.0


@dataclass
class Record:
    id: str
    question: str  # redacted
    flags: list[str]
    status: str
    retrieval: list[dict[str, Any]] = field(default_factory=list)
    data: list[dict[str, Any]] = field(default_factory=list)
    draft: list[dict[str, Any]] = field(default_factory=list)
    check: dict[str, Any] = field(default_factory=dict)


def default_zetaris() -> Any:
    """A Zetaris client, or raise. Imported lazily so tests never touch it."""
    from dalil import zetaris as z

    return z.Client(timeout=ZETARIS_TIMEOUT)


class NoSource(RuntimeError):
    """Raised by default_data_source when no open-data source is available."""


def default_data_source() -> Any:
    """The open-data source DALIL_OPEN_DATA picks (dalil.datasource), or raise
    NoSource with the reason. Zetaris when configured and answering, else the
    local database, by default."""
    from dalil import datasource

    source, why = datasource.resolve(zetaris_timeout=ZETARIS_TIMEOUT)
    if source is None:
        raise NoSource(why)
    if why:  # auto mode fell back to the local database
        print(f"open data: {why}", file=sys.stderr)
    return source


class Pipeline:
    def __init__(self, store: Store, retriever: Searcher, passages: dict[str, dict[str, Any]],
                 drafter: Drafter | None = None,
                 zetaris_factory: Callable[[], Any] | None = default_data_source, k: int = K,
                 memory: Any = None, screener: safety.Screener | None = None,
                 rules: tuple[Any, ...] = RULES, manifest: Path | str | None = None, redraft: bool = True):
        self.store = store
        self.retriever = retriever
        self.passages = passages
        self.drafter = drafter or ExtractiveDrafter()
        # Builds the open-data source: anything with sql(statement, limit) whose
        # result has dicts(). The name is kept from when only Zetaris existed.
        self.zetaris_factory = zetaris_factory
        self.k = k
        self._zetaris: Any = None
        # Review memory (dalil.hmem.HMem or a stand-in with recall()); None turns it off.
        self.memory = memory
        # The knowledge base's own ID formats and fixed sentences (dalil/kb.py).
        self.screener = screener or safety.DEFAULT_SCREENER
        self.fixed_uncited = lang.all_fixed({WEAK_REFUSAL, FEE_CAVEAT, LAW_CAVEAT, *self.screener.fixed_texts.values()})
        # Open-data rules that apply to this knowledge base; () for none.
        self.rules = tuple(rules)
        # When the checker cuts every claim the model wrote, quote the sources
        # with the extractive drafter instead of holding an empty answer.
        self.redraft = redraft
        self.titles = document_titles(manifest)
        self.laws = manifest_laws(manifest)

    @classmethod
    def from_build(cls, store: Store | None = None, passages_path: Path | str = DEFAULT_PASSAGES,
                   **kwargs: Any) -> "Pipeline":
        return cls(store or Store(), make_retriever(passages_path), load_passages(passages_path), **kwargs)

    @classmethod
    def from_profile(cls, profile: Any, store: Store | None = None, **kwargs: Any) -> "Pipeline":
        """A pipeline over one knowledge-base profile (dalil.kb.Profile): its
        passages, store, screener, manifest and open data."""
        from dalil import datasource

        if profile.open_data_dir is None:
            factory = None
        else:
            def factory() -> Any:
                env = {**os.environ, "DALIL_OPEN_DATA_DIR": str(profile.open_data_dir)}
                source, why = datasource.resolve(env, zetaris_timeout=ZETARIS_TIMEOUT)
                if source is None:
                    raise NoSource(why)
                if why:
                    print(f"open data: {why}", file=sys.stderr)
                return source
        kwargs.setdefault("zetaris_factory", factory)
        kwargs.setdefault("screener", profile.screener)
        kwargs.setdefault("rules", RULES if profile.open_data_rules else ())
        kwargs.setdefault("manifest", profile.manifest)
        return cls.from_build(store or Store(profile.store), passages_path=profile.passages, **kwargs)

    def data_source(self) -> tuple[Any, str]:
        """The open-data source and "", or None and why it is unavailable."""
        if self._zetaris is not None:
            return self._zetaris, ""
        if self.zetaris_factory is None:
            return None, "Open data not configured; lookup skipped"
        try:
            source = self.zetaris_factory()
            source.sql("SELECT 1")
        except NoSource as e:
            return None, str(e)
        except Exception as e:  # neither source puts credentials in its messages
            return None, f"Open data source unreachable; lookup skipped ({type(e).__name__}: {str(e)[:160]})"
        self._zetaris = source
        return source, ""

    # The old name, for callers written before the local database existed.
    zetaris = data_source

    def data_source_status(self) -> dict[str, Any]:
        """The health endpoint's data_source block. A cached source that stops
        answering is dropped, so the next question resolves again (in auto
        mode, Zetaris down means the local database takes over)."""
        from dalil import datasource

        if self.zetaris_factory is None and not self.rules:
            # A knowledge base with no open data at all, such as kb/uk.
            return {**datasource.status(None), "label": "Open data: none for this knowledge base"}
        source, _ = self.data_source()
        out = datasource.status(source)
        if source is not None and not out["connected"]:
            self._zetaris = None
            source, _ = self.data_source()
            out = datasource.status(source)
        return out

    def recall(self, qid: str, question: str, english: str = "") -> Guidance:
        """Reviews of similar questions, from the review memory, and log what was used."""
        if self.memory is None:
            return Guidance()
        recalled = self.memory.recall(question)
        if recalled is None:
            self.store.event(qid, "recalled", {"available": False})
            return Guidance()
        g = guidance(question, recalled.get("memories") or [], english=self._to_english(question, english))
        self.store.event(qid, "recalled", {
            "available": True, "similar_reviews": g.similar,
            "passages_left_out": sorted(g.avoid), "notes": len(g.notes),
        })
        return g

    def _to_english(self, question: str, english: str) -> Callable[[str], str] | None:
        """English for the review memory's question matching: this question's
        translation, English text unchanged, others translated once and cached."""
        translate = self.translator()
        if translate is None:
            return None
        cache: dict[str, str] = getattr(self, "_english_cache", {})
        if len(cache) > ENGLISH_CACHE_MAX:
            cache = {}
        self._english_cache = cache
        failed: set[str] = set()  # this request only: a later request tries again
        if english:
            cache[question] = english

        budget = [MEMORY_TRANSLATIONS]

        def to_english(text: str) -> str:
            """English for `text`, or "" when it cannot be had. A failure is
            cached too, and each request translates at most a few memories,
            so a slow model cannot hold a request for long."""
            if lang.language(text) == lang.ENGLISH:
                return text
            if text in cache:
                return cache[text]
            if text in failed or budget[0] <= 0:
                return ""
            budget[0] -= 1
            try:
                cache[text] = translate(text)
            except Exception:
                failed.add(text)
                return ""
            return cache[text]
        return to_english

    def translator(self) -> Callable[[str], str] | None:
        """The model's translate(), when the drafter has a model."""
        d = self.drafter
        for candidate in (d, getattr(d, "primary", None), getattr(getattr(d, "inner", None), "primary", None)):
            if candidate is not None and callable(getattr(candidate, "translate", None)):
                return candidate.translate
        return None

    def english_query(self, qid: str, question: str) -> str:
        """For a question that is not in English, its English translation, or ""."""
        if lang.language(question) == lang.ENGLISH:
            return ""
        translate = self.translator()
        if translate is None:
            return ""
        try:
            english = translate(question)
        except Exception as e:  # the model is down: search with the question alone
            self.store.event(qid, "translated", {"ok": False, "error": type(e).__name__})
            return ""
        english = english if lang.language(english) == lang.ENGLISH else ""
        self.store.event(qid, "translated", {"ok": bool(english), "english": english})
        return english

    def search(self, question: str, avoid: set[str], english: str = "") -> list[Hit]:
        """Top k hits, leaving out `avoid` and filling up from further down the ranking.

        For a question about what a law covers, the scope article of each law
        in the results comes first, in the question's language, because it is
        the law's own answer and its wording is often too far from the
        question's to rank high.
        """
        # A redaction placeholder such as "[NATIONAL INSURANCE]" is not what
        # the question is about, so it is not searched for.
        query, english_query = _PLACEHOLDER.sub(" ", question), _PLACEHOLDER.sub(" ", english)
        # Twice as deep as needed, so that keeping to a named subject below
        # still leaves k passages.
        depth = 2 * self.k + len(avoid)
        hits = self.retriever.search(query, depth)
        if english:
            hits = fuse(self.retriever.search(english_query, depth), hits)
        hits = [h for h in hits if h.passage_id not in avoid]
        hits = self.on_subject(english or question, hits)[: self.k]
        if not is_scope_question(question) or not self.laws:
            return hits
        law_docs = list(dict.fromkeys(h.doc_id for h in hits[:SCOPE_FROM] if h.doc_id in self.laws))
        have = {h.passage_id for h in hits}
        question_lang = lang.language(question)
        # The same law in the question's language, when the results only hold another.
        keys = {self.laws[d].key for d in law_docs}
        law_docs += [d for d, ref in self.laws.items() if ref.key in keys and d not in law_docs]
        wanted = [d for d in law_docs if self._language(d) == question_lang] or law_docs
        extra = [pid for pid in scope_passages(wanted, self.passages)
                 if pid not in have and pid not in avoid][:SCOPE_ADD]
        return [self._hit(pid) for pid in extra] + hits

    @staticmethod
    def _all_claims_cut(check: dict[str, Any]) -> bool:
        claims = [s for s in check["sentences"] if s["kind"] == "claim"]
        return bool(claims) and not any(s["kept"] for s in claims)

    def on_subject(self, question: str, hits: list[Hit]) -> list[Hit]:
        """When the question names a service that has its own page, and that
        page is in the results, keep only passages that can be about it: that
        page, the laws, and passages that name the service themselves.

        "Can a Golden Visa holder sponsor their parents?" otherwise gets the
        rules for employees' families and for UAE citizens' parents, which
        read close to the question and do not apply to it."""
        titles, laws = getattr(self, "titles", {}), getattr(self, "laws", {})
        subjects = subject_docs(question, titles)
        # The subject's own page must rank near the top: a page that matches
        # only further down is not what the question is about.
        if not subjects or not any(h.doc_id in subjects for h in hits[:SUBJECT_RANK]):
            return hits
        names = [titles[d].split("|")[0] for d in subjects]
        # What the question asks about the subject: its words, less the
        # subject's own name and words for the person holding it.
        asked = set(tokenize(question)) - set().union(*(title_terms(n) for n in names)) - _ROLE_WORDS \
            - safety.subject_generic() - self._common_words()
        # Two of the asked words, or all of them when fewer: one shared word,
        # such as "medical" in "medical fitness tests for dependants", does
        # not make another page's passage answer "which medical conditions
        # make someone unfit".
        need = min(2, len(asked))
        return [h for h in hits if h.doc_id in subjects or h.doc_id in laws
                or (names_subject(f"{titles.get(h.doc_id, '')} {h.heading or ''} {h.text}", names)
                    and len(set(tokenize(h.text)) & asked) >= need)]

    def _common_words(self) -> set[str]:
        """Words in more than a fifth of the passages, such as "uae": they say
        nothing about what a passage is about."""
        lexical = getattr(self.retriever, "lexical", None)
        postings, n = getattr(lexical, "_postings", None), len(getattr(lexical, "passages", []) or [])
        if not postings or not n:
            return set()
        if getattr(self, "_common_cache", None) is None:
            self._common_cache = {t for t, post in postings.items() if len(post) > n / 5}
        return self._common_cache

    def _language(self, doc_id: str) -> str:
        for p in self.passages.values():
            if p["doc_id"] == doc_id:
                return p.get("language") or ""
        return ""

    def _hit(self, pid: str) -> Hit:
        p = self.passages[pid]
        return Hit(passage_id=pid, doc_id=p["doc_id"], page=int(p["page"]), score=0.0, heading=p.get("heading"),
                   text=p["text"], start=int(p["start"]), end=int(p["end"]), kind=p.get("kind") or "")

    def answer(self, question: str) -> Record:
        screened = self.screener.screen(question)
        # From here on only the redacted text exists.
        qid = self.store.create_question(screened.redacted_question, screened.flags)
        self.store.event(qid, "screened", {"flags": screened.flags})
        try:
            return self._answer(qid, screened)
        except Exception as e:
            self.store.event(qid, "failed", {"error": type(e).__name__, "message": str(e)[:300]})
            self.store.save_record(qid, [], [], [], {"status": "held", "answer": "", "sentences": []}, "held")
            self.store.event(qid, "queued", {"status": "held"})
            return Record(qid, screened.redacted_question, screened.flags, "held")

    def _answer(self, qid: str, screened: safety.ScreenResult) -> Record:
        q = screened.redacted_question
        hits: list[Hit] = []
        english = ""  # the question in English, when it was asked in another language
        data: list[DataResult] = []
        rules: list[Any] = []
        client = None
        if screened.is_rule_evasion:
            self.store.event(qid, "retrieved", {"skipped": "rule_evasion", "hits": 0})
            draft = [refusal(self.screener.refusal_text), refusal(self.screener.refusal_follow_up)]
        elif screened.is_vague:
            # Too few key words to know the subject: ask, rather than guess.
            self.store.event(qid, "retrieved", {"skipped": "vague_question", "hits": 0})
            draft = [refusal(self.screener.ask_for_detail)]
            if screened.has_personal_data:
                draft.append(caveat(self.screener.personal_data_caveat))
        else:
            english = self.english_query(qid, q)
            past = self.recall(qid, q, english)
            hits = self.search(q, past.avoid, english)
            self.store.event(qid, "retrieved", {
                "k": self.k, "hits": len(hits),
                "top": [{"passage_id": h.passage_id, "score": round(h.score, 3)} for h in hits[:3]],
            })
            # Open data answers how many or how much. A question that only
            # mentions a subject a rule covers, such as "Can I get my Emirates
            # ID within 24 hours?", gets no yearly count it did not ask for.
            rules = match_rules(q, self.rules) if asks_for_a_count(q) else []
            if rules:
                client, why = self.data_source()
                run = (lambda sql: client.sql(sql, limit=200).dicts()) if client else None
                kind = _kind(client)
                data = run_rules(rules, run, why, source=kind)
                if client is not None and all(d.note.startswith("lookup failed") for d in data):
                    self._zetaris = None  # resolve again next time; auto mode can fall back
                self.store.event(qid, "data_looked_up", {
                    "source": kind or None,
                    "tables": [d.table for d in data],
                    "rows": [len(d.rows) for d in data],
                    "notes": [d.note for d in data if d.note],
                })
            draft = self.drafter.draft(q, hits, data, notes=past.notes, **({"english": english} if english else {}))
            if screened.has_personal_data:
                draft = [*draft, caveat(self.screener.personal_data_caveat)]
        # Fixed refusals and caveats go out in the question's language.
        question_lang = lang.language(q)
        draft = [{**s, "text": lang.fixed(s["text"], question_lang)} if s.get("kind") in ("refusal", "caveat") else s
                 for s in draft]
        ran = "fixed refusal" if (screened.is_rule_evasion or screened.is_vague) else (
            getattr(self.drafter, "last_used", "") or type(self.drafter).__name__)
        error = "" if (screened.is_rule_evasion or screened.is_vague) else getattr(self.drafter, "last_error", "")
        self.store.event(qid, "drafted", {
            "drafter": ran, **({"fallback_reason": error} if error else {}),
            "sentences": len(draft),
            "kinds": [s.get("kind", "claim") for s in draft],
        })
        draft = facts_in_language(draft, data, question_lang)
        drafted = len(draft)
        draft = drop_repeats(draft, data)
        repeats = drafted - len(draft)
        drafted = len(draft)
        draft = drop_internal_names(draft, data)
        internal = drafted - len(draft)
        draft, attached = attach_data(draft, data)
        draft, unrelated = drop_unrelated_passages(draft, data)
        if client is None and any("sql" in c for s in draft for c in s.get("citations") or []):
            client, _ = self.data_source()
        if not draft:
            # Nothing is left to check, for example when every drafted
            # sentence was dropped as a repeat. A reviewer never gets an
            # empty answer: Dalil says it cannot answer, and records why.
            draft = [refusal(lang.fixed(WEAK_REFUSAL, question_lang))]
            self.store.event(qid, "emptied", {"reason": "no drafted sentence was left to check",
                                              "drafter": ran})
        subjects = {d: self.titles[d] for d in subject_docs(english or q, self.titles)
                    if any(h.doc_id == d for h in hits)}
        verifier = Verifier(self.passages, data_lookup(client) if client else None,
                            allowed_sql=[d.sql for d in data], allowed_uncited=self.fixed_uncited,
                            allowed_passages=[h.passage_id for h in hits], titles=self.titles, laws=self.laws,
                            similarity=getattr(self.retriever, "similarities", None), question=english or q,
                            language=question_lang,
                            subjects=subjects,
                            data_facts={d.sql: " ".join(f for f in (d.fact, d.fact_ar, d.meaning) if f)
                                        for d in data if d.fact})
        check = verifier.check_answer(draft).to_dict()
        added = restate_data(draft, check, data, question_lang, counted=counted_lookups(q, english, rules))
        if added:
            draft = added
            check = verifier.check_answer(draft).to_dict()
        # Only for English questions: the extractive drafter matches words, and
        # most sources are English, so for another language it finds the few
        # documents in that language whether or not they answer.
        if self.redraft and question_lang == lang.ENGLISH and self._all_claims_cut(check) \
                and not isinstance(self.drafter, ExtractiveDrafter):
            # The model's every claim failed the checker. Quote the sources
            # instead: the extractive drafter copies whole source sentences,
            # so whatever it keeps is the documents' own words.
            # Quote from the page the question is about, when it names one.
            # Two passages can hold the same sentence; it is quoted once.
            redraft = drop_repeats(ExtractiveDrafter().draft(q, [h for h in hits if h.doc_id in subjects] or hits,
                                                             data), data)
            if screened.has_personal_data:
                redraft = [*redraft, caveat(self.screener.personal_data_caveat)]
            redraft = [{**s, "text": lang.fixed(s["text"], question_lang)}
                       if s.get("kind") in ("refusal", "caveat") else s for s in redraft]
            second = verifier.check_answer(redraft).to_dict()
            if lang.language(" ".join(s["text"] for s in second["sentences"] if s["kept"])) == question_lang \
                    and any(s["kept"] and s["kind"] == "claim" for s in second["sentences"]) \
                    and second["status"] != "held":
                self.store.event(qid, "redrafted", {"drafter": "ExtractiveDrafter",
                                                    "reason": "the checker cut every claim the model wrote",
                                                    "cut_first": [s["reasons"] for s in check["sentences"]
                                                                  if not s["kept"]]})
                draft, check = redraft, second
        if self.redraft and self._all_claims_cut(check) \
                and not any(s["kept"] and s["kind"] == "refusal" for s in check["sentences"]):
            # Every claim was cut and no quote-only redraft could stand in.
            # The resident gets Dalil's fixed refusal rather than nothing, and
            # the reviewer still sees every cut sentence and why.
            draft = [*draft, refusal(lang.fixed(WEAK_REFUSAL, question_lang))]
            check = verifier.check_answer(draft).to_dict()
            self.store.event(qid, "declined", {"reason": "the checker cut every claim the writer drafted",
                                               "cut": sum(1 for s in check["sentences"] if not s["kept"])})
        # The checked sentences follow the draft's order; carry the mark of a
        # sentence Dalil's data rule stated, so the screen can show it.
        for checked, drafted in zip(check["sentences"], draft):
            if drafted.get("restated"):
                checked["restated"] = True
        kept = sum(s["kept"] for s in check["sentences"])
        self.store.event(qid, "verified", {"status": check["status"], "kept": kept,
                                           "cut": len(check["sentences"]) - kept,
                                           **({"data_restated": sum(1 for s in draft if s.get("restated"))}
                                              if added else {}),
                                           **({"data_citations_attached": attached} if attached else {}),
                                           **({"repeats_dropped": repeats} if repeats else {}),
                                           **({"internal_names_dropped": internal} if internal else {}),
                                           **({"unrelated_passages_dropped": unrelated} if unrelated else {})})
        status = "held" if check["status"] == "held" else "awaiting_review"
        retrieval = [asdict(h) for h in hits]
        data_dicts = [d.to_dict() for d in data]
        self.store.save_record(qid, retrieval, data_dicts, draft, check, status)
        self.store.event(qid, "queued", {"status": status})
        return Record(qid, q, screened.flags, status, retrieval, data_dicts, draft, check)


def _kind(source: Any) -> str:
    if source is None:
        return ""
    from dalil.datasource import kind_of

    return kind_of(source)


def _numbers(text: str) -> set[str]:
    from dalil.verify import numbers
    return numbers(text)


def drop_repeats(draft: list[dict[str, Any]], data: list[DataResult] = ()) -> list[dict[str, Any]]:
    """The draft without claims that repeat an earlier one.

    A small model sometimes writes the same sentence several times, or
    follows "EHS carried out 1424313 exams in 2025." with "The data value is
    1424313." A claim is a repeat when its text matches an earlier claim word
    for word, or when every number in it is an open-data value (not a year
    alone) that an earlier claim already stated. One copy says it."""
    values = {n for res in data if res.rows for k, v in res.rows[0].items()
              if v is not None and k.lower() != "year" for n in _numbers(str(v))}
    seen: set[str] = set()
    said: set[str] = set()
    out = []
    for s in draft:
        if s.get("kind", "claim") == "claim":
            # Case, spacing and punctuation do not make a sentence a new one.
            key = " ".join(re.sub(r"[^\w]+", " ", str(s.get("text", ""))).split()).casefold()
            nums = _numbers(str(s.get("text", "")))
            if key in seen or (nums & values and nums <= said):
                continue
            seen.add(key)
            said |= nums
        out.append(s)
    return out


def facts_in_language(draft: list[dict[str, Any]], data: list[DataResult], language: str
                      ) -> list[dict[str, Any]]:
    """A rule's English fact sentence in the question's language. The extractive
    drafter, which answers when the model fails, writes every data fact from
    the English template; for an Arabic question the Arabic one is used."""
    if language != lang.ARABIC:
        return draft
    swap = {}
    for res in data:
        if res.rows and res.fact and res.fact_ar:
            english = DataResult(res.table, res.sql, res.rows, fact=res.fact)
            arabic = DataResult(res.table, res.sql, res.rows, fact=res.fact_ar)
            for en, ar in zip(data_claims([english]), data_claims([arabic])):
                swap[en["text"]] = ar["text"]
    return [{**s, "text": swap[s["text"]]} if s.get("kind", "claim") == "claim" and s.get("text") in swap else s
            for s in draft]


def drop_internal_names(draft: list[dict[str, Any]], data: list[DataResult]) -> list[dict[str, Any]]:
    """The draft without claims that name an open-data table, such as "The data
    is from the open_ehs_medical_fitness table." A table name is Dalil's own
    identifier and tells a resident nothing; the screen shows the source."""
    names = {n.lower() for res in data for n in (res.table, res.table.split(".")[-1])}
    if not names:
        return draft
    return [s for s in draft if s.get("kind", "claim") != "claim"
            or not any(n in str(s.get("text", "")).lower() for n in names)]


def attach_data(draft: list[dict[str, Any]], data: list[DataResult]) -> tuple[list[dict[str, Any]], int]:
    """Give a drafted claim the data citations for the lookup values it states.

    A small model often writes "The number of exams is 1424313." and cites a
    passage, or nothing that holds the number. When a claim states a value
    (other than a year) from the row of a lookup Dalil ran for this question,
    the citation to that column is added, with the year too if the claim
    states it. The checker then reruns the query and tests the wording as for
    any data citation, so this can only keep a sentence the data backs.
    Returns the new draft and how many citations were added.
    """
    added = 0
    out = []
    for s in draft:
        if s.get("kind", "claim") != "claim":
            out.append(s)
            continue
        stated = _numbers(s.get("text", ""))
        cits = list(s.get("citations") or [])
        have = {(c.get("sql"), c.get("column")) for c in cits if "sql" in c}
        new = []
        for res in data:
            if not res.rows:
                continue
            row = res.rows[0]
            cols = [k for k, v in row.items() if v not in (None, "") and _numbers(str(v))
                    and _numbers(str(v)) <= stated]
            if not any(k.lower() != "year" for k in cols):
                continue
            new += [{"table": res.table, "sql": res.sql, "column": k, "value": str(row[k])}
                    for k in cols if (res.sql, k) not in have]
        if new:
            added += len(new)
            s = {**s, "citations": cits + new, "data_attached": len(new)}
        out.append(s)
    return out, added


def drop_unrelated_passages(draft: list[dict[str, Any]], data: list[DataResult]
                            ) -> tuple[list[dict[str, Any]], int]:
    """On a claim that states a lookup's value and cites that lookup, put the
    data citations first, and keep a passage citation only if its quote shares
    a subject word with the claim.

    The model tends to add a passage to every data sentence, such as a page on
    cancelling a visa next to "ICP handles 3763072 residence permit renewals".
    The reviewer opens that source first, and it backs nothing in the sentence.
    The data citation carries the claim; a passage that names the same subject
    stays. Returns the new draft and how many citations were taken off."""
    from dalil.retrieve import tokenize
    sqls = {d.sql for d in data if d.rows}
    dropped = 0
    out = []
    for s in draft:
        cits = s.get("citations") or []
        if s.get("kind", "claim") != "claim" or not any(c.get("sql") in sqls and _states(s, c) for c in cits):
            out.append(s)
            continue
        words = {t for t in tokenize(str(s.get("text", ""))) if not t.isdigit()}
        # The data citations come first: they carry the claim, so the screen
        # numbers them 1, 2, and a reviewer opens the lookup before any page.
        keep = [c for c in cits if "sql" in c] + [
            c for c in cits if "sql" not in c
            and words & {t for t in tokenize(str(c.get("quote") or "")) if not t.isdigit()}]
        dropped += len(cits) - len(keep)
        out.append({**s, "citations": keep} if keep != cits else s)
    return out, dropped


def _states(sentence: Mapping[str, Any], citation: Mapping[str, Any]) -> bool:
    """True when the sentence's text holds the data citation's value."""
    value = _numbers(str(citation.get("value") or ""))
    return bool(value) and value <= _numbers(sentence.get("text", ""))


def _drop_unstated_data(sentence: dict[str, Any]) -> dict[str, Any]:
    cits = sentence.get("citations") or []
    keep = [c for c in cits if "sql" not in c or _states(sentence, c)]
    if len(keep) == len(cits) or sentence.get("kind", "claim") != "claim":
        return sentence
    return {**sentence, "citations": keep, "unstated_data_dropped": len(cits) - len(keep)}


_COUNT_QUESTION = re.compile(r"\bhow\s+many\b|\bnumber\s+of\b|كم\s+عدد|عدد", re.I)


def counted_lookups(question: str, english: str, rules: Any) -> set[str]:
    """The queries of rules a question asks the count of: the rule matches in a
    sentence of the question that asks how many."""
    out = set()
    for text in (question, english):
        for sentence in re.split(r"[.?!؟\n]+", text or ""):
            if _COUNT_QUESTION.search(sentence):
                out |= {r.sql for r in rules or () if r.pattern.search(sentence)}
    return out


def restate_data(draft: list[dict[str, Any]], check: dict[str, Any], data: list[DataResult],
                 language: str = "en", counted: set[str] | frozenset[str] = frozenset()
                 ) -> list[dict[str, Any]] | None:
    """A new draft when the model tried to state an open-data result but no
    kept sentence states it; else None.

    A sentence carries a lookup only when it cites the query and states the
    cited value. A data citation on a sentence that does not state its value
    backs nothing, so it is taken off that sentence; the sentence still has to
    pass on its other citations.

    "Tried" means a drafted claim cites the lookup's query or writes one of the
    values in its row other than the year. The rule's own fixed sentence (dalil/opendata.py) then
    states the value, with a citation the checker reruns like any other. The
    model's cut sentence stays in the record for the reviewer to see. A lookup
    the model left alone is left out: it judged the data off the question,
    unless the question asks how many, which the lookup answers.
    """
    sentences = check.get("sentences", [])
    out = [_drop_unstated_data(s) for s in draft]
    changed = out != draft
    extra = []
    for res in data:
        if not res.rows or not res.fact:
            continue
        carried = any(s["kept"] and any(c.get("ok") and c.get("sql") == res.sql and _states(s, c)
                                        for c in s.get("citations", []))
                      for s in sentences)
        if carried:
            continue
        # A year alone does not show intent: passage claims carry years too.
        values = _numbers(" ".join(str(v) for k, v in res.rows[0].items() if v is not None and k.lower() != "year"))
        tried = any(s.get("kind", "claim") == "claim" and (
            any(c.get("sql") == res.sql for c in s.get("citations") or []) or _numbers(s.get("text", "")) & values)
            for s in draft)
        if tried or res.sql in counted:
            # In the question's language: the Arabic sentence when there is one.
            said = DataResult(res.table, res.sql, res.rows, fact=res.fact_ar if language == lang.ARABIC and res.fact_ar
                              else res.fact)
            extra += [{**claim, "restated": True} for claim in data_claims([said])]
    if not extra:
        return out if changed else None
    # After the model's claims, before the fixed caveats and refusals.
    at = max((i + 1 for i, s in enumerate(out) if s.get("kind", "claim") == "claim"), default=0)
    return out[:at] + extra + out[at:]


def fuse(first: list[Hit], second: list[Hit], rrf_k: int = 60) -> list[Hit]:
    """Two rankings in one by reciprocal rank, each passage keeping its best
    BM25 score, so the weak-retrieval gate sees the English query's matches."""
    score: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for ranking in (first, second):
        for rank, h in enumerate(ranking, start=1):
            score[h.passage_id] = score.get(h.passage_id, 0.0) + 1.0 / (rrf_k + rank)
            old = best.get(h.passage_id)
            best[h.passage_id] = h if old is None else Hit(**{**asdict(h), "score": max(h.score, old.score),
                                                              "dense": max(h.dense, old.dense)})
    return [best[p] for p in sorted(score, key=lambda p: -score[p])]


_default: Pipeline | None = None


def answer(question: str) -> Record:
    """Answer through the default pipeline over build/passages.jsonl and build/dalil.db."""
    global _default
    if _default is None:
        _default = Pipeline.from_build()
    return _default.answer(question)
