"""The local HTTP app: ask, inspect, review.

Serves the JSON API under /api and, when the folder exists, the reviewer UI
in web/ at /. It binds to 127.0.0.1 by default (scripts/serve.py refuses any
non-loopback host): page text and the original documents are private.

    uv run python scripts/serve.py
"""

from __future__ import annotations

import csv
import functools
import json
import mimetypes
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dalil import datasource
from dalil.corpus import doc_id as manifest_doc_id
from dalil.corpus import load_manifest, resolve_corpus_dir
from dalil.draft import ExtractiveDrafter, FallbackDrafter, LLMDrafter
from dalil.hmem import HMem
from dalil.pipeline import Pipeline, cited_passages
from dalil.safety import redact
from dalil.store import ReviewError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCUMENTS = REPO_ROOT / "build" / "documents.jsonl"
DEFAULT_WEB = REPO_ROOT / "web"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
ALLOWED_HOSTS = ("127.0.0.1", "localhost")
MAX_QUESTION_CHARS = 2000


class QuestionIn(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class ReviewIn(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str = Field(min_length=1, max_length=200)
    note: str = Field(default="", max_length=4000)


_SITE_SUFFIX = re.compile(r"\s*\|\s*The Official (?:Platform|Portal) of the UAE Government\s*$", re.IGNORECASE)


def display_title(title: str | None) -> str | None:
    """A page title without the portal's site-name suffix, for the screen."""
    return _SITE_SUFFIX.sub("", title) if title else title


def load_documents(path: Path | str = DEFAULT_DOCUMENTS) -> dict[str, dict[str, Any]]:
    """doc_id -> {title, publisher, source_url, pages: {page: text}} from build/documents.jsonl."""
    out: dict[str, dict[str, Any]] = {}
    path = Path(path)
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            out[d["doc_id"]] = {
                "title": display_title(d.get("title")), "publisher": d.get("publisher"),
                "source_url": d.get("source_url"),
                "pages": {int(p["page"]): p.get("text") or "" for p in d.get("pages") or []},
            }
    return out


def default_zetaris_probe() -> bool:
    """True when Zetaris itself answers. The health endpoint no longer uses it by
    default: it reports the source the pipeline actually uses."""
    try:
        from dalil import zetaris as z

        z.Client(timeout=5).sql("SELECT 1")
        return True
    except Exception:
        return False


@functools.cache
def open_table_meta(table: str) -> dict[str, str | None]:
    """Title and Bayanat page for an open-data table, from data/open/tables.csv and SOURCES.csv."""
    base = Path(__file__).resolve().parent.parent / "data" / "open"
    name = table.split(".")[-1].lower()
    try:
        with (base / "tables.csv").open(newline="", encoding="utf-8") as fh:
            files = {r["table"].lower(): r["file"] for r in csv.DictReader(fh)}
        with (base / "SOURCES.csv").open(newline="", encoding="utf-8") as fh:
            sources = {r["file"]: r for r in csv.DictReader(fh)}
    except OSError:
        return {}
    src = sources.get(files.get(name, ""), {})
    # Combined files list each source page, separated by "; ". Link the first.
    url = (src.get("portal_url") or "").split(";")[0].strip()
    return {"title": src.get("title"), "source_url": url or None}


def data_kind(result: Mapping[str, Any]) -> str:
    """Which source ran a stored lookup. Records saved before the local
    database existed carry no source; rows in them came from Zetaris."""
    if "source" in result:
        return result.get("source") or ""
    return datasource.KIND_ZETARIS if result.get("rows") else ""


@dataclass
class KnowledgeBase:
    """One loaded knowledge base: its pipeline (passages, store, screener,
    open data), its page text and the original files it may serve."""

    id: str
    name: str
    country: str
    pipeline: Pipeline
    documents: Mapping[str, Mapping[str, Any]]
    manifest: dict[str, str]
    corpus: Path | None
    labels: Mapping[str, str] = field(default_factory=dict)  # ask_label, example_question

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "country": self.country,
                "documents": len(self.documents) or len(self.manifest), "passages": len(self.pipeline.passages),
                **self.labels}


class Library:
    """The knowledge bases the app can serve, and which one is active.

    Profiles load on first use and stay in memory, so switching back is
    instant. A switch changes what new questions are answered from; a question
    already asked stays with the knowledge base that answered it.
    """

    def __init__(self, active: KnowledgeBase, choices: Mapping[str, Mapping[str, Any]] | None = None,
                 build: Callable[[str], KnowledgeBase] | None = None):
        self.loaded: dict[str, KnowledgeBase] = {active.id: active}
        self.active = active
        # id -> {"name", "country", "documents", "passages"} for the selector, before loading.
        self.choices = dict(choices or {active.id: active.summary()})
        self.build = build
        self._lock = threading.Lock()

    def switch(self, kb_id: str) -> KnowledgeBase:
        with self._lock:
            if kb_id not in self.loaded:
                if kb_id not in self.choices or self.build is None:
                    raise KeyError(kb_id)
                self.loaded[kb_id] = self.build(kb_id)
            self.active = self.loaded[kb_id]
            return self.active

    def preload(self, ids: Iterable[str] | None = None) -> None:
        """Load knowledge bases ahead of a switch, so the switch itself is instant.
        Run in a background thread; a failure leaves that one to load on switch."""
        for kb_id in list(ids if ids is not None else self.choices):
            with self._lock:
                if kb_id in self.loaded or self.build is None:
                    continue
                try:
                    self.loaded[kb_id] = self.build(kb_id)
                except Exception as e:
                    print(f"warning: knowledge base {kb_id} not preloaded ({type(e).__name__}: {e})",
                          file=sys.stderr)

    def list(self) -> list[dict[str, Any]]:
        out = []
        for kb_id, info in sorted(self.choices.items()):
            kb = self.loaded.get(kb_id)
            out.append({**info, **(kb.summary() if kb else {}), "id": kb_id,
                        "active": kb_id == self.active.id, "loaded": kb is not None})
        return out

    def for_question(self, qid: str) -> KnowledgeBase:
        """The loaded knowledge base whose store holds this question, active first."""
        for kb in (self.active, *[k for k in self.loaded.values() if k is not self.active]):
            if kb.pipeline.store.get(qid) is not None:
                return kb
        return self.active

    def for_document(self, did: str) -> KnowledgeBase:
        for kb in (self.active, *[k for k in self.loaded.values() if k is not self.active]):
            if did in kb.documents or did in kb.manifest:
                return kb
        return self.active


class KnowledgeBaseIn(BaseModel):
    id: str = Field(min_length=1, max_length=32)


def create_app(
    pipeline: Pipeline | None,
    documents: Mapping[str, Mapping[str, Any]] | None = None,
    manifest: Mapping[str, str] | None = None,
    corpus_dir: Path | str | None = None,
    web_dir: Path | str | None = DEFAULT_WEB,
    zetaris_probe: Callable[[], bool] | None = None,
    library: Library | None = None,
) -> FastAPI:
    """Serve one knowledge base (pipeline, documents, manifest, corpus_dir), or
    several through `library`. manifest maps doc_id -> file name inside
    corpus_dir; only those files are served."""
    if library is None:
        if pipeline is None:
            raise ValueError("create_app needs a pipeline or a library")
        corpus = Path(corpus_dir).expanduser().resolve() if corpus_dir else None
        library = Library(KnowledgeBase("default", "Knowledge base", "", pipeline, dict(documents or {}),
                                        dict(manifest or {}), corpus))
    lib = library
    app = FastAPI(title="Dalil", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.library = lib
    # Binding to loopback is not enough on its own: a page that rebinds its own
    # hostname to 127.0.0.1 could read the private page text. Only accept
    # requests addressed to this machine by name.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(ALLOWED_HOSTS))

    @app.middleware("http")
    async def same_origin_only(request: Request, call_next: Callable[..., Any]) -> Any:
        """A browser sends Origin on cross-site requests. Only this app's own page
        may call the API, so another site cannot approve answers or read text."""
        origin = request.headers.get("origin")
        if origin is not None and urlsplit(origin).netloc != request.headers.get("host", ""):
            return JSONResponse({"detail": "cross-site requests are not accepted"}, status_code=403)
        return await call_next(request)

    def detail(qid: str) -> dict[str, Any]:
        kb = lib.for_question(qid)
        rec = kb.pipeline.store.get(qid)
        if rec is None:
            raise HTTPException(404, "no such question")

        def doc_meta(did: str | None) -> Mapping[str, Any]:
            return kb.documents.get(did or "", {})

        retrieval = [
            {"passage_id": h["passage_id"], "doc_id": h["doc_id"], "title": doc_meta(h["doc_id"]).get("title"),
             "page": h["page"], "score": h["score"], "heading": h["heading"], "text": h["text"],
             "start": h["start"], "end": h["end"]}
            for h in rec["retrieval"]
        ]
        data = [{"table": d["table"], "sql": d["sql"], "rows": d["rows"], "note": d["note"],
                 "source": data_kind(d), "source_label": datasource.source_label(data_kind(d))}
                for d in rec["data"]]
        check = rec["check"]
        if check:
            check = json.loads(json.dumps(check))
            for s in check.get("sentences", []):
                for c in s.get("citations", []):
                    if c.get("doc_id"):
                        meta = doc_meta(c["doc_id"])
                        c["title"], c["source_url"] = meta.get("title"), meta.get("source_url")
                    else:
                        meta = open_table_meta(c.get("table") or "")
                        c["title"], c["source_url"] = meta.get("title") or c.get("table"), meta.get("source_url")
                        ran = next((d for d in data if d["table"] == c.get("table") and d["sql"] == c.get("sql")),
                                   None) or next((d for d in data if d["table"] == c.get("table")), None)
                        c["source_label"] = ran["source_label"] if ran else None
        return {
            "id": rec["id"], "question": rec["question"], "flags": rec["flags"], "status": rec["status"],
            "created_at": rec["created_at"], "retrieval": retrieval, "data": data, "check": check,
            "review": rec["review"], "audit": rec["audit"], "knowledge_base": kb.id,
        }

    @app.post("/api/questions")
    def post_question(body: QuestionIn) -> dict[str, Any]:
        if not body.question.strip():
            raise HTTPException(422, "question is empty")
        rec = lib.active.pipeline.answer(body.question)
        return {"id": rec.id, "status": rec.status}

    @app.get("/api/questions")
    def list_questions() -> list[dict[str, Any]]:
        return lib.active.pipeline.store.list()

    @app.get("/api/questions/{qid}")
    def get_question(qid: str) -> dict[str, Any]:
        return detail(qid)

    @app.post("/api/questions/{qid}/review")
    def review(qid: str, body: ReviewIn) -> dict[str, Any]:
        kb = lib.for_question(qid)
        # Reviewers type free text too. Redact it before it reaches the store or
        # H-MEM's append-only ledger, which can never be scrubbed afterwards.
        # Both the UAE formats and this knowledge base's own are removed.
        reviewer, _ = kb.pipeline.screener.redact(redact(body.reviewer)[0])
        note, _ = kb.pipeline.screener.redact(redact(body.note)[0])
        try:
            kb.pipeline.store.review(qid, body.decision, reviewer, note)
        except KeyError:
            raise HTTPException(404, "no such question") from None
        except ReviewError as e:
            raise HTTPException(409, str(e)) from None
        remember_review(kb, qid, body.decision, reviewer, note)
        return detail(qid)

    def remember_review(kb: KnowledgeBase, qid: str, decision: str, reviewer: str, note: str) -> None:
        """Write the decision to the review memory, and log the outcome either way."""
        memory = kb.pipeline.memory
        store = kb.pipeline.store
        if memory is None:
            return
        try:
            out = write_memory(kb, memory, qid, decision, reviewer, note)
        except Exception as e:  # the review is saved; never turn it into an error here
            store.event(qid, "memory_written", {"available": False, "error": type(e).__name__})
            return
        if out is None:
            store.event(qid, "memory_written", {"available": False})
        else:
            store.event(qid, "memory_written", {"available": True, "memory_id": out["memory"]["id"],
                                                "kind": out["memory"].get("kind"), "passages": out["passages"]})

    def write_memory(kb: KnowledgeBase, memory: Any, qid: str, decision: str, reviewer: str,
                     note: str) -> dict[str, Any] | None:
        rec = kb.pipeline.store.get(qid) or {}
        check = rec.get("check") or {}
        cited = cited_passages(check)
        out = memory.remember(decision, qid, rec.get("question", ""), reviewer, cited,
                              answer=check.get("answer", ""), note=note)
        if not out or not isinstance(out.get("memory"), dict) or not out["memory"].get("id"):
            return None
        return {**out, "passages": len(cited)}

    @app.get("/api/documents/{did}/pages/{page}")
    def get_page(did: str, page: int) -> dict[str, Any]:
        doc = lib.for_document(did).documents.get(did)
        if doc is None or page not in doc["pages"]:
            raise HTTPException(404, "no such page")
        return {"doc_id": did, "title": doc.get("title"), "publisher": doc.get("publisher"),
                "source_url": doc.get("source_url"), "page": page, "text": doc["pages"][page]}

    @app.get("/api/documents/{did}/file")
    def get_file(did: str) -> FileResponse:
        kb = lib.for_document(did)
        name = kb.manifest.get(did)
        if not name or kb.corpus is None:
            raise HTTPException(404, "no such document")
        path = (kb.corpus / name).resolve()
        if path.parent != kb.corpus or not path.is_file():
            raise HTTPException(404, "document file not found")
        media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if media == "text/html":
            media = "text/html; charset=utf-8"
        # The originals are scraped pages with their own scripts. Sandbox them so
        # nothing in them can call this API from this origin.
        return FileResponse(path, media_type=media, headers={
            "Content-Security-Policy": "sandbox",
            "X-Content-Type-Options": "nosniff",
        })

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        kb = lib.active
        pipeline = kb.pipeline
        memory = pipeline.memory.health() if pipeline.memory is not None else None
        source = pipeline.data_source_status()
        # zetaris stays for older screens: true only when Zetaris itself serves the data.
        zetaris = bool(zetaris_probe()) if zetaris_probe is not None else (
            source["connected"] and source["kind"] == datasource.KIND_ZETARIS)
        return {"ok": True, "zetaris": zetaris, "data_source": source, "passages": len(pipeline.passages),
                "knowledge_base": kb.summary(),
                "memory": bool(memory), "memories": (memory or {}).get("memories", 0)}

    @app.get("/api/knowledge-bases")
    def knowledge_bases() -> list[dict[str, Any]]:
        return lib.list()

    @app.post("/api/knowledge-base")
    def switch_knowledge_base(body: KnowledgeBaseIn) -> dict[str, Any]:
        started = time.monotonic()
        try:
            kb = lib.switch(body.id)
        except KeyError:
            raise HTTPException(404, "no such knowledge base") from None
        return {"knowledge_base": kb.summary(), "seconds": round(time.monotonic() - started, 2)}

    if web_dir is not None and Path(web_dir).is_dir():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    return app


def manifest_map(path: Path | str | None = None) -> dict[str, str]:
    rows = load_manifest(path) if path else load_manifest()
    return {manifest_doc_id(r["file"]): r["file"] for r in rows}


def build_knowledge_base(profile: Any) -> KnowledgeBase:
    """Load one profile (dalil.kb.Profile) the way the reviewer screen uses it."""
    # The model drafts when it answers; otherwise the extractive drafter does,
    # and the audit trail records which one ran and why.
    drafter = FallbackDrafter(LLMDrafter(audience=profile.audience), ExtractiveDrafter())
    memory = HMem() if profile.review_memory else None
    pipeline = Pipeline.from_profile(profile, memory=memory, drafter=drafter)
    corpus = profile.corpus_dir or resolve_corpus_dir()
    try:
        files = manifest_map(profile.manifest)
    except Exception:  # no manifest: pages still show, originals are not served
        files = {}
    kb = KnowledgeBase(profile.id, profile.name, profile.country, pipeline, load_documents(profile.documents),
                       files, Path(corpus).expanduser().resolve(), profile.labels)
    try:  # embed the passages now, so the first question after a switch is not the slow one
        pipeline.retriever.search("warm up", 1)
    except Exception:
        pass
    return kb


def create_default_app() -> FastAPI:
    """The app over the knowledge base DALIL_KB names (default "uae": build/,
    data/open/ and the private corpus in $DALIL_CORPUS_DIR or ./corpus), with
    every other profile under kb/ ready to switch to from the screen."""
    from dalil import kb as kbmod

    profiles = {}
    for kb_id in kbmod.available():
        try:
            profiles[kb_id] = kbmod.load(kb_id)
        except kbmod.ProfileError as e:
            print(f"warning: knowledge base {kb_id} skipped: {e}", file=sys.stderr)
    active = kbmod.active_id()
    if active not in profiles:
        raise kbmod.ProfileError(f"DALIL_KB={active!r} is not a usable knowledge base; have: "
                                 f"{', '.join(sorted(profiles)) or 'none'}")
    choices = {i: p.summary(kbmod.count_documents(p.documents), kbmod.count_documents(p.passages))
               for i, p in profiles.items()}
    library = Library(build_knowledge_base(profiles[active]), choices,
                      build=lambda kb_id: build_knowledge_base(profiles[kb_id]))
    # The others load in the background, so switching on the screen is instant.
    threading.Thread(target=library.preload, name="preload-knowledge-bases", daemon=True).start()
    return create_app(None, library=library)
