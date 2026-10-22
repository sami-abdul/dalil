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
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from fastapi import FastAPI, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dalil.corpus import doc_id as manifest_doc_id
from dalil.corpus import load_manifest, resolve_corpus_dir
from dalil.hmem import HMem
from dalil.pipeline import Pipeline
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


def create_app(
    pipeline: Pipeline,
    documents: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, str] | None = None,
    corpus_dir: Path | str | None = None,
    web_dir: Path | str | None = DEFAULT_WEB,
    zetaris_probe: Callable[[], bool] = default_zetaris_probe,
) -> FastAPI:
    """manifest maps doc_id -> file name inside corpus_dir; only those files are served."""
    store = pipeline.store
    manifest = dict(manifest or {})
    corpus = Path(corpus_dir).expanduser().resolve() if corpus_dir else None
    app = FastAPI(title="Dalil", docs_url=None, redoc_url=None, openapi_url=None)
    # Binding to loopback is not enough on its own: a page that rebinds its own
    # hostname to 127.0.0.1 could read the private page text. Only accept
    # requests addressed to this machine by name.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(ALLOWED_HOSTS))

    def doc_meta(did: str | None) -> Mapping[str, Any]:
        return documents.get(did or "", {})

    def detail(qid: str) -> dict[str, Any]:
        rec = store.get(qid)
        if rec is None:
            raise HTTPException(404, "no such question")
        retrieval = [
            {"passage_id": h["passage_id"], "doc_id": h["doc_id"], "title": doc_meta(h["doc_id"]).get("title"),
             "page": h["page"], "score": h["score"], "heading": h["heading"], "text": h["text"],
             "start": h["start"], "end": h["end"]}
            for h in rec["retrieval"]
        ]
        data = [{"table": d["table"], "sql": d["sql"], "rows": d["rows"], "note": d["note"]} for d in rec["data"]]
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
        return {
            "id": rec["id"], "question": rec["question"], "flags": rec["flags"], "status": rec["status"],
            "created_at": rec["created_at"], "retrieval": retrieval, "data": data, "check": check,
            "review": rec["review"], "audit": rec["audit"],
        }

    @app.post("/api/questions")
    def post_question(body: QuestionIn) -> dict[str, Any]:
        if not body.question.strip():
            raise HTTPException(422, "question is empty")
        rec = pipeline.answer(body.question)
        return {"id": rec.id, "status": rec.status}

    @app.get("/api/questions")
    def list_questions() -> list[dict[str, Any]]:
        return store.list()

    @app.get("/api/questions/{qid}")
    def get_question(qid: str) -> dict[str, Any]:
        return detail(qid)

    @app.post("/api/questions/{qid}/review")
    def review(qid: str, body: ReviewIn) -> dict[str, Any]:
        try:
            store.review(qid, body.decision, body.reviewer, body.note)
        except KeyError:
            raise HTTPException(404, "no such question") from None
        except ReviewError as e:
            raise HTTPException(409, str(e)) from None
        remember_review(qid, body.decision, body.reviewer, body.note)
        return detail(qid)

    def remember_review(qid: str, decision: str, reviewer: str, note: str) -> None:
        """Write the decision to the review memory, and log the outcome either way."""
        memory = pipeline.memory
        if memory is None:
            return
        rec = store.get(qid) or {}
        check = rec.get("check") or {}
        cited = sorted({c["passage_id"] for s in check.get("sentences", []) if s.get("kept")
                        for c in s.get("citations", []) if c.get("passage_id")})
        out = memory.remember(decision, qid, rec.get("question", ""), reviewer, cited,
                              answer=check.get("answer", ""), note=note)
        if out is None:
            store.event(qid, "memory_written", {"available": False})
        else:
            store.event(qid, "memory_written", {"available": True, "memory_id": out["memory"]["id"],
                                                "kind": out["memory"]["kind"], "passages": len(cited)})

    @app.get("/api/documents/{did}/pages/{page}")
    def get_page(did: str, page: int) -> dict[str, Any]:
        doc = documents.get(did)
        if doc is None or page not in doc["pages"]:
            raise HTTPException(404, "no such page")
        return {"doc_id": did, "title": doc.get("title"), "publisher": doc.get("publisher"),
                "source_url": doc.get("source_url"), "page": page, "text": doc["pages"][page]}

    @app.get("/api/documents/{did}/file")
    def get_file(did: str) -> FileResponse:
        name = manifest.get(did)
        if not name or corpus is None:
            raise HTTPException(404, "no such document")
        path = (corpus / name).resolve()
        if path.parent != corpus or not path.is_file():
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
        memory = pipeline.memory.health() if pipeline.memory is not None else None
        return {"ok": True, "zetaris": bool(zetaris_probe()), "passages": len(pipeline.passages),
                "memory": bool(memory), "memories": (memory or {}).get("memories", 0)}

    if web_dir is not None and Path(web_dir).is_dir():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    return app


def manifest_map(path: Path | str | None = None) -> dict[str, str]:
    rows = load_manifest(path) if path else load_manifest()
    return {manifest_doc_id(r["file"]): r["file"] for r in rows}


def create_default_app() -> FastAPI:
    """The app over build/ and the private corpus in $DALIL_CORPUS_DIR or ./corpus."""
    return create_app(Pipeline.from_build(memory=HMem()), load_documents(), manifest_map(), resolve_corpus_dir())
