"""Build the ITU knowledge base with ITU's reference code.

    uv run --extra itu python scripts/build_itu_kb.py [--out ~/hackathons/itu-kb] [--no-archive]

Stores Dalil's passages, ITU's readiness framework, Dalil's use case and the
review decisions in ITU's KnowledgeBase class (ChromaDB), in its four
collections: kb_main, kb_framework, kb_usecase and kb_decisions. ITU's own
chunker is not used: it joins words with single spaces and drops page numbers,
and Dalil's citations need exact character spans on numbered pages.

Embeddings come from Dalil's local NVIDIA Nemotron-3-Embed-1B (DALIL_EMBED_URL,
default 127.0.0.1:8791), swapped in for ITU's Ollama call at run time. The
build is checked: counts, every stored passage equals its page slice, and a
test query. The knowledge base holds document text whose publishers forbid
republishing, so it is written outside the repo and packed as a private
archive for the organisers. Prints counts and ids only, never document text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "itu" / "reference"))

DEFAULT_OUT = Path("~/hackathons/itu-kb").expanduser()
PASSAGES = ROOT / "build" / "passages.jsonl"
DOCUMENTS = ROOT / "build" / "documents.jsonl"
USE_CASE = ROOT / "docs" / "itu" / "use-case.md"
REVIEWS_DB = ROOT / "build" / "dalil.db"
COUNTRY = "United Arab Emirates"

# Manifest topics that are policy or law rather than a service procedure.
_POLICY = re.compile(r"^(AI policy|digital government|data protection|data law|labour: Labour Relations law)", re.I)
_STRATEGY = re.compile(r"strategy", re.I)


def categorise(topic: str) -> tuple[str, str]:
    """(source_type, document_category) in ITU's vocabulary for a manifest topic."""
    if _STRATEGY.search(topic) and _POLICY.search(topic):
        return "national_strategy", "AI_Strategies"
    if _POLICY.search(topic):
        return "policy_document", "AI_Strategies"
    return "government_service_page", "Government_Services"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def use_case_sections(text: str) -> list[tuple[str, str]]:
    """(heading, body) for each section of the use-case document."""
    parts = re.split(r"(?m)^## ", text)
    return [(p.split("\n", 1)[0].strip(), p.strip()) for p in parts[1:] if p.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-archive", action="store_true")
    args = ap.parse_args()

    from dalil.corpus import load_manifest, doc_id as manifest_doc_id
    from dalil.embed import HTTPEmbedder
    from dalil.retrieve import Retriever
    from server.knowledge import ingest
    from server.knowledge.kb import KnowledgeBase

    embedder = HTTPEmbedder()
    if not embedder.available():
        print(f"error: no embedding endpoint answers at {embedder.url}", file=sys.stderr)
        return 1
    # ITU's ingest functions call Ollama for embeddings. Swap in Dalil's
    # embedder, so ITU's own code runs unchanged.
    ingest.embed_texts = lambda texts, **_: embedder.embed_passages(list(texts))

    out = args.out
    kb_dir = out / "chromadb"
    if kb_dir.exists():
        shutil.rmtree(kb_dir)  # a fresh build every time; this directory is build output
    kb_dir.mkdir(parents=True)
    kb = KnowledgeBase(persist_dir=str(kb_dir))

    manifest = {manifest_doc_id(r["file"]): r for r in load_manifest()}
    passages = read_jsonl(PASSAGES)
    pages = {(d["doc_id"], p["page"]): p["text"] for d in read_jsonl(DOCUMENTS) for p in d["pages"]}

    # kb_main: Dalil's passages, text exactly as stored, embedded as Dalil embeds them.
    texts = [p["text"] for p in passages]
    vectors = embedder.embed_passages([Retriever.index_text(p) for p in passages])
    metadatas = []
    for p in passages:
        row = manifest.get(p["doc_id"], {})
        source_type, category = categorise(row.get("topic", ""))
        metadatas.append({
            "passage_id": p["passage_id"], "doc_id": p["doc_id"], "page": int(p["page"]),
            "start": int(p["start"]), "end": int(p["end"]), "heading": p.get("heading") or "",
            "title": row.get("title", ""), "publisher": row.get("publisher", ""),
            "source_url": row.get("source_url", ""), "language": p.get("language", ""),
            "kind": p.get("kind", ""), "country": COUNTRY, "source_type": source_type,
            "document_category": category, "filename": row.get("file", ""),
        })
    kb.add_documents(texts=texts, embeddings=vectors, metadatas=metadatas,
                     collection_name="kb_main", ids=[p["passage_id"] for p in passages])

    # kb_framework: ITU's 13 dimensions and 6 factors, through ITU's own function.
    framework = ingest.ingest_framework_definitions(kb)

    # kb_usecase: Dalil's use case, one document per section.
    sections = use_case_sections(USE_CASE.read_text(encoding="utf-8"))
    kb.add_documents(
        texts=[body for _, body in sections],
        embeddings=embedder.embed_passages([body for _, body in sections]),
        metadatas=[{"section": heading, "entity_type": "use_case_section", "country": COUNTRY,
                    "source_type": "use_case"} for heading, _ in sections],
        collection_name="kb_usecase", ids=[f"usecase-{i + 1}" for i in range(len(sections))],
    )

    # kb_decisions: every review decision so far, from Dalil's store.
    decisions = 0
    if REVIEWS_DB.is_file():
        con = sqlite3.connect(REVIEWS_DB)
        rows = con.execute(
            "SELECT r.question_id, r.decision, r.note, r.at, q.question_redacted "
            "FROM reviews r JOIN questions q ON q.id = r.question_id").fetchall()
        for qid, decision, note, at, question in rows:
            text = f"{decision}: {question}" + (f"\nReviewer note: {note}" if note else "")
            kb.add_decision(text, embedder.embed_passages([text])[0],
                            {"question_id": qid, "decision": decision, "timestamp": at, "country": COUNTRY})
            decisions += 1

    # Checks.
    stats = kb.stats()
    main_col = kb.main
    stored = main_col.get(ids=[p["passage_id"] for p in passages], include=["documents", "metadatas"])
    mismatches = sum(
        1 for doc, meta in zip(stored["documents"], stored["metadatas"])
        if pages.get((meta["doc_id"], meta["page"]), "")[meta["start"]:meta["end"]] != doc
    )
    probe = "How is end-of-service gratuity calculated for a full-time worker?"
    top = kb.search(embedder.embed_query(probe), n_results=3)
    print(f"collections: {json.dumps(stats)}")
    print(f"kb_main: {len(passages)} passages, {len(stored['ids'])} stored, page-slice mismatches: {mismatches}")
    print(f"kb_framework: {framework} definitions; kb_usecase: {len(sections)} sections; kb_decisions: {decisions}")
    print(f"test query top 3: {top['ids'][0]}")
    if mismatches or len(stored["ids"]) != len(passages):
        print("error: the stored passages do not match the pages", file=sys.stderr)
        return 1

    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    info = {
        "built_utc": built, "collections": stats, "embedding_model": embedder.model,
        "passages": len(passages), "page_slice_mismatches": mismatches,
        "itu_reference": "CrashingGuru/ITUAIReadiness@84010da, kb.py and ingest.py unchanged",
        "terms": "Built from public UAE government publications; most publishers' terms allow personal or "
                 "research use only. For the ITU AI for Good Lab evaluation; do not republish.",
    }
    (out / "BUILD.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    if not args.no_archive:
        archive = out / f"dalil-itu-kb-{built[:10]}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(kb_dir, arcname="chromadb")
            tar.add(out / "BUILD.json", arcname="BUILD.json")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        print(f"archive: {archive} ({archive.stat().st_size // 1024} KB) sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
