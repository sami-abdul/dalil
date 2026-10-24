"""Build the ITU knowledge base with ITU's reference code.

    uv run --extra itu python scripts/build_itu_kb.py [--out ~/hackathons/itu-kb] [--no-archive]

Stores Dalil's passages, ITU's readiness framework, Dalil's use case and the
review decisions in ITU's KnowledgeBase class (ChromaDB), in its four
collections: kb_main, kb_framework, kb_usecase and kb_decisions. ITU's own
chunker is not used: it joins words with single spaces and drops page numbers,
and Dalil's citations need exact character spans on numbered pages.

Embeddings come from Dalil's local NVIDIA Nemotron-3-Embed-1B (DALIL_EMBED_URL,
default 127.0.0.1:8792), swapped in for ITU's Ollama call at run time. The
build is checked: every collection's count, every stored passage against its
page slice, a reopen from disk, and a test query. It is built in a temporary
directory and moved into place only if every check passes.

The knowledge base holds document text whose publishers forbid republishing,
so --out must be outside the repo, and the build packs it as a private archive
for the organisers. Passage embeddings come from Dalil's cache in the
git-ignored build/; nothing else is written inside the repo. Prints counts and
ids only, never document text.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ITU_REFERENCE = ROOT / "itu" / "reference"

DEFAULT_OUT = Path("~/hackathons/itu-kb").expanduser()
PASSAGES = ROOT / "build" / "passages.jsonl"
DOCUMENTS = ROOT / "build" / "documents.jsonl"
USE_CASE = ROOT / "docs" / "itu" / "use-case.md"
REVIEWS_DB = ROOT / "build" / "dalil.db"
COUNTRY = "United Arab Emirates"

# Manifest topics that are policy or law rather than a service procedure.
_POLICY = re.compile(r"^(AI policy|digital government|data protection|data law)|\blaw\b|\bdecree\b", re.I)
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


def review_decisions(db: Path) -> list[dict]:
    """Every review decision with its question, note and the passages the answer cited."""
    if not db.is_file():
        return []
    con = sqlite3.connect(db)
    try:
        rows = con.execute(
            'SELECT r.question_id, r.decision, r.note, r.at, q.question_redacted, rec."check" '
            "FROM reviews r JOIN questions q ON q.id = r.question_id "
            "LEFT JOIN records rec ON rec.question_id = r.question_id").fetchall()
    except sqlite3.OperationalError as e:
        print(f"warning: no review decisions read from {db}: {e}", file=sys.stderr)
        return []
    finally:
        con.close()
    out = []
    for qid, decision, note, at, question, check in rows:
        cited = sorted({c["passage_id"] for s in (json.loads(check or "{}").get("sentences") or [])
                        for c in s.get("citations") or [] if c.get("passage_id")})
        out.append({"question_id": qid, "decision": decision, "note": note or "", "at": at,
                    "question": question, "passage_ids": cited})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-archive", action="store_true")
    args = ap.parse_args()
    out = args.out.expanduser().resolve()
    if out == ROOT or ROOT in out.parents:
        print(f"error: --out must be outside the repo; {out} is inside {ROOT}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ITU_REFERENCE))

    from dalil.corpus import load_manifest, doc_id as manifest_doc_id
    from dalil.embed import HTTPEmbedder
    from dalil.retrieve import Retriever
    from server.knowledge import ingest
    from server.knowledge.kb import KnowledgeBase

    embedder = HTTPEmbedder()  # passages: cached in build/embeddings, as Dalil's search uses them
    other = HTTPEmbedder(cache_dir=None)  # framework, use case and decisions: never cached
    if not embedder.available():
        print(f"error: no embedding endpoint answers at {embedder.url}", file=sys.stderr)
        return 1
    if not hasattr(ingest, "embed_texts"):
        print("error: ITU's ingest module has no embed_texts to swap; check the vendored version", file=sys.stderr)
        return 1
    # ITU's ingest functions call Ollama for embeddings. Swap in Dalil's
    # embedder, so ITU's own code runs unchanged.
    swapped = {"calls": 0}

    def embed_texts(texts, **_):
        swapped["calls"] += 1
        return other.embed_passages(list(texts))

    ingest.embed_texts = embed_texts

    out.mkdir(parents=True, exist_ok=True)
    # A fresh directory per build: ChromaDB caches clients by path.
    for stale in out.glob("chromadb.building-*"):
        shutil.rmtree(stale)
    work = out / f"chromadb.building-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
    kb = KnowledgeBase(persist_dir=str(work))

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
    dims = json.loads((ITU_REFERENCE / "data" / "framework" / "dimensions.json").read_text(encoding="utf-8"))
    expected_framework = len(dims["dimensions"]) + len(dims["factors"])

    # kb_usecase: Dalil's use case, one document per section.
    sections = use_case_sections(USE_CASE.read_text(encoding="utf-8"))
    kb.add_documents(
        texts=[body for _, body in sections],
        embeddings=other.embed_passages([body for _, body in sections]),
        metadatas=[{"section": heading, "entity_type": "use_case_section", "country": COUNTRY,
                    "source_type": "use_case"} for heading, _ in sections],
        collection_name="kb_usecase", ids=[f"usecase-{i + 1}" for i in range(len(sections))],
    )

    # kb_decisions: every review decision so far, from Dalil's store.
    decisions = review_decisions(REVIEWS_DB)
    for d in decisions:
        text = f"{d['decision']}: {d['question']}" + (f"\nReviewer note: {d['note']}" if d["note"] else "")
        kb.add_decision(text, other.embed_passages([text])[0], {
            "question_id": d["question_id"], "decision": d["decision"], "timestamp": d["at"],
            "country": COUNTRY, "passage_ids": ",".join(d["passage_ids"]),
        })

    # Checks: first on the open client, then again after reopening from disk.
    stats = kb.stats()
    expected = {"kb_main": len(passages), "kb_framework": expected_framework,
                "kb_usecase": len(sections), "kb_decisions": len(decisions)}
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
    print(f"kb_framework: {framework} definitions; kb_usecase: {len(sections)} sections; "
          f"kb_decisions: {len(decisions)}; ITU embed_texts swapped in for {swapped['calls']} call(s)")
    print(f"test query top 3: {top['ids'][0]}")
    problems = []
    if mismatches or len(stored["ids"]) != len(passages):
        problems.append("the stored passages do not match the pages")
    if framework != expected_framework or not swapped["calls"]:
        problems.append("ITU's framework definitions did not all load through the swapped embedder")
    if len(sections) < 4:
        problems.append("the use case document has fewer sections than expected")
    if problems:
        print("error: " + "; ".join(problems) + f"; nothing replaced in {out}", file=sys.stderr)
        return 1

    # Close the writer, then reopen from disk, so the archive holds only what
    # a fresh client can read back.
    del kb, main_col, stored
    gc.collect()
    from chromadb.api.shared_system_client import SharedSystemClient
    SharedSystemClient.clear_system_cache()
    reopened = KnowledgeBase(persist_dir=str(work)).stats()
    SharedSystemClient.clear_system_cache()
    if reopened != stats or any(stats.get(k) != v for k, v in expected.items()):
        print(f"error: counts after reopening {reopened} differ from {expected}; nothing replaced", file=sys.stderr)
        return 1
    kb_dir = out / "chromadb"
    if kb_dir.exists():
        shutil.rmtree(kb_dir)
    work.rename(kb_dir)

    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    info = {
        "built_utc": built, "collections": stats, "embedding_model": embedder.model,
        "passages": len(passages), "page_slice_mismatches": mismatches,
        "itu_reference": "CrashingGuru/ITUAIReadiness@84010da, kb.py and ingest.py unchanged",
        "terms": "Built from public UAE government publications; most publishers' terms allow personal or "
                 "research use only. For the ITU AI for Good Lab evaluation; do not republish.",
    }
    tmp = out / "BUILD.json.tmp"
    tmp.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out / "BUILD.json")
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
