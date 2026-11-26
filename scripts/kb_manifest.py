"""Write or check the knowledge base manifest, data/KNOWLEDGE-BASE.json.

    uv run python scripts/kb_manifest.py            # write it from build/
    uv run python scripts/kb_manifest.py --check    # compare a rebuild with it

The documents themselves may not be republished, so the repository carries
this manifest instead of the knowledge base. For each document it records
where the file came from, its SHA-256, its pages, how many passages Dalil cut
from it, and a SHA-256 over those passages' records. Someone who fetches the
same files and runs the build steps in the README gets the same passages, and
--check proves it, document by document, without any text leaving their machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dalil.chunk import PASSAGE_FIELDS, read_jsonl  # noqa: E402

OUT = ROOT / "data" / "KNOWLEDGE-BASE.json"


def passage_hash(records: list[dict]) -> str:
    """SHA-256 over a document's passage records, one canonical JSON line each."""
    h = hashlib.sha256()
    for r in records:
        h.update(json.dumps({k: r[k] for k in PASSAGE_FIELDS}, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build(documents: Path, passages: Path) -> dict:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for p in read_jsonl(passages):
        by_doc[p["doc_id"]].append(p)
    docs = []
    for d in read_jsonl(documents):
        recs = by_doc.get(d["doc_id"], [])
        docs.append({
            "doc_id": d["doc_id"], "file": d["file"], "title": " ".join(d["title"].split()),
            "publisher": d["publisher"], "source_url": d["source_url"], "retrieved_utc": d["retrieved_utc"],
            "language": d["language"], "kind": d["kind"], "used_for_answers": not d["is_terms_page"],
            "file_sha256": d["sha256"], "pages": len(d["pages"]), "passages": len(recs),
            "passages_sha256": passage_hash(recs),
        })
    answerable = [d for d in docs if d["used_for_answers"]]
    return {
        "about": ("Dalil's knowledge base, described without its text. The documents' publishers do not "
                  "allow republishing; fetch them with scripts/fetch_corpus.py and rebuild with the steps "
                  "in the README, then run scripts/kb_manifest.py --check."),
        "documents": len(docs), "documents_used_for_answers": len(answerable),
        "passages": sum(d["passages"] for d in docs),
        "languages": sorted({d["language"] for d in docs}),
        "items": docs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare build/ with the manifest instead of writing it")
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    args = ap.parse_args()
    now = build(args.build / "documents.jsonl", args.build / "passages.jsonl")
    if not args.check:
        OUT.write_text(json.dumps(now, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}: {now['documents']} documents, {now['passages']} passages")
        return 0
    want = json.loads(OUT.read_text(encoding="utf-8"))
    have = {d["doc_id"]: d for d in now["items"]}
    bad = 0
    for d in want["items"]:
        got = have.get(d["doc_id"])
        problems = [] if got else ["missing"]
        for key in ("file_sha256", "pages", "passages", "passages_sha256"):
            if got and got[key] != d[key]:
                problems.append(f"{key} differs")
        if problems:
            bad += 1
            print(f"{d['doc_id']}: {', '.join(problems)}")
    print(f"{len(want['items']) - bad}/{len(want['items'])} documents match; "
          f"{now['passages']} passages built, {want['passages']} in the manifest")
    return 1 if bad or now["passages"] != want["passages"] else 0


if __name__ == "__main__":
    sys.exit(main())
