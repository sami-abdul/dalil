"""Download the documents in data/MANIFEST.csv into the corpus directory.

    uv run python scripts/fetch_corpus.py [--corpus-dir DIR] [--only FILE ...] [--delay SECONDS]

Each source_url is saved as corpus/<file>. A file already present whose
SHA-256 matches the manifest is skipped. Pages change, so a download whose
hash differs is kept and reported, not treated as a failure. Failed
downloads are reported too. The script exits 0 unless the manifest cannot
be read, so one dead link does not stop the rest.

The files are saved as bytes and never opened, rendered or run here.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dalil.corpus import DEFAULT_MANIFEST, CorpusError, load_manifest, resolve_corpus_dir, sha256_file

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
MAX_BYTES = 100 * 1024 * 1024
TIMEOUT = 60


def download(url: str, dest: Path) -> str:
    """Fetch url into dest atomically and return the SHA-256 of what was saved."""
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"not an http(s) URL: {url}")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
            "Accept-Language": "en,ar;q=0.8",
        },
    )
    tmp = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response, tmp.open("wb") as fh:
            while chunk := response.read(1 << 16):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError(f"larger than {MAX_BYTES} bytes")
                digest.update(chunk)
                fh.write(chunk)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus-dir", type=Path, default=None, help="default: $DALIL_CORPUS_DIR, else ./corpus")
    parser.add_argument("--only", action="append", default=[], metavar="FILE", help="fetch only this manifest file (repeatable)")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between requests (default 3)")
    parser.add_argument("--force", action="store_true", help="download even when the file is present and matches")
    args = parser.parse_args(argv)

    try:
        rows = load_manifest(args.manifest)
    except CorpusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {r["file"] for r in rows}
        for name in sorted(unknown):
            print(f"warning: {name} is not in the manifest", file=sys.stderr)
        rows = [r for r in rows if r["file"] in wanted]

    corpus_dir = resolve_corpus_dir(args.corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    skipped, fetched, mismatched, failed = [], [], [], []
    first = True
    for row in rows:
        dest = corpus_dir / row["file"]
        expected = row["sha256"].lower()
        if dest.is_file() and not args.force and sha256_file(dest) == expected:
            skipped.append(row["file"])
            print(f"ok       {row['file']} (present, hash matches)")
            continue
        if not first:
            time.sleep(args.delay)
        first = False
        try:
            actual = download(row["source_url"], dest)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            failed.append(row["file"])
            print(f"FAILED   {row['file']}: {exc}")
            continue
        fetched.append(row["file"])
        if actual == expected:
            print(f"fetched  {row['file']}")
        else:
            mismatched.append(row["file"])
            print(f"MISMATCH {row['file']}: sha256 {actual[:12]}..., manifest {expected[:12]}... (kept; the page may have changed)")

    print()
    print(f"present: {len(skipped)}, fetched: {len(fetched)}, hash mismatches: {len(mismatched)}, failed: {len(failed)}")
    if failed:
        print("failed: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
