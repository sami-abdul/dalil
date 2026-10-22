"""Run Dalil's local app.

    uv run python scripts/serve.py [--host 127.0.0.1] [--port 8787] [--no-memory]

Binds to loopback only: the API serves private page text and documents, so
any other host is refused.

Also starts the H-MEM bridge (hmem/bridge.ts, needs Node 22.7 or later) unless
one already answers or --no-memory is given, and stops it on exit.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from dalil.app import DEFAULT_HOST, DEFAULT_PORT, create_default_app  # noqa: E402
from dalil.hmem import HMem  # noqa: E402

LOOPBACK = {"127.0.0.1", "localhost", "::1"}
MIN_NODE = (22, 7)


def node_version(node: str) -> tuple[int, int]:
    out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
    major, minor = (out.lstrip("v").split(".") + ["0"])[:2]
    return int(major), int(minor)


def start_memory() -> subprocess.Popen | None:
    """Start the H-MEM bridge if none is running, and wait until it answers.

    The bridge's port comes from DALIL_HMEM_PORT, and DALIL_HMEM_URL is set to
    match, so the app always calls the bridge this script started.
    """
    port = os.environ.setdefault("DALIL_HMEM_PORT", "8788")
    os.environ["DALIL_HMEM_URL"] = f"http://127.0.0.1:{port}"
    if HMem().health():
        print(f"h-mem bridge already running on port {port}")
        return None
    node = shutil.which("node")
    if node is None:
        print("node not found: running without review memory", file=sys.stderr)
        return None
    if node_version(node) < MIN_NODE:
        print(f"node {'.'.join(map(str, MIN_NODE))} or later is needed for the H-MEM bridge: "
              "running without review memory", file=sys.stderr)
        return None
    proc = subprocess.Popen([node, "--experimental-transform-types", "--no-warnings",
                             str(ROOT / "hmem" / "bridge.ts")], cwd=ROOT)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print(f"h-mem bridge exited with code {proc.returncode}: running without review memory",
                  file=sys.stderr)
            return None
        if HMem().health():
            return proc
        time.sleep(0.25)
    print("h-mem bridge did not answer within 15 seconds: running without review memory", file=sys.stderr)
    stop(proc)
    return None


def stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-memory", action="store_true", help="do not start the H-MEM bridge")
    args = ap.parse_args()
    if args.host not in LOOPBACK:
        print(f"refusing to bind {args.host}: Dalil serves private text, use 127.0.0.1", file=sys.stderr)
        return 2
    bridge = None if args.no_memory else start_memory()
    # Dense retrieval: use the embedding server if one answers. The retriever
    # reads DALIL_EMBED_URL; without it, search is BM25 alone.
    from dalil.embed import HTTPEmbedder
    embedder = HTTPEmbedder()
    if embedder.available():
        os.environ["DALIL_EMBED_URL"] = embedder.url
        print(f"dense retrieval on, embeddings from {embedder.url}")
    else:
        os.environ.pop("DALIL_EMBED_URL", None)
        print(f"no embedding server answers at {embedder.url}: search is BM25 alone")
    try:
        uvicorn.run(create_default_app(), host=args.host, port=args.port, log_level="info")
    finally:
        if bridge is not None:
            stop(bridge)
    return 0


if __name__ == "__main__":
    sys.exit(main())
