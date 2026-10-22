"""Run Dalil's local app.

    uv run python scripts/serve.py [--host 127.0.0.1] [--port 8787] [--no-memory]

Binds to loopback only: the API serves private page text and documents, so
any other host is refused.

Also starts the H-MEM bridge (hmem/bridge.ts, needs Node 22.7 or later) unless
one already answers or --no-memory is given, and stops it on exit.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from dalil.app import DEFAULT_HOST, DEFAULT_PORT, create_default_app  # noqa: E402
from dalil.hmem import HMem  # noqa: E402


def start_memory() -> subprocess.Popen | None:
    """Start the H-MEM bridge if none is running. Returns the process we started."""
    if HMem().health():
        print("h-mem bridge already running")
        return None
    node = shutil.which("node")
    if node is None:
        print("node not found: running without review memory", file=sys.stderr)
        return None
    return subprocess.Popen([node, "--experimental-transform-types", "--no-warnings",
                             str(ROOT / "hmem" / "bridge.ts")], cwd=ROOT)

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


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
    try:
        uvicorn.run(create_default_app(), host=args.host, port=args.port, log_level="info")
    finally:
        if bridge is not None:
            bridge.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
