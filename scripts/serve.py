"""Run Dalil's local app.

    uv run python scripts/serve.py [--host 127.0.0.1] [--port 8787]

Binds to loopback only: the API serves private page text and documents, so
any other host is refused.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from dalil.app import DEFAULT_HOST, DEFAULT_PORT, create_default_app  # noqa: E402

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    if args.host not in LOOPBACK:
        print(f"refusing to bind {args.host}: Dalil serves private text, use 127.0.0.1", file=sys.stderr)
        return 2
    uvicorn.run(create_default_app(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
