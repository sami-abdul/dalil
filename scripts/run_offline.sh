#!/usr/bin/env bash
# Run all of Dalil on one machine, with no network: both models, then the app.
#
#   LLAMA_SERVER=/path/to/llama-server \
#   LLM_MODEL=/path/to/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf \
#   EMBED_MODEL=/path/to/nemotron-3-embed-1b-q4_k_m.gguf \
#   DALIL_CORPUS_DIR=/path/to/corpus scripts/run_offline.sh          # start; Ctrl-C stops all three
#
# Ports, on 127.0.0.1 only: DALIL_PORT (8787, the reviewer screen), DALIL_HMEM_PORT
# (8788, the review memory), DALIL_LLM_PORT (8790), DALIL_EMBED_PORT (8792).
# Logs go to build/run/. It needs build/passages.jsonl, made by the build steps in
# the README, and `uv sync` done once while online. Nothing here downloads.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${LLAMA_SERVER:?set LLAMA_SERVER to a llama.cpp llama-server binary}"
: "${LLM_MODEL:?set LLM_MODEL to the Nemotron 3 Nano 4B GGUF file}"
: "${EMBED_MODEL:?set EMBED_MODEL to the Nemotron-3-Embed-1B GGUF file}"
[ -f "$ROOT/build/passages.jsonl" ] || { echo "build/passages.jsonl is missing; run the build steps first" >&2; exit 1; }
export DALIL_LLM_PORT="${DALIL_LLM_PORT:-8790}" DALIL_EMBED_PORT="${DALIL_EMBED_PORT:-8792}"
export DALIL_LLM_URL="http://127.0.0.1:$DALIL_LLM_PORT/v1" DALIL_EMBED_URL="http://127.0.0.1:$DALIL_EMBED_PORT/v1"
export HF_HUB_OFFLINE=1 ANONYMIZED_TELEMETRY=False UV_OFFLINE=1
mkdir -p "$ROOT/build/run"
pids=()
trap '[ ${#pids[@]} -gt 0 ] && kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null' EXIT INT TERM

wait_for() { # name url
  for _ in $(seq 1 300); do curl -sf -m 2 "$2" >/dev/null && { echo "$1: up"; return; }; sleep 0.5; done
  echo "$1 did not start; see build/run/$1.log" >&2; exit 1
}

# A model server already answering on its port is used as it is, and left running.
up() { curl -sf -m 2 "$1" >/dev/null 2>&1; }
if up "http://127.0.0.1:$DALIL_EMBED_PORT/health"; then echo "embed: already up on $DALIL_EMBED_PORT"; else
  MODEL="$EMBED_MODEL" "$ROOT/scripts/serve_nemotron_embed.sh" >"$ROOT/build/run/embed.log" 2>&1 & pids+=($!)
fi
if up "http://127.0.0.1:$DALIL_LLM_PORT/health"; then echo "llm: already up on $DALIL_LLM_PORT"; else
  MODEL="$LLM_MODEL" "$ROOT/scripts/serve_nemotron.sh" >"$ROOT/build/run/llm.log" 2>&1 & pids+=($!)
fi
wait_for embed "http://127.0.0.1:$DALIL_EMBED_PORT/health"
wait_for llm "http://127.0.0.1:$DALIL_LLM_PORT/health"
cd "$ROOT"
uv run --offline python scripts/serve.py --host 127.0.0.1 --port "${DALIL_PORT:-8787}" \
    >"$ROOT/build/run/dalil.log" 2>&1 & pids+=($!)
wait_for dalil "http://127.0.0.1:${DALIL_PORT:-8787}/api/health"
echo "ready: http://127.0.0.1:${DALIL_PORT:-8787}   (Ctrl-C stops Dalil and both models)"
wait
