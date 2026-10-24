#!/usr/bin/env bash
# Run NVIDIA Nemotron-3-Embed-1B locally for Dalil's dense retrieval, on 127.0.0.1 only.
#
#   LLAMA_SERVER=/path/to/llama-server MODEL=/path/to/nemotron-3-embed-1b-q4_k_m.gguf scripts/serve_nemotron_embed.sh
#
# Model: a Q4_K_M GGUF of nvidia/Nemotron-3-Embed-1B-BF16, from
# https://huggingface.co/zenmagnets/Nemotron-3-Embed-1B-Q4_K_M-GGUF, file
# nemotron-3-embed-1b-q4_k_m.gguf, SHA-256
# 9a74166f51dbc280073748fa199bea49283bd21f7f9280f2dec2b4d975ddfd1d, under the
# OpenMDW 1.1 licence. The quantisation is a community build, not NVIDIA's.
set -euo pipefail
: "${LLAMA_SERVER:?set LLAMA_SERVER to a llama.cpp llama-server binary}"
: "${MODEL:?set MODEL to the embedding GGUF file}"
PORT="${DALIL_EMBED_PORT:-8792}"
THREADS="${DALIL_EMBED_THREADS:-4}"  # queries are short; the corpus is embedded once and cached
exec "$LLAMA_SERVER" -m "$MODEL" --host 127.0.0.1 --port "$PORT" -t "$THREADS" -c 8192 -b 8192 -ub 8192 \
    --embeddings --pooling mean --alias nemotron-embed
