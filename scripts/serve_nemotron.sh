#!/usr/bin/env bash
# Run NVIDIA Nemotron 3 Nano 4B locally for Dalil's drafter, on 127.0.0.1 only.
#
#   LLAMA_SERVER=/path/to/llama-server MODEL=/path/to/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf scripts/serve_nemotron.sh
#
# Model: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF, file
# NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf, SHA-256
# be5d9a656a51922f24f1f09a759cebb694e1f5d9728bf0ef9f8c972c5a0b5ef2, under the
# NVIDIA Nemotron Open Model License. Server: llama.cpp's llama-server.
set -euo pipefail
: "${LLAMA_SERVER:?set LLAMA_SERVER to a llama.cpp llama-server binary}"
: "${MODEL:?set MODEL to the Nemotron GGUF file}"
PORT="${DALIL_LLM_PORT:-8790}"
THREADS="${DALIL_LLM_THREADS:-$(( $(nproc) * 4 / 5 ))}"
exec "$LLAMA_SERVER" -m "$MODEL" --host 127.0.0.1 --port "$PORT" -t "$THREADS" -c 16384 --jinja --alias nemotron
