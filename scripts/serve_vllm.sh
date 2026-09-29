#!/usr/bin/env bash
# Start the vLLM server for the self-hosted backend (run inside WSL).
# Usage: scripts/serve_vllm.sh [model]   (default: Qwen/Qwen2.5-7B-Instruct-AWQ)
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
VENV="${VLLM_VENV:-$HOME/venvs/egypt-law-rag-serve}"

# FlashInfer JIT-compiles its sampling kernel with nvcc, which needs a full CUDA toolkit.
# We decode greedily (temperature 0), so PyTorch's native sampler costs nothing.
export VLLM_USE_FLASHINFER_SAMPLER=0

# 0.75 leaves GPU memory for the embedding model; 8192 tokens fits ~10 retrieved articles + answer.
exec "$VENV/bin/vllm" serve "$MODEL" \
  --gpu-memory-utilization 0.75 \
  --max-model-len 8192 \
  --port 8001
