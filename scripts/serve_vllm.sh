#!/usr/bin/env bash
# Start the vLLM server for the self-hosted backend (run inside WSL).
# Usage: scripts/serve_vllm.sh [model]   (default: Qwen/Qwen2.5-7B-Instruct-AWQ)
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
VENV="${VLLM_VENV:-$HOME/venvs/egypt-law-rag-serve}"

# FlashInfer JIT-compiles its sampling kernel with nvcc, which needs a full CUDA toolkit.
# We decode greedily (temperature 0), so PyTorch's native sampler costs nothing.
export VLLM_USE_FLASHINFER_SAMPLER=0

# 0.60 of the 16 GB leaves room for the question-embedding model on the GPU next to the Windows desktop
# (~2.3 GB): at 0.75 it fell back to the CPU and became the bottleneck under load (reports/locust_summary.md),
# while vLLM's KV cache stayed under 6% used at 50 concurrent users. 8192 tokens fits ~10 articles + answer.
exec "$VENV/bin/vllm" serve "$MODEL" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.60}" \
  --max-model-len 8192 \
  --port 8001
