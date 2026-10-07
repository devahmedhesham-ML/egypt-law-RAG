#!/usr/bin/env bash
# CI quality gate on the self-hosted GPU runner: RAGAS faithfulness of the production system (Qwen2.5 on vLLM)
# on the 20 CI questions; exits non-zero below the threshold in params.yaml, which fails the CI job.
#
# Uses the machine's app venv (RAG_VENV) with this checkout's code first on the path. Reuses a running vLLM
# server; otherwise starts scripts/serve_vllm.sh and stops it again at the end.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${RAG_VENV:-$HOME/venvs/egypt-law-rag}"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$PWD/src"            # this checkout's code, not the venv's editable install
export LANGFUSE_TRACING_ENABLED=false
# Gate runs accumulate in one MLflow store on the runner, so the trend is visible across sessions.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///$HOME/.egypt-law-rag/mlflow-gate.db}"
mkdir -p "$HOME/.egypt-law-rag"

VLLM_URL="${VLLM_BASE_URL:-http://localhost:8001/v1}"
started_vllm=""
if ! curl -sf "$VLLM_URL/models" >/dev/null; then
  echo "Starting vLLM for the gate…"
  scripts/serve_vllm.sh > /tmp/ci-gate-vllm.log 2>&1 &
  started_vllm=$!
  trap '[ -n "$started_vllm" ] && kill "$started_vllm" 2>/dev/null || true' EXIT
  for _ in $(seq 1 120); do
    curl -sf "$VLLM_URL/models" >/dev/null && break
    kill -0 "$started_vllm" 2>/dev/null || { echo "vLLM exited:"; tail -20 /tmp/ci-gate-vllm.log; exit 1; }
    sleep 5
  done
  curl -sf "$VLLM_URL/models" >/dev/null || { echo "vLLM did not start in 10 minutes"; exit 1; }
fi

python -m rag.ingest.ensure
python -m rag.evaluation.gate
