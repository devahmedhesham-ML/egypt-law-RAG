# syntax=docker/dockerfile:1.7
# Egypt Law RAG API: /ask and /health on port 8000, with the corpus, the Chroma index and the
# embedding model inside the image. Build: docker compose build (or docker build -t egypt-law-rag-api .)

# --- 1) data: the articles and index that dvc.lock pins, from the public S3 remote (no AWS account needed)
FROM python:3.12-slim AS data
RUN pip install --no-cache-dir "dvc[s3]==3.67.1"
WORKDIR /repo
COPY .dvc/config .dvc/config
COPY dvc.yaml dvc.lock params.yaml ./
RUN dvc config --local core.no_scm true \
 && dvc pull build_corpus index \
 && test -s data/processed/articles.json && test -f data/index/chroma/chroma.sqlite3

# --- 2) runtime: CPU-only dependencies and the embedding model baked in (runs offline)
FROM python:3.12-slim AS runtime
COPY requirements-api.lock /tmp/requirements-api.lock
# uv is mounted for the install only, and torch's own test suite is dropped: neither is needed at runtime.
RUN --mount=from=ghcr.io/astral-sh/uv:0.12,source=/uv,target=/usr/local/bin/uv \
    uv pip install --system --no-cache --torch-backend cpu -r /tmp/requirements-api.lock \
 && rm -rf /usr/local/lib/python3.12/site-packages/torch/test

ARG EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
ARG EMBED_REVISION=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
ENV HF_HOME=/opt/hf
# Pinned revision, recorded as "main" so the app (which loads by name, offline) finds exactly this one.
RUN python - <<'EOF'
import os, pathlib
from huggingface_hub import snapshot_download
model, rev = os.environ["EMBED_MODEL"], os.environ["EMBED_REVISION"]
snapshot = pathlib.Path(snapshot_download(model, revision=rev))
refs = snapshot.parents[1] / "refs"
refs.mkdir(exist_ok=True)
(refs / "main").write_text(rev)
EOF
# The hub cache writes some metadata as root-only; touch only those files (keeps this layer tiny).
RUN find "$HF_HOME" ! -perm -a+r -exec chmod a+rX {} +
RUN HF_HUB_OFFLINE=1 python -c "import os; from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer(os.environ['EMBED_MODEL'], device='cpu'); \
print('embedding dim', m.encode(['warm-up'], prompt_name='query').shape[-1])"

RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY pyproject.toml params.yaml ./
COPY src ./src
COPY --from=data --chown=app /repo/data/processed/articles.json data/processed/articles.json
COPY --from=data --chown=app /repo/data/index/chroma data/index/chroma

ARG GIT_SHA=unknown
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    LANGFUSE_RELEASE=${GIT_SHA} \
    LANGFUSE_TRACING_ENVIRONMENT=docker
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["python", "-m", "rag.api", "--host", "0.0.0.0", "--port", "8000"]
