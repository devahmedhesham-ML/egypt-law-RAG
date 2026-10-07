# Egypt Law RAG

Arabic/English question answering over the Egyptian Civil Code, with answers cited by article number.
(ITI × MLOps MENA, Final Project 2: LLM / RAG.)

## Quick start: Q&A on any machine (Docker)

```bash
git clone https://github.com/devahmedhesham-ML/egypt-law-RAG.git && cd egypt-law-RAG
cp .env.example .env         # optional keys (Langfuse tracing, Bedrock); vLLM needs none
docker compose up --build    # vLLM (Qwen2.5-7B on the GPU) + the API on http://localhost:8000
```

```bash
curl localhost:8000/health
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{"question": "ما حكم هبة الأموال المستقبلة؟"}'
# {"answer":"هبة الأموال المستقبلة باطلة [Article 492].","sources":["Article 492"]}
curl -N -X POST localhost:8000/ask/stream -H 'Content-Type: application/json' -d '{"question": "What is a lease?"}'
```

The generative model is **Qwen/Qwen2.5-7B-Instruct-AWQ served by vLLM** (the `vllm` service, NVIDIA GPU with 16 GB;
the first start downloads the ~5.5 GB model). The API image needs only Docker to build: it pulls the corpus and the
Chroma index pinned in `dvc.lock` from the public S3 bucket (no AWS account, no DVC install) and bakes in the
embedding model, so it runs offline except for the LLM call. Variants: a vLLM server already running on the host
(`scripts/serve_vllm.sh`) → `VLLM_BASE_URL=http://host.docker.internal:8001/v1 docker compose up api`; no GPU →
`LLM_BACKEND=bedrock docker compose up api` with the optional Bedrock key in `.env`. Without an LLM, `/health` still
works and `/ask` answers 503 with the reason.

## API

`python -m rag.api` (app venv, port 8000) or the Docker image above ([src/rag/api/app.py](src/rag/api/app.py)):

| Endpoint | Request → response |
|---|---|
| `POST /ask` | `{"question": str}` → `{"answer": str, "sources": ["Article 492", ...]}`: the articles the answer cites that were in its retrieved context (top 5). Empty, blank or missing question → **422**. LLM or index unavailable → 503 with the reason. Langfuse trace id in the `X-Trace-Id` header. |
| `POST /ask/stream` | same request; the answer as plain text, token by token (`curl -N` shows it arriving), then `Sources: ...` |
| `GET /health` | `{"status": "healthy", "documents_indexed": 1149}`, or 503 `unhealthy` without an index |

Handlers are async: retrieval runs in a worker thread and the call to vLLM is awaited. Every response carries an
`X-Release` header (the build), which the canary rollout uses.

[reports/curl_checks.md](reports/curl_checks.md) holds a real run of every case; regenerate it with
`scripts/curl_checks.sh > reports/curl_checks.md` against a running API. The API, BentoML and the evaluation share
one path, [src/rag/pipeline.py](src/rag/pipeline.py): retrieve → answer → check citations. The image installs only
`requirements-api.lock` (CPU torch, 2.8 GB image); regenerate it after editing `requirements-api.txt` with the command
under Setup below.

## Setup (WSL Ubuntu + CUDA)

Three virtual environments, each built from a lock file, because each one pins a different torch version:

| venv | lock file | used for |
|---|---|---|
| `~/venvs/egypt-law-rag` | `requirements.lock` | app: ingestion, embeddings, Chroma, RAGAS, BentoML, MLflow, DVC |
| `~/venvs/egypt-law-rag-serve` | `requirements-serve.lock` | `vllm serve` (OpenAI-compatible API) |
| `~/venvs/egypt-law-rag-quantize` | `requirements-quantize.lock` | one-off AWQ-4bit quantization |

```bash
sudo apt install -y build-essential   # C compiler: vLLM/Triton compile GPU kernels at start-up
curl -LsSf https://astral.sh/uv/install.sh | sh
for n in "" -serve -quantize; do
  uv venv --python 3.12 ~/venvs/egypt-law-rag$n
  VIRTUAL_ENV=~/venvs/egypt-law-rag$n uv pip sync requirements$n.lock
done
cp .env.example .env   # fill in keys
source ~/venvs/egypt-law-rag/bin/activate
dvc pull               # fetch the source PDF (public bucket, no AWS account needed)
```

The `requirements*.txt` files state intent. After editing one, regenerate its lock file:

```bash
uv pip compile --python-version 3.12 --python-platform x86_64-manylinux_2_28 requirements.txt -o requirements.lock
# API image: CPU torch, versions constrained to the app lock
grep -vE "^(torch|nvidia-|triton|cuda-)" requirements.lock | grep -E "^[a-zA-Z0-9_.-]+==" > /tmp/api-constraints.txt
uv pip compile requirements-api.txt -c /tmp/api-constraints.txt --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 --torch-backend cpu --no-header -o requirements-api.lock
```

## Corpus and index

Two DVC stages turn the PDF into a searchable index ([dvc.yaml](dvc.yaml), settings in [params.yaml](params.yaml)):

```bash
dvc repro                   # both stages, non-interactive; or run them one at a time:
python -m rag.corpus.build  # PDF → data/processed/articles.json (+ corpus_report.json), ~3.5 s
python -m rag.ingest        # articles → GPU embeddings → data/index/chroma (+ index_report.json), ~30 s
```

**`rag.corpus.build`** reads each column of the bilingual table top to bottom and splits it on its own markers ("Article N", "مادة (n)"), joining the two sides by number; table rows serve as a cross-check. It produces 1,149 records (1,093 live, 56 repealed and flagged) with a numbered, bilingual hierarchy (part > book > chapter > section > topic > subtopic). Arabic is rebuilt from glyph positions: lam-alef ligatures, digit order (confirmed against the English), mirrored brackets and the font's private-use ligatures are repaired. Plan and design: [docs/plans/corpus-build.md](docs/plans/corpus-build.md).

Checks **warn, never stop the build**: Arabic/English numbers, articles cut off at page breaks, paragraph and list sequences, reversed numbers, length ratios, extraction artifacts, and reference articles checked word for word. The current 11 warnings are all defects in the source PDF (e.g. no Arabic "مادة ١٠٢٢"); they are listed in the console's Corpus view and in `corpus_report.json`.

**`rag.ingest`** shows those warnings and asks before continuing (`--yes` for scripts, `--stop-on-warning` to refuse). It embeds one chunk per article (heading path + Arabic + English), so the top 10 hits are always 10 different articles, with `Qwen/Qwen3-Embedding-0.6B`, then writes Chroma and runs smoke queries in both languages.

GPU use: the first replica embeds the longest batch to measure its real footprint, then more replicas start only if they fit in free VRAM minus a 2 GB margin **and** the GPU has idle compute. Measured on an RTX 4070 Ti SUPER (1,149 chunks, 298k tokens):

| Batch budget | Replicas | Embedding | Tokens/s | Peak VRAM |
|---|---|---|---|---|
| 16,384 tokens | 1 | 11.3 s | 26.4k | 7.7 GB |
| 16,384 tokens | 2 | 12.9 s | 23.1k | 11.6 GB |
| 4,096 tokens | 4 | 10.1 s | 29.5k | 11.9 GB |
| **4,096 tokens** | **1** | **8.8 s** | **33.9k** | **4.7 GB** |

One replica already keeps this GPU ~91% busy, so auto mode stays at one; `--replicas N` forces more (always capped at what fits).

## LLM inference

One OpenAI-compatible client ([src/rag/llm/](src/rag/llm/)) drives the backends, chosen with `llm.backend` in
[params.yaml](params.yaml) (or `LLM_BACKEND`). **The main model everywhere is Qwen2.5 on vLLM**; Bedrock is an
optional extra:

| backend | server | model |
|---|---|---|
| `vllm` (main) | `vllm serve` on :8001 (`scripts/serve_vllm.sh`, `VLLM_BASE_URL`), or the compose `vllm` service | **`Qwen/Qwen2.5-7B-Instruct-AWQ`** |
| `bedrock` (optional) | Amazon Bedrock's OpenAI-compatible endpoint (`OPENAI_BASE_URL`, key `Bedrock_API_key`) | `openai.gpt-oss-120b` |

The model answers only from the retrieved articles and cites them inline as `[Article 492]`. Every answer is checked: any cited article that was not retrieved is flagged as a hallucination.

```bash
# vLLM (WSL): starts the serve venv's server on :8001 (the API owns :8000) with the project's flags
scripts/serve_vllm.sh

# app venv
pytest                                        # unit tests, no GPU/AWS needed
python scripts/llm_smoke.py --backend vllm    # real call: 3 questions, streamed, citations checked
```

## Test console

A local web UI for manual and user testing (`src/rag/ui/`):

```bash
python -m rag.ui          # app venv, from the repo root -> http://localhost:7860
```

| View | What it does |
|---|---|
| Ask | Streamed answer from vLLM (main) or Bedrock (optional); context retrieved from the index per question (top-k, articles named by number first) or picked by hand; citations are clickable and checked against the context |
| Compare | Same question and context on both backends side by side, with a latency/tokens/citations summary |
| Status | Live health of every pipeline stage; planned stages are listed so gaps stay visible |
| Corpus | All 1,149 articles with their bilingual hierarchy and source pages, the build's warnings, and a "Random 20" eyeball check (a 19-article sample until the corpus is built) |
| Retrieval | Search the index directly: ranked articles with similarity scores, hierarchy and text |
| Traces | Every answer and search from this tab, with a link to its Langfuse trace (one tab = one Langfuse session) |
| Evaluation | Planned: what it will test, what it needs first, and a preview of its layout |
| Feedback log | Every tester rating (right/wrong, reason tags, comment) from `data/feedback/feedback.jsonl`, downloadable; ratings are also scored on the answer's trace |

When a stage lands, update its entry in [src/rag/ui/status.py](src/rag/ui/status.py) so testers see it.

## Evaluation set and chunking experiments (MLflow)

[eval/questions.jsonl](eval/questions.jsonl) holds 62 questions: 28 topics from all four books, each asked in Arabic
and in English, plus 2 that name an article by number and 4 out-of-scope questions. Each question lists the articles a
correct answer rests on and a short reference answer. It is the project's **accepted evaluation set** (not reviewed by
a legal expert); 20 of its questions (`"ci": true`) form the CI quality gate. How it was built and its limits:
[docs/evaluation-dataset.md](docs/evaluation-dataset.md).

```bash
python -m rag.experiments.chunking                 # retrieval metrics only, ~5 min for 8 configs (GPU)
python -m rag.experiments.chunking --faithfulness  # + Qwen2.5 answers judged by RAGAS (judge: Qwen2.5)
python -m rag.experiments.chunking --faithfulness-only  # re-score stored rankings, no re-embedding
python -m rag.experiments.chunking --report-only   # rebuild reports/chunking_experiments.md from MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db  # experiment "chunking" → select runs → Compare
```

Each entry under `experiments.chunking` in [params.yaml](params.yaml) is one MLflow run: a chunking strategy
(`article`, `window` with `chunk_size`/`overlap` in tokens, or `per_language`) and an embedding model. The run builds a
throwaway index under `data/experiments/`, retrieves for every question exactly as production does (articles named in
the question first, then the best chunk per article), and logs:

- params: `strategy`, `chunk_size`, `overlap`, `embedding_model`, `top_k` (tags: `answer_model`, `judge_model`)
- metrics: `hit_at_1`, `hit_at_5`, `recall_at_5`, `mrr`, `ndcg_at_5` (overall, `_ar`, `_en`), `ar_en_top1_agreement`,
  the top-1 similarity for in-scope vs out-of-scope questions, chunk counts and timings, and with `--faithfulness`,
  RAGAS `faithfulness` (Qwen2.5 answers from the top 5 articles; Qwen2.5 judges by default, `--judge-backend bedrock`
  is optional)
- artifact: `per_question.json` with every question's ranking, answer and score

Chunking changes only which articles are retrieved: the model always receives whole articles. Results:
[reports/chunking_experiments.md](reports/chunking_experiments.md).

Results (8 runs, 58 in-scope questions; one question = 0.017 of hit@1, so treat small gaps as ties; faithfulness:
Qwen2.5 answers, Qwen2.5 judge):

| Run | hit@1 | recall@5 | MRR | AR/EN same top-1 | faithfulness |
|---|---|---|---|---|---|
| **article, Qwen3-Embedding-0.6B (production)** | 0.741 | 0.931 | 0.819 | 0.607 | 0.813 |
| article, bge-m3 | 0.759 | 0.931 | 0.835 | 0.714 | 0.806 |
| window 512/64, Qwen3 | 0.724 | 0.931 | 0.808 | 0.571 | 0.813 |
| window 256/32, Qwen3 | 0.724 | 0.897 | 0.805 | 0.571 | 0.823 |
| per_language, Qwen3 | 0.655 | 0.897 | 0.767 | 0.357 | 0.813 |
| window 128/16, Qwen3 | 0.466 | 0.690 | 0.567 | 0.393 | 0.741 |

- **Whole articles are the right unit.** Every split retrieves worse; 128-token windows lose the context that makes an
  article findable (hit@1 0.47). Splitting Arabic from English loses the bilingual chunk's help across languages: the
  two versions of a question agree on the top article far less often (0.61 → 0.36).
- **bge-m3 ties Qwen3 on whole articles** (one question apart) and agrees more across languages, but separates
  out-of-scope questions worse: their best match scores 0.49 with bge-m3 against 0.42 with Qwen3 (in-scope: ~0.65
  for both), which matters for a future "no relevant article" threshold.
- **Faithfulness is 0.74–0.84** with Qwen2.5 answering (0.87–0.92 when Bedrock answered and judged): mostly within
  noise, lowest for the config that retrieves worst. The 7B judge is noisy per question
  ([reports/judge_comparison.md](reports/judge_comparison.md)); read its averages, not single scores.
- **Cost is small for every run.** Production's embeddings are 4.5 MB of vectors (a 22.5 MB Chroma index), built in
  about 9 s on the RTX 4070 Ti SUPER with the GPU ~80% busy and one 2.1 GB model copy. Per question, retrieval takes
  ~15 ms to embed on the GPU (~56 ms on the CPU, as in Docker) plus ~10 ms of vector search. bge-m3 embeds about
  twice as fast.
- Production is **article + Qwen3-Embedding-0.6B**, registered in the MLflow Model Registry as
  `civil-code-retrieval` with the alias **`production`** (`python -m rag.experiments.register`: best recall@5, then
  within one question of the best hit@1, then the largest in-/out-of-scope gap). It loads as
  `mlflow.pyfunc.load_model("models:/civil-code-retrieval@production")` and returns the top-5 article numbers.
  Screenshots: [compare view](reports/mlflow_chunking_compare.png), [registry](reports/mlflow_registry_production.png).

## Serving: BentoML, streaming, load test, canary

[src/rag/serving/service.py](src/rag/serving/service.py) wraps the same pipeline in BentoML; vLLM serves the model:

```bash
scripts/serve_vllm.sh                                        # vLLM: Qwen/Qwen2.5-7B-Instruct-AWQ on :8001
bentoml serve rag.serving.service:RagService --port 3000     # async /ask and /ask_stream
curl -N -X POST localhost:3000/ask_stream -H 'Content-Type: application/json' -d '{"question": "ما حكم هبة الأموال المستقبلة؟"}'
```

**Load test** ([loadtest/locustfile.py](loadtest/locustfile.py), 50 concurrent users, 3 minutes, questions from the
evaluation set, 80% `/ask` and 20% `/ask_stream`): **p95 12 s for a full answer, 8.7 s to the first streamed text, 0
failures in 899 requests, 5 requests/s** on one RTX 4070 Ti SUPER; one question alone takes 0.9 s. The limit is vLLM's
generation throughput (~460 tokens/s shared by all answers), not the API. Details, before/after and what would lower
it: [reports/locust_summary.md](reports/locust_summary.md); full report: [reports/locust_report.html](reports/locust_report.html).

### Canary rollout

[deploy/canary/](deploy/canary/) runs two versions of the API behind nginx and splits traffic by weight; every
response carries `X-Release`, and nginx logs which version served each request.

```bash
STABLE_IMAGE=ghcr.io/devahmedhesham-ml/egypt-law-rag-api:<current sha> \
CANARY_IMAGE=ghcr.io/devahmedhesham-ml/egypt-law-rag-api:<new sha> \
  docker compose -f deploy/canary/docker-compose.yml up -d    # entry point: http://localhost:8080
deploy/canary/set_weights.sh 90 10    # 1. 10% of traffic on the canary (reloads nginx, no downtime)
deploy/canary/set_weights.sh 50 50    # 2. widen while it stays healthy
deploy/canary/set_weights.sh 0 100    # 3. promote; or `100 0` to roll back at any step
```

Promote only while the canary matches stable on: 5xx rate (nginx log, `release=canary`), p95 latency (rerun Locust
against :8080), and answer quality, meaning the faithfulness gate run against the canary image and its Langfuse
scores (`citations_outside_context`, tester ratings) filtered by `environment=canary`. A canary that stops answering
leaves rotation automatically (`max_fails`).

## CI/CD (GitHub Actions)

[.github/workflows/ci.yml](.github/workflows/ci.yml) runs on every pull request and on `main`:

1. **lint**: `ruff check .` (rules in `pyproject.toml`)
2. **test**: `pytest` on CPU (`requirements-ci.lock`)
3. **index**: `python -m rag.ingest.ensure`: use the local index, else `dvc pull` it from the public remote, else
   build it (corpus first if missing), then a retrieval smoke test
4. **docker**: build the API image and push it to GHCR (`ghcr.io/devahmedhesham-ml/egypt-law-rag-api`), tagged with the
   commit, the PR and `latest` on `main`
5. **quality-gate**: `scripts/ci_gate.sh` → `python -m rag.evaluation.gate`: RAGAS faithfulness of the production system
   (Qwen2.5 on vLLM) on the 20 CI questions; **fails below 0.75**. It needs the GPU, so it runs on a self-hosted runner
   labelled `gpu` (`scripts/setup_gpu_runner.sh`, then `~/actions-runner/run-gpu.sh`), only when the repository variable
   `GPU_RUNNER` is `true`, and never for pull requests from forks. Last result: [reports/faithfulness_gate.md](reports/faithfulness_gate.md).

## Index maintenance

```bash
python -m rag.ingest.ensure                                  # make sure a usable index exists (local → pull → build)
python -m rag.ingest.batch data/samples/batch_test_document.json   # add/replace articles without a rebuild
```

`rag.ingest.batch` embeds only new or changed articles (unchanged ones are skipped, so re-runs are free) and merges
them into the corpus file the API cites from. Tested with a new document in
[reports/batch_reindex.md](reports/batch_reindex.md): a hypothetical Article 1150 on electronic signatures goes from
absent to the top result (0.795) for a matching question after a 0.8 s update.

## Tracing (Langfuse)

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL` in `.env` (see `.env.example`); without them, or
with `LANGFUSE_TRACING_ENABLED=false` (set by the tests), tracing is a no-op. Configuration lives in
[src/rag/tracing.py](src/rag/tracing.py): environment `development` unless `LANGFUSE_TRACING_ENVIRONMENT` is set, and
the git commit as the release.

| Trace | Steps (observation type) | Scores |
|---|---|---|
| `answer-question` (console Ask/Compare, `scripts/llm_smoke.py`) | `retrieve-articles` (retriever) → `embed-question` (embedding), `generate-answer` (generation: prompt, model, tokens, time to first token, reasoning), `check-citations` (evaluator) | `citations_outside_context`, `answer_has_citations`, `tester_rating`, `tester_issue` |
| `search-articles` (Retrieval view) | `retrieve-articles` → `embed-question`; `load-embedding-model` on the first search | |
| `index-corpus` (`python -m rag.ingest`) | `check-corpus`, `embed-chunks` (embedding, token usage), `write-index`, `run-smoke-queries` (evaluator) | `corpus_warnings`, `smoke_top1_rate` |
| `build-corpus` (`python -m rag.corpus.build`) | `extract-pages`, `validate-records` | `build_warnings` |

Console traces carry the tab's session id and tags (`ask`/`compare`, backend, context mode); the two traces of one
Compare run share a `group_id` in their metadata. Errors (e.g. vLLM down, an expired key) are ERROR-level; a Stop press is a
WARNING with the partial answer kept. Every answer card links to its trace.

## Data

`data/raw/egyptian_civil_code.pdf` is tracked with DVC: git stores only the `.dvc` pointer file, and the PDF itself lives in S3 (`s3://amzn-egypt-law-rag/dvc`, eu-north-1). The pipeline outputs (`articles.json`, the Chroma index) are DVC-tracked the same way through `dvc.lock`; their reports are small and committed to git.

The `dvc/` prefix of the bucket is publicly readable, so anyone can `dvc pull` without an AWS account. Only the owner can write (`dvc push`), using `aws login`: install AWS CLI v2 inside WSL and share one login with Windows:

```bash
ln -sfn /mnt/c/Users/<you>/.aws ~/.aws   # reuse the Windows ~/.aws
aws login
dvc push
```
