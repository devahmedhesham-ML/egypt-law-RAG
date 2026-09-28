# Egypt Law RAG

Arabic/English question answering over the Egyptian Civil Code, with answers cited by article number.
(ITI × MLOps MENA, Final Project 2: LLM / RAG.)

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

One OpenAI-compatible client ([src/rag/llm/](src/rag/llm/)) drives two backends, chosen with `llm.backend` in [params.yaml](params.yaml):

| backend | server | model |
|---|---|---|
| `bedrock` | Amazon Bedrock's OpenAI-compatible endpoint (`OPENAI_BASE_URL`, key `Bedrock_API_key`) | `openai.gpt-oss-120b` |
| `vllm` | local `vllm serve` (`VLLM_BASE_URL`) | `Qwen/Qwen2.5-7B-Instruct-AWQ` |

The model answers only from the retrieved articles and cites them inline as `[Article 492]`. Every answer is checked: any cited article that was not retrieved is flagged as a hallucination.

```bash
# vLLM (WSL): starts the serve venv's server on :8000 with the project's flags
scripts/serve_vllm.sh

# app venv
pytest                                        # unit tests, no GPU/AWS needed
python scripts/llm_smoke.py --backend bedrock # real call: 3 questions, streamed, citations checked
python scripts/llm_smoke.py --backend vllm
```

## Test console

A local web UI for manual and user testing (`src/rag/ui/`):

```bash
python -m rag.ui          # app venv, from the repo root -> http://localhost:7860
```

| View | What it does |
|---|---|
| Ask | Streamed answer from Bedrock or vLLM; context retrieved from the index per question (top-k, articles named by number first) or picked by hand; citations are clickable and checked against the context |
| Compare | Same question and context on both backends side by side, with a latency/tokens/citations summary |
| Status | Live health of every pipeline stage; planned stages are listed so gaps stay visible |
| Corpus | All 1,149 articles with their bilingual hierarchy and source pages, the build's warnings, and a "Random 20" eyeball check (a 19-article sample until the corpus is built) |
| Retrieval | Search the index directly: ranked articles with similarity scores, hierarchy and text |
| Traces | Every answer and search from this tab, with a link to its Langfuse trace (one tab = one Langfuse session) |
| Evaluation | Planned: what it will test, what it needs first, and a preview of its layout |
| Feedback log | Every tester rating (right/wrong, reason tags, comment) from `data/feedback/feedback.jsonl`, downloadable; ratings are also scored on the answer's trace |

When a stage lands, update its entry in [src/rag/ui/status.py](src/rag/ui/status.py) so testers see it.

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
Compare run share a `group_id` in their metadata. Errors (e.g. an expired Bedrock key) are ERROR-level; a Stop press is a
WARNING with the partial answer kept. Every answer card links to its trace.

## Data

`data/raw/egyptian_civil_code.pdf` is tracked with DVC: git stores only the `.dvc` pointer file, and the PDF itself lives in S3 (`s3://amzn-egypt-law-rag/dvc`, eu-north-1). The pipeline outputs (`articles.json`, the Chroma index) are DVC-tracked the same way through `dvc.lock`; their reports are small and committed to git.

The `dvc/` prefix of the bucket is publicly readable, so anyone can `dvc pull` without an AWS account. Only the owner can write (`dvc push`), using `aws login`: install AWS CLI v2 inside WSL and share one login with Windows:

```bash
ln -sfn /mnt/c/Users/<you>/.aws ~/.aws   # reuse the Windows ~/.aws
aws login
dvc push
```
