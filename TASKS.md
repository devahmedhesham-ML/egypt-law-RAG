# Tasks: Project 2 · LLM / RAG (Arabic Legal Document Q&A)

Source: MLOps Practitioner Handbook pp. 63–69 (rules, Project 2 brief, 10-point rubric, Project 2 checklist).
Optimization items come from the "What you ship" table on p. 66, because the p. 69 checklist stops before them.
`R0x` = rubric row on p. 67.

## Course rules (p. 63)
- [ ] At least one GitHub commit per session (5 sessions)
- [ ] Review another student's project, 300+ words covering setup, code, a strength, 2 improvements and an extension (R09)

## Waiting on you
- [x] Docker Desktop repaired after the full disk (`wsl --shutdown`, new version 29.8.2; images and containers intact); WSL integration on; 60 GB freed
- [x] Canary demo run end to end (`reports/canary_test.md`)
- [ ] Keep `~/actions-runner/run-gpu.sh` running when PRs should get the GPU quality gate (or `scripts/setup_gpu_runner.sh --remove`)
- [ ] Arabic spot-check of 20 articles (Corpus view → Random 20)
- [ ] Peer review of another student's project (R09)
- [ ] Delete the unused IAM access keys from `.env` and deactivate them in AWS IAM

## Done: R04–R07 (vLLM main model, CI/CD, BentoML serving)
- [x] Qwen2.5-7B-Instruct-AWQ on vLLM is the main model everywhere (params, API, Docker, console, evaluation); Bedrock is optional
- [x] Evaluation set accepted as is; 20 questions flagged `ci: true` for the quality gate
- [x] Faithfulness: Qwen2.5 answers and judges (Bedrock judge optional); 8 chunking runs re-scored; judge reliability checked (`reports/judge_comparison.md`)
- [x] MLflow Registry: `civil-code-retrieval` v2 with alias `production` (article + Qwen3-Embedding); screenshots in `reports/`
- [x] FastAPI handlers async; `POST /ask/stream` streams tokens; `X-Release` header
- [x] BentoML service (`rag.serving.service`) with async `/ask` and streaming `/ask_stream`
- [x] Locust at 50 users: p95 12 s (`/ask`), 8.7 s to first streamed text, 0 failures (`reports/locust_summary.md`); bottleneck = vLLM generation throughput
- [x] vLLM GPU share 0.75 → 0.60 so the embedding model runs on the GPU next to it
- [x] Batch re-indexing (`rag.ingest.batch`) tested with a new document (`reports/batch_reindex.md`)
- [x] Index fallback `rag.ingest.ensure`: local → dvc pull → build (used by CI)
- [x] Canary rollout: nginx weighted split + `set_weights.sh`, documented in the README and tested (`deploy/canary/`, `reports/canary_test.md`)
- [x] CI workflow: lint → test → index → Docker build/push to GHCR → faithfulness gate on a self-hosted GPU runner; PR #1 green (gate 0.802 in CI, index pulled from DVC) and merged; image public at `ghcr.io/devahmedhesham-ml/egypt-law-rag-api`
- [x] Quality gate passes locally: faithfulness 0.787 ≥ 0.75 on the 20 CI questions
- [x] `docker-compose.yml` starts vLLM + API: verified (both healthy in 80 s; curl checks pass with Qwen2.5 in the container, streaming works)
- [x] Own vLLM image (`Dockerfile.vllm`, 8.0 GB instead of the official 21.6 GB): only `requirements-serve.lock`, gcc kept (Triton needs it), video/audio packages removed; same curl answers, faithfulness 0.804 / 0.792, Locust p95 12 s; built and pushed by `.github/workflows/vllm-image.yml` (public at `ghcr.io/devahmedhesham-ml/egypt-law-rag-vllm:0.30.0`, 3.89 GB download; CI gate 0.804 against it); fresh clone + `docker compose up --build` verified with the GHCR image (curl answers identical)

## Done: API, Docker, evaluation set, MLflow chunking experiments (R02–R04)
- [x] vLLM moved to port 8001 so the API owns 8000
- [x] `rag.pipeline`: one retrieve → answer → check-citations path shared by the API, evaluation and (next) BentoML
- [x] `python -m rag.api`: `/ask` + `/health`, 422 on empty/blank/missing questions, 503 with the reason when the LLM or index is down
- [x] Docker image (2.6 GB, CPU): corpus + Chroma index pulled from S3 at build, embedding model baked in, runs offline; verified from a fresh clone
- [x] `eval/questions.jsonl`: 62 questions (28 topics × AR/EN, 2 by number, 4 out of scope), consistency-tested
- [x] Chunking strategies `article` (default) / `window` (size, overlap) / `per_language`; retrieval maps chunks back to articles
- [x] `python -m rag.experiments.chunking --faithfulness`: 8 MLflow runs (strategy × chunk_size/overlap × embedding model), retrieval metrics + RAGAS faithfulness judged by Bedrock
- [x] Report explains every column, compares runs question by question, and adds embedding cost/speed/hardware (`reports/chunking_experiments.md`)
- [x] `docs/evaluation-dataset.md`: how the evaluation set was built, checked, and its limits
- [x] Chroma searches no longer modify the DVC-tracked index (read-only copy); MLflow experiment shown in the classic runs view
- [x] 85 tests passing; API verified locally, in Docker and from a fresh clone; console verified against Bedrock and vLLM (:8001)

## Done: Langfuse tracing and follow-ups
- [x] One `answer-question` trace per answer: retriever → embedding, generation (prompt, tokens, time to first token, reasoning), citation evaluator
- [x] `search-articles`, `index-corpus` and `build-corpus` traces; batch jobs flush before exit
- [x] Scores: `citations_outside_context`, `answer_has_citations`, `tester_rating`, `tester_issue`, `corpus_warnings`, `smoke_top1_rate`, `build_warnings`
- [x] Console: per-tab session, "Trace ↗" link on every answer and search, Traces view, live Langfuse status check
- [x] Audited real traces against the Langfuse best-practices page; fixed Stop handling, model-load span, service name, smoke sessions
- [x] Tests disable tracing (`tests/conftest.py`): 58 passing
- [x] `dvc repro` + `dvc push` after the tracing edits (dvc.lock committed)

## Done: build corpus + GPU ingestion into Chroma
- [x] `params.yaml`: `corpus:` and `ingest:` sections
- [x] `rag.corpus`: column streams split on markers + table rows as cross-check, Arabic rebuild, bilingual numbered hierarchy
- [x] `rag.corpus.build`: progress bar, warnings summary, `articles.json` + `corpus_report.json` (170 pages in ~3.5 s)
- [x] Checks: AR/EN number match, full articles (cut-offs, paragraphs, lists, reversed numbers), gaps, artifacts, golden records
- [x] Run the build on the whole PDF and review every warning: 1,149 articles, 11 warnings, all source issues
- [x] `rag.ingest`: warning gate, bilingual chunks, multi-replica GPU embedding with progress, Chroma write, smoke queries (4/4 top-1)
- [x] Measure: 1 replica is fastest on this GPU (33k tok/s); auto mode adds replicas only if they fit AND the GPU has idle cores
- [x] Tests (extraction, hierarchy, chunks, replica planner, warning gate): 53 passing
- [x] Console: full corpus in the Corpus view (warnings, Random 20, breadcrumbs), index status, context search
- [x] `dvc.yaml` stages `build_corpus` + `index`, `dvc repro`, commit and push
- [x] `dvc push` of the corpus and index
- [x] Replace the expired Bedrock API key in `.env`

## 1. Corpus (Step 0, pp. 64–65, 69)
Plan: [docs/plans/corpus-build.md](docs/plans/corpus-build.md)
- [x] Civil Code PDF converted to structured JSON, one record per article (1,149), DVC-tracked (`dvc.lock`)
- [ ] Extraction validated: numbers are integers and repealed articles flagged (done); Arabic spot-checked on 20 articles (yours: Corpus view → Random 20)
- [x] Source PDF tracked with DVC; `dvc pull` fetches it from the public S3 bucket
- [x] Derived JSON tracked with DVC too; `dvc repro` rebuilds the JSON from the PDF (R05)

## 2. Ingestion and retrieval (p. 69)
- [x] Embedding model installed and tested: Qwen/Qwen3-Embedding-0.6B (recall@1 7/8, recall@3 8/8 on the sample)
- [x] Ingestion pipeline: JSON → chunk by article → embed → vector store (Chroma, 1,149 vectors)
- [x] `dvc repro` re-indexes the corpus reproducibly from tracked documents (R05)
- [x] Wire retrieval into the console: Ask/Compare search the index per question (top-k); Retrieval inspector view
- [x] Retrieval in the production `/ask` (FastAPI, through `rag.pipeline`); the BentoML wrapper is in section 5
- [x] Batch re-indexing script tested with at least one new document (`reports/batch_reindex.md`)

## 3. Code, API, Docker (Module 1 · R01–R03, p. 69)
- [x] `src/` layout with `pyproject.toml`; `pip install -e .` works (R01)
- [x] LLM layer: one OpenAI-compatible client over Bedrock (`gpt-oss-120b`) and vLLM, inline article citations, citation check
- [x] FastAPI `/ask`: `{question: str}` → `{answer: str, sources: list[str]}`, with sources as article citations, not chunk IDs (R02)
- [x] Pydantic rejects an empty question: 422 returned and tested with curl, see `reports/curl_checks.md` (R02)
- [x] `/health` returns `{status: healthy, documents_indexed: N}` (R02)
- [x] Dockerfile includes the vector store and embedded documents; `docker compose up` starts on port 8000 (R03)
- [x] README: exactly 3 commands to run Q&A on any machine (R03, R10)

## 4. Tracking, versioning, CI (Module 2 · R04–R06, p. 69)
- [x] MLflow logs each config: `chunk_size`, `overlap`, `embedding_model`, `faithfulness` (R04)
- [x] 5+ runs compared: 8 runs (`reports/chunking_experiments.md`), MLflow screenshot `reports/mlflow_chunking_compare.png` (R04)
- [x] Best chunking config registered in the MLflow Registry, alias `production` (R04)
- [x] GitHub Actions: lint → test → rebuild index → push Docker image (R06)
- [x] CI fails if RAGAS faithfulness < 0.75 on a 20-question test set (R06; self-hosted GPU runner)

## 5. Production serving (Module 3 · R07, pp. 66, 69)
- [x] vLLM serves the generative LLM, model name in README (`Qwen/Qwen2.5-7B-Instruct-AWQ`, `scripts/serve_vllm.sh`, compose `vllm` service)
- [x] BentoML wraps the RAG pipeline with an async `/ask` endpoint
- [x] Streaming: tokens appear progressively in `curl -N` output (`/ask_stream`, `/ask/stream`)
- [x] Locust report at 50 concurrent users in `/reports/`, with p95 latency documented
- [x] Canary rollout config documented in README

## 6. Optimization (Module 4, p. 66)
- [ ] Own AWQ-4bit quantization of the generative model (calibrated on Arabic articles)
- [ ] Re-ranker distillation
- [ ] RAGAS before vs after optimization

## 7. Monitoring and observability (Module 5 · R08, pp. 66, 69)
- [ ] RAGAS on 50+ questions, all 4 metrics logged (faithfulness done on 58 questions for every chunking run; answer relevancy, context precision and recall to add)
- [ ] RAGAS results stored in MLflow, with the trend visible across sessions (faithfulness is in the chunking runs; no per-session trend yet)
- [ ] Grafana panel for RAGAS faithfulness; screenshot in README (R08)
- [ ] Alert: faithfulness < 0.80 triggers a notification; threshold documented (R08)
- [x] Langfuse tracing over the whole app: answers, searches, index and corpus builds, citation and tester scores (Langfuse Cloud)
- [ ] Langfuse **self-hosted** (the handbook asks for it; same keys/env vars, only `LANGFUSE_BASE_URL` changes)
- [ ] Model prices in Langfuse for `openai.gpt-oss-120b` (Bedrock) so cost shows per trace
- [ ] RAGAS faithfulness score attached to each Langfuse trace
- [ ] Cosine embedding drift and token cost tracked

## 8. README and architecture (R10)
- [x] 3-command setup that the reviewer runs without asking anything (compose starts vLLM too, no key needed on a GPU machine; verified)
- [ ] Architecture diagram covering all 5 sessions
- [ ] Session changelog

## Done outside the handbook list
- [x] Test console (`python -m rag.ui`): Ask, Compare, Status, Corpus, Retrieval, Traces, feedback log
- [x] Public S3 DVC remote with anonymous pulls; lock files for the app, serve and quantize venvs

## Housekeeping (yours)
- [x] Commit this `TASKS.md`
- [ ] Decide whether to commit `.claude/skills/langfuse`, `.vscode/` and `test.ipynb`
- [ ] Optional: free disk space (bge-m3 cache 4.3 GB in `~/.cache/huggingface`, `data/experiments/` 287 MB)
