# Chunking experiments

Which way of cutting the Civil Code into chunks, and which embedding model, puts the right article in front of the model? Each row below is one configuration: the corpus is chunked and embedded its way, then every question goes through the same retrieval production uses, and the result is scored. Production today: **article__Qwen3-Embedding-0.6B**.

> Generated 2026-10-07 by `python -m rag.experiments.chunking --faithfulness` (git afa1e3d) from the latest MLflow run of each configuration (`mlflow.db`, experiment `chunking`; open it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).
> The 62 questions are the project's accepted evaluation set ([docs/evaluation-dataset.md](../docs/evaluation-dataset.md)); they have not been reviewed by a legal expert.

## Results

| Run | Strategy | Chunk size / overlap | Embedding model | Chunks | hit_at_1 | recall_at_5 | mrr | ndcg_at_5 | hit_at_1_ar | hit_at_1_en | ar_en_top1_agreement | faithfulness | faithfulness_judged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| article__Qwen3-Embedding-0.6B | article | whole article | Qwen3-Embedding-0.6B | 1149 | 0.741 | **0.931** | 0.819 | 0.844 | 0.724 | **0.759** | 0.607 | 0.813 | 57/58 |
| per_language__Qwen3-Embedding-0.6B | per_language | whole article | Qwen3-Embedding-0.6B | 2242 | 0.655 | 0.897 | 0.767 | 0.792 | 0.621 | 0.690 | 0.357 | 0.813 | 58/58 |
| window-128o16__Qwen3-Embedding-0.6B | window | 128 / 16 | Qwen3-Embedding-0.6B | 9744 | 0.466 | 0.690 | 0.567 | 0.592 | 0.448 | 0.483 | 0.393 | 0.741 | 57/58 |
| window-256o32__Qwen3-Embedding-0.6B | window | 256 / 32 | Qwen3-Embedding-0.6B | 1842 | 0.724 | 0.897 | 0.805 | 0.823 | 0.724 | 0.724 | 0.571 | 0.823 | 58/58 |
| window-512o64__Qwen3-Embedding-0.6B | window | 512 / 64 | Qwen3-Embedding-0.6B | 1181 | 0.724 | **0.931** | 0.808 | 0.836 | 0.724 | 0.724 | 0.571 | 0.813 | 57/58 |
| article__bge-m3 | article | whole article | bge-m3 | 1149 | **0.759** | **0.931** | **0.835** | **0.854** | **0.759** | **0.759** | **0.714** | 0.806 | 58/58 |
| per_language__bge-m3 | per_language | whole article | bge-m3 | 2242 | 0.741 | 0.914 | 0.816 | 0.832 | **0.759** | 0.724 | 0.571 | **0.843** | 58/58 |
| window-256o32__bge-m3 | window | 256 / 32 | bge-m3 | 1740 | 0.690 | 0.914 | 0.789 | 0.815 | 0.655 | 0.724 | **0.714** | 0.830 | 57/58 |

Best value per column in bold. Every metric is an average over the **58 in-scope questions** (the out-of-scope ones have no right article to find), so **one question moves a metric by 0.017**: a gap of one or two questions between two runs is not a real difference. The sections below explain each column, then show the same results question by question.

## Where the right article landed

The 58 in-scope questions counted by where the run ranked the first relevant article. That rank is what `hit_at_1` (rank 1), `recall_at_5` (rank ≤ 5) and `mrr` (1 ÷ rank) are calculated from.

| Run | 1st | 2nd–5th | 6th–10th | not in top 10 | right article reaches the model (top 5) |
|---|---|---|---|---|---|
| article__Qwen3-Embedding-0.6B | 43 | 11 | 1 | 3 | 54 of 58 |
| per_language__Qwen3-Embedding-0.6B | 38 | 14 | 4 | 2 | 52 of 58 |
| window-128o16__Qwen3-Embedding-0.6B | 27 | 13 | 3 | 15 | 40 of 58 |
| window-256o32__Qwen3-Embedding-0.6B | 42 | 10 | 2 | 4 | 52 of 58 |
| window-512o64__Qwen3-Embedding-0.6B | 42 | 12 | 1 | 3 | 54 of 58 |
| article__bge-m3 | 44 | 10 | 1 | 3 | 54 of 58 |
| per_language__bge-m3 | 43 | 10 | 3 | 2 | 53 of 58 |
| window-256o32__bge-m3 | 40 | 13 | 2 | 3 | 53 of 58 |

## Compared with article__Qwen3-Embedding-0.6B, question by question

Averages can hide trade-offs, so each run is compared with the baseline on the same questions. **Won**: the run gets the question right and the baseline does not; **lost**: the reverse. "Right" means rank 1 for hit@1, and top 5 for the top-5 columns.

| Run | hit@1 won | hit@1 lost | top-5 won | top-5 lost | faithfulness vs baseline |
|---|---|---|---|---|---|
| per_language__Qwen3-Embedding-0.6B | 3 | 8 | 1 | 3 | +0.000 |
| window-128o16__Qwen3-Embedding-0.6B | 1 | 17 | 0 | 14 | -0.072 |
| window-256o32__Qwen3-Embedding-0.6B | 3 | 4 | 1 | 3 | +0.009 |
| window-512o64__Qwen3-Embedding-0.6B | 0 | 1 | 0 | 0 | +0.000 |
| article__bge-m3 | 10 | 9 | 2 | 2 | -0.007 |
| per_language__bge-m3 | 9 | 9 | 2 | 3 | +0.030 |
| window-256o32__bge-m3 | 9 | 12 | 1 | 2 | +0.017 |

Read the two columns together: the net (won − lost) is what moves the average, but large won *and* lost counts with a small net mean the two runs get *different* questions right, which a single average does not show.

## Embedding cost and speed

Measured on an **NVIDIA GeForce RTX 4070 Ti SUPER** (16.0 GB VRAM, driver 591.86, CUDA 13.0, torch 2.14.0+cu130) and an **AMD Ryzen 7 9700X 8-Core Processor** (16 threads, 15.2 GB RAM visible to the OS), Linux-6.6.87.2-microsoft-standard-WSL2, on 2026-10-07. Under WSL the GPU is shared with the Windows desktop, so the idle load before each run is shown next to the numbers.

### Size of the embeddings

| Run | Vectors | Dimensions | Raw vectors (float32) | Chroma index on disk |
|---|---|---|---|---|
| article__Qwen3-Embedding-0.6B | 1,149 | 1024 | 4.5 MB | 22.5 MB |
| per_language__Qwen3-Embedding-0.6B | 2,242 | 1024 | 8.8 MB | 31.3 MB |
| window-128o16__Qwen3-Embedding-0.6B | 9,744 | 1024 | 38.1 MB | 101.5 MB |
| window-256o32__Qwen3-Embedding-0.6B | 1,842 | 1024 | 7.2 MB | 28.5 MB |
| window-512o64__Qwen3-Embedding-0.6B | 1,181 | 1024 | 4.6 MB | 22.6 MB |
| article__bge-m3 | 1,149 | 1024 | 4.5 MB | 22.6 MB |
| per_language__bge-m3 | 2,242 | 1024 | 8.8 MB | 31.3 MB |
| window-256o32__bge-m3 | 1,740 | 1024 | 6.8 MB | 26.6 MB |

### Building the index: embedding the whole corpus

| Run | Embedding time | Chunks/s | Tokens/s | Per chunk (batched) | GPU busy, mean / max | Idle before: busy / VRAM | VRAM of the model copy | GPU memory peak (whole device) |
|---|---|---|---|---|---|---|---|---|
| article__Qwen3-Embedding-0.6B | 8.7 s | 133 | 34,455 | 7.5 ms | 80% / 92% | 12% / 2.60 GB | 2.16 GB | 4.71 GB of 15.99 GB |
| per_language__Qwen3-Embedding-0.6B | 10.7 s | 209 | 38,184 | 4.8 ms | 85% / 92% | 13% / 2.60 GB | 2.19 GB | 4.61 GB of 15.99 GB |
| window-128o16__Qwen3-Embedding-0.6B | 31.5 s | 309 | 43,276 | 3.2 ms | 88% / 92% | 15% / 2.61 GB | 2.15 GB | 4.59 GB of 15.99 GB |
| window-256o32__Qwen3-Embedding-0.6B | 10.6 s | 174 | 37,518 | 5.7 ms | 82% / 93% | 4% / 4.00 GB | 2.19 GB | 7.39 GB of 15.99 GB |
| window-512o64__Qwen3-Embedding-0.6B | 8.4 s | 140 | 36,024 | 7.1 ms | 83% / 95% | 22% / 2.59 GB | 2.18 GB | 4.74 GB of 15.99 GB |
| article__bge-m3 | 3.7 s | 312 | 76,649 | 3.2 ms | 63% / 83% | 4% / 5.12 GB | 2.08 GB | 4.15 GB of 15.99 GB |
| per_language__bge-m3 | 4.5 s | 493 | 84,671 | 2.0 ms | 65% / 82% | 0% / 3.99 GB | 3.13 GB | 5.49 GB of 15.99 GB |
| window-256o32__bge-m3 | 4.2 s | 418 | 86,974 | 2.4 ms | 66% / 85% | 0% / 2.57 GB | 2.08 GB | 4.04 GB of 15.99 GB |

### Answering a question: retrieval latency

| Run | Embed the question, GPU (p50 / p95) | Embed the question, CPU (p50 / p95) | Vector search (p50 / p95) | Model VRAM when answering |
|---|---|---|---|---|
| article__Qwen3-Embedding-0.6B | 14.9 / 17.9 ms | 52.6 / 69.3 ms | 10.1 / 11.3 ms | 1.13 GB |
| per_language__Qwen3-Embedding-0.6B | 15.5 / 20.1 ms | 51.0 / 67.1 ms | 11.7 / 12.8 ms | 1.13 GB |
| window-128o16__Qwen3-Embedding-0.6B | 14.5 / 19.2 ms | 54.1 / 66.8 ms | 12.4 / 14.9 ms | 1.13 GB |
| window-256o32__Qwen3-Embedding-0.6B | 16.1 / 18.4 ms | 54.9 / 68.2 ms | 12.0 / 13.4 ms | 1.13 GB |
| window-512o64__Qwen3-Embedding-0.6B | 14.7 / 17.7 ms | 54.2 / 63.3 ms | 12.3 / 13.6 ms | 1.13 GB |
| article__bge-m3 | 6.0 / 7.7 ms | 64.1 / 84.3 ms | 9.9 / 10.9 ms | 1.07 GB |
| per_language__bge-m3 | 5.7 / 6.7 ms | 61.6 / 80.7 ms | 11.4 / 12.2 ms | 1.07 GB |
| window-256o32__bge-m3 | 5.6 / 6.7 ms | 61.4 / 81.3 ms | 12.4 / 14.0 ms | 1.07 GB |

How each number is measured:

- **Vectors / dimensions**: one vector per chunk; both models output 1,024 numbers per vector.
- **Raw vectors** = chunks × dimensions × 4 bytes (Chroma stores float32). **Chroma index on disk** is the run's whole index directory: the vectors, the HNSW search graph, and a SQLite file with every chunk's text and metadata, which is why it is several times the raw size.
- **Embedding time**: the embedding phase only, for the whole corpus: fp16 on the GPU, one model copy, chunks sorted by length and batched up to 4,096 padded tokens. Model loading is logged separately (`embed_load_s`). **Per chunk (batched)** = time ÷ chunks: throughput, not the latency of embedding a single chunk.
- **GPU busy**: NVML utilization (the share of time the GPU's cores were executing work), sampled every 0.1 s during the embedding phase. It is device-wide, so it includes the idle load measured in the second before the run started (**Idle before**).
- **VRAM of the model copy**: the embedder measures how much device memory one loaded model uses while embedding the longest batch; it plans how many copies fit from this number. **GPU memory peak** is the highest memory use of the whole device during embedding, idle usage included.
- **Embed the question**: each of the 62 questions embedded on its own, exactly as production does per request, after 3 warm-up questions; p50 is the median, p95 the value 95% of questions stay under. GPU runs fp16 like the production Retriever when the GPU has room; CPU runs fp32 on all threads, like the Docker image (no GPU). **Model VRAM when answering** is PyTorch's peak allocation during these single questions.
- **Vector search**: one Chroma query per question against the run's index, after a warm-up query, fetching enough chunks to rank 10 distinct articles. Retrieval latency per question ≈ question embedding + vector search.
- These were measured in a separate pass (`--perf-only`, same code and hardware) and added to the same MLflow runs; the retrieval and faithfulness results above come from the original runs.

## How a run works

1. **Chunk** the 1,149 articles with the run's strategy (below). Every chunk starts with the article's heading path and `Article N | مادة N`, so even a window from the middle of an article says where it comes from.
2. **Embed** every chunk with the run's embedding model on the GPU and write a throwaway Chroma index (`data/experiments/`; the production index is never touched).
3. **Retrieve** for every question exactly as production does: embed the question (with the model's query prompt, if it has one), fetch the nearest chunks, keep each article's best-scoring chunk, and put any article the question names by number ("What does Article 505 say?") first. The result is a ranked list of up to 10 different articles.
4. **Score retrieval** by comparing that list with the question's relevant articles (the retrieval metrics).
5. **Answer and judge**: the top 5 articles, always as whole articles whatever the chunking, go to `Qwen/Qwen2.5-7B-Instruct-AWQ` with the production prompt; RAGAS then judges how faithful the answer is to those articles (the same model acts as judge).
6. **Log** the configuration, the metrics and a `per_question.json` (every question's ranking, answer and score) as one MLflow run.

Chunking only changes *which* articles are retrieved: the model always receives whole articles.

## The questions

`eval/questions.jsonl` has 62 questions: 28 topics asked once in Arabic and once in English, 2 that name an article by number, and 4 out of scope (outside the Civil Code, e.g. criminal penalties) that no article answers. The 58 in-scope questions (29 Arabic, 29 English) each list their **relevant articles**: the article(s) a correct answer rests on. 56 have one; 2 have more (e.g. `q26-ar`: Articles 968 and 969). How the set was built, and its limits: [docs/evaluation-dataset.md](../docs/evaluation-dataset.md).

## What each column means and how it is calculated

### The configuration

- **Run**: `strategy[-size o overlap]__model`, the run's name in MLflow.
- **Strategy**: how articles become chunks.
  - `article`: one chunk per article, Arabic and English together (what production uses).
  - `window`: articles longer than the chunk size are cut into overlapping windows; shorter articles stay whole.
  - `per_language`: two chunks per article, one Arabic and one English (repealed articles keep one short chunk).
- **Chunk size / overlap**: `window` only, in tokens of the run's embedding model. The size includes the heading and article number repeated at the top of each window; the overlap is how many tokens neighbouring windows share, so a sentence cut at a boundary still appears whole in one of them. The heading path is bilingual and often long, so small windows are mostly heading: each window keeps at least overlap + 16 tokens of article text even when the heading alone nearly fills the size, which is why some chunks run past it (see *Longest chunk* below).
- **Embedding model**: turns chunks and questions into vectors; retrieval ranks chunks by cosine similarity to the question. Qwen3-Embedding-0.6B is production's; bge-m3 is the comparison.
- **Chunks**: vectors in the run's index. 1,149 = one per article; `per_language` makes one Arabic and one English chunk per live article plus one per repealed article; `window` adds a chunk for every extra window of a long article:

| Run | Chunks | Avg tokens per chunk | Longest chunk | Embedding time |
|---|---|---|---|---|
| article__Qwen3-Embedding-0.6B | 1149 | 259 | 1091 | 8.7 s |
| per_language__Qwen3-Embedding-0.6B | 2242 | 183 | 649 | 10.7 s |
| window-128o16__Qwen3-Embedding-0.6B | 9744 | 140 | 172 | 31.5 s |
| window-256o32__Qwen3-Embedding-0.6B | 1842 | 215 | 259 | 10.6 s |
| window-512o64__Qwen3-Embedding-0.6B | 1181 | 257 | 513 | 8.4 s |
| article__bge-m3 | 1149 | 245 | 1071 | 3.7 s |
| per_language__bge-m3 | 2242 | 172 | 625 | 4.5 s |
| window-256o32__bge-m3 | 1740 | 208 | 258 | 4.2 s |

### Retrieval metrics

Each is computed per question from its ranked list, then averaged over the 58 in-scope questions. Ranks start at 1.

| Column | Value for one question | How it is calculated | What the average tells you |
|---|---|---|---|
| `hit_at_1` | 1 or 0 | 1 if the article ranked first is a relevant one | share of questions where the top article is right |
| `recall_at_5` | 0 to 1 | relevant articles in the top 5 ÷ all relevant articles (with one relevant article: 1 or 0) | share of the needed articles that actually reach the model, which receives the top 5 |
| `mrr` | 1, 1/2, 1/3 … or 0 | 1 ÷ rank of the first relevant article; 0 if it is not in the top 10 (mean reciprocal rank) | how high the right article sits: 1.0 means always first, 0.5 means second on average |
| `ndcg_at_5` | 0 to 1 | each relevant article at rank r ≤ 5 earns 1/log₂(r+1); the sum is divided by the best possible sum (all relevant articles at the top). With one relevant article: 1.000, 0.631, 0.500, 0.431, 0.387 for ranks 1 to 5 | like recall@5, but a hit at rank 1 counts more than a hit at rank 5 |
| `hit_at_1_ar`, `hit_at_1_en` | 1 or 0 | `hit_at_1`, averaged over the 29 Arabic or the 29 English questions only | whether one language is served better than the other |
| `ar_en_top1_agreement` | 1 or 0 per topic | over the 28 topics asked in both languages: 1 if the Arabic and English versions put the same article first, right or wrong | consistency across languages: the same legal question should find the same article in either language |

### Answer metric: faithfulness (RAGAS)

For each in-scope question, `Qwen/Qwen2.5-7B-Instruct-AWQ` (the production model) answers from the run's top 5 articles, and RAGAS scores the answer in two LLM steps with `Qwen/Qwen2.5-7B-Instruct-AWQ` as the judge:

1. **Split** the answer into short standalone statements ("a gift of future property is void").
2. **Check** each statement against the 5 articles: supported or not.

`faithfulness` for one answer = supported statements ÷ all statements, so an answer with 4 statements of which 3 are supported scores 0.75. The column is the average over all judged answers. It measures **grounding, not correctness**: an answer that faithfully repeats the wrong article still scores high, which is why it is read together with the retrieval metrics.

`faithfulness_judged` = how many of the 58 answers received a score. An answer is left out when the model fails after retries, the judge errors, or the answer yields no statements.

## Worked example

From article__Qwen3-Embedding-0.6B: question `q04-en`, "At what point is a contract concluded between two people?" — relevant article: **89**. The run's top 5: 97, **89** ✓, 543, 694, 95.

- `hit_at_1` = **0**: the first article (97) is not relevant.
- `recall_at_5` = 1/1 = **1.000**: the relevant article is in the top 5, so it reaches the model.
- `mrr` = 1/2 = **0.500**.
- `ndcg_at_5` = (1/log₂(2+1)) ÷ 1.000 = **0.631**.

Averaging these per-question values gives the table: article__Qwen3-Embedding-0.6B ranks the right article first for 43 of 58 questions, so its `hit_at_1` is 43/58 = 0.741.

## Out-of-scope questions: can a run tell there is no answer?

MLflow also logs the cosine similarity of the best match. For a future "no relevant article" threshold, in-scope questions should score clearly higher than out-of-scope ones; the gap is what such a threshold would have to work with.

| Run | Best-match similarity, in scope | Out of scope | Gap |
|---|---|---|---|
| article__Qwen3-Embedding-0.6B | 0.658 | 0.418 | 0.240 |
| per_language__Qwen3-Embedding-0.6B | 0.665 | 0.411 | 0.254 |
| window-128o16__Qwen3-Embedding-0.6B | 0.653 | 0.426 | 0.227 |
| window-256o32__Qwen3-Embedding-0.6B | 0.663 | 0.416 | 0.247 |
| window-512o64__Qwen3-Embedding-0.6B | 0.659 | 0.418 | 0.241 |
| article__bge-m3 | 0.653 | 0.488 | 0.165 |
| per_language__bge-m3 | 0.650 | 0.480 | 0.170 |
| window-256o32__bge-m3 | 0.651 | 0.482 | 0.169 |

Also in MLflow, per run: `hit_at_5` and an `_ar` / `_en` version of every retrieval metric, `faithfulness_ar` / `faithfulness_en`, `chunks_truncated` (chunks longer than the model reads; 0 in every run), and timings.

## Questions article__Qwen3-Embedding-0.6B misses

The relevant article is not in its top 5, so the model never sees it. Worth checking during the review: the question may be hard, or its relevant articles may be incomplete.

| Question | Relevant | Retrieved instead (top 3) | Found in the top 5 by |
|---|---|---|---|
| `q06-ar` هل يعد كتمان أحد الطرفين عمداً لواقعة مهمة أثناء التعاقد تدليساً يبرر إبطال العقد؟ | 125 | 161, 120, 549 | 4 of 7 other runs (per_language__Qwen3-Embedding-0.6B, article__bge-m3, per_language__bge-m3, window-256o32__bge-m3) |
| `q07-en` Someone took advantage of my obvious recklessness to make me sign a grossly one-sided contract. What can I do, and how long do I have? | 129 | 140, 172, 455 | 3 of 7 other runs (window-256o32__Qwen3-Embedding-0.6B, article__bge-m3, per_language__bge-m3) |
| `q28-ar` هل يستطيع الدائن والمدين إنشاء حق امتياز بمجرد اتفاقهما عليه؟ | 1130 | 279, 235, 1073 | 0 of 7 other runs |
| `q28-en` Can a creditor and debtor create a privileged right simply by agreeing on it? | 1130 | 279, 244, 241 | 0 of 7 other runs |

## Caveats

- **Not legally reviewed.** The questions were accepted as the evaluation set by the project owner; their relevant articles and reference answers have not been checked by a legal expert.
- **Small sample.** 58 questions: differences of one or two questions (0.017–0.034) are noise.
- **Articles named by number** are found by the number lookup, not the embedding, so every run gets those questions right.
- **Self-judging.** `Qwen/Qwen2.5-7B-Instruct-AWQ` both answers and judges faithfulness, which can be lenient with its own phrasing, and a 7B judge is less reliable than a larger one; read faithfulness as a comparison between runs, not as an absolute score.

MLflow comparison view (screenshot): [mlflow_chunking_compare.png](mlflow_chunking_compare.png).
