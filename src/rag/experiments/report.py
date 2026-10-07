"""reports/chunking_experiments.md, built from MLflow: the latest run of every config in the grid.

Everything in the report is computed from the logged metrics and each run's per_question.json, so it can be
rebuilt at any time without rerunning anything: python -m rag.experiments.chunking --report-only
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

EXPERIMENT = "chunking"


def run_name(cfg: dict) -> str:
    size = f"-{cfg['chunk_size']}o{cfg.get('overlap', 0)}" if cfg["strategy"] == "window" else ""
    return f"{cfg['strategy']}{size}__{cfg['model'].split('/')[-1]}"


def latest_runs(mlflow, grid: list[dict]) -> list[dict]:
    """The latest finished run of each grid config, in grid order: {name, cfg, metrics, rows}."""
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    runs = mlflow.search_runs([exp.experiment_id], filter_string="attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"], output_format="list")
    latest = {}
    for r in runs:
        latest.setdefault(r.info.run_name, r)
    out = []
    for cfg in grid:
        r = latest.get(run_name(cfg))
        if r is None:
            continue
        rows = mlflow.artifacts.load_dict(f"runs:/{r.info.run_id}/per_question.json")["questions"]
        out.append({"name": run_name(cfg), "cfg": cfg, "metrics": dict(r.data.metrics), "tags": dict(r.data.tags),
                    "rows": rows})
    return out


def production_config(grid: list[dict], ingest: dict) -> dict | None:
    """The grid entry that matches what params.yaml `ingest` uses in production, if the grid has it."""
    prod = {**(ingest.get("chunking") or {"strategy": "article"}), "model": ingest["model"]}
    for cfg in grid:
        if cfg["model"] == prod["model"] and cfg["strategy"] == prod["strategy"] and (
                cfg["strategy"] != "window" or (cfg.get("chunk_size"), cfg.get("overlap", 0))
                == (prod.get("chunk_size"), prod.get("overlap") or 0)):
            return cfg
    return None


def first_rank(row: dict) -> int | None:
    """1-based rank of the first relevant article in the question's ranked list (None: not in the top 10)."""
    rel = set(row["relevant_articles"])
    return next((i + 1 for i, n in enumerate(row["ranked"]) if n in rel), None)


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _f(v: float | None) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def _mb(v: float | None, digits: int = 1) -> str:
    return "—" if v is None else f"{v:,.{digits}f} MB"


def _gb(mb: float | None) -> str:
    return "—" if mb is None else f"{mb / 1024:.2f} GB"


def _ms(m: dict, prefix: str) -> str:
    p50, p95 = m.get(f"{prefix}_ms_p50"), m.get(f"{prefix}_ms_p95")
    return "—" if p50 is None else f"{p50:.1f} / {p95:.1f} ms"


def _embedding_cost(add, runs: list[dict], base: dict) -> None:
    measured = [r for r in runs if "vectors_mb" in r["metrics"]]
    hw = next((r["tags"] for r in [base, *measured] if "hw_gpu" in r["tags"] or "hw_cpu" in r["tags"]), {})
    add("## Embedding cost and speed\n")
    if hw:
        os_name = hw.get("hw_os", "").split("-x86_64")[0]
        add(f"Measured on an **{hw.get('hw_gpu', 'no GPU')}** ({hw.get('hw_gpu_vram_gb', '?')} GB VRAM, driver "
            f"{hw.get('hw_driver', '?')}, CUDA {hw.get('hw_cuda', '?')}, torch {hw.get('hw_torch', '?')}) and an "
            f"**{hw.get('hw_cpu', '?')}** ({hw.get('hw_cpu_threads', '?')} threads, {hw.get('hw_ram_gb', '?')} GB RAM "
            f"visible to the OS), {os_name}, on {hw.get('perf_measured_at', '?')}. Under WSL the GPU is shared with the "
            "Windows desktop, so the idle load before each run is shown next to the numbers.\n")
    add("### Size of the embeddings\n")
    add("| Run | Vectors | Dimensions | Raw vectors (float32) | Chroma index on disk |")
    add("|---|---|---|---|---|")
    for r in measured:
        m = r["metrics"]
        add(f"| {r['name']} | {m.get('chunks', 0):,.0f} | {m.get('embed_dim', 0):.0f} | {_mb(m.get('vectors_mb'))} | "
            f"{_mb(m.get('index_disk_mb'))} |")
    add("")
    add("### Building the index: embedding the whole corpus\n")
    add("| Run | Embedding time | Chunks/s | Tokens/s | Per chunk (batched) | GPU busy, mean / max | "
        "Idle before: busy / VRAM | VRAM of the model copy | GPU memory peak (whole device) |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in measured:
        m = r["metrics"]
        util = (f"{m['gpu_util_mean']:.0f}% / {m['gpu_util_max']:.0f}%" if "gpu_util_mean" in m else "—")
        idle = (f"{m['gpu_util_idle']:.0f}% / {_gb(m.get('vram_idle_mb'))}" if "gpu_util_idle" in m else "—")
        peak = (f"{_gb(m.get('vram_peak_device_mb'))} of {_gb(m.get('vram_total_mb'))}"
                if "vram_peak_device_mb" in m else "—")
        add(f"| {r['name']} | {m.get('embed_s', 0):.1f} s | {m.get('embed_chunks_per_s', 0):,.0f} | "
            f"{m.get('embed_tokens_per_s', 0):,.0f} | {m.get('embed_ms_per_chunk', 0):.1f} ms | {util} | {idle} | "
            f"{_gb(m.get('vram_replica_mb'))} | {peak} |")
    add("")
    add("### Answering a question: retrieval latency\n")
    add("| Run | Embed the question, GPU (p50 / p95) | Embed the question, CPU (p50 / p95) | Vector search (p50 / p95) | "
        "Model VRAM when answering |")
    add("|---|---|---|---|---|")
    for r in measured:
        m = r["metrics"]
        add(f"| {r['name']} | {_ms(m, 'query_gpu')} | {_ms(m, 'query_cpu')} | {_ms(m, 'search')} | "
            f"{_gb(m.get('query_gpu_vram_mb'))} |")
    add("")
    add("How each number is measured:\n")
    add("- **Vectors / dimensions**: one vector per chunk; both models output 1,024 numbers per vector.")
    add("- **Raw vectors** = chunks × dimensions × 4 bytes (Chroma stores float32). **Chroma index on disk** is the "
        "run's whole index directory: the vectors, the HNSW search graph, and a SQLite file with every chunk's text and "
        "metadata, which is why it is several times the raw size.")
    add("- **Embedding time**: the embedding phase only, for the whole corpus: fp16 on the GPU, one model copy, chunks "
        "sorted by length and batched up to 4,096 padded tokens. Model loading is logged separately (`embed_load_s`). "
        "**Per chunk (batched)** = time ÷ chunks: throughput, not the latency of embedding a single chunk.")
    add("- **GPU busy**: NVML utilization (the share of time the GPU's cores were executing work), sampled every 0.1 s "
        "during the embedding phase. It is device-wide, so it includes the idle load measured in the second before the "
        "run started (**Idle before**).")
    add("- **VRAM of the model copy**: the embedder measures how much device memory one loaded model uses while embedding "
        "the longest batch; it plans how many copies fit from this number. **GPU memory peak** is the highest memory "
        "use of the whole device during embedding, idle usage included.")
    add("- **Embed the question**: each of the 62 questions embedded on its own, exactly as production does per request, "
        "after 3 warm-up questions; p50 is the median, p95 the value 95% of questions stay under. GPU runs fp16 like the "
        "production Retriever when the GPU has room; CPU runs fp32 on all threads, like the Docker image (no GPU). "
        "**Model VRAM when answering** is PyTorch's peak allocation during these single questions.")
    add("- **Vector search**: one Chroma query per question against the run's index, after a warm-up query, fetching "
        "enough chunks to rank 10 distinct articles. Retrieval latency per question ≈ question embedding + vector search.")
    add("- These were measured in a separate pass (`--perf-only`, same code and hardware) and added to the same MLflow "
        "runs; the retrieval and faithfulness results above come from the original runs.\n")


def build_report(runs: list[dict], questions: list[dict], production: dict | None, k: int, git: str,
                 command: str) -> str:
    scoped = [q for q in questions if q["relevant_articles"]]
    n = len(scoped)
    n_ar = sum(q["lang"] == "ar" for q in scoped)
    n_en = n - n_ar
    pairs = sum(1 for p in {q["pair"] for q in scoped}
                if {q["lang"] for q in scoped if q["pair"] == p} == {"ar", "en"})
    draft = sum(q.get("status") == "draft" for q in questions)
    accepted = sum(q.get("status") == "accepted" for q in questions)
    tags = next((r.get("tags", {}) for r in runs if r.get("tags", {}).get("answer_model")), {})
    answer_model = tags.get("answer_model", "the production model")
    judge_model = tags.get("judge_model", "an LLM judge")
    self_judged = answer_model == judge_model
    base = next((r for r in runs if production and r["cfg"] == production), runs[0])
    is_prod = bool(production) and base["cfg"] == production
    base_label = f"**{base['name']}**" if is_prod else (
        f"not in this grid, so **{base['name']}** (the first run) serves as the baseline")
    faith = any("faithfulness" in r["metrics"] for r in runs)
    step = 1 / n
    L: list[str] = []
    add = L.append

    # --- header ------------------------------------------------------------------------------------------
    add("# Chunking experiments\n")
    add("Which way of cutting the Civil Code into chunks, and which embedding model, puts the right article in front "
        "of the model? Each row below is one configuration: the corpus is chunked and embedded its way, then every "
        f"question goes through the same retrieval production uses, and the result is scored. Production today: "
        f"{base_label}.\n")
    add(f"> Generated {datetime.now(UTC):%Y-%m-%d} by `{command}` (git {git}) from the latest MLflow run of each "
        "configuration (`mlflow.db`, experiment `chunking`; open it with "
        "`mlflow ui --backend-store-uri sqlite:///mlflow.db`).")
    if accepted:
        add(f"> The {len(questions)} questions are the project's accepted evaluation set "
            "([docs/evaluation-dataset.md](../docs/evaluation-dataset.md)); they have not been reviewed by a legal "
            "expert.")
    if draft:
        add(f"> **{draft} of {len(questions)} questions are drafts awaiting legal review** "
            "([eval/README.md](../eval/README.md)), so every number here is provisional.")
    add("")

    # --- results -----------------------------------------------------------------------------------------
    cols = ["hit_at_1", f"recall_at_{k}", "mrr", f"ndcg_at_{k}", "hit_at_1_ar", "hit_at_1_en", "ar_en_top1_agreement"]
    if faith:
        cols += ["faithfulness", "faithfulness_judged"]
    add("## Results\n")
    add("| Run | Strategy | Chunk size / overlap | Embedding model | Chunks | " + " | ".join(cols) + " |")
    add("|" + "---|" * (5 + len(cols)))
    best = {c: max(r["metrics"].get(c, float("-inf")) for r in runs) for c in cols}
    for r in runs:
        cfg, m = r["cfg"], r["metrics"]
        size = f"{cfg['chunk_size']} / {cfg.get('overlap', 0)}" if cfg["strategy"] == "window" else "whole article"
        cells = []
        for c in cols:
            v = m.get(c)
            if c == "faithfulness_judged":
                cells.append("—" if v is None else f"{v:.0f}/{n}")
            else:
                cells.append("—" if v is None else (f"**{v:.3f}**" if v == best[c] else f"{v:.3f}"))
        add(f"| {r['name']} | {cfg['strategy']} | {size} | {cfg['model'].split('/')[-1]} | {m.get('chunks', 0):.0f} | "
            + " | ".join(cells) + " |")
    add("")
    add(f"Best value per column in bold. Every metric is an average over the **{n} in-scope questions** (the out-of-scope "
        f"ones have no right article to find), so **one question moves a metric by {step:.3f}**: a gap of one or two "
        "questions between two runs is not a real difference. The sections below explain each column, then show the "
        "same results question by question.\n")

    # --- where the right article landed --------------------------------------------------------------------
    add("## Where the right article landed\n")
    add(f"The {n} in-scope questions counted by where the run ranked the first relevant article. That rank is what "
        f"`hit_at_1` (rank 1), `recall_at_{k}` (rank ≤ {k}) and `mrr` (1 ÷ rank) are calculated from.\n")
    add(f"| Run | 1st | 2nd–{k}th | {k + 1}th–10th | not in top 10 | right article reaches the model (top {k}) |")
    add("|---|---|---|---|---|---|")
    for r in runs:
        ranks = [first_rank(q) for q in r["rows"] if q["relevant_articles"]]
        top1 = sum(x == 1 for x in ranks)
        mid = sum(x is not None and 1 < x <= k for x in ranks)
        low = sum(x is not None and x > k for x in ranks)
        miss = sum(x is None for x in ranks)
        add(f"| {r['name']} | {top1} | {mid} | {low} | {miss} | {top1 + mid} of {n} |")
    add("")

    # --- question by question vs production ---------------------------------------------------------------
    add(f"## Compared with {base['name']}, question by question\n")
    add("Averages can hide trade-offs, so each run is compared with the baseline on the same questions. "
        "**Won**: the run gets the question right and the baseline does not; **lost**: the reverse. "
        f"\"Right\" means rank 1 for hit@1, and top {k} for the top-{k} columns.\n")
    add(f"| Run | hit@1 won | hit@1 lost | top-{k} won | top-{k} lost |" + (" faithfulness vs baseline |" if faith else ""))
    add("|---|---|---|---|---|" + ("---|" if faith else ""))
    base_rows = {q["id"]: q for q in base["rows"]}
    for r in runs:
        if r is base:
            continue
        w1 = l1 = wk = lk = 0
        for q in r["rows"]:
            b = base_rows.get(q["id"])
            if not q["relevant_articles"] or b is None:
                continue
            rq, rb = first_rank(q), first_rank(b)
            w1 += rq == 1 and rb != 1
            l1 += rb == 1 and rq != 1
            in_q, in_b = rq is not None and rq <= k, rb is not None and rb <= k
            wk += in_q and not in_b
            lk += in_b and not in_q
        line = f"| {r['name']} | {w1} | {l1} | {wk} | {lk} |"
        if faith:
            d = r["metrics"].get("faithfulness", float("nan")) - base["metrics"].get("faithfulness", float("nan"))
            line += f" {d:+.3f} |"
        add(line)
    add("")
    add("Read the two columns together: the net (won − lost) is what moves the average, but large won *and* lost counts "
        "with a small net mean the two runs get *different* questions right, which a single average does not show.\n")

    # --- embedding cost and speed ------------------------------------------------------------------------
    if any("vectors_mb" in r["metrics"] for r in runs):
        _embedding_cost(add, runs, base)

    # --- how a run works -----------------------------------------------------------------------------------
    add("## How a run works\n")
    add("1. **Chunk** the 1,149 articles with the run's strategy (below). Every chunk starts with the article's heading "
        "path and `Article N | مادة N`, so even a window from the middle of an article says where it comes from.")
    add("2. **Embed** every chunk with the run's embedding model on the GPU and write a throwaway Chroma index "
        "(`data/experiments/`; the production index is never touched).")
    add("3. **Retrieve** for every question exactly as production does: embed the question (with the model's query "
        "prompt, if it has one), fetch the nearest chunks, keep each article's best-scoring chunk, and put any article "
        "the question names by number (\"What does Article 505 say?\") first. The result is a ranked list of up to 10 "
        "different articles.")
    add("4. **Score retrieval** by comparing that list with the question's relevant articles (the retrieval metrics).")
    if faith:
        add(f"5. **Answer and judge**: the top {k} articles, always as whole articles whatever the chunking, go to "
            f"`{answer_model}` with the production prompt; RAGAS then judges how faithful the answer is to those articles "
            "(the same model acts as judge).")
        add("6. **Log** the configuration, the metrics and a `per_question.json` (every question's ranking, answer and "
            "score) as one MLflow run.")
    else:
        add("5. **Log** the configuration, the metrics and a `per_question.json` as one MLflow run.")
    add("")
    add("Chunking only changes *which* articles are retrieved: the model always receives whole articles.\n")

    # --- the questions -------------------------------------------------------------------------------------
    add("## The questions\n")
    by_number = sum(q["type"] == "by_number" for q in questions)
    multi = [q for q in scoped if len(q["relevant_articles"]) > 1]
    add(f"`eval/questions.jsonl` has {len(questions)} questions: {pairs} topics asked once in Arabic and once in English, "
        f"{by_number} that name an article by number, and {len(questions) - n} out of scope (outside the Civil Code, "
        f"e.g. criminal penalties) that no article answers. The {n} in-scope questions ({n_ar} Arabic, {n_en} English) "
        "each list their **relevant articles**: the article(s) a correct answer rests on. "
        + (f"{n - len(multi)} have one; {len(multi)} have more (e.g. `{multi[0]['id']}`: Articles "
           f"{' and '.join(map(str, multi[0]['relevant_articles']))}). " if multi else "Each has exactly one. ")
        + "How the set was built, and its limits: [docs/evaluation-dataset.md](../docs/evaluation-dataset.md).\n")

    # --- columns -------------------------------------------------------------------------------------------
    add("## What each column means and how it is calculated\n")
    add("### The configuration\n")
    add("- **Run**: `strategy[-size o overlap]__model`, the run's name in MLflow.")
    add("- **Strategy**: how articles become chunks.")
    add("  - `article`: one chunk per article, Arabic and English together (what production uses).")
    add("  - `window`: articles longer than the chunk size are cut into overlapping windows; shorter articles stay whole.")
    add("  - `per_language`: two chunks per article, one Arabic and one English (repealed articles keep one short chunk).")
    add("- **Chunk size / overlap**: `window` only, in tokens of the run's embedding model. The size includes the heading "
        "and article number repeated at the top of each window; the overlap is how many tokens neighbouring windows "
        "share, so a sentence cut at a boundary still appears whole in one of them. The heading path is bilingual and "
        "often long, so small windows are mostly heading: each window keeps at least overlap + 16 tokens of article "
        "text even when the heading alone nearly fills the size, which is why some chunks run past it (see *Longest "
        "chunk* below).")
    add("- **Embedding model**: turns chunks and questions into vectors; retrieval ranks chunks by cosine similarity to "
        "the question. Qwen3-Embedding-0.6B is production's; bge-m3 is the comparison.")
    add("- **Chunks**: vectors in the run's index. 1,149 = one per article; `per_language` makes one Arabic and one "
        "English chunk per live article plus one per repealed article; `window` adds a chunk for every extra window of "
        "a long article:\n")
    add("| Run | Chunks | Avg tokens per chunk | Longest chunk | Embedding time |")
    add("|---|---|---|---|---|")
    for r in runs:
        m = r["metrics"]
        add(f"| {r['name']} | {m.get('chunks', 0):.0f} | {m.get('chunk_tokens_mean', 0):.0f} | "
            f"{m.get('chunk_tokens_max', 0):.0f} | {m.get('embed_s', 0):.1f} s |")
    add("")

    add("### Retrieval metrics\n")
    add(f"Each is computed per question from its ranked list, then averaged over the {n} in-scope questions. Ranks "
        "start at 1.\n")
    add("| Column | Value for one question | How it is calculated | What the average tells you |")
    add("|---|---|---|---|")
    add("| `hit_at_1` | 1 or 0 | 1 if the article ranked first is a relevant one | share of questions where the top "
        "article is right |")
    add(f"| `recall_at_{k}` | 0 to 1 | relevant articles in the top {k} ÷ all relevant articles (with one relevant "
        f"article: 1 or 0) | share of the needed articles that actually reach the model, which receives the top {k} |")
    add("| `mrr` | 1, 1/2, 1/3 … or 0 | 1 ÷ rank of the first relevant article; 0 if it is not in the top 10 "
        "(mean reciprocal rank) | how high the right article sits: 1.0 means always first, 0.5 means second on "
        "average |")
    add(f"| `ndcg_at_{k}` | 0 to 1 | each relevant article at rank r ≤ {k} earns 1/log₂(r+1); the sum is divided by the "
        "best possible sum (all relevant articles at the top). With one relevant article: 1.000, 0.631, 0.500, 0.431, "
        f"0.387 for ranks 1 to 5 | like recall@{k}, but a hit at rank 1 counts more than a hit at rank {k} |")
    add(f"| `hit_at_1_ar`, `hit_at_1_en` | 1 or 0 | `hit_at_1`, averaged over the {n_ar} Arabic or the {n_en} English "
        "questions only | whether one language is served better than the other |")
    add(f"| `ar_en_top1_agreement` | 1 or 0 per topic | over the {pairs} topics asked in both languages: 1 if the Arabic "
        "and English versions put the same article first, right or wrong | consistency across languages: the same "
        "legal question should find the same article in either language |")
    add("")

    if faith:
        add("### Answer metric: faithfulness (RAGAS)\n")
        add(f"For each in-scope question, `{answer_model}` (the production model) answers from the run's top {k} "
            f"articles, and RAGAS scores the answer in two LLM steps with `{judge_model}` as the judge:\n")
        add("1. **Split** the answer into short standalone statements (\"a gift of future property is void\").")
        add(f"2. **Check** each statement against the {k} articles: supported or not.")
        add("")
        add("`faithfulness` for one answer = supported statements ÷ all statements, so an answer with 4 statements of "
            "which 3 are supported scores 0.75. The column is the average over all judged answers. It measures "
            "**grounding, not correctness**: an answer that faithfully repeats the wrong article still scores high, "
            "which is why it is read together with the retrieval metrics.\n")
        add(f"`faithfulness_judged` = how many of the {n} answers received a score. An answer is left out when the model "
            "fails after retries, the judge errors, or the answer yields no statements.\n")

    # --- worked example ------------------------------------------------------------------------------------
    example = None
    for want in (2, 3, 4, 5):
        example = next((q for q in base["rows"] if q["relevant_articles"] and q["type"] != "by_number"
                        and first_rank(q) == want), None)
        if example:
            break
    if example:
        rank = first_rank(example)
        rel = example["relevant_articles"]
        top = example["ranked"][:k]
        add("## Worked example\n")
        add(f"From {base['name']}: question `{example['id']}`, \"{_cell(example['question'])}\" — relevant article: "
            f"**{', '.join(map(str, rel))}**. The run's top {k}: "
            + ", ".join(f"**{a}** ✓" if a in rel else str(a) for a in top) + ".\n")
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
        dcg = sum(1 / math.log2(i + 2) for i, a in enumerate(top) if a in rel)
        found = len(set(rel) & set(top))
        add(f"- `hit_at_1` = **0**: the first article ({top[0]}) is not relevant.")
        add(f"- `recall_at_{k}` = {found}/{len(rel)} = **{found / len(rel):.3f}**: the relevant article is in the top {k}, "
            "so it reaches the model.")
        add(f"- `mrr` = 1/{rank} = **{1 / rank:.3f}**.")
        add(f"- `ndcg_at_{k}` = (1/log₂({rank}+1)) ÷ {ideal:.3f} = **{dcg / ideal:.3f}**.")
        hits = sum(q.get("hit_at_1", 0) for q in base["rows"] if q["relevant_articles"])
        add("")
        add(f"Averaging these per-question values gives the table: {base['name']} ranks the right article first for "
            f"{hits:.0f} of {n} questions, so its `hit_at_1` is {hits:.0f}/{n} = {hits / n:.3f}.\n")

    # --- other logged numbers ------------------------------------------------------------------------------
    add("## Out-of-scope questions: can a run tell there is no answer?\n")
    add("MLflow also logs the cosine similarity of the best match. For a future \"no relevant article\" threshold, "
        "in-scope questions should score clearly higher than out-of-scope ones; the gap is what such a threshold "
        "would have to work with.\n")
    add("| Run | Best-match similarity, in scope | Out of scope | Gap |")
    add("|---|---|---|---|")
    for r in runs:
        m = r["metrics"]
        i, o = m.get("top1_score_in_scope"), m.get("top1_score_out_of_scope")
        add(f"| {r['name']} | {_f(i)} | {_f(o)} | {_f(i - o) if i is not None and o is not None else '—'} |")
    add("")
    add(f"Also in MLflow, per run: `hit_at_{k}` and an `_ar` / `_en` version of every retrieval metric, "
        "`faithfulness_ar` / `faithfulness_en`, `chunks_truncated` (chunks longer than the model reads; 0 in every run), "
        "and timings.\n")

    # --- misses --------------------------------------------------------------------------------------------
    misses = [q for q in base["rows"] if q["relevant_articles"] and (first_rank(q) or 99) > k]
    if misses:
        others = [r for r in runs if r is not base]
        add(f"## Questions {base['name']} misses\n")
        add(f"The relevant article is not in its top {k}, so the model never sees it. Worth checking during the review: "
            "the question may be hard, or its relevant articles may be incomplete.\n")
        add(f"| Question | Relevant | Retrieved instead (top 3) | Found in the top {k} by |")
        add("|---|---|---|---|")
        for q in misses:
            found_by = [r["name"] for r in others
                        for o in r["rows"] if o["id"] == q["id"] and (first_rank(o) or 99) <= k]
            add(f"| `{q['id']}` {_cell(q['question'])} | {', '.join(map(str, q['relevant_articles']))} | "
                f"{', '.join(map(str, q['ranked'][:3]))} | "
                f"{len(found_by)} of {len(others)} other runs" + (f" ({', '.join(found_by)})" if found_by else "")
                + " |")
        add("")

    # --- caveats -------------------------------------------------------------------------------------------
    add("## Caveats\n")
    if draft:
        add("- **Draft questions.** The relevant articles and reference answers have not been reviewed by a legal "
            "reader yet.")
    elif accepted:
        add("- **Not legally reviewed.** The questions were accepted as the evaluation set by the project owner; their "
            "relevant articles and reference answers have not been checked by a legal expert.")
    add(f"- **Small sample.** {n} questions: differences of one or two questions ({step:.3f}–{2 * step:.3f}) are noise.")
    add("- **Articles named by number** are found by the number lookup, not the embedding, so every run gets those "
        "questions right.")
    if faith and self_judged:
        add(f"- **Self-judging.** `{answer_model}` both answers and judges faithfulness, which can be lenient with its "
            "own phrasing, and a 7B judge is less reliable than a larger one; read faithfulness as a comparison between "
            "runs, not as an absolute score.")
    add("")
    add("MLflow comparison view (screenshot): [mlflow_chunking_compare.png](mlflow_chunking_compare.png).")
    return "\n".join(L) + "\n"
