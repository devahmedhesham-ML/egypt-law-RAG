"""Chunking experiments: python -m rag.experiments.chunking [--faithfulness] [--only N ...]

For each config in params.yaml `experiments.chunking` (strategy, chunk_size, overlap, embedding model):
chunk the corpus → embed on the GPU → a throwaway Chroma index → retrieve for every question in
eval/questions.jsonl the way production does (articles named in the question first, then the best
chunk per article) → retrieval metrics → one MLflow run. With --faithfulness, Bedrock also answers
each in-scope question from its top-k articles and RAGAS judges how faithful the answer is to them.

View: mlflow ui --backend-store-uri sqlite:///mlflow.db  →  experiment "chunking", compare runs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from rag.ingest.chunks import build_chunks, chunks_per_article, query_text
from rag.ingest.embed import GpuEmbedder
from rag.ingest.store import best_per_article, collection_name, search, write_index
from rag.retrieval import merge_hits, numbers_in_question

REPO_ROOT = Path(__file__).resolve().parents[3]
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
EXPERIMENT = "chunking"
DEPTH = 10  # articles ranked per question (MRR, nDCG look this deep)


def run_name(cfg: dict) -> str:
    size = f"-{cfg['chunk_size']}o{cfg.get('overlap', 0)}" if cfg["strategy"] == "window" else ""
    return f"{cfg['strategy']}{size}__{cfg['model'].split('/')[-1]}"


# --- metrics --------------------------------------------------------------------------------
def question_metrics(ranked: list[int], relevant: list[int], k: int) -> dict:
    """Binary relevance metrics for one in-scope question."""
    rel = set(relevant)
    first = next((i + 1 for i, n in enumerate(ranked) if n in rel), None)
    dcg = sum(1 / math.log2(i + 2) for i, n in enumerate(ranked[:k]) if n in rel)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return {
        "hit_at_1": float(bool(ranked[:1]) and ranked[0] in rel),
        f"hit_at_{k}": float(any(n in rel for n in ranked[:k])),
        f"recall_at_{k}": len(rel & set(ranked[:k])) / len(rel),
        "mrr": 1 / first if first else 0.0,
        f"ndcg_at_{k}": dcg / ideal if ideal else 0.0,
    }


def aggregate(rows: list[dict], k: int) -> dict[str, float]:
    """Means overall and per language, plus how often the Arabic and English versions agree on the top article."""
    keys = ["hit_at_1", f"hit_at_{k}", f"recall_at_{k}", "mrr", f"ndcg_at_{k}"]
    scoped = [r for r in rows if r["relevant_articles"]]
    out: dict[str, float] = {}
    for key in keys:
        out[key] = statistics.fmean(r[key] for r in scoped)
        for lang in ("ar", "en"):
            vals = [r[key] for r in scoped if r["lang"] == lang]
            if vals:
                out[f"{key}_{lang}"] = statistics.fmean(vals)
    pairs: dict[str, dict[str, int]] = {}
    for r in scoped:
        if r["ranked"]:
            pairs.setdefault(r["pair"], {})[r["lang"]] = r["ranked"][0]
    both = [p for p in pairs.values() if len(p) == 2]
    if both:
        out["ar_en_top1_agreement"] = statistics.fmean(float(p["ar"] == p["en"]) for p in both)
    top_in = [r["top_score"] for r in scoped if r["top_score"] is not None]
    top_out = [r["top_score"] for r in rows if not r["relevant_articles"] and r["top_score"] is not None]
    if top_in and top_out:  # the gap a "no relevant article" threshold would need
        out["top1_score_in_scope"] = statistics.fmean(top_in)
        out["top1_score_out_of_scope"] = statistics.fmean(top_out)
    return out


# --- faithfulness (optional) -------------------------------------------------------------------
class Faithfulness:
    """Bedrock answers from the retrieved articles; RAGAS (judge: the same Bedrock model) scores faithfulness."""

    def __init__(self, articles: dict[int, dict], concurrency: int = 8) -> None:
        from rag.llm import get_backend
        from rag.llm.factory import load_llm_params

        self.params = load_llm_params()
        self.backend = get_backend(self.params, backend="bedrock")
        self.judge_model = self.params["bedrock"]["model"]
        self.articles = articles
        self.concurrency = concurrency
        self.cache: dict[tuple, tuple[str, float]] = {}

    def _metric(self):
        """RAGAS faithfulness judged by Bedrock. Instructor's TOOLS mode: RAGAS defaults to JSON mode, in which
        gpt-oss returns malformed JSON. Built per batch, so the async client never outlives its event loop."""
        import instructor
        from openai import AsyncOpenAI
        from ragas.llms.base import InstructorLLM
        from ragas.metrics.collections import Faithfulness as RagasFaithfulness

        client = AsyncOpenAI(api_key=os.environ["Bedrock_API_key"],
                             base_url=os.environ.get("OPENAI_BASE_URL", self.params["bedrock"]["base_url"]))
        llm = InstructorLLM(client=instructor.from_openai(client, mode=instructor.Mode.TOOLS), model=self.judge_model,
                            provider="openai", max_tokens=4096)
        return RagasFaithfulness(llm=llm)

    def _context(self, numbers: list[int]) -> list[str]:
        out = []
        for n in numbers:
            a = self.articles[n]
            body = "\n".join(x for x in (a.get("text_ar") or a.get("repeal_note_ar"),
                                         a.get("text_en") or a.get("repeal_note")) if x)
            out.append(f"[Article {n}]\n{body}")
        return out

    async def _one(self, metric, sem: asyncio.Semaphore, question: str, numbers: list[int]) -> tuple[str, float]:
        from rag.llm import answer

        from rag.llm import LLMError

        key = (question, tuple(numbers))
        if key not in self.cache:  # same question + same articles → same answer: judge it once
            async with sem:
                context = [self.articles[n] for n in numbers]
                text, score = "", float("nan")
                for attempt in range(3):  # the client already retries twice; this covers longer network blips
                    try:
                        result = await asyncio.to_thread(answer, self.backend, question, context,
                                                         max_tokens=self.params["max_tokens"], temperature=0.0)
                        text = result.text
                        break
                    except LLMError as e:
                        print(f"  answer failed (attempt {attempt + 1}/3) on {question[:40]!r}: {e}"[:300])
                        await asyncio.sleep(5 * (attempt + 1))
                if text.strip():
                    try:
                        score = (await metric.ascore(user_input=question, response=text,
                                                     retrieved_contexts=self._context(numbers))).value
                    except Exception as e:  # noqa: BLE001 - one failed judgement must not sink the run
                        print(f"  faithfulness failed on {question[:40]!r}: {type(e).__name__}: {e}"[:300])
                self.cache[key] = (text, score)
        return self.cache[key]

    def score(self, items: list[tuple[str, list[int]]]) -> list[tuple[str, float]]:
        async def run():
            metric, sem = self._metric(), asyncio.Semaphore(self.concurrency)
            return await asyncio.gather(*(self._one(metric, sem, q, nums) for q, nums in items))
        return asyncio.run(run())


# --- one config -------------------------------------------------------------------------------
def run_config(cfg: dict, records: list[dict], questions: list[dict], exp: dict, ingest: dict, *,
               faith: Faithfulness | None, console: Console) -> tuple[dict, list[dict], dict]:
    from transformers import AutoTokenizer

    k = exp["top_k"]
    model = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(model)
    chunks = build_chunks(records, ingest["normalize_arabic"], strategy=cfg["strategy"],
                          chunk_size=cfg.get("chunk_size"), overlap=cfg.get("overlap", 0), tokenizer=tokenizer)
    tokens = [len(ids) for ids in tokenizer([c.embed_text for c in chunks])["input_ids"]]
    max_len = min(ingest["max_chunk_tokens"], tokenizer.model_max_length or ingest["max_chunk_tokens"])
    embedder = GpuEmbedder(model, tokens_per_batch=ingest["tokens_per_batch"], max_replicas=1,
                           margin_gb=ingest["vram_margin_gb"], max_seq_len=max_len, log=console.print)
    queries = [query_text(q["question"], ingest["normalize_arabic"]) for q in questions]
    started = time.perf_counter()
    vectors, query_vectors, stats = embedder.run([c.embed_text for c in chunks], tokens, queries)
    index_dir = REPO_ROOT / exp["index_dir"] / run_name(cfg).lower()
    name = collection_name("civil_code", model)
    write_index(index_dir, name, chunks, vectors, {"embedding_model": model, "strategy": cfg["strategy"]})
    fanout = chunks_per_article(cfg["strategy"])
    rows = []
    for q, hits in zip(questions, search(index_dir, name, query_vectors, k=(DEPTH + 2) * fanout)):
        semantic = best_per_article(hits)
        named = numbers_in_question(q["question"])
        ranked = [h.article_number for h in merge_hits(named, semantic, DEPTH)]
        row = {**q, "ranked": ranked, "top_score": semantic[0]["score"] if semantic else None}
        if q["relevant_articles"]:
            row.update(question_metrics(ranked, q["relevant_articles"], k))
        rows.append(row)
    metrics = aggregate(rows, k)
    if faith is not None:
        scoped = [r for r in rows if r["relevant_articles"]]
        results = faith.score([(r["question"], r["ranked"][:k]) for r in scoped])
        for r, (text, score) in zip(scoped, results):
            r["answer"], r["faithfulness"] = text, score
        vals = [s for _, s in results if not math.isnan(s)]
        if vals:
            metrics["faithfulness"] = statistics.fmean(vals)
            for lang in ("ar", "en"):
                lv = [r["faithfulness"] for r in scoped if r["lang"] == lang and not math.isnan(r["faithfulness"])]
                if lv:
                    metrics[f"faithfulness_{lang}"] = statistics.fmean(lv)
        metrics["faithfulness_judged"] = len(vals)
    info = {
        "chunks": len(chunks), "chunk_tokens_mean": statistics.fmean(tokens), "chunk_tokens_max": max(tokens),
        "chunks_truncated": sum(t > max_len for t in tokens), "embed_s": stats.embed_s,
        "index_s": time.perf_counter() - started, "device": stats.device,
    }
    return metrics, rows, info


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    exp, ingest, corpus = params["experiments"], params["ingest"], params["corpus"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", type=int, nargs="*", help="run only these config indexes (0-based)")
    ap.add_argument("--faithfulness", action="store_true", help="also answer with Bedrock and score with RAGAS")
    ap.add_argument("--concurrency", type=int, default=8, help="parallel Bedrock requests for --faithfulness")
    ap.add_argument("--report", default="reports/chunking_experiments.md")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")  # hundreds of eval calls would flood the traces
    import mlflow

    console = Console()
    records = json.loads((REPO_ROOT / corpus["output"]).read_text(encoding="utf-8"))
    questions_path = REPO_ROOT / exp["questions"]
    questions = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    configs = [c for i, c in enumerate(exp["chunking"]) if args.only is None or i in args.only]
    faith = (Faithfulness({r["article_number"]: r for r in records}, concurrency=args.concurrency)
             if args.faithfulness else None)
    draft = sum(q.get("status") == "draft" for q in questions)
    if draft:
        console.print(f"[yellow]! {draft}/{len(questions)} questions are still drafts (eval/README.md): "
                      "treat these scores as provisional.[/]")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    for cfg in configs:
        name = run_name(cfg)
        console.rule(f"[bold]{name}")
        metrics, rows, info = run_config(cfg, records, questions, exp, ingest, faith=faith, console=console)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"strategy": cfg["strategy"], "chunk_size": cfg.get("chunk_size") or "whole article",
                               "overlap": cfg.get("overlap", 0), "embedding_model": cfg["model"],
                               "normalize_arabic": ingest["normalize_arabic"], "top_k": exp["top_k"],
                               "judge_model": faith.judge_model if faith else "none"})
            mlflow.log_metrics({**metrics, **{k: float(v) for k, v in info.items() if isinstance(v, int | float)}})
            mlflow.set_tags({"git_sha": git_sha(), "eval_questions": len(questions), "eval_draft": draft,
                             "eval_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest()[:12],
                             "device": info["device"]})
            mlflow.log_dict({"config": cfg, "questions": rows}, "per_question.json")
        console.print({k: round(v, 3) for k, v in metrics.items()})

    # The table and report cover the latest run of every config in the grid, including ones from earlier
    # invocations (so rerunning a single config with --only still yields the full comparison).
    k = exp["top_k"]
    summary = latest_runs(mlflow, exp["chunking"])
    cols = ["hit_at_1", f"recall_at_{k}", "mrr", f"ndcg_at_{k}", "hit_at_1_ar", "hit_at_1_en", "ar_en_top1_agreement"]
    if any("faithfulness" in m for _, _, m, _ in summary):
        cols += ["faithfulness", "faithfulness_judged"]
    t = Table(title=f"Chunking experiments ({len(questions)} questions, top-{k})", title_justify="left")
    fmt = {"faithfulness_judged": "{:.0f}"}
    for c in ["run", "chunks", *cols]:
        t.add_column(c, justify="right" if c != "run" else "left")
    for name, _, m, info in summary:
        t.add_row(name, str(info["chunks"]), *(fmt.get(c, "{:.3f}").format(m.get(c, float("nan"))) for c in cols))
    console.print(t)
    write_report(REPO_ROOT / args.report, summary, cols, len(questions), draft, k, "faithfulness" in cols,
                 scoped=sum(bool(q["relevant_articles"]) for q in questions))
    console.print(f"MLflow: mlflow ui --backend-store-uri {TRACKING_URI}  (experiment '{EXPERIMENT}')")
    return 0


def latest_runs(mlflow, grid: list[dict]) -> list[tuple[str, dict, dict, dict]]:
    """(name, config, metrics, info) for the latest finished MLflow run of each grid config, in grid order."""
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    runs = mlflow.search_runs([exp.experiment_id], filter_string="attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"], output_format="list")
    latest = {}
    for r in runs:
        latest.setdefault(r.info.run_name, r)
    out = []
    for cfg in grid:
        r = latest.get(run_name(cfg))
        if r is not None:
            m = dict(r.data.metrics)
            out.append((run_name(cfg), cfg, m, {"chunks": int(m.get("chunks", 0))}))
    return out


def write_report(path: Path, summary: list, cols: list[str], n: int, draft: int, k: int, faith: bool,
                 scoped: int = 0) -> None:
    head = "| Run | Strategy | Chunk size / overlap | Embedding model | Chunks | " + " | ".join(cols) + " |"
    lines = [
        "# Chunking experiments", "",
        f"Generated by `python -m rag.experiments.chunking{' --faithfulness' if faith else ''}` "
        f"(git {git_sha()}); every row is an MLflow run in `mlflow.db`, experiment `chunking`.",
        f"{n} questions from `eval/questions.jsonl`"
        + (f", **{draft} still drafts awaiting review, so these numbers are provisional**" if draft else "") + ".",
        f"Metrics are over the {scoped} in-scope questions; `@{k}` = the {k} articles sent to the model. Retrieval "
        "works as in production: articles named in the question first, then the best chunk per article. "
        f"With {scoped} questions one question moves hit@1 by {1 / max(scoped, 1):.3f}, so small gaps are noise.", "",
        head, "|" + "---|" * (5 + len(cols)),
    ]
    best = {c: max(m.get(c, float("-inf")) for _, _, m, _ in summary) for c in cols}
    for name, cfg, m, info in summary:
        size = f"{cfg['chunk_size']} / {cfg.get('overlap', 0)}" if cfg["strategy"] == "window" else "whole article"
        cells = []
        for c in cols:
            v = m.get(c)
            if c == "faithfulness_judged":
                cells.append("—" if v is None else f"{v:.0f}/{scoped}")
            else:
                cells.append("—" if v is None else (f"**{v:.3f}**" if v == best[c] else f"{v:.3f}"))
        lines.append(f"| {name} | {cfg['strategy']} | {size} | {cfg['model'].split('/')[-1]} | {info['chunks']} | "
                     + " | ".join(cells) + " |")
    lines += ["", "Best value per column in bold. Screenshot of the MLflow comparison view: "
              "[mlflow_chunking_compare.png](mlflow_chunking_compare.png)."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
