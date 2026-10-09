"""Chunking experiments: python -m rag.experiments.chunking [--faithfulness] [--only N ...]

For each config in params.yaml `experiments.chunking` (strategy, chunk_size, overlap, embedding model):
chunk the corpus → embed on the GPU → a throwaway Chroma index → retrieve for every question in
eval/questions.jsonl the way production does (articles named in the question first, then the best
chunk per article) → retrieval metrics → one MLflow run. With --faithfulness, the production model (Qwen2.5 on
vLLM) also answers each in-scope question from its top-k articles and RAGAS judges how faithful the answer is
to them (judge: Qwen2.5 by default, Bedrock optional).

View: mlflow ui --backend-store-uri sqlite:///mlflow.db  →  experiment "chunking", compare runs.
"""

from __future__ import annotations

import argparse
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

from rag.evaluation.faithfulness import FaithfulnessScorer, summarize
from rag.experiments.report import EXPERIMENT, build_report, latest_runs, production_config, run_name
from rag.ingest.chunks import build_chunks, chunks_per_article, query_text
from rag.ingest.embed import GpuEmbedder
from rag.ingest.store import best_per_article, collection_name, search, write_index
from rag.llm.factory import backend_for
from rag.retrieval import merge_hits, numbers_in_question

REPO_ROOT = Path(__file__).resolve().parents[3]
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
DEPTH = 10  # articles ranked per question (MRR, nDCG look this deep)


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


# --- one config -------------------------------------------------------------------------------
def embed_and_index(cfg: dict, records: list[dict], questions: list[dict], exp: dict, ingest: dict, *,
                    console: Console) -> dict:
    """Chunk → embed on the GPU (utilization sampled) → throwaway Chroma index → cost and speed measurements."""
    from transformers import AutoTokenizer

    from rag.experiments import perf

    model = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(model)
    chunks = build_chunks(records, ingest["normalize_arabic"], strategy=cfg["strategy"],
                          chunk_size=cfg.get("chunk_size"), overlap=cfg.get("overlap", 0), tokenizer=tokenizer)
    tokens = [len(ids) for ids in tokenizer([c.embed_text for c in chunks])["input_ids"]]
    max_len = min(ingest["max_chunk_tokens"], tokenizer.model_max_length or ingest["max_chunk_tokens"])
    embedder = GpuEmbedder(model, tokens_per_batch=ingest["tokens_per_batch"], max_replicas=1,
                           margin_gb=ingest["vram_margin_gb"], max_seq_len=max_len, log=console.print)
    queries = [query_text(q["question"], ingest["normalize_arabic"]) for q in questions]
    with perf.GpuSampler() as sampler:
        sampler.idle(1.0)
        vectors, query_vectors, stats = embedder.run([c.embed_text for c in chunks], tokens, queries)
    index_dir = REPO_ROOT / exp["index_dir"] / run_name(cfg).lower()
    name = collection_name("civil_code", model)
    started = time.perf_counter()
    write_index(index_dir, name, chunks, vectors, {"embedding_model": model, "strategy": cfg["strategy"]})
    fanout = chunks_per_article(cfg["strategy"])

    # cost and speed: reports/chunking_experiments.md, section "Embedding cost and speed"
    console.print("Measuring query latency (GPU, CPU) and search latency…")
    measured = {
        "chunks": len(chunks), "chunk_tokens_mean": statistics.fmean(tokens), "chunk_tokens_max": max(tokens),
        "chunks_truncated": sum(t > max_len for t in tokens),
        "embed_dim": int(vectors.shape[1]), "vectors_mb": vectors.shape[0] * vectors.shape[1] * 4 / perf.MB,
        "index_disk_mb": perf.dir_size_mb(index_dir), "index_write_s": time.perf_counter() - started,
        "embed_s": stats.embed_s, "embed_load_s": stats.load_s, "embed_chunks_per_s": stats.texts_per_s,
        "embed_tokens_per_s": stats.tokens_per_s, "embed_ms_per_chunk": 1000 * stats.embed_s / len(chunks),
        "embed_replicas": stats.replicas, "embed_batches": stats.batches,
        "vram_replica_mb": stats.per_replica_mb, "vram_total_mb": stats.total_mb,
        **sampler.last(stats.embed_s),
        **perf.query_latency(model, queries, "cuda", max_len),
        **perf.query_latency(model, queries, "cpu", max_len),
        **perf.search_latency(index_dir, name, query_vectors, k=(DEPTH + 2) * fanout),
    }
    return {"chunks": chunks, "query_vectors": query_vectors, "index_dir": index_dir, "name": name,
            "fanout": fanout, "device": stats.device, "measured": measured}


def add_faithfulness(rows: list[dict], faith: FaithfulnessScorer, k: int) -> dict[str, float]:
    """Answer and judge every in-scope row from its top-k ranked articles; fills row answer/faithfulness."""
    scoped = [r for r in rows if r["relevant_articles"]]
    for r, judged in zip(scoped, faith.score([(r["question"], r["ranked"][:k]) for r in scoped])):
        r["answer"], r["faithfulness"] = judged.answer, judged.score
    return summarize(scoped)


def run_config(cfg: dict, records: list[dict], questions: list[dict], exp: dict, ingest: dict, *,
               faith: FaithfulnessScorer | None, console: Console) -> tuple[dict, list[dict], dict]:
    k = exp["top_k"]
    built = embed_and_index(cfg, records, questions, exp, ingest, console=console)
    index_dir, name, fanout = built["index_dir"], built["name"], built["fanout"]
    rows = []
    for q, hits in zip(questions, search(index_dir, name, built["query_vectors"], k=(DEPTH + 2) * fanout)):
        semantic = best_per_article(hits)
        named = numbers_in_question(q["question"])
        ranked = [h.article_number for h in merge_hits(named, semantic, DEPTH)]
        row = {**q, "ranked": ranked, "top_score": semantic[0]["score"] if semantic else None}
        if q["relevant_articles"]:
            row.update(question_metrics(ranked, q["relevant_articles"], k))
        rows.append(row)
    metrics = aggregate(rows, k)
    if faith is not None:
        metrics.update(add_faithfulness(rows, faith, k))
    return metrics, rows, {**built["measured"], "device": built["device"]}


def measure_existing(mlflow, cfg: dict, records: list[dict], questions: list[dict], exp: dict, ingest: dict, *,
                     console: Console) -> bool:
    """--perf-only: re-embed a config with the measurements on and add them to its latest MLflow run.

    Retrieval metrics and faithfulness in that run are left as they were (embedding the same corpus with the
    same model gives the same vectors up to floating-point noise, so the retrieval results stand).
    """
    from rag.experiments import perf

    run = latest_run(mlflow, cfg)
    if run is None:
        console.print(f"[yellow]! no finished run for {run_name(cfg)}: run it first[/]")
        return False
    built = embed_and_index(cfg, records, questions, exp, ingest, console=console)
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_metrics({k: float(v) for k, v in built["measured"].items()})
        mlflow.set_tags({**perf.hardware_info(), "perf_measured_at": time.strftime("%Y-%m-%d"),
                         "perf_git_sha": git_sha()})
    return True


def latest_run(mlflow, cfg: dict):
    exp_id = mlflow.get_experiment_by_name(EXPERIMENT).experiment_id
    runs = mlflow.search_runs([exp_id], filter_string=f"attributes.run_name = '{run_name(cfg)}' and "
                              "attributes.status = 'FINISHED'", order_by=["attributes.start_time DESC"],
                              max_results=1, output_format="list")
    return runs[0] if runs else None


def faithfulness_existing(mlflow, cfg: dict, faith: FaithfulnessScorer, k: int, *, console: Console) -> bool:
    """--faithfulness-only: answer + judge from the rankings stored in a config's latest run, and log the scores
    into that run. Retrieval is not redone: the answers depend only on which articles were retrieved."""
    run = latest_run(mlflow, cfg)
    if run is None:
        console.print(f"[yellow]! no finished run for {run_name(cfg)}: run it first[/]")
        return False
    data = mlflow.artifacts.load_dict(f"runs:/{run.info.run_id}/per_question.json")
    metrics = add_faithfulness(data["questions"], faith, k)
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"answer_model": faith.answer_model, "judge_model": faith.judge_model,
                         "faithfulness_measured_at": time.strftime("%Y-%m-%d")})
        mlflow.log_dict(data, "per_question.json")
    console.print({key: round(v, 3) for key, v in metrics.items()})
    return True


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    exp, ingest, corpus, ev = params["experiments"], params["ingest"], params["corpus"], params["evaluation"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", type=int, nargs="*", help="run only these config indexes (0-based)")
    ap.add_argument("--faithfulness", action="store_true", help="also answer and score RAGAS faithfulness")
    ap.add_argument("--faithfulness-only", action="store_true",
                    help="answer + judge from the rankings in each config's latest run (no re-embedding)")
    ap.add_argument("--answer-backend", default=backend_for(ev["answer_backend"]), choices=["vllm", "bedrock"])
    ap.add_argument("--judge-backend", default=backend_for(ev["judge_backend"]), choices=["vllm", "bedrock"])
    ap.add_argument("--concurrency", type=int, default=ev["concurrency"], help="parallel answer/judge requests")
    ap.add_argument("--report", default="reports/chunking_experiments.md")
    ap.add_argument("--report-only", action="store_true", help="rebuild the report from MLflow without running anything")
    ap.add_argument("--perf-only", action="store_true",
                    help="measure embedding cost/speed/hardware and add it to each config's latest run")
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
    configs = [c for i, c in enumerate(exp["chunking"]) if (args.only is None or i in args.only) and not args.report_only]
    want_faith = (args.faithfulness or args.faithfulness_only) and configs and not args.perf_only
    faith = (FaithfulnessScorer({r["article_number"]: r for r in records}, answer_backend=args.answer_backend,
                                judge_backend=args.judge_backend, concurrency=args.concurrency)
             if want_faith else None)
    draft = sum(q.get("status") == "draft" for q in questions)
    if draft and not args.report_only:
        console.print(f"[yellow]! {draft}/{len(questions)} questions are still drafts (eval/README.md): "
                      "treat these scores as provisional.[/]")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    # Classic runs view (table, charts, Compare): without this tag MLflow 3 may show the GenAI evaluation layout,
    # whose Overview expects MLflow Tracing (we trace with Langfuse) and stays empty.
    mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
    from rag.experiments.perf import hardware_info

    for cfg in configs:
        name = run_name(cfg)
        console.rule(f"[bold]{name}")
        if args.perf_only:
            measure_existing(mlflow, cfg, records, questions, exp, ingest, console=console)
            continue
        if args.faithfulness_only:
            faithfulness_existing(mlflow, cfg, faith, exp["top_k"], console=console)
            continue
        metrics, rows, info = run_config(cfg, records, questions, exp, ingest, faith=faith, console=console)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"strategy": cfg["strategy"], "chunk_size": cfg.get("chunk_size") or "whole article",
                               "overlap": cfg.get("overlap", 0), "embedding_model": cfg["model"],
                               "normalize_arabic": ingest["normalize_arabic"], "top_k": exp["top_k"]})
            mlflow.log_metrics({**metrics, **{k: float(v) for k, v in info.items() if isinstance(v, int | float)}})
            mlflow.set_tags({"git_sha": git_sha(), "eval_questions": len(questions), "eval_draft": draft,
                             "eval_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest()[:12],
                             "device": info["device"], **hardware_info(), "perf_measured_at": time.strftime("%Y-%m-%d"),
                             **({"answer_model": faith.answer_model, "judge_model": faith.judge_model,
                                 "faithfulness_measured_at": time.strftime("%Y-%m-%d")} if faith else {})})
            mlflow.log_dict({"config": cfg, "questions": rows}, "per_question.json")
        console.print({k: round(v, 3) for k, v in metrics.items()})

    # The table and report cover the latest run of every config in the grid, including ones from earlier
    # invocations (so rerunning a single config with --only still yields the full comparison).
    k = exp["top_k"]
    runs = latest_runs(mlflow, exp["chunking"])
    cols = ["hit_at_1", f"recall_at_{k}", "mrr", f"ndcg_at_{k}", "ar_en_top1_agreement", "faithfulness"]
    t = Table(title=f"Chunking experiments ({len(questions)} questions, top-{k})", title_justify="left")
    for c in ["run", "chunks", *cols]:
        t.add_column(c, justify="right" if c != "run" else "left")
    for r in runs:
        m = r["metrics"]
        t.add_row(r["name"], f"{m.get('chunks', 0):.0f}", *(f"{m.get(c, float('nan')):.3f}" for c in cols))
    console.print(t)
    command = "python -m rag.experiments.chunking" + (" --faithfulness" if any(
        "faithfulness" in r["metrics"] for r in runs) else "")
    report = build_report(runs, questions, production_config(exp["chunking"], ingest), k, git_sha(), command)
    (REPO_ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / args.report).write_text(report, encoding="utf-8")
    console.print(f"Report: {args.report}")
    console.print(f"MLflow: mlflow ui --backend-store-uri {TRACKING_URI}  (experiment '{EXPERIMENT}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
