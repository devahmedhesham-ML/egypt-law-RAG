"""RAGAS with all four metrics on the evaluation set (rubric R08 checklist).

    python -m rag.evaluation.ragas_eval [--label baseline] [--rerank | --no-rerank]
                                        [--answer-backend vllm|bedrock] [--judge-backend vllm|bedrock]

The production path answers every in-scope question (58): retrieve top 5 → answer, each one an `answer-question`
trace. The judge (Qwen2.5 on vLLM by default; LLM_BACKEND or the flags switch it to Bedrock) then scores:

- faithfulness: the answer's claims are supported by the retrieved articles
- answer relevancy: questions generated back from the answer resemble the one asked (embeddings: Qwen3-Embedding)
- context precision: the retrieved articles that support the reference answer are ranked high
- context recall: the reference answer's claims are found in the retrieved articles

plus judge-free retrieval numbers from the labelled article numbers: hit@1, MRR and recall@5.

Each run is one MLflow run in the `ragas` experiment (the trend across sessions) and writes reports/ragas_eval.md.
With tracing on, every question's scores are attached to its Langfuse trace. With PUSHGATEWAY_URL set, the means go to
the Prometheus Pushgateway that Grafana's faithfulness panel and alert read (deploy/monitoring).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from rag.evaluation.faithfulness import article_context, make_judge
from rag.evaluation.gate import _git_sha, load_questions
from rag.llm.factory import backend_for, load_llm_params

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "ragas"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
RETRIEVAL = ("hit_at_1", "mrr", "recall_at_5")


def retrieval_scores(retrieved: list[int], relevant: list[int]) -> dict[str, float]:
    """hit@1, reciprocal rank of the first relevant article, and the share of relevant articles in the top k."""
    rank = next((i for i, n in enumerate(retrieved, 1) if n in relevant), None)
    return {"hit_at_1": float(bool(retrieved) and retrieved[0] in relevant),
            "mrr": 1.0 / rank if rank else 0.0,
            "recall_at_5": len(set(retrieved[:5]) & set(relevant)) / len(relevant)}


def summarize(rows: list[dict]) -> dict[str, float]:
    """Mean of every metric overall and per language, ignoring failed judgements, plus how many were judged."""
    out: dict[str, float] = {}
    for name in METRICS + RETRIEVAL:
        for lang in (None, "ar", "en"):
            vals = [r[name] for r in rows if (lang is None or r["lang"] == lang) and not math.isnan(r[name])]
            if vals:
                out[name if lang is None else f"{name}_{lang}"] = statistics.fmean(vals)
        if name in METRICS:
            out[f"{name}_judged"] = float(sum(not math.isnan(r[name]) for r in rows))
    return out


class QueryEmbeddings:
    """RAGAS embeddings backed by the retriever's own query encoder (Qwen3-Embedding), already loaded."""

    def __new__(cls, encode):
        from ragas.embeddings import BaseRagasEmbedding

        class _Emb(BaseRagasEmbedding):
            def embed_text(self, text: str, **kwargs) -> list[float]:
                return encode(text)[0].tolist()

            async def aembed_text(self, text: str, **kwargs) -> list[float]:
                return await asyncio.to_thread(self.embed_text, text)

        return _Emb()


async def _judge(metrics: dict, q: dict, answer: str, contexts: list[str]) -> dict[str, float]:
    calls = {
        "faithfulness": lambda: metrics["faithfulness"].ascore(user_input=q["question"], response=answer,
                                                               retrieved_contexts=contexts),
        "answer_relevancy": lambda: metrics["answer_relevancy"].ascore(user_input=q["question"], response=answer),
        "context_precision": lambda: metrics["context_precision"].ascore(
            user_input=q["question"], reference=q["reference"], retrieved_contexts=contexts),
        "context_recall": lambda: metrics["context_recall"].ascore(
            user_input=q["question"], retrieved_contexts=contexts, reference=q["reference"]),
    }

    async def one(name):
        try:
            return float((await calls[name]()).value)
        except Exception as e:  # noqa: BLE001 - one failed judgement must not sink the run
            print(f"  {name} failed on {q['id']}: {type(e).__name__}: {e}"[:300])
            return float("nan")

    values = await asyncio.gather(*(one(n) for n in METRICS))
    return dict(zip(METRICS, values))


def run(questions: list[dict], *, answer_backend: str, judge_backend: str, concurrency: int,
        rerank: bool | None, session: str) -> dict:
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    from rag import tracing
    from rag.pipeline import Pipeline

    pipe = Pipeline(backend_name=answer_backend, rerank=rerank)
    pipe.warm_up()
    params = load_llm_params()
    started = time.perf_counter()

    async def main():
        llm, client = make_judge(params, judge_backend)
        relevancy = AnswerRelevancy(llm=llm, embeddings=QueryEmbeddings(pipe.retriever._load()))
        # The judge otherwise writes its back-generated question in English even for an Arabic answer, which halves
        # the cosine with the Arabic question (0.44 → 0.74 on q02-ar with this line).
        relevancy.prompt.instruction += " Write the question in the same language as the answer."
        metrics = {"faithfulness": Faithfulness(llm=llm), "answer_relevancy": relevancy,
                   "context_precision": ContextPrecision(llm=llm), "context_recall": ContextRecall(llm=llm)}
        sem = asyncio.Semaphore(concurrency)

        async def one(q):
            async with sem:
                row = {"id": q["id"], "lang": q["lang"], "question": q["question"], "relevant": q["relevant_articles"],
                       "retrieved": [], "answer": "", "trace_id": None, **{m: float("nan") for m in METRICS}}
                try:
                    res = await pipe.aask(q["question"], temperature=0.0, tags=["ragas-eval"], session_id=session)
                except Exception as e:  # noqa: BLE001
                    print(f"  answer failed on {q['id']}: {type(e).__name__}: {e}"[:300])
                    row.update(retrieval_scores([], q["relevant_articles"]))
                    return row
                row["retrieved"] = [a["article_number"] for a in res.context]
                row["answer"], row["trace_id"] = res.result.text, res.trace_id
                row.update(retrieval_scores(row["retrieved"], q["relevant_articles"]))
                if res.result.text.strip():
                    row.update(await _judge(metrics, q, res.result.text, [article_context(a) for a in res.context]))
                return row

        try:
            return await asyncio.gather(*(one(q) for q in questions))
        finally:
            await client.close()

    rows = asyncio.run(main())
    if tracing.enabled():  # each question's scores on its own answer-question trace
        lf = tracing.client()
        for r in rows:
            for name in METRICS:
                if r["trace_id"] and not math.isnan(r[name]):
                    lf.create_score(trace_id=r["trace_id"], name=name, value=r[name], data_type="NUMERIC",
                                    comment=f"RAGAS ({session})")
        lf.flush()
    return {"rows": rows, "metrics": summarize(rows), "answer_model": params[answer_backend]["model"],
            "judge_model": params[judge_backend]["model"], "rerank": pipe.rerank,
            "elapsed_s": time.perf_counter() - started}


def write_report(path: Path, result: dict, label: str) -> None:
    m, rows = result["metrics"], result["rows"]

    def f(key):
        return f"{m[key]:.3f}" if key in m else "–"

    lines = [
        "# RAGAS evaluation (all four metrics)", "",
        f"Generated by `python -m rag.evaluation.ragas_eval` on {datetime.now(UTC):%Y-%m-%d %H:%M} UTC, run "
        f"**{label}**: {len(rows)} in-scope questions, answers by `{result['answer_model']}`, judged by "
        f"`{result['judge_model']}`, re-ranker {'on' if result['rerank'] else 'off'}, {result['elapsed_s']:.0f} s. "
        "Every run is also in MLflow (experiment `ragas`).", "",
        "| Metric | All | Arabic | English | Judged |", "|---|---|---|---|---|",
    ]
    for name in METRICS:
        lines.append(f"| {name.replace('_', ' ')} | **{f(name)}** | {f(name + '_ar')} | {f(name + '_en')} | "
                     f"{m.get(name + '_judged', 0):.0f}/{len(rows)} |")
    for name in RETRIEVAL:
        lines.append(f"| {name.replace('_at_', '@').replace('_', ' ')} (exact, no judge) | **{f(name)}** | "
                     f"{f(name + '_ar')} | {f(name + '_en')} | {len(rows)}/{len(rows)} |")
    lines += ["", "What each metric means:", "",
              "- **faithfulness**: share of the answer's claims supported by the retrieved articles.",
              "- **answer relevancy**: how closely questions generated back from the answer match the question "
              "(cosine of Qwen3-Embedding vectors); low when the answer drifts or declines.",
              "- **context precision**: whether the retrieved articles that support the reference answer are "
              "ranked first (rank-weighted).",
              "- **context recall**: share of the reference answer's claims found in the retrieved articles.",
              "- **hit@1 / MRR / recall@5**: exact retrieval scores from the labelled article numbers.", "",
              "| Question | Relevant | Retrieved | Faith. | Relev. | Prec. | Recall |", "|---|---|---|---|---|---|---|"]

    def g(v):
        return "–" if math.isnan(v) else f"{v:.2f}"

    for r in rows:
        lines.append(f"| `{r['id']}` {r['question'][:70]} | {', '.join(map(str, r['relevant']))} | "
                     f"{', '.join(map(str, r['retrieved']))} | {g(r['faithfulness'])} | {g(r['answer_relevancy'])} | "
                     f"{g(r['context_precision'])} | {g(r['context_recall'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_mlflow(result: dict, label: str, args: argparse.Namespace) -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(EXPERIMENT)
    mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
    with mlflow.start_run(run_name=f"{label}-{datetime.now(UTC):%Y%m%d-%H%M}"):
        mlflow.log_params({"label": label, "n_questions": len(result["rows"]), "rerank": result["rerank"],
                           "answer_backend": args.answer_backend, "judge_backend": args.judge_backend})
        mlflow.log_metrics({**result["metrics"], "elapsed_s": result["elapsed_s"]})
        mlflow.set_tags({"answer_model": result["answer_model"], "judge_model": result["judge_model"],
                         "git_sha": _git_sha()})
        mlflow.log_dict({"rows": result["rows"]}, "ragas_questions.json")
        if args.report.exists():
            mlflow.log_artifact(str(args.report))


def push_metrics(result: dict, label: str, url: str, finished: float | None = None) -> None:
    """Means to the Prometheus Pushgateway (job `ragas_eval`, one group per label): Grafana's panel and alert."""
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    reg = CollectorRegistry()
    for name in METRICS + RETRIEVAL:
        if name in result["metrics"]:
            Gauge(f"rag_ragas_{name}", f"RAGAS {name} (mean over the evaluation set)", registry=reg).set(
                result["metrics"][name])
    Gauge("rag_ragas_last_run_timestamp_seconds", "When the last RAGAS run finished", registry=reg).set(
        finished or time.time())
    push_to_gateway(url, job="ragas_eval", grouping_key={"label": label}, registry=reg)


def push_latest(url: str) -> int:
    """Grafana from existing results: the latest MLflow run of every label, pushed to the Pushgateway."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}"))
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["start_time DESC"])
    for label, run in runs.groupby("params.label").head(1).set_index("params.label").iterrows():
        metrics = {c.removeprefix("metrics."): v for c, v in run.items() if c.startswith("metrics.") and v == v}
        push_metrics({"metrics": metrics}, label, url, finished=run["end_time"].timestamp())
        print(f"pushed {label}: faithfulness {metrics.get('faithfulness', float('nan')):.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    ev = params["evaluation"]
    ap = argparse.ArgumentParser(description="RAGAS with all four metrics on the evaluation set")
    ap.add_argument("--label", default="production", help="name of this run (MLflow, Grafana)")
    ap.add_argument("--answer-backend", default=backend_for(ev["answer_backend"]), choices=["vllm", "bedrock"])
    ap.add_argument("--judge-backend", default=backend_for(ev["judge_backend"]), choices=["vllm", "bedrock"])
    ap.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None,
                    help="force the re-ranker on or off (default: params.yaml retrieval.rerank)")
    ap.add_argument("--concurrency", type=int, default=ev["concurrency"])
    ap.add_argument("--limit", type=int, help="first N questions only (smoke runs)")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "reports" / "ragas_eval.md")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--push-latest", action="store_true",
                    help="push the latest MLflow run of each label to PUSHGATEWAY_URL without running anything")
    args = ap.parse_args(argv)
    if args.push_latest:
        return push_latest(os.environ.get("PUSHGATEWAY_URL", "http://localhost:9091"))

    questions = load_questions(REPO_ROOT / params["experiments"]["questions"], "all")[:args.limit]
    session = f"ragas-{args.label}-{datetime.now(UTC):%Y%m%d-%H%M}"
    result = run(questions, answer_backend=args.answer_backend, judge_backend=args.judge_backend,
                 concurrency=args.concurrency, rerank=args.rerank, session=session)
    write_report(args.report, result, args.label)
    if not args.no_mlflow:
        log_mlflow(result, args.label, args)
    if os.environ.get("PUSHGATEWAY_URL"):
        push_metrics(result, args.label, os.environ["PUSHGATEWAY_URL"])
    m = result["metrics"]
    print("  ".join(f"{k} {m[k]:.3f}" for k in METRICS + RETRIEVAL if k in m), f"({len(questions)} questions)")
    print(json.dumps({k: round(v, 4) for k, v in m.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
