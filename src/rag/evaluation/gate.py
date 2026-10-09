"""CI quality gate: RAGAS faithfulness of the production system on the 20-question CI set.

    python -m rag.evaluation.gate [--min 0.75] [--judge-backend vllm|bedrock] [--report reports/faithfulness_gate.md]

Each `ci: true` question in eval/questions.jsonl goes through production retrieval (the Chroma index, top 5) and
the production model (Qwen2.5 on vLLM); RAGAS scores how faithful each answer is to the articles it was given.
Exits 1 when the mean faithfulness is below the threshold (params.yaml evaluation.gate_min_faithfulness), so the
CI job fails. Every run is also logged to the MLflow experiment "faithfulness-gate" (a trend across sessions).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from rag.evaluation.faithfulness import FaithfulnessScorer, summarize
from rag.llm.factory import backend_for

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "faithfulness-gate"


def load_questions(path: Path, subset: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [q for q in rows if q["relevant_articles"] and (subset == "all" or q.get(subset))]


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return os.environ.get("GITHUB_SHA", "unknown")[:7]


def run_gate(questions: list[dict], *, judge_backend: str, answer_backend: str, concurrency: int) -> dict:
    from rag.pipeline import Pipeline

    pipe = Pipeline(backend_name=answer_backend)
    started = time.perf_counter()
    retrieved = [[a["article_number"] for a in pipe.retrieve(q["question"])[1]] for q in questions]
    scorer = FaithfulnessScorer(pipe.articles, answer_backend=answer_backend, judge_backend=judge_backend,
                                concurrency=concurrency)
    judged = scorer.score([(q["question"], nums) for q, nums in zip(questions, retrieved)])
    rows = [{"id": q["id"], "lang": q["lang"], "question": q["question"], "relevant": q["relevant_articles"],
             "retrieved": nums, "answer": j.answer, "faithfulness": j.score}
            for q, nums, j in zip(questions, retrieved, judged)]
    return {"rows": rows, "metrics": summarize(rows), "answer_model": scorer.answer_model,
            "judge_model": scorer.judge_model, "elapsed_s": time.perf_counter() - started}


def write_report(path: Path, result: dict, threshold: float, passed: bool) -> None:
    m, rows = result["metrics"], result["rows"]
    score = m.get("faithfulness", float("nan"))
    lines = [
        "# Faithfulness quality gate\n",
        f"{'✅ PASSED' if passed else '❌ FAILED'}: mean RAGAS faithfulness **{score:.3f}** on {len(rows)} questions "
        f"(threshold {threshold:.2f}); {m.get('faithfulness_judged', 0):.0f} answers judged.\n",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `python -m rag.evaluation.gate` (git {_git_sha()}). "
        f"Answers: `{result['answer_model']}` from the production index's top 5 articles; judge: "
        f"`{result['judge_model']}`. Arabic {m.get('faithfulness_ar', float('nan')):.3f}, English "
        f"{m.get('faithfulness_en', float('nan')):.3f}. Took {result['elapsed_s']:.0f} s.\n",
        "Faithfulness = supported statements ÷ all statements in the answer (RAGAS): it checks that the answer sticks "
        "to the articles it was given, not that it is legally correct.\n",
        "| Question | Relevant | Retrieved (top 5) | Faithfulness | Answer (start) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        f = "—" if r["faithfulness"] != r["faithfulness"] else f"{r['faithfulness']:.2f}"
        ans = r["answer"][:120].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{r['id']}` {r['question'][:70].replace('|', '/')} | {', '.join(map(str, r['relevant']))} | "
                     f"{', '.join(map(str, r['retrieved']))} | {f} | {ans} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_mlflow(result: dict, threshold: float, passed: bool, subset: str) -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(EXPERIMENT)
    mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
    with mlflow.start_run(run_name=f"gate-{datetime.now(UTC):%Y%m%d-%H%M}"):
        mlflow.log_params({"questions": subset, "n_questions": len(result["rows"]), "threshold": threshold})
        mlflow.log_metrics({**result["metrics"], "passed": float(passed), "elapsed_s": result["elapsed_s"]})
        mlflow.set_tags({"answer_model": result["answer_model"], "judge_model": result["judge_model"],
                         "git_sha": _git_sha(), "ci": os.environ.get("GITHUB_ACTIONS", "false")})
        mlflow.log_dict({"rows": result["rows"]}, "gate_questions.json")


def main(argv: list[str] | None = None) -> int:
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    ev = params["evaluation"]
    ap = argparse.ArgumentParser(description="RAGAS faithfulness quality gate")
    ap.add_argument("--min", type=float, default=ev["gate_min_faithfulness"], help="fail below this mean")
    ap.add_argument("--questions", default=ev["gate_questions"], help="question flag to select (ci), or 'all'")
    ap.add_argument("--answer-backend", default=backend_for(ev["answer_backend"]), choices=["vllm", "bedrock"])
    ap.add_argument("--judge-backend", default=backend_for(ev["judge_backend"]), choices=["vllm", "bedrock"])
    ap.add_argument("--concurrency", type=int, default=ev["concurrency"])
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "reports" / "faithfulness_gate.md")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
    questions = load_questions(REPO_ROOT / params["experiments"]["questions"], args.questions)
    result = run_gate(questions, judge_backend=args.judge_backend, answer_backend=args.answer_backend,
                      concurrency=args.concurrency)
    score = result["metrics"].get("faithfulness", float("nan"))
    judged = result["metrics"].get("faithfulness_judged", 0)
    passed = judged >= len(questions) * 0.8 and score >= args.min  # too many unjudged answers is a failure too
    write_report(args.report, result, args.min, passed)
    if not args.no_mlflow:
        log_mlflow(result, args.min, passed, args.questions)
    print(f"faithfulness {score:.3f} on {len(questions)} questions ({judged:.0f} judged), threshold {args.min:.2f}: "
          f"{'PASSED' if passed else 'FAILED'} — {args.report}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
