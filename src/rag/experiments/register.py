"""Register the best chunking config in the MLflow Model Registry and promote it (alias "production").

    python -m rag.experiments.register [--dry-run]

Selection rule, applied to the latest run of every config in the "chunking" experiment:
1. keep the runs with the best recall@5 (the right article reaches the model);
2. of those, keep the ones within one question of the best hit@1 (smaller gaps are noise on this set);
3. pick the one that best separates in-scope from out-of-scope questions (top-1 similarity gap), which a
   future "no relevant article" threshold needs.
The registered model is a pyfunc: questions in, the production top-k article numbers out. It records the
config, the embedding model, the generative model (Qwen2.5 on vLLM) and the DVC hash of the index it expects,
and loads inside this project (the package installed with `pip install -e .`, the index from `dvc pull`).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow.pyfunc  # noqa: E402
import yaml  # noqa: E402

from rag.experiments.report import EXPERIMENT, latest_runs, run_name  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "civil-code-retrieval"
ALIAS = "production"


def select(runs: list[dict], n_questions: int) -> tuple[dict, list[str]]:
    """The chosen run and the reasoning, one line per step."""
    best_recall = max(r["metrics"]["recall_at_5"] for r in runs)
    step1 = [r for r in runs if r["metrics"]["recall_at_5"] >= best_recall - 1e-9]
    best_hit = max(r["metrics"]["hit_at_1"] for r in step1)
    step2 = [r for r in step1 if r["metrics"]["hit_at_1"] >= best_hit - 1 / n_questions - 1e-9]

    def gap(r):
        return r["metrics"].get("top1_score_in_scope", 0) - r["metrics"].get("top1_score_out_of_scope", 0)

    chosen = max(step2, key=gap)
    why = [
        f"best recall@5 = {best_recall:.3f}: {', '.join(r['name'] for r in step1)}",
        f"within one question of the best hit@1 ({best_hit:.3f}): {', '.join(r['name'] for r in step2)}",
        "largest in-scope vs out-of-scope similarity gap: "
        + ", ".join(f"{r['name']} {gap(r):.3f}" for r in sorted(step2, key=gap, reverse=True)),
    ]
    return chosen, why


class RetrievalModel(mlflow.pyfunc.PythonModel):
    """pyfunc body: a DataFrame or list of questions → the top-k article numbers production would retrieve."""

    def load_context(self, context) -> None:  # noqa: D401
        from rag.retrieval import Retriever

        cfg = yaml.safe_load(Path(context.artifacts["config"]).read_text(encoding="utf-8"))
        self.top_k = cfg["top_k"]
        self.retriever = Retriever()

    def predict(self, context, model_input, params=None):
        questions = list(model_input["question"]) if hasattr(model_input, "columns") else list(model_input)
        return [[h.article_number for h in self.retriever.retrieve(q, self.top_k).hits] for q in questions]


def _index_md5() -> str:
    lock = yaml.safe_load((REPO_ROOT / "dvc.lock").read_text(encoding="utf-8"))
    return next(o["md5"] for o in lock["stages"]["index"]["outs"] if o["path"] == "data/index/chroma")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="show the selection, register nothing")
    args = ap.parse_args(argv)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import json

    import mlflow
    import mlflow.pyfunc
    from mlflow import MlflowClient

    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    runs = latest_runs(mlflow, params["experiments"]["chunking"])
    questions = [json.loads(x) for x in (REPO_ROOT / params["experiments"]["questions"]).read_text(
        encoding="utf-8").splitlines() if x.strip()]
    n = sum(bool(q["relevant_articles"]) for q in questions)
    chosen, why = select(runs, n)
    print(f"Chosen: {chosen['name']}")
    for line in why:
        print(f"  · {line}")
    prod = {**(params["ingest"].get("chunking") or {"strategy": "article"}), "model": params["ingest"]["model"]}
    if chosen["cfg"]["strategy"] != prod["strategy"] or chosen["cfg"]["model"] != prod["model"]:
        print(f"! params.yaml ingest still uses {prod}: update it and run `dvc repro` before promoting.")
        return 1
    if args.dry_run:
        return 0

    exp_id = mlflow.get_experiment_by_name(EXPERIMENT).experiment_id
    run = mlflow.search_runs([exp_id], filter_string=f"attributes.run_name = '{chosen['name']}' and "
                             "attributes.status = 'FINISHED'", order_by=["attributes.start_time DESC"],
                             max_results=1, output_format="list")[0]
    config = {**chosen["cfg"], "chunk_size": chosen["cfg"].get("chunk_size"), "top_k": params["retrieval"]["top_k"],
              "normalize_arabic": params["ingest"]["normalize_arabic"], "llm": params["llm"]["vllm"]["model"],
              "index_dvc_md5": _index_md5(), "selected_because": why}
    cfg_path = REPO_ROOT / "data" / "experiments" / "production_config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    client = MlflowClient()
    # Log into the chunking run, which must end FINISHED whatever happens: an exception inside a run marks it
    # FAILED, and the selection above only considers finished runs.
    try:
        mlflow.start_run(run_id=run.info.run_id)
        info = mlflow.pyfunc.log_model(name="retrieval", python_model=RetrievalModel(),
                                       artifacts={"config": str(cfg_path)},
                                       # loads inside this project (pip install -e . and dvc pull for the index)
                                       pip_requirements=["egypt-law-rag", "sentence-transformers>=3.0",
                                                         "chromadb==1.5.9", "pyyaml>=6.0", "numpy"],
                                       registered_model_name=MODEL_NAME)
    finally:
        mlflow.end_run(status="FINISHED")  # the chunking run stays FINISHED whatever happened
    version = info.registered_model_version
    m = chosen["metrics"]
    client.update_registered_model(MODEL_NAME, description=(
        "Retrieval for Egyptian Civil Code Q&A: which chunking + embedding config production uses. Answers come "
        f"from {config['llm']} on vLLM. Chosen by rag.experiments.register (rule in its docstring)."))
    client.update_model_version(MODEL_NAME, version, description=(
        f"{chosen['name']}: hit@1 {m['hit_at_1']:.3f}, recall@5 {m['recall_at_5']:.3f}, MRR {m['mrr']:.3f}, "
        f"faithfulness {m.get('faithfulness', float('nan')):.3f}. " + " | ".join(why)))
    for key, value in {"strategy": chosen["cfg"]["strategy"], "embedding_model": chosen["cfg"]["model"],
                       "llm": config["llm"], "index_dvc_md5": config["index_dvc_md5"],
                       "chunking_run": run_name(chosen["cfg"])}.items():
        client.set_model_version_tag(MODEL_NAME, version, key, str(value))
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    print(f"Registered {MODEL_NAME} v{version} and set alias '{ALIAS}' (models:/{MODEL_NAME}@{ALIAS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
