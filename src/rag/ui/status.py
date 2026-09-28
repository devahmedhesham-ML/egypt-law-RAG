"""Live status of every pipeline stage, planned ones included.

When a stage lands, flip its entry here (or give it a live check) so testers see it in the console.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import httpx
import yaml
from openai import OpenAI

from rag.llm.factory import load_llm_params
from rag.ui.data import CORPUS_PATH, FEEDBACK_PATH, REPO_ROOT, SOURCE_PDF, load_articles, load_report

WORKING, OFFLINE, PLANNED = "working", "offline", "planned"


@dataclass(frozen=True)
class Stage:
    id: str
    group: str
    title: str
    state: str
    detail: str
    next_step: str = ""


def _source_pdf() -> Stage:
    if SOURCE_PDF.exists():
        return Stage("source_pdf", "Data", "Source PDF (DVC)", WORKING, "Civil Code PDF pulled from the public S3 remote.")
    return Stage("source_pdf", "Data", "Source PDF (DVC)", OFFLINE, "PDF not in the working tree.", "Run `dvc pull`.")


def _corpus() -> Stage:
    if CORPUS_PATH.exists():
        n = len(load_articles().articles)
        report = load_report() or {}
        warnings = report.get("counts", {}).get("warnings")
        extra = f" {warnings} build warnings to review (Corpus view)." if warnings else ""
        return Stage("corpus", "Data", "Structured corpus", WORKING,
                     f"{n:,} articles in data/processed/articles.json, bilingual numbered hierarchy.{extra}")
    n = len(load_articles().articles)
    return Stage(
        "corpus", "Data", "Structured corpus", PLANNED,
        f"PDF to one JSON record per article, as a DVC stage with validation tests. "
        f"The console uses a {n}-article sample fixture meanwhile.",
        "python -m rag.corpus.build",
    )


def _index() -> Stage:
    cfg = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))["ingest"]
    title = "Vector index (Chroma)"
    if not CORPUS_PATH.exists():
        return Stage("index", "Retrieval", title, PLANNED, "Needs the structured corpus first.",
                     "python -m rag.corpus.build, then python -m rag.ingest")
    from rag.ingest.store import collection_name, index_count

    count = index_count(REPO_ROOT / cfg["index_dir"], collection_name(cfg["collection"], cfg["model"]))
    corpus_n = len(load_articles().articles)
    if count is None:
        return Stage("index", "Retrieval", title, OFFLINE, "No index yet.", "python -m rag.ingest")
    if count != corpus_n:
        return Stage("index", "Retrieval", title, OFFLINE,
                     f"Index holds {count:,} articles, the corpus {corpus_n:,}: it is stale.", "python -m rag.ingest")
    return Stage("index", "Retrieval", title, WORKING,
                 f"{count:,} articles, one bilingual chunk each, embedded with {cfg['model']}.")


def _retrieval() -> Stage:
    if importlib.util.find_spec("rag.retrieval") and CORPUS_PATH.exists():
        return Stage("retrieval", "Retrieval", "Retrieval in the answer path", WORKING,
                     "Ask and Compare search the index for each question (top-k, articles named by number first). "
                     "The Retrieval view shows the ranked hits. BentoML /ask comes next.")
    return Stage(
        "retrieval", "Retrieval", "Retrieval in the answer path", PLANNED,
        "Search the index for each question and pass the top articles to the model. "
        "Until then you pick the context articles by hand in Ask and Compare.",
        "TASKS: Retrieval",
    )


def _bedrock() -> Stage:
    params = load_llm_params()["bedrock"]
    title = f"Bedrock · {params['model']}"
    key = os.environ.get("Bedrock_API_key")
    if not key:
        return Stage("llm_bedrock", "Generation", title, OFFLINE, "Bedrock_API_key is not set.", "Add it to .env.")
    base_url = os.environ.get("OPENAI_BASE_URL", params["base_url"])
    try:
        models = {m.id for m in OpenAI(api_key=key, base_url=base_url, timeout=6, max_retries=0).models.list()}
    except Exception as e:  # noqa: BLE001 - any failure means "offline", shown to the tester
        return Stage("llm_bedrock", "Generation", title, OFFLINE, f"Endpoint check failed: {e}", "Check the key and OPENAI_BASE_URL.")
    if params["model"] not in models:
        return Stage("llm_bedrock", "Generation", title, OFFLINE, f"{params['model']} not offered by the endpoint.")
    return Stage("llm_bedrock", "Generation", title, WORKING, f"OpenAI-compatible endpoint reachable ({base_url}).")


def _vllm() -> Stage:
    params = load_llm_params()["vllm"]
    title = f"vLLM · {params['model']}"
    base_url = os.environ.get("VLLM_BASE_URL", params["base_url"])
    try:
        served = [m["id"] for m in httpx.get(f"{base_url}/models", timeout=2).json()["data"]]
    except Exception:  # noqa: BLE001
        return Stage("llm_vllm", "Generation", title, OFFLINE, f"No server at {base_url}.", "Start it in WSL: scripts/serve_vllm.sh")
    if params["model"] not in served:
        return Stage("llm_vllm", "Generation", title, OFFLINE, f"Server is up but serves {served}, not {params['model']}.")
    return Stage("llm_vllm", "Generation", title, WORKING, f"Serving on {base_url}.")


def _static() -> list[Stage]:
    feedback_n = sum(1 for _ in FEEDBACK_PATH.open(encoding="utf-8")) if FEEDBACK_PATH.exists() else 0
    langfuse = "Keys found in .env; " if os.environ.get("LANGFUSE_PUBLIC_KEY") else "No keys in .env yet; "
    return [
        Stage("citations", "Generation", "Citation check", WORKING, "Flags any cited article that was not in the context."),
        Stage("streaming", "Generation", "Streaming answers", WORKING, "Token-by-token streaming with usage and latency."),
        Stage("serve", "Serving", "BentoML /ask", PLANNED,
              "Production API with streaming. The console talks to its own dev server meanwhile.", "TASKS: Serve"),
        Stage("tracing", "Observability", "Langfuse tracing", PLANNED,
              langfuse + "retriever and generation spans not wired yet.", "TASKS: Langfuse tracing"),
        Stage("feedback", "Observability", "Tester feedback", WORKING,
              f"{feedback_n} ratings recorded in data/feedback/feedback.jsonl."),
        Stage("mlflow", "Evaluation", "MLflow experiments", PLANNED,
              "Chunking experiments: chunk_size, overlap, embedding model, faithfulness.", "TASKS: MLflow"),
        Stage("ragas", "Evaluation", "RAGAS evaluation", PLANNED,
              "Faithfulness, answer relevancy, context precision/recall on 50+ questions.", "TASKS: RAGAS"),
        Stage("awq", "Optimization", "Own AWQ-4bit quantization", PLANNED,
              "Calibrated on Arabic articles; vLLM serves the official Qwen AWQ build meanwhile.", "TASKS: Optimize"),
        Stage("monitoring", "Observability", "Drift and cost monitoring", PLANNED,
              "Embedding drift (MMD, domain classifier) and token cost over time.", "TASKS: Monitor"),
    ]


_LIVE: list[Callable[[], Stage]] = [_source_pdf, _corpus, _index, _retrieval, _bedrock, _vllm]


def collect_status() -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(_LIVE)) as pool:
        live = list(pool.map(lambda check: check(), _LIVE))
    return [asdict(s) for s in live + _static()]
