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
from openai import OpenAI

from rag.llm.factory import load_llm_params
from rag.ui.data import CORPUS_PATH, FEEDBACK_PATH, SOURCE_PDF, load_articles

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
        return Stage("corpus", "Data", "Structured corpus", WORKING, f"{n} articles in data/processed/articles.json.")
    n = len(load_articles().articles)
    return Stage(
        "corpus", "Data", "Structured corpus", PLANNED,
        f"PDF to one JSON record per article, as a DVC stage with validation tests. "
        f"The console uses a {n}-article sample fixture meanwhile.",
        "TASKS: Corpus",
    )


def _retrieval() -> Stage:
    if importlib.util.find_spec("rag.retrieval"):
        return Stage("retrieval", "Retrieval", "Embeddings + vector search", WORKING, "rag.retrieval is installed.")
    return Stage(
        "retrieval", "Retrieval", "Embeddings + vector search", PLANNED,
        "Multilingual embeddings in Chroma, one chunk per article. Until then you pick the context articles by hand.",
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


_LIVE: list[Callable[[], Stage]] = [_source_pdf, _corpus, _retrieval, _bedrock, _vllm]


def collect_status() -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(_LIVE)) as pool:
        live = list(pool.map(lambda check: check(), _LIVE))
    return [asdict(s) for s in live + _static()]
