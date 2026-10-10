"""Build the configured backend from params.yaml + environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from rag.llm.base import LLMBackend
from rag.llm.openai_compat import OpenAICompatBackend

PARAMS_PATH = Path(__file__).resolve().parents[3] / "params.yaml"
ENV_FILE = PARAMS_PATH.parent / ".env"


def load_llm_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    """params.yaml `llm`, with VLLM_MODEL (environment or .env) choosing which model vLLM serves, e.g. the small
    Qwen/Qwen2.5-1.5B-Instruct-AWQ for GPUs with ~2 GB free. Every vLLM call (answers, judge, teacher) follows it."""
    params = yaml.safe_load(path.read_text(encoding="utf-8"))["llm"]
    load_dotenv(ENV_FILE)
    if os.environ.get("VLLM_MODEL"):
        params["vllm"]["model"] = os.environ["VLLM_MODEL"]
    return params


def backend_for(configured: str) -> str:
    """The one switch: LLM_BACKEND (environment or .env) overrides a configured backend, so setting it moves every
    LLM call (answers, RAGAS judge, question generation) to the same backend. Unset, the configured one is used."""
    load_dotenv(ENV_FILE)
    return os.environ.get("LLM_BACKEND") or configured


def default_backend(params: dict[str, Any] | None = None) -> str:
    """LLM_BACKEND > params.yaml llm.backend."""
    return backend_for((params or load_llm_params())["backend"])


def get_backend(params: dict[str, Any] | None = None, backend: str | None = None) -> LLMBackend:
    load_dotenv(ENV_FILE)
    params = params or load_llm_params()
    name = backend or default_backend(params)
    if name == "bedrock":
        key = os.environ.get("Bedrock_API_key")
        if not key:
            raise ValueError("Bedrock_API_key is not set (add it to .env)")
        cfg = params["bedrock"]
        base_url = os.environ.get("OPENAI_BASE_URL", cfg["base_url"])
        return OpenAICompatBackend("bedrock", cfg["model"], base_url, api_key=key)
    if name == "vllm":
        cfg = params["vllm"]
        return OpenAICompatBackend("vllm", cfg["model"], os.environ.get("VLLM_BASE_URL", cfg["base_url"]))
    raise ValueError(f"unknown llm backend {name!r} (expected 'bedrock' or 'vllm')")
