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


def load_llm_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["llm"]


def get_backend(params: dict[str, Any] | None = None, backend: str | None = None) -> LLMBackend:
    load_dotenv()
    params = params or load_llm_params()
    name = backend or params["backend"]
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
