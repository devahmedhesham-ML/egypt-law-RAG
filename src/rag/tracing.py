"""Langfuse client for the whole app: one place for configuration and trace conventions.

Conventions (treat names like an API: dashboards and evaluators match on them):
- Traces: answer-question, search-articles, index-corpus, build-corpus.
- Observations: retrieve-articles (retriever), embed-question / embed-chunks (embedding),
  generate-answer (generation), check-citations (evaluator), plus pipeline spans.
- Scores: citations_outside_context, answer_has_citations, tester_rating, tester_issue,
  smoke_top1_rate, corpus_warnings, build_warnings.

Keys come from .env (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL). Without keys,
or with LANGFUSE_TRACING_ENABLED=false (the tests), every call is a no-op.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_release() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:  # noqa: BLE001 - no git: no release tag
        return "unknown"


@lru_cache
def client():
    """The process-wide Langfuse client (environment and release set before it is created)."""
    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")
    os.environ.setdefault("LANGFUSE_RELEASE", _git_release())
    os.environ.setdefault("OTEL_SERVICE_NAME", "egypt-law-rag")
    from langfuse import get_client

    return get_client()


def enabled() -> bool:
    load_dotenv(REPO_ROOT / ".env")
    return (os.environ.get("LANGFUSE_TRACING_ENABLED", "true").lower() != "false"
            and bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(os.environ.get("LANGFUSE_SECRET_KEY")))


def trace_link(trace_id: str | None) -> str | None:
    if not trace_id or not enabled():
        return None
    try:
        return client().get_trace_url(trace_id=trace_id)
    except Exception:  # noqa: BLE001 - a missing link must never break a request
        return None


def str_metadata(**values) -> dict[str, str]:
    """propagate_attributes() takes string values only."""
    return {k: str(v) for k, v in values.items() if v is not None}
