"""Question → retrieved articles → cited answer: the RAG path shared by the API, evaluation and serving.

Traced as one `answer-question` trace (retrieve-articles, generate-answer, check-citations inside).
The test console keeps its own streaming path but uses the same Retriever and answer code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from langfuse import propagate_attributes

from rag import tracing
from rag.llm import AnswerResult, LLMBackend, answer, get_backend
from rag.llm.factory import load_llm_params
from rag.retrieval import Retrieval, Retriever

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_params() -> dict:
    return yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))


def load_corpus(path: Path | None = None) -> dict[int, dict]:
    """articles.json as {article_number: record}."""
    path = path or REPO_ROOT / load_params()["corpus"]["output"]
    return {a["article_number"]: a for a in json.loads(path.read_text(encoding="utf-8"))}


@dataclass(frozen=True)
class RagAnswer:
    question: str
    result: AnswerResult
    retrieval: Retrieval
    context: list[dict]
    trace_id: str | None = None

    @property
    def sources(self) -> list[str]:
        """Articles the answer cites that were in its context, in citation order: 'Article 492'."""
        return [f"Article {n}" for n in self.result.citations.valid]


class Pipeline:
    """Holds the corpus, retriever and LLM backend; everything loads lazily on first use."""

    def __init__(self, *, retriever: Retriever | None = None, articles: dict[int, dict] | None = None,
                 backend: LLMBackend | None = None, backend_name: str | None = None) -> None:
        params = load_params()
        self.top_k: int = params["retrieval"]["top_k"]
        self.llm_params = load_llm_params()
        self._retriever = retriever
        self._articles = articles
        self._backend = backend
        # LLM_BACKEND (env) > params.yaml llm.backend
        self.backend_name = backend_name or os.environ.get("LLM_BACKEND") or self.llm_params["backend"]

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever()
        return self._retriever

    @property
    def articles(self) -> dict[int, dict]:
        if self._articles is None:
            self._articles = load_corpus()
        return self._articles

    @property
    def backend(self) -> LLMBackend:
        if self._backend is None:
            self._backend = get_backend(self.llm_params, backend=self.backend_name)
        return self._backend

    def warm_up(self) -> None:
        """Load the corpus and the embedding model now, so the first question is not slow."""
        _ = self.articles
        self.retriever._load()

    def retrieve(self, question: str, k: int | None = None) -> tuple[Retrieval, list[dict]]:
        """Top-k hits and the article records sent to the model (hits missing from the corpus are dropped)."""
        retrieval = self.retriever.retrieve(question, k or self.top_k)
        context = [self.articles[h.article_number] for h in retrieval.hits if h.article_number in self.articles]
        return retrieval, context

    def ask(self, question: str, *, k: int | None = None, max_tokens: int | None = None,
            temperature: float | None = None, tags: list[str] | None = None,
            session_id: str | None = None) -> RagAnswer:
        """Retrieve, answer and check citations under one `answer-question` trace."""
        lf = tracing.client()
        k = k or self.top_k
        with lf.start_as_current_observation(as_type="span", name="answer-question",
                                             input={"question": question}) as root, propagate_attributes(
            trace_name="answer-question", session_id=session_id,
            tags=[*(tags or []), self.backend_name, "retrieved-context"],
            metadata=tracing.str_metadata(backend=self.backend_name, context_mode="retrieve", top_k=k),
        ):
            try:
                retrieval, context = self.retrieve(question, k)
                result = answer(self.backend, question, context,
                                max_tokens=max_tokens or self.llm_params["max_tokens"],
                                temperature=self.llm_params["temperature"] if temperature is None else temperature)
            except Exception as e:
                root.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
                raise
            root.update(output={"answer": result.text, "cited_articles": result.citations.cited},
                        metadata={"context_articles": [a["article_number"] for a in context]})
            trace_id = lf.get_current_trace_id() if tracing.enabled() else None
        return RagAnswer(question, result, retrieval, context, trace_id)

    def documents_indexed(self) -> int | None:
        """Vectors in the Chroma collection (one per article), or None if the index is missing."""
        from rag.ingest.store import index_count

        return index_count(self.retriever.index_dir, self.retriever.collection)
