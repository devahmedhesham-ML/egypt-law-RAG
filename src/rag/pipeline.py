"""Question → retrieved articles → cited answer: the RAG path shared by the API, evaluation and serving.

Traced as one `answer-question` trace (retrieve-articles, generate-answer, check-citations inside).
The test console keeps its own streaming path but uses the same Retriever and answer code.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import AsyncIterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml
from langfuse import propagate_attributes

from rag import tracing
from rag.llm import AnswerResult, LLMBackend, answer, answer_async, answer_stream_async, get_backend
from rag.llm.factory import default_backend, load_llm_params
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
                 backend: LLMBackend | None = None, backend_name: str | None = None,
                 rerank: bool | None = None, reranker=None) -> None:
        params = load_params()
        self.top_k: int = params["retrieval"]["top_k"]
        # Re-ranker: retrieve `rerank_candidates`, re-order them, keep the top k. RAG_RERANK (env) > params.yaml
        # retrieval.rerank; off by default because the API image is CPU-only (seconds per question there).
        env = os.environ.get("RAG_RERANK", "").strip().lower()
        default = env in ("1", "true", "yes") if env else params["retrieval"].get("rerank", False)
        self.rerank: bool = default if rerank is None else rerank
        self.rerank_candidates: int = params["retrieval"].get("rerank_candidates", 20)
        self._reranker = reranker
        self._reranker_lock = threading.Lock()  # concurrent requests must load it once
        self.llm_params = load_llm_params()
        self._retriever = retriever
        self._articles = articles
        self._backend = backend
        # LLM_BACKEND (environment or .env) > params.yaml llm.backend
        self.backend_name = backend_name or default_backend(self.llm_params)

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
        if self.rerank:
            _ = self.reranker

    @property
    def reranker(self):
        with self._reranker_lock:
            if self._reranker is None:
                from rag.rerank import Reranker

                self._reranker = Reranker()
        return self._reranker

    def retrieve(self, question: str, k: int | None = None) -> tuple[Retrieval, list[dict]]:
        """Top-k hits and the article records sent to the model (hits missing from the corpus are dropped).
        With the re-ranker on, the top `rerank_candidates` are re-ordered by it first."""
        k = k or self.top_k
        retrieval = self.retriever.retrieve(question, self.rerank_candidates if self.rerank else k)
        hits = [h for h in retrieval.hits if h.article_number in self.articles]
        if self.rerank:
            hits = self.reranker.rerank(question, hits, self.articles)[:k]
            retrieval = Retrieval(hits=hits, **{f: getattr(retrieval, f) for f in Retrieval.__dataclass_fields__
                                                if f != "hits"})
        context = [self.articles[h.article_number] for h in hits]
        return retrieval, context

    @contextmanager
    def _trace(self, question: str, k: int, tags: list[str] | None, session_id: str | None):
        """The `answer-question` root span with the trace's name, session, tags and metadata."""
        lf = tracing.client()
        with lf.start_as_current_observation(as_type="span", name="answer-question",
                                             input={"question": question}) as root, propagate_attributes(
            trace_name="answer-question", session_id=session_id,
            tags=[*(tags or []), self.backend_name, "retrieved-context"],
            metadata=tracing.str_metadata(backend=self.backend_name, context_mode="retrieve", top_k=k),
        ):
            yield root, (lf.get_current_trace_id() if tracing.enabled() else None)

    @staticmethod
    def _close(root, result: AnswerResult, context: list[dict]) -> None:
        root.update(output={"answer": result.text, "cited_articles": result.citations.cited},
                    metadata={"context_articles": [a["article_number"] for a in context]})

    def _gen_args(self, max_tokens: int | None, temperature: float | None) -> dict:
        return {"max_tokens": max_tokens or self.llm_params["max_tokens"],
                "temperature": self.llm_params["temperature"] if temperature is None else temperature}

    def ask(self, question: str, *, k: int | None = None, max_tokens: int | None = None,
            temperature: float | None = None, tags: list[str] | None = None,
            session_id: str | None = None) -> RagAnswer:
        """Retrieve, answer and check citations under one `answer-question` trace."""
        k = k or self.top_k
        with self._trace(question, k, tags, session_id) as (root, trace_id):
            try:
                retrieval, context = self.retrieve(question, k)
                result = answer(self.backend, question, context, **self._gen_args(max_tokens, temperature))
            except Exception as e:
                root.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
                raise
            self._close(root, result, context)
        return RagAnswer(question, result, retrieval, context, trace_id)

    async def aask(self, question: str, *, k: int | None = None, max_tokens: int | None = None,
                   temperature: float | None = None, tags: list[str] | None = None,
                   session_id: str | None = None) -> RagAnswer:
        """ask() for async servers: retrieval (CPU/GPU-bound) in a worker thread, the LLM call awaited."""
        k = k or self.top_k
        with self._trace(question, k, tags, session_id) as (root, trace_id):
            try:
                retrieval, context = await asyncio.to_thread(self.retrieve, question, k)
                result = await answer_async(self.backend, question, context, **self._gen_args(max_tokens, temperature))
            except Exception as e:
                root.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
                raise
            self._close(root, result, context)
        return RagAnswer(question, result, retrieval, context, trace_id)

    async def astream(self, question: str, *, k: int | None = None, max_tokens: int | None = None,
                      temperature: float | None = None, tags: list[str] | None = None,
                      session_id: str | None = None) -> AsyncIterator[str | RagAnswer]:
        """Yields the answer's text as it is generated, then one RagAnswer (sources, citations, trace id)."""
        k = k or self.top_k
        with self._trace(question, k, tags, session_id) as (root, trace_id):
            try:
                retrieval, context = await asyncio.to_thread(self.retrieve, question, k)
                async for item in answer_stream_async(self.backend, question, context,
                                                      **self._gen_args(max_tokens, temperature)):
                    if isinstance(item, AnswerResult):
                        self._close(root, item, context)
                        yield RagAnswer(question, item, retrieval, context, trace_id)
                    else:
                        yield item
            except (GeneratorExit, asyncio.CancelledError):
                root.update(level="WARNING", status_message="client disconnected before the answer finished")
                raise
            except Exception as e:
                root.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
                raise

    def documents_indexed(self) -> int | None:
        """Vectors in the Chroma collection (one per article), or None if the index is missing."""
        from rag.ingest.store import index_count

        return index_count(self.retriever.index_dir, self.retriever.collection)
