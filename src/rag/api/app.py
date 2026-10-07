"""Production Q&A API (rubric R02).

POST /ask         {"question": "..."} → {"answer": "...", "sources": ["Article 492", ...]}
POST /ask/stream  {"question": "..."} → the answer as plain text, token by token, then "Sources: ..."
GET  /health      → {"status": "healthy", "documents_indexed": 1149}

Handlers are async: retrieval runs in a worker thread and the LLM call is awaited, so one process serves
many questions at once.

`sources` are the articles the answer cites that were in its retrieved context (never chunk ids).
An empty or whitespace-only question is rejected by Pydantic with 422. The model is Qwen2.5 on vLLM
(params.yaml); LLM_BACKEND=bedrock switches to the optional Bedrock backend. Each answer is an `answer-question` Langfuse trace,
its id returned in the X-Trace-Id header.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, StringConstraints

from rag import tracing
from rag.llm import LLMError
from rag.pipeline import Pipeline, RagAnswer

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class AskRequest(BaseModel):
    question: Question


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    documents_indexed: int


def create_app(pipeline: Pipeline | None = None, *, warm_up: bool | None = None) -> FastAPI:
    pipeline = pipeline or Pipeline()
    if warm_up is None:
        warm_up = os.environ.get("RAG_WARMUP", "true").lower() != "false"

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if warm_up:
            try:
                pipeline.warm_up()  # embedding model loaded before the first question, not during it
            except Exception as e:  # noqa: BLE001 - /health reports it; the server still starts
                print(f"warm-up failed: {type(e).__name__}: {e}")
        yield
        tracing.client().flush()

    app = FastAPI(title="Egypt Law RAG", version="0.1.0", lifespan=lifespan,
                  description="Questions on the Egyptian Civil Code, answered with article citations.")
    release = os.environ.get("APP_RELEASE") or os.environ.get("LANGFUSE_RELEASE") or "dev"

    @app.middleware("http")
    async def release_header(request, call_next):
        """Which build answered: lets a canary rollout (deploy/canary) tell stable and canary traffic apart."""
        response = await call_next(request)
        response.headers["X-Release"] = release
        return response

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest, response: Response) -> AskResponse:
        try:
            result = await pipeline.aask(req.question, tags=["api"])
        except (LLMError, ValueError) as e:  # backend down, bad or missing key
            raise HTTPException(503, f"LLM unavailable: {e}") from e
        except Exception as e:  # noqa: BLE001 - index missing or unreadable
            raise HTTPException(503, f"retrieval unavailable: {type(e).__name__}: {e}") from e
        if result.trace_id:
            response.headers["X-Trace-Id"] = result.trace_id
        return AskResponse(answer=result.result.text, sources=result.sources)

    @app.post("/ask/stream")
    async def ask_stream(req: AskRequest) -> StreamingResponse:
        """The answer token by token (curl -N shows it arriving), then a "Sources:" line."""
        async def tokens() -> AsyncIterator[str]:
            try:
                async for item in pipeline.astream(req.question, tags=["api", "stream"]):
                    if isinstance(item, RagAnswer):
                        yield "\n\nSources: " + (", ".join(item.sources) or "none") + "\n"
                    else:
                        yield item
            except (LLMError, ValueError) as e:  # the status line is already sent: report in the body
                yield f"\n[error] LLM unavailable: {e}\n"
            except Exception as e:  # noqa: BLE001
                yield f"\n[error] {type(e).__name__}: {e}\n"

        return StreamingResponse(tokens(), media_type="text/plain; charset=utf-8")

    @app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
    async def health():
        n = await asyncio.to_thread(pipeline.documents_indexed)
        if not n:
            return JSONResponse({"status": "unhealthy", "documents_indexed": 0}, status_code=503)
        return HealthResponse(status="healthy", documents_indexed=n)

    return app


app = create_app()
