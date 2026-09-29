"""Production Q&A API (rubric R02).

POST /ask     {"question": "..."} → {"answer": "...", "sources": ["Article 492", ...]}
GET  /health  → {"status": "healthy", "documents_indexed": 1149}

`sources` are the articles the answer cites that were in its retrieved context (never chunk ids).
An empty or whitespace-only question is rejected by Pydantic with 422. The backend comes from
LLM_BACKEND (bedrock | vllm), else params.yaml. Each answer is an `answer-question` Langfuse trace,
its id returned in the X-Trace-Id header.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StringConstraints

from rag import tracing
from rag.llm import LLMError
from rag.pipeline import Pipeline

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

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, response: Response) -> AskResponse:
        try:
            result = pipeline.ask(req.question, tags=["api"])
        except (LLMError, ValueError) as e:  # backend down, bad or missing key
            raise HTTPException(503, f"LLM unavailable: {e}") from e
        except Exception as e:  # noqa: BLE001 - index missing or unreadable
            raise HTTPException(503, f"retrieval unavailable: {type(e).__name__}: {e}") from e
        if result.trace_id:
            response.headers["X-Trace-Id"] = result.trace_id
        return AskResponse(answer=result.result.text, sources=result.sources)

    @app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
    def health():
        n = pipeline.documents_indexed()
        if not n:
            return JSONResponse({"status": "unhealthy", "documents_indexed": 0}, status_code=503)
        return HealthResponse(status="healthy", documents_indexed=n)

    return app


app = create_app()
