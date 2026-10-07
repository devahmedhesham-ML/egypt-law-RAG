"""BentoML service wrapping the RAG pipeline (rubric R07): vLLM serves Qwen2.5, BentoML serves the API.

    bentoml serve rag.serving.service:RagService --port 3000

POST /ask         {"question": "..."} → {"answer": "...", "sources": ["Article 492", ...]}   (async)
POST /ask_stream  {"question": "..."} → the answer token by token (curl -N), then "Sources: ..."
GET  /readyz, /livez, /healthz         BentoML's own probes; /metrics for Prometheus

Both endpoints are async: retrieval runs in a worker thread and the call to vLLM is awaited, so one worker
serves many requests at once. The same pipeline as the FastAPI app (rag.pipeline), so answers are identical.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

import bentoml
from pydantic import StringConstraints

from rag.llm import LLMError

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


@bentoml.service(
    name="egypt_law_rag",
    traffic={"timeout": 300, "concurrency": 128},  # requests wait on vLLM, not on this process
    workers=1,  # one copy of the embedding model; async handlers give the concurrency
)
class RagService:
    def __init__(self) -> None:
        from rag.pipeline import Pipeline

        self.pipeline = Pipeline()
        self.pipeline.warm_up()  # embedding model loaded before the first request

    @bentoml.api
    async def ask(self, question: Question) -> dict:
        try:
            result = await self.pipeline.aask(question, tags=["bentoml"])
        except (LLMError, ValueError) as e:
            raise bentoml.exceptions.ServiceUnavailable(f"LLM unavailable: {e}") from e
        return {"answer": result.result.text, "sources": result.sources}

    @bentoml.api
    async def ask_stream(self, question: Question) -> AsyncGenerator[str, None]:
        from rag.pipeline import RagAnswer

        try:
            async for item in self.pipeline.astream(question, tags=["bentoml", "stream"]):
                if isinstance(item, RagAnswer):
                    yield "\n\nSources: " + (", ".join(item.sources) or "none") + "\n"
                else:
                    yield item
        except (LLMError, ValueError) as e:  # headers are already sent: report in the body
            yield f"\n[error] LLM unavailable: {e}\n"
