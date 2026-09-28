"""Dev server for the test console: JSON API + static single-page UI."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.llm import AnswerResult, LLMBackend, LLMError, answer_stream, get_backend
from rag.llm.factory import load_llm_params
from rag.ui import data
from rag.ui.status import collect_status

STATIC = Path(__file__).parent / "static"

load_dotenv()
app = FastAPI(title="Egypt Law RAG · test console")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@lru_cache
def backend_for(name: str) -> LLMBackend:
    return get_backend(backend=name)


@lru_cache
def get_retriever():
    from rag.retrieval import Retriever

    return Retriever()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    backend: Literal["bedrock", "vllm"]
    article_numbers: list[int] = Field(default_factory=list, max_length=50)
    retrieve: bool = False  # True: search the index for the context instead of article_numbers
    top_k: int = Field(5, ge=1, le=20)
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(1024, ge=64, le=4096)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(10, ge=1, le=50)


def _hits_json(retrieval, by_number: dict) -> dict:
    return {
        "device": retrieval.device,
        "latency_s": retrieval.latency_s,
        "hits": [{"article_number": h.article_number, "score": h.score, "by_number": h.by_number,
                  "in_corpus": h.article_number in by_number} for h in retrieval.hits],
    }


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    tags: list[str] = Field(default_factory=list, max_length=20)
    comment: str = Field("", max_length=2000)
    question: str
    backend: str
    model: str
    answer: str
    article_numbers: list[int]
    cited: list[int]
    invalid: list[int]
    latency_s: float | None = None
    view: Literal["ask", "compare"] = "ask"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config() -> dict:
    params = load_llm_params()
    return {
        "backends": {
            "bedrock": {"label": "Bedrock", "model": params["bedrock"]["model"]},
            "vllm": {"label": "vLLM", "model": params["vllm"]["model"]},
        },
        "default_backend": params["backend"],
        "temperature": params["temperature"],
        "max_tokens": params["max_tokens"],
        "langfuse_url": os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST"),
    }


@app.get("/api/status")
def status() -> dict:
    return {"stages": collect_status()}


@app.get("/api/articles")
def articles() -> dict:
    s = data.load_articles()
    return {"source": s.source, "note": s.note, "articles": s.articles}


@app.get("/api/corpus/report")
def corpus_report() -> dict:
    """Build counts and the warnings to review (notes are left out: they are expected quirks)."""
    report = data.load_report()
    if report is None:
        return {"built": False}
    return {
        "built": True,
        "generated_at": report.get("generated_at"),
        "counts": report.get("counts", {}),
        "warnings": [i for i in report.get("issues", []) if i.get("level") == "warning"],
    }


def _ndjson(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


@app.post("/api/ask")
def ask(req: AskRequest) -> StreamingResponse:
    by_number = {a["article_number"]: a for a in data.load_articles().articles}
    unknown = [n for n in req.article_numbers if n not in by_number]
    if unknown:
        raise HTTPException(400, f"unknown article numbers: {unknown}")
    context = [by_number[n] for n in req.article_numbers]

    def events() -> Iterator[str]:
        nonlocal context
        try:
            if req.retrieve:
                retrieval = get_retriever().retrieve(req.question, req.top_k)
                context = [by_number[h.article_number] for h in retrieval.hits if h.article_number in by_number]
                yield _ndjson({"type": "retrieved", **_hits_json(retrieval, by_number)})
            backend = backend_for(req.backend)
            yield _ndjson({"type": "start", "backend": backend.name, "model": backend.model})
            for item in answer_stream(
                backend, req.question, context, max_tokens=req.max_tokens, temperature=req.temperature
            ):
                if isinstance(item, AnswerResult):
                    c, r = item.citations, item.llm
                    yield _ndjson({
                        "type": "done",
                        "text": item.text,
                        "citations": {"cited": c.cited, "valid": c.valid, "invalid": c.invalid},
                        "metrics": {
                            "model": r.model,
                            "input_tokens": r.input_tokens,
                            "output_tokens": r.output_tokens,
                            "latency_s": round(r.latency_s, 2),
                            "stop_reason": r.stop_reason,
                        },
                    })
                else:
                    yield _ndjson({"type": "delta", "text": item})
        except (LLMError, ValueError) as e:
            yield _ndjson({"type": "error", "message": str(e)})
        except Exception as e:  # noqa: BLE001 - e.g. index missing: show it in the answer card
            yield _ndjson({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/api/search")
def search_index(req: SearchRequest) -> dict:
    """Retrieval only: the top-k articles for a question, with scores (Retrieval view)."""
    by_number = {a["article_number"]: a for a in data.load_articles().articles}
    try:
        retrieval = get_retriever().retrieve(req.question, req.k)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"retrieval unavailable: {type(e).__name__}: {e}") from e
    return _hits_json(retrieval, by_number)


@app.post("/api/feedback")
def add_feedback(fb: FeedbackRequest) -> dict:
    data.FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **fb.model_dump()}
    with data.FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.get("/api/feedback")
def list_feedback(limit: int = 200) -> dict:
    if not data.FEEDBACK_PATH.exists():
        return {"entries": []}
    lines = data.FEEDBACK_PATH.read_text(encoding="utf-8").splitlines()
    return {"entries": [json.loads(line) for line in reversed(lines[-limit:]) if line.strip()]}


@app.get("/api/feedback.jsonl")
def download_feedback() -> FileResponse:
    if not data.FEEDBACK_PATH.exists():
        raise HTTPException(404, "no feedback yet")
    return FileResponse(data.FEEDBACK_PATH, media_type="application/x-ndjson", filename="feedback.jsonl")
