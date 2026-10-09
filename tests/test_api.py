"""Production API: /ask contract, 422 on empty questions, /health, errors. Fake retriever and LLM: no GPU, no network."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.llm.openai_compat import OpenAICompatBackend
from rag.pipeline import Pipeline
from rag.retrieval import Retriever

ARTICLES = {
    492: {"article_number": 492, "text_ar": "تقع هبة الأموال المستقبلة باطلة.", "text_en": "A gift of future property is void.",
          "is_repealed": False},
    487: {"article_number": 487, "text_ar": "لا تتم الهبة إلا إذا قبلها الموهوب له.", "text_en": "A gift is complete only on acceptance.",
          "is_repealed": False},
}


class Completions:
    def __init__(self, text):
        self.text, self.calls = text, []

    def create(self, **req):
        self.calls.append(req)
        message = NS(content=self.text, model_extra={})
        return NS(choices=[NS(message=message, finish_reason="stop")], usage=NS(prompt_tokens=50, completion_tokens=9))


class AsyncCompletions(Completions):
    async def create(self, **req):
        self.calls.append(req)
        if not req.get("stream"):
            message = NS(content=self.text, model_extra={})
            return NS(choices=[NS(message=message, finish_reason="stop")],
                      usage=NS(prompt_tokens=50, completion_tokens=9))

        async def chunks():
            for piece in (self.text[:10], self.text[10:]):
                yield NS(choices=[NS(delta=NS(content=piece, model_extra={}), finish_reason=None)], usage=None)
            yield NS(choices=[], usage=NS(prompt_tokens=50, completion_tokens=9))
        return chunks()


def make_client(text="هبة الأموال المستقبلة باطلة [Article 492]، وانظر [Article 999].", *, indexed=1149, backend=None):
    completions = AsyncCompletions(text)
    backend = backend or OpenAICompatBackend("vllm", "qwen", "http://x/v1", client=NS(chat=NS(completions=completions)),
                                             async_client=NS(chat=NS(completions=completions)))
    retriever = Retriever(encode=lambda q: np.ones((1, 4), dtype=np.float32),
                          searcher=lambda v, k: [{"article_number": 492, "score": 0.75},
                                                 {"article_number": 487, "score": 0.56}][:k])
    pipeline = Pipeline(retriever=retriever, articles=ARTICLES, backend=backend, backend_name="vllm")
    pipeline.documents_indexed = lambda: indexed
    return TestClient(create_app(pipeline, warm_up=False)), completions


def test_ask_returns_answer_and_cited_sources_only():
    client, completions = make_client()
    res = client.post("/ask", json={"question": "  ما حكم هبة الأموال المستقبلة؟  "})
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"answer", "sources"}
    assert body["sources"] == ["Article 492"]  # 999 was cited but never retrieved: not a source
    prompt = completions.calls[0]["messages"][1]["content"]
    assert "ما حكم هبة الأموال المستقبلة؟" in prompt and "[Article 487]" in prompt


def test_answer_without_citations_has_no_sources():
    client, _ = make_client("The provided articles do not cover this.")
    assert client.post("/ask", json={"question": "What is the penalty for theft?"}).json()["sources"] == []


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "   \n "}, {}, {"question": None}, {"q": "x"},
                                  {"question": "x" * 2001}])
def test_empty_or_invalid_question_is_rejected_with_422(body):
    client, completions = make_client()
    res = client.post("/ask", json=body)
    assert res.status_code == 422
    assert not completions.calls  # nothing reached the LLM


def test_health_reports_indexed_documents():
    client, _ = make_client()
    assert client.get("/health").json() == {"status": "healthy", "documents_indexed": 1149}


def test_health_is_503_without_an_index():
    client, _ = make_client(indexed=None)
    res = client.get("/health")
    assert res.status_code == 503 and res.json() == {"status": "unhealthy", "documents_indexed": 0}


def test_llm_failure_is_a_503_with_the_reason():
    from rag.llm import LLMError

    class Down:
        name, model = "vllm", "qwen"

        async def agenerate(self, *a, **kw):
            raise LLMError("vllm: server not reachable at http://localhost:8001/v1 (is `vllm serve` running?)")

    client, _ = make_client(backend=Down())
    res = client.post("/ask", json={"question": "What is a gift?"})
    assert res.status_code == 503 and "vllm serve" in res.json()["detail"]


def test_stream_sends_tokens_then_sources():
    client, completions = make_client()
    with client.stream("POST", "/ask/stream", json={"question": "ما حكم هبة الأموال المستقبلة؟"}) as res:
        assert res.status_code == 200 and res.headers["content-type"].startswith("text/plain")
        pieces = [p for p in res.iter_text() if p]
    body = "".join(pieces)  # the test client buffers the body; arrival in parts is checked with curl -N
    assert body.startswith("هبة الأموال") and body.rstrip().endswith("Sources: Article 492")
    assert completions.calls[0]["stream"] is True


def test_stream_rejects_empty_question():
    client, _ = make_client()
    assert client.post("/ask/stream", json={"question": " "}).status_code == 422


def test_backend_can_be_chosen_in_dot_env(monkeypatch, tmp_path):
    from rag.llm import factory

    (tmp_path / ".env").write_text("LLM_BACKEND=bedrock\n")
    monkeypatch.setattr(factory, "ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    assert Pipeline().backend_name == "bedrock"


def test_environment_wins_over_dot_env(monkeypatch, tmp_path):
    from rag.llm import factory

    (tmp_path / ".env").write_text("LLM_BACKEND=bedrock\n")
    monkeypatch.setattr(factory, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    assert Pipeline().backend_name == "vllm"


def test_one_switch_moves_the_evaluation_too(monkeypatch, tmp_path):
    from rag.evaluation import gate
    from rag.llm import factory

    monkeypatch.setattr(factory, "ENV_FILE", tmp_path / ".env")  # no .env
    monkeypatch.setenv("LLM_BACKEND", "bedrock")
    captured = {}
    monkeypatch.setattr(gate, "run_gate", lambda qs, **kw: captured.update(kw) or {"metrics": {}, "rows": []})
    monkeypatch.setattr(gate, "write_report", lambda *a, **kw: None)
    gate.main(["--no-mlflow"])
    assert captured["answer_backend"] == captured["judge_backend"] == "bedrock"


def test_every_response_names_the_release(monkeypatch):
    monkeypatch.setenv("APP_RELEASE", "canary")
    client, _ = make_client()
    assert client.get("/health").headers["X-Release"] == "canary"
