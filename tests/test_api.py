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


def make_client(text="هبة الأموال المستقبلة باطلة [Article 492]، وانظر [Article 999].", *, indexed=1149, backend=None):
    completions = Completions(text)
    backend = backend or OpenAICompatBackend("bedrock", "gpt", "http://x/v1",
                                             client=NS(chat=NS(completions=completions)))
    retriever = Retriever(encode=lambda q: np.ones((1, 4), dtype=np.float32),
                          searcher=lambda v, k: [{"article_number": 492, "score": 0.75},
                                                 {"article_number": 487, "score": 0.56}][:k])
    pipeline = Pipeline(retriever=retriever, articles=ARTICLES, backend=backend, backend_name="bedrock")
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
        name, model = "bedrock", "gpt"

        def generate(self, *a, **kw):
            raise LLMError("bedrock error on gpt: invalid or expired Bedrock_API_key")

    client, _ = make_client(backend=Down())
    res = client.post("/ask", json={"question": "What is a gift?"})
    assert res.status_code == 503 and "expired Bedrock_API_key" in res.json()["detail"]
