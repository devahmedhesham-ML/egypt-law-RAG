"""Test console API with a fake backend: no network, no GPU."""

import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from rag.llm.openai_compat import OpenAICompatBackend
from rag.ui import data, server, status


class FakeCompletions:
    def create(self, **req):
        text = "هبة الأموال المستقبلة باطلة [Article 492] وانظر [Article 999]."
        usage = NS(prompt_tokens=100, completion_tokens=12)
        return iter([
            NS(choices=[NS(delta=NS(content=text[:20]), finish_reason=None)], usage=None),
            NS(choices=[NS(delta=NS(content=text[20:]), finish_reason="stop")], usage=None),
            NS(choices=[], usage=usage),
        ])


@pytest.fixture
def client(monkeypatch, tmp_path):
    fake = OpenAICompatBackend("bedrock", "openai.gpt-oss-120b", "http://x/v1",
                               client=NS(chat=NS(completions=FakeCompletions())))
    monkeypatch.setattr(server, "backend_for", lambda name: fake)
    monkeypatch.setattr(data, "FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(status, "FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    # No built corpus unless a test writes one: independent of data/processed on this machine.
    for module in (data, status):
        monkeypatch.setattr(module, "CORPUS_PATH", tmp_path / "articles.json")
    monkeypatch.setattr(data, "REPORT_PATH", tmp_path / "corpus_report.json")
    # Skip live network checks (Bedrock endpoint, vLLM server).
    monkeypatch.setattr(status, "_LIVE", [status._source_pdf, status._corpus, status._retrieval])
    return TestClient(server.app)


def test_index_and_static_assets_are_served(client):
    assert "Test console" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200


def test_articles_fall_back_to_sample_fixture(client):
    body = client.get("/api/articles").json()
    numbers = [a["article_number"] for a in body["articles"]]
    assert body["source"] == "sample"
    assert 492 in numbers and 505 in numbers
    assert any(a.get("is_repealed") for a in body["articles"])


def test_built_corpus_replaces_the_sample_and_exposes_its_warnings(client, tmp_path):
    record = {"article_number": 492, "section_title_en": "Gifts", "topic_title_en": "Elements of a Gift",
              "text_ar": "تقع هبة الأموال المستقبلة باطلة.", "text_en": "A gift of future property is void.",
              "is_repealed": False}
    (tmp_path / "articles.json").write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")
    body = client.get("/api/articles").json()
    assert body["source"] == "corpus" and body["articles"][0]["topic"] == "Gifts: Elements of a Gift"
    assert client.get("/api/corpus/report").json() == {"built": False}
    report = {"generated_at": "t", "counts": {"warnings": 1},
              "issues": [{"level": "warning", "code": "CUT_OFF_AR", "message": "m", "article": 492},
                         {"level": "info", "code": "PAGE_BREAK", "message": "m"}]}
    (tmp_path / "corpus_report.json").write_text(json.dumps(report), encoding="utf-8")
    got = client.get("/api/corpus/report").json()
    assert got["built"] and [w["code"] for w in got["warnings"]] == ["CUT_OFF_AR"]


def test_status_lists_planned_stages(client):
    stages = {s["id"]: s for s in client.get("/api/status").json()["stages"]}
    assert stages["retrieval"]["state"] == "planned"
    assert stages["ragas"]["state"] == "planned"
    assert stages["citations"]["state"] == "working"


def test_ask_streams_deltas_then_checked_result(client):
    res = client.post("/api/ask", json={"question": "ما حكم هبة الأموال المستقبلة؟", "backend": "bedrock", "article_numbers": [492, 505]})
    events = [json.loads(line) for line in res.text.splitlines()]
    assert [e["type"] for e in events] == ["start", "delta", "delta", "done"]
    done = events[-1]
    assert done["citations"] == {"cited": [492, 999], "valid": [492], "invalid": [999]}
    assert done["metrics"]["input_tokens"] == 100 and done["metrics"]["stop_reason"] == "stop"


def test_ask_rejects_unknown_articles(client):
    res = client.post("/api/ask", json={"question": "q", "backend": "bedrock", "article_numbers": [123456]})
    assert res.status_code == 400


def test_feedback_roundtrip(client):
    entry = {
        "rating": "down", "tags": ["Irrelevant citation"], "comment": "cites 999",
        "question": "q", "backend": "bedrock", "model": "openai.gpt-oss-120b", "answer": "a",
        "article_numbers": [492], "cited": [492, 999], "invalid": [999], "latency_s": 0.8,
    }
    assert client.post("/api/feedback", json=entry).json() == {"ok": True, "scored": False}
    saved = client.get("/api/feedback").json()["entries"]
    assert len(saved) == 1 and saved[0]["tags"] == ["Irrelevant citation"] and "ts" in saved[0]
    assert client.get("/api/feedback.jsonl").status_code == 200


def test_feedback_on_a_trace_becomes_langfuse_scores(client, monkeypatch):
    scores = []
    monkeypatch.setattr(server.tracing, "client", lambda: NS(create_score=lambda **kw: scores.append(kw)))
    monkeypatch.setattr(server.tracing, "enabled", lambda: True)
    entry = {
        "rating": "down", "tags": ["Wrong conclusion", "Too long"], "comment": "wrong article",
        "question": "q", "backend": "vllm", "model": "m", "answer": "a", "article_numbers": [492],
        "cited": [492], "invalid": [], "trace_id": "abc123",
    }
    assert client.post("/api/feedback", json=entry).json() == {"ok": True, "scored": True}
    assert [(s["name"], s["value"]) for s in scores] == [
        ("tester_rating", "wrong"), ("tester_issue", "Wrong conclusion"), ("tester_issue", "Too long")]
    assert all(s["trace_id"] == "abc123" and s["data_type"] == "CATEGORICAL" for s in scores)
