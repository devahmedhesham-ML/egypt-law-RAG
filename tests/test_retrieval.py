"""Retrieval: article numbers named in the question, merging with semantic hits, and the /api/ask path."""

import json

import numpy as np
from fastapi.testclient import TestClient

from rag.retrieval import Hit, Retrieval, Retriever, merge_hits, numbers_in_question


def test_article_numbers_in_arabic_and_english_questions():
    assert numbers_in_question("ماذا تقول المادة رقم 801") == [801]
    assert numbers_in_question("ما نص مادة (٤٩٢)؟") == [492]
    assert numbers_in_question("Compare Article 60 and articles 505") == [60, 505]
    assert numbers_in_question("What does Article 5000 say?") == []  # outside the code
    assert numbers_in_question("At what age does a person reach majority?") == []


def test_named_articles_come_first_without_duplicates():
    semantic = [{"article_number": 505, "score": 0.6}, {"article_number": 801, "score": 0.5},
                {"article_number": 506, "score": 0.4}]
    hits = merge_hits([801], semantic, k=3)
    assert [(h.article_number, h.by_number) for h in hits] == [(801, True), (505, False), (506, False)]


def test_retriever_embeds_the_question_and_searches():
    calls = {}

    def search(vector, k):
        calls["k"] = k
        return [{"article_number": 44, "score": 0.7}, {"article_number": 46, "score": 0.6}]

    r = Retriever(encode=lambda q: np.ones((1, 4), dtype=np.float32), searcher=search)
    out = r.retrieve("What does Article 60 say?", k=2)
    assert calls["k"] == 3  # one extra so a named article cannot push out a semantic hit
    assert [h.article_number for h in out.hits] == [60, 44]


def test_ask_retrieves_its_context(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS

    from rag.llm.openai_compat import OpenAICompatBackend
    from rag.ui import data, server

    sent = {}

    class Completions:
        def create(self, **req):
            sent["prompt"] = req["messages"][1]["content"]
            usage = NS(prompt_tokens=10, completion_tokens=3)
            return iter([NS(choices=[NS(delta=NS(content="باطلة [Article 492]."), finish_reason="stop")], usage=None),
                         NS(choices=[], usage=usage)])

    backend = OpenAICompatBackend("vllm", "qwen", "http://x/v1", client=NS(chat=NS(completions=Completions())))
    fake = NS(retrieve=lambda q, k: Retrieval([Hit(492, 0.75), Hit(487, 0.56)], "cpu", 0.05))
    monkeypatch.setattr(server, "backend_for", lambda name: backend)
    monkeypatch.setattr(server, "get_retriever", lambda: fake)
    monkeypatch.setattr(data, "CORPUS_PATH", tmp_path / "missing.json")  # sample fixture has 492 but not 487
    res = TestClient(server.app).post("/api/ask", json={"question": "ما حكم هبة الأموال المستقبلة؟", "backend": "vllm",
                                                        "retrieve": True, "top_k": 2})
    events = [json.loads(line) for line in res.text.splitlines()]
    assert [e["type"] for e in events][:2] == ["retrieved", "start"]
    assert [(h["article_number"], h["in_corpus"]) for h in events[0]["hits"]] == [(492, True), (487, False)]
    assert "[Article 492]" in sent["prompt"] and "[Article 487]" not in sent["prompt"]
    assert events[-1]["citations"]["valid"] == [492]
