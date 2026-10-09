"""RAGAS evaluation helpers, re-ranker training-data filters and the pipeline's re-ranking step (no GPU, no network)."""

import math

import numpy as np

from rag.evaluation.ragas_eval import retrieval_scores, summarize
from rag.pipeline import Pipeline
from rag.rerank.data import parse
from rag.retrieval import Retriever


def test_retrieval_scores():
    assert retrieval_scores([5, 9, 7], [9]) == {"hit_at_1": 0.0, "mrr": 0.5, "recall_at_5": 1.0}
    assert retrieval_scores([9, 1], [9, 4]) == {"hit_at_1": 1.0, "mrr": 1.0, "recall_at_5": 0.5}
    assert retrieval_scores([], [9]) == {"hit_at_1": 0.0, "mrr": 0.0, "recall_at_5": 0.0}


def test_summarize_ignores_failed_judgements_and_counts_judged():
    nan = float("nan")
    base = {"answer_relevancy": 0.5, "context_precision": 1.0, "context_recall": 1.0,
            "hit_at_1": 1.0, "mrr": 1.0, "recall_at_5": 1.0}
    rows = [{"lang": "ar", "faithfulness": 1.0, **base}, {"lang": "en", "faithfulness": nan, **base},
            {"lang": "en", "faithfulness": 0.5, **base}]
    m = summarize(rows)
    assert m["faithfulness"] == 0.75 and m["faithfulness_judged"] == 2
    assert m["faithfulness_ar"] == 1.0 and m["faithfulness_en"] == 0.5
    assert not any(isinstance(v, float) and math.isnan(v) for v in m.values())


def test_generated_questions_are_filtered():
    good = '{"ar": "هل يجوز للمستأجر أن يؤجر العين لغيره؟", "en": "Can a tenant sublet the property?"}'
    assert parse(good, 593) == {"ar": "هل يجوز للمستأجر أن يؤجر العين لغيره؟", "en": "Can a tenant sublet the property?"}
    assert parse('{"ar": "What is article 593 about?", "en": "x"}', 593) is None       # Arabic missing, too short
    assert parse('{"ar": "ما حكم المادة 593 في الإيجار؟", "en": "What does article 593 say?"}', 593) is None  # names it
    assert parse("no json here", 1) is None


ARTICLES = {n: {"article_number": n, "text_ar": "نص", "text_en": "text", "is_repealed": False} for n in (1, 2, 3)}


class FakeReranker:
    def rerank(self, question, hits, articles):
        return sorted(hits, key=lambda h: -h.article_number)  # "prefers" higher article numbers


def make_pipeline(rerank):
    retriever = Retriever(encode=lambda q: np.ones((1, 4), dtype=np.float32),
                          searcher=lambda v, k: [{"article_number": n, "score": 0.9 - n / 10} for n in (1, 2, 3)][:k])
    return Pipeline(retriever=retriever, articles=ARTICLES, backend=object(), backend_name="vllm",
                    rerank=rerank, reranker=FakeReranker())


def test_rerank_off_keeps_the_retrieval_order():
    retrieval, context = make_pipeline(False).retrieve("q", k=2)
    assert [a["article_number"] for a in context] == [1, 2]


def test_rerank_on_reorders_the_candidates_then_keeps_top_k():
    retrieval, context = make_pipeline(True).retrieve("q", k=2)
    assert [a["article_number"] for a in context] == [3, 2]        # 3 was outside the top 2 before re-ranking
    assert [h.article_number for h in retrieval.hits] == [3, 2]


def test_rag_rerank_env_turns_the_reranker_on(monkeypatch):
    monkeypatch.setenv("RAG_RERANK", "true")
    retriever = Retriever(encode=lambda q: np.ones((1, 4), dtype=np.float32), searcher=lambda v, k: [])
    assert Pipeline(retriever=retriever, articles=ARTICLES, backend=object(), backend_name="vllm").rerank is True
