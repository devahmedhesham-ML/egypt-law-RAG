"""Chunking experiment metrics: hit/recall/MRR/nDCG per question and their aggregation."""

import math

import pytest

from rag.experiments.chunking import aggregate, question_metrics, run_name


def test_metrics_for_a_single_relevant_article():
    m = question_metrics([7, 492, 3, 4, 5], [492], k=5)
    assert m["hit_at_1"] == 0.0 and m["hit_at_5"] == 1.0 and m["recall_at_5"] == 1.0
    assert m["mrr"] == 0.5
    assert m["ndcg_at_5"] == pytest.approx(1 / math.log2(3))


def test_metrics_for_two_relevant_articles_and_a_miss():
    m = question_metrics([968, 1, 2, 3, 4, 969], [968, 969], k=5)
    assert m["hit_at_1"] == 1.0 and m["recall_at_5"] == 0.5  # 969 is ranked 6th, outside the top 5
    assert question_metrics([1, 2, 3], [9], k=5) == {"hit_at_1": 0.0, "hit_at_5": 0.0, "recall_at_5": 0.0,
                                                      "mrr": 0.0, "ndcg_at_5": 0.0}


def test_aggregate_splits_languages_and_measures_agreement():
    rows = [
        {"pair": "q1", "lang": "ar", "relevant_articles": [5], "ranked": [5, 1], "top_score": 0.8,
         **question_metrics([5, 1], [5], 5)},
        {"pair": "q1", "lang": "en", "relevant_articles": [5], "ranked": [1, 5], "top_score": 0.6,
         **question_metrics([1, 5], [5], 5)},
        {"pair": "q9", "lang": "en", "relevant_articles": [], "ranked": [3], "top_score": 0.4},  # out of scope
    ]
    out = aggregate(rows, 5)
    assert out["hit_at_1"] == 0.5 and out["hit_at_1_ar"] == 1.0 and out["hit_at_1_en"] == 0.0
    assert out["ar_en_top1_agreement"] == 0.0  # Arabic ranks 5 first, English ranks 1 first
    assert out["top1_score_in_scope"] == pytest.approx(0.7) and out["top1_score_out_of_scope"] == 0.4


def test_run_names_are_readable_and_distinct():
    assert run_name({"strategy": "article", "model": "Qwen/Qwen3-Embedding-0.6B"}) == "article__Qwen3-Embedding-0.6B"
    assert run_name({"strategy": "window", "chunk_size": 256, "overlap": 32, "model": "BAAI/bge-m3"}) \
        == "window-256o32__bge-m3"
