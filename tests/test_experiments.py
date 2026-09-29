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


def test_report_explains_columns_with_a_worked_example_and_compares_runs():
    from rag.experiments.report import build_report, first_rank

    qs = [{"id": "q1-en", "pair": "q1", "lang": "en", "type": "factual", "question": "What is a sale?",
           "relevant_articles": [418], "status": "draft"},
          {"id": "q1-ar", "pair": "q1", "lang": "ar", "type": "factual", "question": "ما البيع؟",
           "relevant_articles": [418], "status": "draft"},
          {"id": "q9-en", "pair": "q9", "lang": "en", "type": "out_of_scope", "question": "Theft penalty?",
           "relevant_articles": [], "status": "draft"}]

    def run(name, strategy, ranked_en, ranked_ar):
        rows = [{**qs[0], "ranked": ranked_en, "hit_at_1": float(ranked_en[0] == 418)},
                {**qs[1], "ranked": ranked_ar, "hit_at_1": float(ranked_ar[0] == 418)},
                {**qs[2], "ranked": [7]}]
        return {"name": name, "cfg": {"strategy": strategy, "model": "Qwen/Qwen3-Embedding-0.6B"},
                "metrics": {"hit_at_1": 0.5, "chunks": 1149}, "rows": rows}

    prod = run("article__Qwen3-Embedding-0.6B", "article", [97, 418, 3], [418, 2, 3])
    other = run("per_language__Qwen3-Embedding-0.6B", "per_language", [418, 1, 2], [5, 6, 418])
    md = build_report([prod, other], qs, prod["cfg"], 5, "abc123", "python -m rag.experiments.chunking")

    assert first_rank(prod["rows"][0]) == 2 and first_rank({"relevant_articles": [1], "ranked": [2]}) is None
    assert "Production today: **article__Qwen3-Embedding-0.6B**" in md
    assert "`mrr` = 1/2 = **0.500**" in md and "**0.631**" in md  # worked example from the rank-2 question
    assert "| per_language__Qwen3-Embedding-0.6B | 1 | 1 | 0 | 0 |" in md  # won q1-en, lost q1-ar at rank 1
    assert "3 of 3 questions are drafts" in md
    assert "## What each column means and how it is calculated" in md
