"""The evaluation question set stays well-formed: unique ids, real articles, paired languages, cited references."""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
CORPUS = ROOT / "data" / "processed" / "articles.json"


def test_ids_unique_and_fields_present():
    assert len(QUESTIONS) >= 50
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS)
    for q in QUESTIONS:
        assert q["lang"] in ("ar", "en") and q["question"].strip() and q["reference"].strip()
        assert q["type"] in ("factual", "reasoning", "by_number", "out_of_scope")
        assert (q["type"] == "out_of_scope") == (not q["relevant_articles"])


def test_both_languages_are_balanced():
    langs = Counter(q["lang"] for q in QUESTIONS)
    assert abs(langs["ar"] - langs["en"]) <= 2
    pairs = Counter(q["pair"] for q in QUESTIONS if q["type"] in ("factual", "reasoning"))
    assert set(pairs.values()) == {2}  # every topic asked in both languages


def test_references_cite_their_relevant_articles():
    for q in QUESTIONS:
        cited = {int(n) for n in re.findall(r"\d{1,4}", " ".join(re.findall(r"\[[^\]]*\]", q["reference"])))}
        assert set(q["relevant_articles"]) <= cited, q["id"]


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built (dvc pull)")
def test_relevant_articles_exist_in_the_corpus():
    numbers = {a["article_number"] for a in json.loads(CORPUS.read_text(encoding="utf-8"))}
    assert all(n in numbers for q in QUESTIONS for n in q["relevant_articles"])


def test_ci_gate_subset_is_twenty_paired_questions_across_the_code():
    ci = [q for q in QUESTIONS if q.get("ci")]
    assert len(ci) == 20 and all(q["relevant_articles"] for q in ci)
    assert Counter(q["lang"] for q in ci) == {"ar": 10, "en": 10}
    assert {q["type"] for q in ci} == {"factual", "reasoning"}
    assert all(q["status"] in ("accepted", "reviewed") for q in QUESTIONS)
