"""Checks on the built corpus (data/processed/articles.json). Skipped until `python -m rag.corpus.build` has run."""

import json
from pathlib import Path

import pytest
import yaml

from rag.corpus.validate import parse_ranges, validate_records

ROOT = Path(__file__).resolve().parents[1]
ARTICLES = ROOT / "data" / "processed" / "articles.json"

# Defects in the source PDF itself, reviewed by hand (see docs/plans/corpus-build.md). Any other warning fails.
KNOWN_SOURCE_ISSUES = {
    ("CUT_OFF_AR", 260), ("CUT_OFF_AR", 813), ("CUT_OFF_AR", 936), ("CUT_OFF_AR", 1115),  # no final period
    ("LENGTH_RATIO", 519),  # Arabic omits a clause the English has
    ("LENGTH_RATIO", 970),  # Arabic carries a later amendment the English lacks
    ("LENGTH_RATIO", 1060),
    ("LENGTH_RATIO", 1021), ("EMPTY_TEXT_AR", 1022), ("AR_NUMBER_MISSING", 1022),  # no "مادة ١٠٢٢" in the PDF
}

pytestmark = pytest.mark.skipif(not ARTICLES.exists(), reason="corpus not built: python -m rag.corpus.build")


@pytest.fixture(scope="module")
def records():
    return json.loads(ARTICLES.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))["corpus"]


def test_every_article_once_and_repealed_flagged(records, cfg):
    numbers = [r["article_number"] for r in records]
    assert numbers == list(range(cfg["first_article"], cfg["last_article"] + 1))
    assert {r["article_number"] for r in records if r["is_repealed"]} == parse_ranges(cfg["repealed"])


def test_no_warnings_beyond_known_source_defects(records, cfg):
    warnings = {(i.code, i.article) for i in validate_records(records, cfg) if i.level == "warning"}
    assert warnings - KNOWN_SOURCE_ISSUES == set()


def test_live_articles_are_bilingual_and_placed(records):
    for r in records:
        if r["is_repealed"]:
            assert r["repeal_note"] and not r["text_en"]
            continue
        assert r["text_en"], r["article_number"]
        assert r["section_number"] is not None or r["chapter_number"] is not None, r["article_number"]
        if r["article_number"] != 1022:
            assert r["text_ar"] and r["ar_number"] == r["article_number"], r["article_number"]


def test_hierarchy_numbers_and_bilingual_titles(records):
    by = {r["article_number"]: r for r in records}
    assert by[88]["part_number"] is None and by[88]["section_number"] == 3  # preliminary chapter: sections only
    assert (by[89]["part_number"], by[89]["book_number"], by[89]["chapter_number"], by[89]["section_number"],
            by[89]["topic_number"], by[89]["subtopic_title_en"]) == (1, 1, 1, 1, 1, "Consent")
    assert (by[802]["part_number"], by[802]["part_title_en"], by[802]["part_title_ar"]) == (2, "Real Rights",
                                                                                          "الحقوق العينية")
    assert by[1149]["book_number"] == 4
    for r in records:
        for level in ("part", "book", "chapter", "section"):
            if r[f"{level}_number"] is not None:
                assert r[f"{level}_title_en"] and r[f"{level}_title_ar"], (r["article_number"], level)
