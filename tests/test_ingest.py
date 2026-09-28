"""Chunks, batching, replica planning, the warning gate and a Chroma round trip (no GPU needed)."""

import hashlib
import json

import numpy as np

from rag.ingest.__main__ import gate, report_warnings
from rag.ingest.chunks import build_chunks, chunk_metadata, chunk_text, heading_path, query_text
from rag.ingest.embed import make_batches, plan_replicas
from rag.ingest.store import collection_name, index_count, search, write_index

A44 = {
    "article_number": 44, "part_number": None, "part_title_en": None, "part_title_ar": None,
    "book_number": None, "book_title_en": None, "book_title_ar": None,
    "chapter_number": None, "chapter_title_en": None, "chapter_title_ar": None,
    "section_number": 2, "section_title_en": "Persons", "section_title_ar": "الأشخاص",
    "topic_number": 1, "topic_title_en": "Individuals", "topic_title_ar": "الشخص الطبيعي",
    "subtopic_title_en": None, "subtopic_title_ar": None,
    "text_ar": "(١) كل شخص بلغ سن الرشد.\n(٢) وسن الرشد هى إحدى وعشرون سنة ميلادية كاملة.",
    "text_en": "All persons attaining majority have full legal capacity.", "is_repealed": False,
    "repeal_note": None, "repeal_note_ar": None, "source_pages": [6], "ar_number": 44,
    "citation": "Egyptian Civil Code, Article 44",
}
A60 = {**A44, "article_number": 60, "text_ar": "", "text_en": "", "is_repealed": True, "ar_number": None,
       "repeal_note": "Articles 54-80 have been repealed by Presidential Decree.", "repeal_note_ar": "ألغيت",
       "source_pages": [7, 8], "citation": "Egyptian Civil Code, Article 60"}


def test_chunk_is_one_bilingual_article_with_its_heading_path():
    assert heading_path(A44) == "Section 2 Persons | الفصل الثاني الأشخاص > Topic 1 Individuals | الشخص الطبيعي"
    text = chunk_text(A44)
    assert text.splitlines()[1] == "Article 44 | مادة 44"
    assert A44["text_ar"] in text and A44["text_en"] in text


def test_repealed_article_is_indexed_with_its_note():
    text = chunk_text(A60)
    assert "(repealed | ملغاة)" in text and "Presidential Decree" in text and "ألغيت" in text


def test_metadata_is_chroma_safe():
    meta = chunk_metadata(A60)
    assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())
    assert meta["source_pages"] == "7,8" and "part_number" not in meta and meta["is_repealed"] is True


def test_normalization_changes_only_what_is_embedded():
    plain, normalized = build_chunks([A44])[0], build_chunks([A44], normalize_arabic=True)[0]
    assert plain.id == "art-0044" and plain.text == normalized.text
    assert "هى" in plain.embed_text and "هي" in normalized.embed_text  # alef maqsura unified
    assert query_text("إلى", True) == "الي"


def test_batches_are_length_sorted_and_within_budget():
    batches = make_batches([10, 500, 30, 2000, 100], tokens_per_batch=2048)
    assert batches == [[3], [1, 4, 2, 0]]
    lengths = [10, 500, 30, 2000, 100]
    assert all(len(b) * max(lengths[i] for i in b) <= 2048 for b in batches)


def test_replica_plan_respects_memory_margin_and_caps():
    assert plan_replicas(13_400, 2_048, 2_080, max_replicas=4, n_batches=77) == 4
    assert plan_replicas(13_400, 2_048, 4_300, max_replicas=4, n_batches=77) == 2
    assert plan_replicas(3_000, 2_048, 2_080, max_replicas=4, n_batches=77) == 1  # never zero
    assert plan_replicas(13_400, 2_048, 2_080, max_replicas=4, n_batches=2) == 2  # no idle replicas


def test_warning_gate():
    assert gate(0, yes=False, stop_on_warning=False, interactive=False)
    assert not gate(3, yes=True, stop_on_warning=True, interactive=True)
    assert gate(3, yes=True, stop_on_warning=False, interactive=False)
    assert not gate(3, yes=False, stop_on_warning=False, interactive=False)  # nobody to ask: stop
    assert gate(3, yes=False, stop_on_warning=False, interactive=True, ask=lambda _: "y")
    assert not gate(3, yes=False, stop_on_warning=False, interactive=True, ask=lambda _: "")


def test_build_report_is_used_only_for_the_same_articles(tmp_path):
    articles = tmp_path / "articles.json"
    articles.write_text("[]", encoding="utf-8")
    report = tmp_path / "report.json"
    issues = [{"level": "warning", "code": "GAP_BEFORE_ARTICLE", "message": "m", "article": 3},
              {"level": "warning", "code": "EMPTY_TEXT_AR", "message": "m", "article": 3},
              {"level": "info", "code": "PAGE_BREAK", "message": "m"}]
    report.write_text(json.dumps({"articles_sha256": hashlib.sha256(b"[]").hexdigest(), "issues": issues}), "utf-8")
    got = report_warnings(report, articles, produced_codes={"EMPTY_TEXT_AR"})
    assert [i.code for i in got] == ["GAP_BEFORE_ARTICLE"]
    articles.write_text("[1]", encoding="utf-8")
    assert [i.code for i in report_warnings(report, articles, set())] == ["REPORT_STALE"]


def test_chroma_round_trip(tmp_path):
    chunks = build_chunks([A44, A60])
    vectors = np.eye(2, 8, dtype=np.float32)
    name = collection_name("civil_code", "Qwen/Qwen3-Embedding-0.6B")
    assert name == "civil_code__qwen3-embedding-0.6b"
    assert write_index(tmp_path / "chroma", name, chunks, vectors, {"embedding_model": "test"}) == 2
    hits = search(tmp_path / "chroma", name, vectors[1:2], k=2)[0]
    assert hits[0]["article_number"] == 60 and hits[0]["score"] > 0.99 and hits[0]["is_repealed"] is True
    assert index_count(tmp_path / "chroma", name) == 2 and index_count(tmp_path / "missing", name) is None
