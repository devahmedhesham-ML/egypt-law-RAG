"""Where the console's articles, build report and feedback live."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "articles.json"
REPORT_PATH = REPO_ROOT / "data" / "processed" / "corpus_report.json"
SAMPLE_PATH = REPO_ROOT / "data" / "samples" / "articles_sample.json"
SOURCE_PDF = REPO_ROOT / "data" / "raw" / "egyptian_civil_code.pdf"
FEEDBACK_PATH = REPO_ROOT / "data" / "feedback" / "feedback.jsonl"


@dataclass(frozen=True)
class ArticleSet:
    articles: list[dict]
    source: str  # "corpus" | "sample"
    note: str


def display_topic(r: dict) -> str:
    """Short label for lists: 'Gifts: Elements of a Gift'."""
    section = r.get("section_title_en") or r.get("chapter_title_en")
    detail = r.get("topic_title_en") or r.get("subtopic_title_en")
    return f"{section}: {detail}" if section and detail else section or detail or ""


def load_articles() -> ArticleSet:
    """The real corpus once it exists; the hand-cleaned sample fixture until then."""
    if CORPUS_PATH.exists():
        articles = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        for a in articles:
            a.setdefault("topic", display_topic(a))
        return ArticleSet(articles, "corpus", f"Full corpus: {len(articles):,} articles")
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    return ArticleSet(sample["articles"], "sample", sample["source"])


def load_report() -> dict | None:
    """The corpus build report (counts and warnings), if the corpus has been built."""
    if not REPORT_PATH.exists():
        return None
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
