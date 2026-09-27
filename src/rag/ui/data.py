"""Where the console's articles and feedback live."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "articles.json"
SAMPLE_PATH = REPO_ROOT / "data" / "samples" / "articles_sample.json"
SOURCE_PDF = REPO_ROOT / "data" / "raw" / "egyptian_civil_code.pdf"
FEEDBACK_PATH = REPO_ROOT / "data" / "feedback" / "feedback.jsonl"


@dataclass(frozen=True)
class ArticleSet:
    articles: list[dict]
    source: str  # "corpus" | "sample"
    note: str


def load_articles() -> ArticleSet:
    """The real corpus once it exists; the hand-cleaned sample fixture until then."""
    if CORPUS_PATH.exists():
        articles = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        return ArticleSet(articles, "corpus", f"Full corpus: {len(articles)} articles")
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    return ArticleSet(sample["articles"], "sample", sample["source"])
