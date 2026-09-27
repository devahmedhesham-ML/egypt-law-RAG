"""The one prompt both backends get, so their answers are comparable."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict


class Article(TypedDict, total=False):
    article_number: int
    text_ar: str
    text_en: str
    is_repealed: bool


SYSTEM_PROMPT = """\
You are a legal assistant for the Egyptian Civil Code (Law No. 131 of 1948).
أنت مساعد قانوني متخصص في القانون المدني المصري.

Rules:
1. Answer ONLY from the articles provided in the user message. Never use outside knowledge of the law.
2. Answer in the language of the question (Arabic question -> Arabic answer, English -> English).
3. Cite every claim inline using exactly this tag, with Western digits, in both languages: [Article 492]
4. Cite only article numbers that appear in the provided articles.
5. If the provided articles do not answer the question, say so plainly instead of guessing.
6. If an article is marked REPEALED, say that it was repealed; do not describe what it used to say.
"""


def format_articles(articles: Sequence[Article]) -> str:
    blocks = []
    for a in articles:
        header = f"[Article {a['article_number']}]" + (" (REPEALED)" if a.get("is_repealed") else "")
        body = "\n".join(t for t in (a.get("text_ar", ""), a.get("text_en", "")) if t)
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def build_user_message(question: str, articles: Sequence[Article]) -> str:
    return f"<articles>\n{format_articles(articles)}\n</articles>\n\n<question>\n{question}\n</question>"
