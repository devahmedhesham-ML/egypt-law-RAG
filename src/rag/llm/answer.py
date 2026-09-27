"""Question + retrieved articles -> cited answer, on whichever backend is configured."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from rag.llm.base import LLMBackend, LLMResult, Message
from rag.llm.citations import CitationCheck, check_citations
from rag.llm.prompts import SYSTEM_PROMPT, Article, build_user_message


@dataclass(frozen=True)
class AnswerResult:
    text: str
    citations: CitationCheck
    llm: LLMResult


def _finish(result: LLMResult, articles: Sequence[Article]) -> AnswerResult:
    retrieved = [a["article_number"] for a in articles]
    return AnswerResult(text=result.text, citations=check_citations(result.text, retrieved), llm=result)


def answer(
    backend: LLMBackend,
    question: str,
    articles: Sequence[Article],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> AnswerResult:
    messages = [Message("user", build_user_message(question, articles))]
    result = backend.generate(SYSTEM_PROMPT, messages, max_tokens=max_tokens, temperature=temperature)
    return _finish(result, articles)


def answer_stream(
    backend: LLMBackend,
    question: str,
    articles: Sequence[Article],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Iterator[str | AnswerResult]:
    """Yield text deltas, then one AnswerResult (citations are checked on the full text)."""
    messages = [Message("user", build_user_message(question, articles))]
    for item in backend.stream(SYSTEM_PROMPT, messages, max_tokens=max_tokens, temperature=temperature):
        yield _finish(item, articles) if isinstance(item, LLMResult) else item
