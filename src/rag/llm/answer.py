"""Question + retrieved articles -> cited answer, on whichever backend is configured.

Traced in Langfuse as a `generate-answer` generation (model, prompt, tokens, time to first token,
reasoning) followed by a `check-citations` evaluator whose result is scored on the trace. When a
caller has an `answer-question` span open, both nest under it; otherwise they form their own trace.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from rag import tracing
from rag.llm.base import LLMBackend, LLMError, LLMResult, Message
from rag.llm.citations import CitationCheck, check_citations
from rag.llm.prompts import SYSTEM_PROMPT, Article, build_user_message


@dataclass(frozen=True)
class AnswerResult:
    text: str
    citations: CitationCheck
    llm: LLMResult


def _generation(backend: LLMBackend, user_message: str, articles: Sequence[Article], max_tokens: int,
                temperature: float):
    return tracing.client().start_as_current_observation(
        as_type="generation",
        name="generate-answer",
        model=backend.model,
        input=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
        model_parameters={"temperature": temperature, "max_tokens": max_tokens},
        metadata={"backend": backend.name, "context_articles": [a["article_number"] for a in articles]},
    )


def _record(gen, result: LLMResult, first_token_at: datetime | None) -> None:
    gen.update(
        output=result.text,
        usage_details={"input": result.input_tokens, "output": result.output_tokens},
        completion_start_time=first_token_at,
        metadata={"stop_reason": result.stop_reason, "latency_s": round(result.latency_s, 3),
                  **({"reasoning": result.reasoning} if result.reasoning else {})},
        **({"level": "WARNING", "status_message": "answer cut off at max_tokens"}
           if result.stop_reason == "length" else {}),
    )


def _finish(result: LLMResult, articles: Sequence[Article]) -> AnswerResult:
    """Check citations against the context, as an evaluator step whose result is scored on the trace."""
    retrieved = [a["article_number"] for a in articles]
    lf = tracing.client()
    with lf.start_as_current_observation(
        as_type="evaluator", name="check-citations", input={"answer": result.text, "context_articles": retrieved},
    ) as ev:
        check = check_citations(result.text, retrieved)
        ev.update(output={"cited": check.cited, "valid": check.valid, "outside_context": check.invalid},
                  **({"level": "WARNING", "status_message": "cites articles that were not in the context"}
                     if check.invalid else {}))
        lf.score_current_trace(name="citations_outside_context", value=len(check.invalid), data_type="NUMERIC",
                               comment=", ".join(map(str, check.invalid)) or None)
        lf.score_current_trace(name="answer_has_citations", value=1 if check.cited else 0, data_type="BOOLEAN")
    return AnswerResult(text=result.text, citations=check, llm=result)


def answer(
    backend: LLMBackend,
    question: str,
    articles: Sequence[Article],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> AnswerResult:
    user_message = build_user_message(question, articles)
    with _generation(backend, user_message, articles, max_tokens, temperature) as gen:
        try:
            result = backend.generate(SYSTEM_PROMPT, [Message("user", user_message)], max_tokens=max_tokens,
                                      temperature=temperature)
        except LLMError as e:
            gen.update(level="ERROR", status_message=str(e))
            raise
        _record(gen, result, None)
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
    user_message = build_user_message(question, articles)
    first_token_at: datetime | None = None
    result: LLMResult | None = None
    parts: list[str] = []
    with _generation(backend, user_message, articles, max_tokens, temperature) as gen:
        try:
            for item in backend.stream(SYSTEM_PROMPT, [Message("user", user_message)], max_tokens=max_tokens,
                                       temperature=temperature):
                if isinstance(item, LLMResult):
                    result = item
                    _record(gen, item, first_token_at)
                else:
                    first_token_at = first_token_at or datetime.now(UTC)
                    parts.append(item)
                    yield item
        except LLMError as e:
            gen.update(level="ERROR", status_message=str(e))
            raise
        except GeneratorExit:  # the consumer stopped reading (e.g. the tester pressed Stop)
            gen.update(level="WARNING", status_message="stopped before the answer finished",
                       output="".join(parts), completion_start_time=first_token_at)
            raise
    if result is not None:
        yield _finish(result, articles)
