"""Backend-agnostic types shared by every LLM backend."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMResult:
    """One completed generation. Token usage and model id feed cost tracking and tracing."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    stop_reason: str


class LLMError(RuntimeError):
    """Backend failure with an actionable message (auth, quota, server down)."""


class LLMBackend(Protocol):
    name: str
    model: str

    def generate(
        self, system: str, messages: list[Message], *, max_tokens: int, temperature: float
    ) -> LLMResult: ...

    def stream(
        self, system: str, messages: list[Message], *, max_tokens: int, temperature: float
    ) -> Iterator[str | LLMResult]:
        """Yield text deltas, then exactly one final LLMResult with the full text and usage."""
        ...
