"""One backend for every OpenAI-compatible server.

Both backends speak the Chat Completions API:
- bedrock: Amazon Bedrock's OpenAI-compatible endpoint (bedrock-mantle), authenticated with a Bedrock API key
- vllm:    a local `vllm serve` process
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import openai
from openai import OpenAI

from rag.llm.base import LLMError, LLMResult, Message

_HINTS = {
    "bedrock": {
        openai.AuthenticationError: "invalid or expired Bedrock_API_key",
        openai.PermissionDeniedError: "model not enabled for this account, or the key lacks access to it",
        openai.RateLimitError: "Bedrock quota hit (Service Quotas > Amazon Bedrock)",
        openai.NotFoundError: "unknown model id (list them with client.models.list())",
    },
    "vllm": {
        openai.NotFoundError: "vLLM serves a different model (check GET /models)",
    },
}


class OpenAICompatBackend:
    def __init__(
        self,
        name: str,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        client: Any | None = None,
        timeout_s: float = 180,
    ) -> None:
        # The timeout covers vLLM's first request after start-up, when it compiles CUDA graphs.
        self.name = name
        self.model = model
        self._base_url = base_url
        self._client = client or OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=2)

    def _messages(self, system: str, messages: list[Message]) -> list[dict]:
        return [{"role": "system", "content": system}] + [
            {"role": m.role, "content": m.content} for m in messages
        ]

    def _error(self, e: openai.OpenAIError) -> LLMError:
        if isinstance(e, openai.APIConnectionError):
            hint = "is `vllm serve` running?" if self.name == "vllm" else "check OPENAI_BASE_URL"
            return LLMError(f"{self.name}: server not reachable at {self._base_url} ({hint})")
        hint = next((h for cls, h in _HINTS.get(self.name, {}).items() if isinstance(e, cls)), None)
        msg = f"{self.name} error on {self.model}: {e}"
        return LLMError(f"{msg} -> {hint}" if hint else msg)

    def generate(
        self, system: str, messages: list[Message], *, max_tokens: int, temperature: float
    ) -> LLMResult:
        start = time.perf_counter()
        try:
            r = self._client.chat.completions.create(
                model=self.model,
                messages=self._messages(system, messages),
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except openai.OpenAIError as e:
            raise self._error(e) from e
        # Reasoning models (gpt-oss) return their chain of thought in a separate field, never in content.
        return LLMResult(
            text=r.choices[0].message.content or "",
            model=self.model,
            input_tokens=r.usage.prompt_tokens,
            output_tokens=r.usage.completion_tokens,
            latency_s=time.perf_counter() - start,
            stop_reason=r.choices[0].finish_reason or "",
            reasoning=_reasoning(r.choices[0].message),
        )

    def stream(
        self, system: str, messages: list[Message], *, max_tokens: int, temperature: float
    ) -> Iterator[str | LLMResult]:
        start = time.perf_counter()
        parts: list[str] = []
        thinking: list[str] = []
        input_tokens = output_tokens = 0
        stop = ""
        try:
            chunks = self._client.chat.completions.create(
                model=self.model,
                messages=self._messages(system, messages),
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in chunks:
                if chunk.choices:
                    choice = chunk.choices[0]
                    if piece := _reasoning(choice.delta):
                        thinking.append(piece)
                    if choice.delta.content:
                        parts.append(choice.delta.content)
                        yield choice.delta.content
                    if choice.finish_reason:
                        stop = choice.finish_reason
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
        except openai.OpenAIError as e:
            raise self._error(e) from e
        yield LLMResult(
            text="".join(parts),
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=time.perf_counter() - start,
            stop_reason=stop,
            reasoning="".join(thinking),
        )


def _reasoning(message) -> str:
    """Thinking text from a message or stream delta: vLLM and Bedrock use reasoning_content or reasoning."""
    extra = getattr(message, "model_extra", None) or {}
    for name in ("reasoning_content", "reasoning"):
        value = getattr(message, name, None) or extra.get(name)
        if isinstance(value, str) and value:
            return value
    return ""
