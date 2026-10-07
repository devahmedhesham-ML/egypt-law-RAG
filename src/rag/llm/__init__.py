"""LLM inference: one interface over Amazon Bedrock and a self-hosted vLLM server."""

from rag.llm.answer import AnswerResult, answer, answer_async, answer_stream, answer_stream_async
from rag.llm.base import LLMBackend, LLMError, LLMResult, Message
from rag.llm.factory import get_backend

__all__ = [
    "AnswerResult",
    "LLMBackend",
    "LLMError",
    "LLMResult",
    "Message",
    "answer",
    "answer_async",
    "answer_stream",
    "answer_stream_async",
    "get_backend",
]
