"""LLM inference: one interface over Amazon Bedrock and a self-hosted vLLM server."""

from rag.llm.answer import AnswerResult, answer, answer_stream
from rag.llm.base import LLMBackend, LLMError, LLMResult, Message
from rag.llm.factory import get_backend

__all__ = [
    "AnswerResult",
    "LLMBackend",
    "LLMError",
    "LLMResult",
    "Message",
    "answer",
    "answer_stream",
    "get_backend",
]
