"""LLM layer tests with fake clients: no GPU, no AWS, safe for CI."""

from types import SimpleNamespace as NS

import httpx
import openai
import pytest

from rag.llm import LLMError, LLMResult, answer, answer_stream
from rag.llm.citations import check_citations, extract_citations
from rag.llm.factory import get_backend
from rag.llm.openai_compat import OpenAICompatBackend
from rag.llm.prompts import build_user_message

ARTICLES = [
    {"article_number": 492, "text_ar": "تقع هبة الأموال المستقبلة باطلة.", "text_en": "A gift of future property is void."},
    {"article_number": 60, "text_ar": "", "text_en": "", "is_repealed": True},
]


# --- citations ---------------------------------------------------------------


def test_extract_citations_dedupes_and_normalizes():
    text = "باطلة [Article 492]. انظر [Art. ٤٩٢] و [article 505] و [Articles 60]."
    assert extract_citations(text) == [492, 505, 60]


def test_extract_citations_accepts_fullwidth_brackets():
    # Seen live from gpt-oss-120b on Bedrock.
    assert extract_citations("لا يجوز الرجوع【Article 502】 و［Article 501］") == [502, 501]


def test_check_citations_flags_hallucinated_article():
    check = check_citations("See [Article 492] and [Article 999].", retrieved=[492, 60])
    assert check.valid == [492]
    assert check.invalid == [999]
    assert not check.ok


def test_prompt_states_answer_language():
    assert build_user_message("ما حكم هبة الأموال المستقبلة؟", ARTICLES).endswith("أجب باللغة العربية.")
    assert build_user_message("How is a partnership defined?", ARTICLES).endswith("Answer in English.")
    assert build_user_message("ما معنى partnership في المادة ٥٠٥؟", ARTICLES).endswith("أجب باللغة العربية.")


def test_prompt_marks_repealed_articles():
    msg = build_user_message("q", ARTICLES)
    assert "[Article 60] (REPEALED)" in msg
    assert "A gift of future property is void." in msg


def test_prompt_gives_the_repeal_note_for_repealed_articles():
    repealed = {"article_number": 60, "is_repealed": True, "text_ar": "", "text_en": "",
                "repeal_note": "Articles 54-80 have been repealed by Presidential Decree.", "repeal_note_ar": "ألغيت"}
    msg = build_user_message("What does Article 60 say?", [repealed])
    assert "[Article 60] (REPEALED)\nألغيت\nArticles 54-80 have been repealed by Presidential Decree." in msg


# --- OpenAI-compatible backend (Bedrock endpoint and vLLM) --------------------


class FakeCompletions:
    """Mimics chat.completions: gpt-oss puts reasoning in a separate field, never in content."""

    def __init__(self, text, error=None):
        self.text, self.error, self.calls = text, error, []

    def create(self, **req):
        self.calls.append(req)
        if self.error:
            raise self.error
        usage = NS(prompt_tokens=130, completion_tokens=18)
        if not req.get("stream"):
            msg = NS(content=self.text, reasoning_content="thinking...")
            return NS(choices=[NS(message=msg, finish_reason="stop")], usage=usage)
        head, tail = self.text[: len(self.text) // 2], self.text[len(self.text) // 2 :]
        return iter(
            [
                NS(choices=[NS(delta=NS(content=None, reasoning_content="thinking..."), finish_reason=None)], usage=None),
                NS(choices=[NS(delta=NS(content=head), finish_reason=None)], usage=None),
                NS(choices=[NS(delta=NS(content=tail), finish_reason="stop")], usage=None),
                NS(choices=[], usage=usage),
            ]
        )


def fake_backend(name="bedrock", text="هبة الأموال المستقبلة باطلة [Article 492].", error=None):
    completions = FakeCompletions(text, error)
    client = NS(chat=NS(completions=completions))
    return OpenAICompatBackend(name, "openai.gpt-oss-120b", "http://x/v1", client=client), completions


def test_answer_sends_system_prompt_and_checks_citations():
    backend, completions = fake_backend()
    result = answer(backend, "ما حكم هبة الأموال المستقبلة؟", ARTICLES)
    req = completions.calls[0]
    assert [m["role"] for m in req["messages"]] == ["system", "user"]
    assert (req["temperature"], req["max_tokens"]) == (0.0, 1024)
    assert "[Article 492]" in req["messages"][1]["content"]
    assert result.citations.valid == [492] and result.citations.ok
    assert (result.llm.input_tokens, result.llm.output_tokens, result.llm.stop_reason) == (130, 18, "stop")
    assert "thinking" not in result.text


def test_stream_yields_deltas_then_result_with_usage():
    backend, completions = fake_backend()
    items = list(answer_stream(backend, "q", ARTICLES))
    deltas, final = items[:-1], items[-1]
    assert completions.calls[0]["stream_options"] == {"include_usage": True}
    assert all(isinstance(d, str) for d in deltas) and len(deltas) == 2
    assert "".join(deltas) == final.text and "thinking" not in final.text
    assert final.llm.input_tokens == 130 and final.citations.valid == [492]


def test_vllm_server_down_becomes_actionable_error():
    err = openai.APIConnectionError(request=httpx.Request("POST", "http://x/v1"))
    backend, _ = fake_backend(name="vllm", error=err)
    with pytest.raises(LLMError, match="vllm serve"):
        answer(backend, "q", ARTICLES)


def test_bedrock_rate_limit_becomes_actionable_error():
    response = httpx.Response(429, request=httpx.Request("POST", "http://x/v1"))
    err = openai.RateLimitError("Too many tokens per day", response=response, body=None)
    backend, _ = fake_backend(name="bedrock", error=err)
    with pytest.raises(LLMError, match="Service Quotas"):
        answer(backend, "q", ARTICLES)


# --- factory -----------------------------------------------------------------

PARAMS = {
    "backend": "vllm",
    "bedrock": {"model": "openai.gpt-oss-120b", "base_url": "https://bedrock-mantle.eu-north-1.api.aws/v1"},
    "vllm": {"model": "qwen", "base_url": "http://localhost:8001/v1"},
}


def test_factory_builds_bedrock_from_env(monkeypatch):
    monkeypatch.setenv("Bedrock_API_key", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://bedrock-mantle.eu-west-1.api.aws/v1")
    backend = get_backend(PARAMS, backend="bedrock")
    assert (backend.name, backend.model) == ("bedrock", "openai.gpt-oss-120b")
    assert backend._base_url.startswith("https://bedrock-mantle.eu-west-1")


def test_factory_requires_bedrock_key(monkeypatch):
    monkeypatch.setattr("rag.llm.factory.load_dotenv", lambda *a: None)
    monkeypatch.delenv("Bedrock_API_key", raising=False)
    with pytest.raises(ValueError, match="Bedrock_API_key"):
        get_backend(PARAMS, backend="bedrock")


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown llm backend"):
        get_backend(PARAMS, backend="anthropic")


def test_llm_result_is_immutable():
    r = LLMResult("t", "m", 1, 2, 0.1, "stop")
    with pytest.raises(AttributeError):
        r.text = "x"  # type: ignore[misc]


def test_vllm_model_env_chooses_the_served_model(monkeypatch, tmp_path):
    from rag.llm import factory

    monkeypatch.setattr(factory, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setenv("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct-AWQ")
    assert factory.load_llm_params()["vllm"]["model"] == "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
    assert get_backend(backend="vllm").model == "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
