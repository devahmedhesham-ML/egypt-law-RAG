"""RAGAS faithfulness: the production model answers from given articles, an LLM judge scores the answer.

Both models default to Qwen2.5 on vLLM (params.yaml `evaluation`); LLM_BACKEND or the flags switch either to Bedrock.
RAGAS asks the judge for structured output through instructor: JSON-schema mode on vLLM (guided decoding),
tools mode on Bedrock (gpt-oss returns malformed JSON in RAGAS's default JSON mode).
"""

from __future__ import annotations

import asyncio
import math
import os
import statistics
from dataclasses import dataclass

from rag.llm import LLMError, answer, get_backend
from rag.llm.factory import load_llm_params


def article_context(article: dict) -> str:
    """One article as the judge sees it: number, Arabic and English text (or the repeal note)."""
    body = "\n".join(x for x in (article.get("text_ar") or article.get("repeal_note_ar"),
                                 article.get("text_en") or article.get("repeal_note")) if x)
    return f"[Article {article['article_number']}]\n{body}"


def make_judge(params: dict, backend: str):
    """(RAGAS judge LLM, its async client) for `backend`. Create it inside the event loop that uses it, and close the
    client there: instructor tools mode on Bedrock, JSON-schema (guided decoding) on vLLM."""
    import instructor
    from openai import AsyncOpenAI
    from ragas.llms.base import InstructorLLM

    cfg = params[backend]
    if backend == "bedrock":
        client = AsyncOpenAI(api_key=os.environ["Bedrock_API_key"],
                             base_url=os.environ.get("OPENAI_BASE_URL", cfg["base_url"]))
        mode = instructor.Mode.TOOLS
    else:
        client = AsyncOpenAI(api_key="EMPTY", base_url=os.environ.get("VLLM_BASE_URL", cfg["base_url"]))
        mode = instructor.Mode.JSON_SCHEMA
    llm = InstructorLLM(client=instructor.from_openai(client, mode=mode), model=cfg["model"],
                        provider="openai", max_tokens=2048, temperature=0.0)
    return llm, client


@dataclass(frozen=True)
class Judged:
    answer: str
    score: float  # NaN when the answer or the judgement failed


def summarize(rows: list[dict]) -> dict[str, float]:
    """faithfulness overall and per language, plus how many answers were judged (rows: lang, faithfulness)."""
    vals = [r for r in rows if not math.isnan(r["faithfulness"])]
    out: dict[str, float] = {"faithfulness_judged": float(len(vals))}
    if vals:
        out["faithfulness"] = statistics.fmean(r["faithfulness"] for r in vals)
        for lang in ("ar", "en"):
            lv = [r["faithfulness"] for r in vals if r["lang"] == lang]
            if lv:
                out[f"faithfulness_{lang}"] = statistics.fmean(lv)
    return out


class FaithfulnessScorer:
    def __init__(self, articles: dict[int, dict], *, answer_backend: str = "vllm", judge_backend: str = "vllm",
                 concurrency: int = 8) -> None:
        self.params = load_llm_params()
        self.backend = get_backend(self.params, backend=answer_backend)
        self.judge_backend = judge_backend
        self.answer_model = self.params[answer_backend]["model"]
        self.judge_model = self.params[judge_backend]["model"]
        self.articles = articles
        self.concurrency = concurrency
        self.cache: dict[tuple, Judged] = {}

    def _metric(self):
        """Built per batch, so the async client never outlives its event loop."""
        from ragas.metrics.collections import Faithfulness

        llm, client = make_judge(self.params, self.judge_backend)
        return Faithfulness(llm=llm), client

    async def _one(self, metric, sem: asyncio.Semaphore, question: str, numbers: list[int]) -> Judged:
        key = (question, tuple(numbers))
        if key not in self.cache:  # same question + same articles → same answer: judge it once
            async with sem:
                context = [self.articles[n] for n in numbers]
                text, score = "", float("nan")
                for attempt in range(3):  # the client already retries twice; this covers longer blips
                    try:
                        result = await asyncio.to_thread(answer, self.backend, question, context,
                                                         max_tokens=self.params["max_tokens"], temperature=0.0)
                        text = result.text
                        break
                    except LLMError as e:
                        print(f"  answer failed (attempt {attempt + 1}/3) on {question[:40]!r}: {e}"[:300])
                        await asyncio.sleep(5 * (attempt + 1))
                if text.strip():
                    try:
                        score = (await metric.ascore(user_input=question, response=text,
                                                     retrieved_contexts=[article_context(a) for a in context])).value
                    except Exception as e:  # noqa: BLE001 - one failed judgement must not sink the run
                        print(f"  faithfulness failed on {question[:40]!r}: {type(e).__name__}: {e}"[:300])
                self.cache[key] = Judged(text, float(score))
        return self.cache[key]

    def score(self, items: list[tuple[str, list[int]]]) -> list[Judged]:
        """(question, article numbers sent to the model) → answer + faithfulness, in order."""
        async def run():
            (metric, client), sem = self._metric(), asyncio.Semaphore(self.concurrency)
            try:
                return await asyncio.gather(*(self._one(metric, sem, q, nums) for q, nums in items))
            finally:
                await client.close()  # inside the loop: closing after asyncio.run ends raises noisy errors
        return asyncio.run(run())
