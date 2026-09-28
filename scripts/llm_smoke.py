"""Smoke-test a real backend: one Arabic and one English question, streamed.

Usage (app venv, from the repo root):
    python scripts/llm_smoke.py --backend vllm
    python scripts/llm_smoke.py --backend bedrock
"""

from __future__ import annotations

import argparse
import sys
import time

from langfuse import propagate_attributes

from rag import tracing
from rag.llm import AnswerResult, LLMError, answer_stream, get_backend
from rag.llm.factory import load_llm_params

# Two real articles (text from the source PDF) so the citation check has a known answer set.
ARTICLES = [
    {
        "article_number": 492,
        "text_ar": "تقع هبة الأموال المستقبلة باطلة.",
        "text_en": "A gift of future property is void.",
    },
    {
        "article_number": 505,
        "text_ar": "الشركة عقد بمقتضاه يلتزم شخصان أو أكثر بأن يساهم كل منهم في مشروع مالي، "
        "بتقديم حصة من مال أو من عمل، لاقتسام ما قد ينشأ عن هذا المشروع من ربح أو من خسارة.",
        "text_en": "Partnership is a contract by which two or more persons undertake to contribute "
        "jointly in an undertaking of a pecuniary nature by the provision of contributions of "
        "property or services, with the object of sharing in the profits or the losses of the undertaking.",
    },
]
QUESTIONS = [
    "ما حكم هبة الأموال المستقبلة؟",
    "How does the Civil Code define a partnership?",
    "What is the penalty for theft?",  # not in the context: must decline, not guess
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["bedrock", "vllm"])
    args = parser.parse_args()

    params = load_llm_params()
    backend = get_backend(params, backend=args.backend)
    print(f"backend={backend.name} model={backend.model}\n")

    failed = False
    lf = tracing.client()
    session = f"smoke-{backend.name}-{time.strftime('%Y%m%d-%H%M%S')}"  # the run's traces, grouped
    try:
        for q in QUESTIONS:  # one answer-question trace per question
            print(f"Q: {q}\nA: ", end="", flush=True)
            with lf.start_as_current_observation(as_type="span", name="answer-question",
                                                 input={"question": q}) as root, propagate_attributes(
                trace_name="answer-question", session_id=session, tags=["smoke", backend.name],
                metadata=tracing.str_metadata(backend=backend.name, context_mode="fixed"),
            ):
                try:
                    for item in answer_stream(
                        backend, q, ARTICLES, max_tokens=params["max_tokens"], temperature=params["temperature"]
                    ):
                        if isinstance(item, AnswerResult):
                            r = item
                        else:
                            print(item, end="", flush=True)
                except LLMError as e:
                    root.update(level="ERROR", status_message=str(e))
                    print(f"\nERROR: {e}")
                    return 1
                root.update(output={"answer": r.text, "cited_articles": r.citations.cited})
            c = r.citations
            print(
                f"\n   citations={c.cited} invalid={c.invalid} | tokens in/out={r.llm.input_tokens}/"
                f"{r.llm.output_tokens} | {r.llm.latency_s:.1f}s | stop={r.llm.stop_reason}\n"
            )
            failed |= not c.ok
    finally:
        lf.flush()
    print("FAIL: hallucinated citations" if failed else "OK: all citations are from the provided articles")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
