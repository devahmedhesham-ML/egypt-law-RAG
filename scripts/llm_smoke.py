"""Smoke-test a real backend: one Arabic and one English question, streamed.

Usage (app venv, from the repo root):
    python scripts/llm_smoke.py --backend vllm
    python scripts/llm_smoke.py --backend bedrock
"""

from __future__ import annotations

import argparse
import sys

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
    for q in QUESTIONS:
        print(f"Q: {q}\nA: ", end="", flush=True)
        try:
            for item in answer_stream(
                backend, q, ARTICLES, max_tokens=params["max_tokens"], temperature=params["temperature"]
            ):
                if isinstance(item, AnswerResult):
                    r = item
                else:
                    print(item, end="", flush=True)
        except LLMError as e:
            print(f"\nERROR: {e}")
            return 1
        c = r.citations
        print(
            f"\n   citations={c.cited} invalid={c.invalid} | tokens in/out={r.llm.input_tokens}/"
            f"{r.llm.output_tokens} | {r.llm.latency_s:.1f}s | stop={r.llm.stop_reason}\n"
        )
        failed |= not c.ok
    print("FAIL: hallucinated citations" if failed else "OK: all citations are from the provided articles")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
