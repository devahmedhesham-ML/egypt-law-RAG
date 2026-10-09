"""Re-ranker training set: synthetic questions written by the LLM from corpus articles, plus retrieval candidates.

    python -m rag.rerank.data [--articles 300] [--backend vllm|bedrock]

1. Sample in-force articles, leaving out every article the evaluation set asks about (so the 62 test questions stay
   unseen, by question and by article).
2. The LLM (LLM_BACKEND or --backend; Qwen2.5 on vLLM by default) writes one Arabic and one English question each
   article answers, the way a non-lawyer would ask, without quoting it or naming its number.
3. Filters: both languages present, sensible length, no article number, no duplicates.
4. The production retriever fetches the top `candidates` articles per question; the source article is added when it
   was not retrieved, so the teacher always sees it.

Writes data/rerank/train.jsonl and data/rerank/val.jsonl (split by article) and a summary.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from rag.llm import Message, get_backend
from rag.llm.factory import default_backend, load_llm_params
from rag.rerank import REPO_ROOT, load_cfg

OUT = REPO_ROOT / "data" / "rerank"
ARABIC = re.compile(r"[؀-ۿ]")
SYSTEM = "You write realistic questions that ordinary people ask about Egyptian civil law. Reply with JSON only."
PROMPT = """Civil Code article:
{en}

{ar}

Write one question in Arabic and one question in English that this article answers, phrased the way a non-lawyer
would ask it about their own situation. Do not quote the article and do not mention any article number.
Reply with JSON only: {{"ar": "...", "en": "..."}}"""


def eval_articles() -> set[int]:
    rows = [json.loads(x) for x in (REPO_ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    return {n for r in rows for n in r["relevant_articles"]}


def parse(text: str, number: int) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    ar, en = str(d.get("ar", "")).strip(), str(d.get("en", "")).strip()
    ok = (ARABIC.search(ar) and not ARABIC.search(en) and 10 <= len(ar) <= 300 and 10 <= len(en) <= 300
          and str(number) not in ar + en)
    return {"ar": ar, "en": en} if ok else None


def main(argv: list[str] | None = None) -> int:
    cfg = load_cfg()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--articles", type=int, default=cfg["articles"])
    ap.add_argument("--backend", choices=["vllm", "bedrock"], default=None, help="default: LLM_BACKEND, else params")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args(argv)

    from rag.pipeline import load_corpus
    from rag.retrieval import Retriever

    articles = load_corpus()
    held_out = eval_articles()
    pool = sorted(n for n, a in articles.items() if not a.get("is_repealed") and n not in held_out)
    chosen = random.Random(cfg["seed"]).sample(pool, min(args.articles, len(pool)))
    backend = get_backend(load_llm_params(), backend=args.backend or default_backend())
    print(f"{len(chosen)} articles (of {len(pool)} in force, {len(held_out)} evaluation articles left out); "
          f"questions by {backend.name}:{backend.model}")

    def ask(n: int) -> tuple[int, dict | None]:
        a = articles[n]
        try:
            res = backend.generate(SYSTEM, [Message("user", PROMPT.format(en=a["text_en"], ar=a["text_ar"]))],
                                   max_tokens=300, temperature=0.7)
            return n, parse(res.text, n)
        except Exception as e:  # noqa: BLE001 - a failed article is skipped and counted
            print(f"  article {n}: {type(e).__name__}: {e}"[:200])
            return n, None

    with ThreadPoolExecutor(args.concurrency) as ex:
        generated = list(ex.map(ask, chosen))
    kept = [(n, q) for n, q in generated if q]

    retriever, k, seen, rows = Retriever(), cfg["candidates"], set(), []
    for n, q in kept:
        for lang in ("ar", "en"):
            if q[lang] in seen:
                continue
            seen.add(q[lang])
            cands = [h.article_number for h in retriever.retrieve(q[lang], k).hits]
            retrieved = n in cands
            if not retrieved:
                cands = cands[:k - 1] + [n]
            rows.append({"query": q[lang], "lang": lang, "positive": n, "candidates": cands,
                         "positive_retrieved": retrieved})

    val_articles = set(random.Random(cfg["seed"] + 1).sample(sorted({r["positive"] for r in rows}),
                                                             max(1, len(kept) // 10)))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", [r for r in rows if r["positive"] not in val_articles]),
                       ("val", [r for r in rows if r["positive"] in val_articles])):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part),
                                           encoding="utf-8")
    summary = {"articles": len(chosen), "articles_with_questions": len(kept), "questions": len(rows),
               "val_articles": len(val_articles), "held_out_eval_articles": len(held_out),
               "positive_retrieved_at_k": sum(r["positive_retrieved"] for r in rows) / max(1, len(rows)),
               "generator": f"{backend.name}:{backend.model}", "candidates": k}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
