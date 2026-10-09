"""Distil a large teacher into Qwen3-Reranker-0.6B (student), then compare them on the test set.

    python -m rag.rerank.distill --teacher-only     # teacher = the LLM backend: vLLM must be up
    python -m rag.rerank.distill --skip-teacher     # train + compare: stop vLLM first, the student trains on the GPU

The teacher (params.yaml rerank.teacher) is either `llm`, the LLM backend (Qwen2.5-7B-Instruct on vLLM by default,
Bedrock with LLM_BACKEND) asked whether each article answers the question, scored log P(yes) − log P(no), or a
Qwen3-Reranker checkpoint such as Qwen/Qwen3-Reranker-4B (needs the GPU to itself, ~8 GB).

1. Teacher: scores every (question, candidate) pair of data/rerank/{train,val}.jsonl (rag.rerank.data) and of the
   evaluation questions' own top-10 candidates → data/rerank/teacher_scores.json.
2. Student: trained to reproduce the teacher's ranking: listwise KL divergence between the teacher's and the
   student's softmax over each question's candidates. The epoch with the best validation agreement is saved to
   data/rerank/student/, which the pipeline loads when params.yaml retrieval.rerank is true.
3. Test (the 58 in-scope evaluation questions, never trained on): hit@1, MRR and recall@5 of the plain retrieval
   order, the teacher, the base student and the distilled student, plus latency per question (GPU, and CPU for the
   student) → reports/reranker.md and an MLflow run in the `reranker` experiment.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import random
import statistics
import time
from datetime import UTC, datetime

from rag.evaluation.ragas_eval import retrieval_scores
from rag.llm.factory import backend_for, load_llm_params
from rag.rerank import REPO_ROOT, Scorer, document, load_cfg

DATA = REPO_ROOT / "data" / "rerank"


def read(name: str) -> list[dict]:
    return [json.loads(x) for x in (DATA / name).read_text(encoding="utf-8").splitlines() if x.strip()]


class LLMTeacher:
    """The LLM backend as teacher: "does this article answer the question? yes/no", scored by the log-probabilities
    of its first token (vLLM returns them). A backend without log-probabilities (Bedrock) falls back to the answer
    itself: +4 for yes, −4 for no."""

    SYSTEM = "You judge whether an Egyptian Civil Code article answers a question. Reply with yes or no only."

    def __init__(self, concurrency: int = 16) -> None:
        params = load_llm_params()
        self.backend = backend_for(params["backend"])
        self.cfg, self.concurrency, self.logprobs = params[self.backend], concurrency, True
        self.label = f"{self.backend}:{self.cfg['model']}"

    def _client(self):
        from openai import AsyncOpenAI

        if self.backend == "bedrock":
            return AsyncOpenAI(api_key=os.environ["Bedrock_API_key"],
                               base_url=os.environ.get("OPENAI_BASE_URL", self.cfg["base_url"]))
        return AsyncOpenAI(api_key="EMPTY", base_url=os.environ.get("VLLM_BASE_URL", self.cfg["base_url"]))

    async def _one(self, client, sem, query: str, doc: str) -> float:
        import openai

        msgs = [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": f"Question: {query}\n\nArticle:\n{doc[:2000]}\n\n"
                                            "Does this article answer the question? Reply yes or no."}]
        async with sem:
            for attempt in range(3):
                try:
                    kw = {"logprobs": True, "top_logprobs": 10} if self.logprobs else {}
                    r = await client.chat.completions.create(model=self.cfg["model"], messages=msgs, max_tokens=1,
                                                             temperature=0.0, **kw)
                    break
                except openai.BadRequestError:
                    if not self.logprobs:
                        raise
                    self.logprobs = False  # backend without log-probabilities: use the answer itself
                except openai.APIError:
                    await asyncio.sleep(2 * (attempt + 1))
            choice = r.choices[0]
            lp: dict[str, float] = {}
            if choice.logprobs and choice.logprobs.content:
                for t in choice.logprobs.content[0].top_logprobs:
                    key = t.token.strip().lower()
                    lp[key] = max(lp.get(key, -100.0), t.logprob)
            if "yes" in lp or "no" in lp:
                return lp.get("yes", -20.0) - lp.get("no", -20.0)
            return 4.0 if (choice.message.content or "").strip().lower().startswith("yes") else -4.0

    def score_rows(self, rows: list[dict], articles: dict) -> tuple[dict[str, list[float]], float]:
        """{query: scores} and the mean seconds per question (its candidates judged in parallel)."""
        async def run():
            client, sem, out, took = self._client(), asyncio.Semaphore(self.concurrency), {}, []
            try:
                for i in range(0, len(rows), 8):  # 8 questions at a time, their candidates in parallel
                    batch = rows[i:i + 8]
                    start = time.perf_counter()
                    res = await asyncio.gather(*(asyncio.gather(*(self._one(client, sem, r["query"],
                                                                            document(articles[n]))
                                                                  for n in r["candidates"])) for r in batch))
                    took.append((time.perf_counter() - start) / len(batch))
                    out.update({r["query"]: list(v) for r, v in zip(batch, res)})
                    if (i // 8) % 10 == 0:
                        print(f"  teacher scored {len(out)}/{len(rows)} questions")
                return out, statistics.fmean(took)
            finally:
                await client.close()
        return asyncio.run(run())


def test_set(k: int) -> list[dict]:
    from rag.evaluation.gate import load_questions
    from rag.retrieval import Retriever

    retriever = Retriever()
    rows = []
    for q in load_questions(REPO_ROOT / "eval" / "questions.jsonl", "all"):
        cands = [h.article_number for h in retriever.retrieve(q["question"], k).hits]
        rows.append({"id": q["id"], "query": q["question"], "lang": q["lang"], "relevant": q["relevant_articles"],
                     "candidates": cands})
    return rows


def free(scorer) -> None:
    import torch

    del scorer.model
    gc.collect()
    torch.cuda.empty_cache()


def score_all(scorer: Scorer, rows: list[dict], articles: dict) -> dict[str, list[float]]:
    out = {}
    for i, r in enumerate(rows, 1):
        out[r["query"]] = scorer.scores(r["query"], [document(articles[n]) for n in r["candidates"]])
        if i % 100 == 0:
            print(f"  scored {i}/{len(rows)}")
    return out


def order(row: dict, scores: list[float]) -> list[int]:
    return [n for _, n in sorted(zip(scores, row["candidates"]), key=lambda p: -p[0])]


def evaluate(rows: list[dict], scores: dict[str, list[float]] | None) -> dict[str, float]:
    per = [retrieval_scores(order(r, scores[r["query"]]) if scores else r["candidates"], r["relevant"]) for r in rows]
    return {m: statistics.fmean(p[m] for p in per) for m in per[0]}


def train(cfg: dict, train_rows: list[dict], val_rows: list[dict], teacher: dict, articles: dict) -> dict:
    import torch

    student = Scorer(cfg["student"], max_doc_tokens=cfg["max_doc_tokens"], train=True)
    student.model.gradient_checkpointing_enable()
    student.model.config.use_cache = False
    opt = torch.optim.AdamW(student.model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    kl = torch.nn.KLDivLoss(reduction="batchmean")
    rng, best, history = random.Random(cfg["seed"]), -1.0, []

    def agreement() -> float:
        """Validation: share of questions where the student's top candidate is the teacher's top candidate."""
        student.model.eval()
        hits = [order(r, student.scores(r["query"], [document(articles[n]) for n in r["candidates"]]))[0]
                == order(r, teacher[r["query"]])[0] for r in val_rows]
        student.model.train()
        return sum(hits) / len(hits)

    for epoch in range(1, cfg["epochs"] + 1):
        rows, losses, start = train_rows[:], [], time.perf_counter()
        rng.shuffle(rows)
        for r in rows:
            s = student.logits(r["query"], [document(articles[n]) for n in r["candidates"]])
            t = torch.tensor(teacher[r["query"]], device=s.device)
            loss = kl(torch.log_softmax(s, -1).unsqueeze(0), torch.softmax(t, -1).unsqueeze(0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        agree = agreement()
        history.append({"epoch": epoch, "train_kl": statistics.fmean(losses), "val_top1_agreement": agree,
                        "seconds": time.perf_counter() - start})
        print(f"  epoch {epoch}: train KL {history[-1]['train_kl']:.4f}, val top-1 agreement with teacher {agree:.3f}")
        if agree > best:
            best = agree
            (REPO_ROOT / cfg["model_dir"]).mkdir(parents=True, exist_ok=True)
            student.model.save_pretrained(REPO_ROOT / cfg["model_dir"])
            student.tok.save_pretrained(REPO_ROOT / cfg["model_dir"])
    free(student)
    return {"history": history, "best_val_agreement": best}


def latency(model: str, cfg: dict, rows: list[dict], articles: dict, device: str, n: int) -> float:
    scorer = Scorer(model, device=device, max_doc_tokens=cfg["max_doc_tokens"])
    docs = [[document(articles[x]) for x in r["candidates"]] for r in rows[:n]]
    scorer.scores(rows[0]["query"], docs[0])  # warm-up
    start = time.perf_counter()
    for r, d in zip(rows[:n], docs):
        scorer.scores(r["query"], d)
    took = (time.perf_counter() - start) / n
    if device == "cuda":
        free(scorer)
    return took


def main(argv: list[str] | None = None) -> int:
    cfg = load_cfg()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--teacher-only", action="store_true", help="score with the teacher, save, and stop")
    ap.add_argument("--skip-teacher", action="store_true", help="reuse data/rerank/teacher_scores.json")
    ap.add_argument("--skip-train", action="store_true", help="reuse data/rerank/student/")
    args = ap.parse_args(argv)
    os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")

    from rag.pipeline import load_corpus

    articles = load_corpus()
    train_rows, val_rows = read("train.jsonl"), read("val.jsonl")
    tests = test_set(cfg["candidates"])
    path, meta_path = DATA / "teacher_scores.json", DATA / "teacher_meta.json"
    if args.skip_teacher:
        teacher = json.loads(path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        start = time.perf_counter()
        if cfg["teacher"] == "llm":
            llm = LLMTeacher()
            teacher, per_q = llm.score_rows(train_rows + val_rows + tests, articles)
            meta = {"teacher": llm.label, "teacher_latency_s": per_q, "logprobs": llm.logprobs}
        else:
            scorer = Scorer(cfg["teacher"], max_doc_tokens=cfg["max_doc_tokens"])
            teacher = score_all(scorer, train_rows + val_rows + tests, articles)
            free(scorer)
            meta = {"teacher": cfg["teacher"], "teacher_latency_s": None}
        meta["seconds"] = time.perf_counter() - start
        path.write_text(json.dumps(teacher, ensure_ascii=False), encoding="utf-8")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"teacher {meta['teacher']} scored {len(teacher)} questions in {meta['seconds']:.0f} s")
        if args.teacher_only:
            print(json.dumps({"teacher on the test set": evaluate(tests, teacher)}))
            return 0

    training = {} if args.skip_train else train(cfg, train_rows, val_rows, teacher, articles)
    student_dir = str(REPO_ROOT / cfg["model_dir"])

    base = Scorer(cfg["student"], max_doc_tokens=cfg["max_doc_tokens"])
    base_scores = score_all(base, tests, articles)
    free(base)
    distilled = Scorer(student_dir, max_doc_tokens=cfg["max_doc_tokens"])
    distilled_scores = score_all(distilled, tests, articles)
    free(distilled)

    results = {"retrieval only (no re-ranker)": evaluate(tests, None),
               f"teacher ({meta['teacher']})": evaluate(tests, teacher),
               "base student Qwen3-Reranker-0.6B": evaluate(tests, base_scores),
               "distilled student (ours)": evaluate(tests, distilled_scores)}
    lat = {"teacher_s": meta["teacher_latency_s"] if cfg["teacher"] == "llm"
           else latency(cfg["teacher"], cfg, tests, articles, "cuda", 20),
           "student_gpu_s": latency(student_dir, cfg, tests, articles, "cuda", 20),
           "student_cpu_s": latency(student_dir, cfg, tests, articles, "cpu", 3)}
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    report = {"results": results, "latency": lat, "training": training, "data": summary, "teacher": meta["teacher"],
              "n_test": len(tests), "n_train": len(train_rows), "n_val": len(val_rows)}
    (DATA / "distill_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(report, cfg)
    log_mlflow(report, cfg)
    print(json.dumps({k: {m: round(v, 3) for m, v in r.items()} for k, r in results.items()}, indent=1))
    print(json.dumps(lat))
    return 0


def write_report(rep: dict, cfg: dict) -> None:
    d, lat = rep["data"], rep["latency"]
    lines = [
        "# Re-ranker distillation", "",
        f"Generated by `python -m rag.rerank.distill` on {datetime.now(UTC):%Y-%m-%d %H:%M} UTC.", "",
        f"**Teacher** `{rep['teacher']}` → **student** `{cfg['student']}`, trained to reproduce the teacher's ranking "
        f"(listwise KL over each question's {cfg['candidates']} retrieved candidates) for {cfg['epochs']} epochs.", "",
        "## Training data (generated, `python -m rag.rerank.data`)", "",
        f"- {d['questions']} synthetic questions ({rep['n_train']} train, {rep['n_val']} validation, split by "
        f"article) written by `{d['generator']}` from {d['articles_with_questions']} of {d['articles']} sampled "
        "in-force articles, one Arabic and one English each.",
        f"- The {d['held_out_eval_articles']} articles the evaluation set asks about were left out, so the test "
        "questions are unseen by question and by article.",
        f"- The source article was among the retrieved candidates for {d['positive_retrieved_at_k']:.0%} of the "
        "questions (added otherwise, so the teacher always sees it).", "",
        f"## Test: the {rep['n_test']} in-scope evaluation questions", "",
        f"The re-ranker re-orders the top {cfg['candidates']} retrieved articles; the top 5 go to the model.", "",
        "| Ranking | hit@1 | MRR | recall@5 |", "|---|---|---|---|"]
    for name, r in rep["results"].items():
        lines.append(f"| {name} | {r['hit_at_1']:.3f} | {r['mrr']:.3f} | {r['recall_at_5']:.3f} |")
    lines += ["", "## Latency per question (re-ranking 10 candidates)", "",
              "| Model | Device | Seconds |", "|---|---|---|",
              f"| teacher ({rep['teacher']}) | {'GPU (vLLM, 10 judgements in parallel)' if ':' in rep['teacher'] else 'GPU'} "
              f"| {lat['teacher_s']:.3f} |",
              f"| distilled student 0.6B | GPU | {lat['student_gpu_s']:.3f} |",
              f"| distilled student 0.6B | CPU | {lat['student_cpu_s']:.2f} |", ""]
    if rep["training"]:
        lines += ["## Training", "", "| Epoch | Train KL | Validation top-1 agreement with teacher | Seconds |",
                  "|---|---|---|---|"]
        lines += [f"| {h['epoch']} | {h['train_kl']:.4f} | {h['val_top1_agreement']:.3f} | {h['seconds']:.0f} |"
                  for h in rep["training"]["history"]]
    (REPO_ROOT / "reports" / "reranker.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_mlflow(rep: dict, cfg: dict) -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    from rag.evaluation.gate import _git_sha

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}"))
    mlflow.set_experiment("reranker")
    mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
    with mlflow.start_run(run_name=f"distill-{datetime.now(UTC):%Y%m%d-%H%M}"):
        mlflow.log_params({k: cfg[k] for k in ("student", "candidates", "max_doc_tokens", "epochs", "lr")}
                          | {"n_train": rep["n_train"], "n_val": rep["n_val"], "n_test": rep["n_test"]})
        short = ["retrieval", "teacher", "base_student", "student"]  # same order as rep["results"]
        mlflow.log_metrics({f"{short[i]}_{m}": v for i, r in enumerate(rep["results"].values()) for m, v in r.items()}
                           | {k: v for k, v in rep["latency"].items() if v is not None})
        mlflow.set_tag("teacher", rep["teacher"])
        mlflow.set_tag("git_sha", _git_sha())
        mlflow.log_artifact(str(REPO_ROOT / "reports" / "reranker.md"))


if __name__ == "__main__":
    raise SystemExit(main())
