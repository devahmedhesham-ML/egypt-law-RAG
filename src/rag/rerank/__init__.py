"""Re-ranker: re-orders the retrieved candidates with a Qwen3-Reranker cross-encoder (rubric Module 4).

Qwen3-Reranker is a causal LM asked "does this Document answer the Query? yes/no"; the relevance score is the logit
of "yes" minus the logit of "no" at the last position. The teacher (4B) and the distilled student (0.6B, trained by
rag.rerank.distill) share this format, so one scorer serves both.
"""

from __future__ import annotations

import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
          "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCT = "Given a question about Egyptian civil law, judge whether this Civil Code article answers it"


def load_cfg() -> dict:
    return yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))["rerank"]


def document(article: dict) -> str:
    """The article as the re-ranker reads it: Arabic and English text (or the repeal note)."""
    body = "\n".join(x for x in (article.get("text_ar") or article.get("repeal_note_ar"),
                                 article.get("text_en") or article.get("repeal_note")) if x)
    return f"Article {article['article_number']}\n{body}"


class Scorer:
    """Relevance logits for (query, document) pairs from a Qwen3-Reranker checkpoint."""

    def __init__(self, model: str, *, device: str | None = None, max_doc_tokens: int = 384, train: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device)
        self.model.train(train)
        self.yes, self.no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")
        self.prefix = self.tok.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tok.encode(SUFFIX, add_special_tokens=False)
        self.max_doc_tokens = max_doc_tokens

    def encode(self, query: str, docs: list[str]):
        rows = []
        for d in docs:
            body = self.tok.encode(f"<Instruct>: {INSTRUCT}\n<Query>: {query}\n<Document>: ", add_special_tokens=False)
            doc = self.tok.encode(d, add_special_tokens=False)[:self.max_doc_tokens]
            rows.append(self.prefix + body + doc + self.suffix)
        width = max(map(len, rows))
        pad = self.tok.pad_token_id
        ids = [[pad] * (width - len(r)) + r for r in rows]
        mask = [[0] * (width - len(r)) + [1] * len(r) for r in rows]
        t = self.torch.tensor
        return t(ids, device=self.device), t(mask, device=self.device)

    def logits(self, query: str, docs: list[str]):
        """Tensor of relevance logits (yes − no), with gradients when the model is training."""
        ids, mask = self.encode(query, docs)
        last = self.model(input_ids=ids, attention_mask=mask).logits[:, -1, :]
        return (last[:, self.yes] - last[:, self.no]).float()

    def scores(self, query: str, docs: list[str]) -> list[float]:
        with self.torch.inference_mode():
            return self.logits(query, docs).tolist()


class Reranker:
    """The pipeline's re-ranker: the distilled student if it exists, else the base student."""

    def __init__(self, model: str | None = None) -> None:
        cfg = load_cfg()
        trained = REPO_ROOT / cfg["model_dir"]
        self.model_name = model or (str(trained) if (trained / "config.json").exists() else cfg["student"])
        self.scorer = Scorer(self.model_name, max_doc_tokens=cfg["max_doc_tokens"])
        self.last_latency_s = 0.0

    def rerank(self, question: str, hits: list, articles: dict[int, dict]) -> list:
        start = time.perf_counter()
        scores = self.scorer.scores(question, [document(articles[h.article_number]) for h in hits])
        self.last_latency_s = time.perf_counter() - start
        return [h for _, h in sorted(zip(scores, hits), key=lambda p: -p[0])]
