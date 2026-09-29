"""Question → top-k articles from the Chroma index.

The question is embedded with the same model and Arabic normalization used at indexing time
(Qwen3's "query" prompt), then Chroma finds the nearest article chunks. Articles the question
names by number ("Article 60", "المادة رقم 801") are looked up directly and put first, because
embedding models match numbers poorly.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from rag import tracing
from rag.ingest.chunks import chunks_per_article, query_text
from rag.ingest.embed import gpu_memory, query_prompt_name
from rag.ingest.store import collection_name, search

REPO_ROOT = Path(__file__).resolve().parents[2]
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
EN_NUMBER = re.compile(r"\barticles?\s+(\d{1,4})\b", re.I)
AR_NUMBER = re.compile(r"(?:المادة|مادة|المواد)\s*(?:رقم\s*)?[(]?\s*([0-9٠-٩]{1,4})")
MIN_FREE_GPU_MB = 3000  # below this (vLLM holds most of the GPU) the query model runs on CPU


@dataclass(frozen=True)
class Hit:
    article_number: int
    score: float | None  # cosine similarity; None when matched by the number in the question
    by_number: bool = False


@dataclass(frozen=True)
class Retrieval:
    hits: list[Hit]
    device: str
    latency_s: float


def numbers_in_question(question: str, first: int = 1, last: int = 1149) -> list[int]:
    found = [int(m.group(1).translate(AR_DIGITS)) for rx in (EN_NUMBER, AR_NUMBER) for m in rx.finditer(question)]
    return [n for n in dict.fromkeys(found) if first <= n <= last]


def merge_hits(named: list[int], semantic: list[dict], k: int) -> list[Hit]:
    """Articles named in the question first, then the nearest chunks, without duplicates, k in total."""
    hits = [Hit(n, None, True) for n in named]
    seen = set(named)
    for h in semantic:
        if h["article_number"] not in seen:
            hits.append(Hit(h["article_number"], h["score"]))
            seen.add(h["article_number"])
    return hits[:max(k, len(named))]


class Retriever:
    def __init__(self, encode: Callable[[str], np.ndarray] | None = None,
                 searcher: Callable[[np.ndarray, int], list[dict]] | None = None) -> None:
        params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
        self.cfg, corpus = params["ingest"], params["corpus"]
        self.first, self.last = corpus["first_article"], corpus["last_article"]
        self.index_dir = REPO_ROOT / self.cfg["index_dir"]
        self.collection = collection_name(self.cfg["collection"], self.cfg["model"])
        self.fanout = chunks_per_article(self.cfg.get("chunking", {}).get("strategy", "article"))
        self._encode = encode
        self._searcher = searcher or (lambda v, k: search(self.index_dir, self.collection, v, k)[0])
        self._lock = threading.Lock()
        self.device = "not loaded"

    def _load(self) -> Callable[[str], np.ndarray]:
        with self._lock:
            if self._encode is None:
                with tracing.client().start_as_current_observation(
                    as_type="span", name="load-embedding-model", input={"model": self.cfg["model"]},
                ) as span:  # first question only: shows why that retrieval is slow
                    from sentence_transformers import SentenceTransformer

                    mem = gpu_memory()
                    self.device = "cuda" if mem and mem.free_mb >= MIN_FREE_GPU_MB else "cpu"
                    kwargs = {}
                    if self.device == "cuda":
                        import torch

                        kwargs["model_kwargs"] = {"dtype": torch.float16}
                    model = SentenceTransformer(self.cfg["model"], device=self.device, **kwargs)
                    normalize = self.cfg["normalize_arabic"]
                    prompt = query_prompt_name(model)
                    self._encode = lambda q: model.encode(
                        [query_text(q, normalize)], prompt_name=prompt, normalize_embeddings=True)
                    span.update(output={"device": self.device},
                                metadata={"free_gpu_mb": mem.free_mb if mem else None,
                                          "min_free_gpu_mb": MIN_FREE_GPU_MB})
        return self._encode

    def retrieve(self, question: str, k: int = 5) -> Retrieval:
        """Traced as `retrieve-articles` (retriever) with an `embed-question` (embedding) step inside."""
        started = time.perf_counter()
        lf = tracing.client()
        with lf.start_as_current_observation(
            as_type="retriever", name="retrieve-articles", input={"question": question, "top_k": k},
            metadata={"collection": self.collection, "embedding_model": self.cfg["model"]},
        ) as span:
            named = numbers_in_question(question, self.first, self.last)
            encode = self._load()
            with lf.start_as_current_observation(
                as_type="embedding", name="embed-question", model=self.cfg["model"], input=question,
                metadata={"device": self.device, "prompt": "query"},
            ) as emb:
                vector = np.asarray(encode(question), dtype=np.float32)
                emb.update(output={"dimensions": int(vector.shape[-1])})
            semantic = self._searcher(vector, (k + len(named)) * self.fanout)  # chunks → k distinct articles
            hits = merge_hits(named, semantic, k)
            latency = round(time.perf_counter() - started, 3)
            span.update(
                output=[{"article_number": h.article_number, "score": h.score, "by_number": h.by_number} for h in hits],
                metadata={"named_in_question": named, "device": self.device, "latency_s": latency},
            )
        return Retrieval(hits, self.device, latency)
