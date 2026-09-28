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

from rag.ingest.chunks import query_text
from rag.ingest.embed import gpu_memory
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
        self._encode = encode
        self._searcher = searcher or (lambda v, k: search(self.index_dir, self.collection, v, k)[0])
        self._lock = threading.Lock()
        self.device = "not loaded"

    def _load(self) -> Callable[[str], np.ndarray]:
        with self._lock:
            if self._encode is None:
                from sentence_transformers import SentenceTransformer

                mem = gpu_memory()
                self.device = "cuda" if mem and mem.free_mb >= MIN_FREE_GPU_MB else "cpu"
                kwargs = {}
                if self.device == "cuda":
                    import torch

                    kwargs["model_kwargs"] = {"dtype": torch.float16}
                model = SentenceTransformer(self.cfg["model"], device=self.device, **kwargs)
                normalize = self.cfg["normalize_arabic"]
                self._encode = lambda q: model.encode(
                    [query_text(q, normalize)], prompt_name="query", normalize_embeddings=True)
        return self._encode

    def retrieve(self, question: str, k: int = 5) -> Retrieval:
        started = time.perf_counter()
        named = numbers_in_question(question, self.first, self.last)
        vector = self._load()(question)
        semantic = self._searcher(np.asarray(vector, dtype=np.float32), k + len(named))
        return Retrieval(merge_hits(named, semantic, k), self.device, round(time.perf_counter() - started, 3))
