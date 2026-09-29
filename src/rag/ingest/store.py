"""Chroma index: one persistent collection per embedding model, cosine space, rebuilt from scratch each run."""

from __future__ import annotations

import atexit
import re
import shutil
import tempfile
import threading
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rag.ingest.chunks import Chunk


def collection_name(base: str, model: str) -> str:
    """'civil_code' + 'Qwen/Qwen3-Embedding-0.6B' → 'civil_code__qwen3-embedding-0.6b'."""
    return re.sub(r"[^a-z0-9._-]+", "-", f"{base}__{model.split('/')[-1].lower()}").strip("-._")[:120]


def _client(path: Path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


_READ_COPIES: dict[tuple[Path, int], Path] = {}
_READ_LOCK = threading.Lock()


def _read_client(path: Path):
    """Chroma rewrites its SQLite file even when only queried, so every search would make DVC report the
    index as changed. Reads go to a private copy (~25 MB), made once per process and per index version."""
    src = path.resolve()
    key = (src, (src / "chroma.sqlite3").stat().st_mtime_ns)
    with _READ_LOCK:
        if key not in _READ_COPIES:
            tmp = Path(tempfile.mkdtemp(prefix="chroma-read-"))
            atexit.register(shutil.rmtree, tmp, True)
            shutil.copytree(src, tmp / "index")
            _READ_COPIES[key] = tmp / "index"
        return _client(_READ_COPIES[key])


def write_index(path: Path, name: str, chunks: list[Chunk], vectors: np.ndarray, metadata: dict,
                batch: int = 256) -> int:
    """Replace the index directory with a fresh collection (no stale articles survive a rebuild)."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    col = _client(path).create_collection(
        name, configuration={"hnsw": {"space": "cosine"}}, metadata=metadata, embedding_function=None)
    for start in tqdm(range(0, len(chunks), batch), desc="Writing Chroma", unit="batch", dynamic_ncols=True):
        part = chunks[start: start + batch]
        col.add(
            ids=[c.id for c in part],
            embeddings=vectors[start: start + batch],
            documents=[c.text for c in part],
            metadatas=[c.metadata for c in part],
        )
    return col.count()


def search(path: Path, name: str, query_vectors: np.ndarray, k: int = 10) -> list[list[dict]]:
    """Top-k articles per query: [{'article_number', 'score' (cosine similarity), 'citation', ...}]."""
    col = _read_client(path).get_collection(name)
    res = col.query(query_embeddings=query_vectors, n_results=k, include=["distances", "metadatas"])
    return [
        [{**meta, "score": round(1 - dist, 4)} for meta, dist in zip(metas, dists)]
        for metas, dists in zip(res["metadatas"], res["distances"])
    ]


def best_per_article(hits: list[dict]) -> list[dict]:
    """Chunk hits (best first) → one hit per article, keeping each article's best chunk and the order."""
    best: dict[int, dict] = {}
    for h in hits:
        best.setdefault(h["article_number"], h)
    return list(best.values())


def index_count(path: Path, name: str) -> int | None:
    """Vectors in the collection, or None if the index is missing."""
    if not (path / "chroma.sqlite3").exists():
        return None
    try:
        return _read_client(path).get_collection(name).count()
    except Exception:  # noqa: BLE001 - missing collection or unreadable index
        return None
