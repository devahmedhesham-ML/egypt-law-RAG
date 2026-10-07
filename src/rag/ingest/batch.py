"""Batch re-indexing: add new or changed articles to an existing index without rebuilding it.

    python -m rag.ingest.batch NEW.json [--index-dir DIR] [--corpus FILE] [--dry-run]

NEW.json is a list of records in the articles.json schema (e.g. a newly added or amended article). Each record
is chunked exactly like a full build (params.yaml ingest.chunking). A chunk whose id is new is added, one whose
text changed is replaced, an unchanged one is skipped, so only added or changed chunks are embedded and a re-run
does nothing. The records are also merged into the corpus file, because the API only cites articles it has.

The default targets are the production index and corpus, which DVC tracks: after a batch update there, record
it with `dvc commit` + `dvc push` (or rebuild everything with `dvc repro`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from rag.ingest.chunks import build_chunks
from rag.ingest.store import _client, collection_name

REPO_ROOT = Path(__file__).resolve().parents[3]
REQUIRED = ("article_number", "is_repealed", "citation", "source_pages")


@dataclass(frozen=True)
class BatchResult:
    added: list[str]
    changed: list[str]
    unchanged: list[str]
    count_before: int
    count_after: int
    embed_s: float


def validate(records: list[dict]) -> list[str]:
    """Problems that would make a record unusable (an empty list means all good)."""
    problems = []
    for i, r in enumerate(records):
        where = f"record {i} (article {r.get('article_number', '?')})"
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            problems.append(f"{where}: missing {', '.join(missing)}")
            continue
        if not isinstance(r["article_number"], int):
            problems.append(f"{where}: article_number must be an integer")
        text = (r.get("repeal_note_ar"), r.get("repeal_note")) if r["is_repealed"] else (r.get("text_ar"), r.get("text_en"))
        if not any(text):
            problems.append(f"{where}: no article text")
    return problems


def default_encoder(model_name: str) -> Callable[[list[str]], np.ndarray]:
    """Documents are embedded like a full build: no query prompt, normalized; GPU if one has room, else CPU."""
    from sentence_transformers import SentenceTransformer

    from rag.ingest.embed import gpu_memory
    from rag.retrieval import MIN_FREE_GPU_MB

    mem = gpu_memory()
    device = "cuda" if mem and mem.free_mb >= MIN_FREE_GPU_MB else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return lambda texts: np.asarray(model.encode(texts, normalize_embeddings=True, batch_size=8), dtype=np.float32)


def batch_reindex(records: list[dict], index_dir: Path, corpus_path: Path | None, *, dry_run: bool = False,
                  encode: Callable[[list[str]], np.ndarray] | None = None, cfg: dict | None = None) -> BatchResult:
    cfg = cfg or yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))["ingest"]
    chunking = cfg.get("chunking") or {"strategy": "article"}
    tokenizer = None
    if chunking["strategy"] == "window":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    chunks = build_chunks(records, cfg["normalize_arabic"], strategy=chunking["strategy"],
                          chunk_size=chunking.get("chunk_size"), overlap=chunking.get("overlap") or 0,
                          tokenizer=tokenizer)
    col = _client(index_dir).get_collection(collection_name(cfg["collection"], cfg["model"]))
    before = col.count()
    existing = col.get(ids=[c.id for c in chunks], include=["documents"])
    old_text = dict(zip(existing["ids"], existing["documents"]))
    added = [c for c in chunks if c.id not in old_text]
    changed = [c for c in chunks if c.id in old_text and old_text[c.id] != c.text]
    unchanged = [c.id for c in chunks if c.id in old_text and old_text[c.id] == c.text]
    todo = added + changed
    embed_s = 0.0
    if todo and not dry_run:
        encoder = encode or default_encoder(cfg["model"])  # model loading is not counted as embedding time
        started = time.perf_counter()
        vectors = encoder([c.embed_text for c in todo])
        embed_s = time.perf_counter() - started
        col.upsert(ids=[c.id for c in todo], embeddings=vectors, documents=[c.text for c in todo],
                   metadatas=[c.metadata for c in todo])
        if corpus_path is not None:
            merge_into_corpus(corpus_path, records)
    return BatchResult([c.id for c in added], [c.id for c in changed], unchanged, before, col.count(), embed_s)


def merge_into_corpus(corpus_path: Path, records: list[dict]) -> None:
    """Replace records with the same article number, append new ones, keep the corpus sorted; atomic write."""
    corpus = {r["article_number"]: r for r in json.loads(corpus_path.read_text(encoding="utf-8"))}
    corpus.update({r["article_number"]: r for r in records})
    tmp = corpus_path.with_suffix(corpus_path.suffix + ".tmp")
    tmp.write_text(json.dumps([corpus[n] for n in sorted(corpus)], ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    tmp.replace(corpus_path)


def main(argv: list[str] | None = None) -> int:
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description="Add new or changed articles to an existing index")
    ap.add_argument("records", type=Path, help="JSON list of article records (articles.json schema)")
    ap.add_argument("--index-dir", type=Path, default=REPO_ROOT / params["ingest"]["index_dir"])
    ap.add_argument("--corpus", type=Path, default=REPO_ROOT / params["corpus"]["output"],
                    help="corpus file to merge the records into (the API cites only articles it has)")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args(argv)

    records = json.loads(args.records.read_text(encoding="utf-8"))
    problems = validate(records)
    if problems:
        print("✗ invalid records:\n  " + "\n  ".join(problems))
        return 2
    r = batch_reindex(records, args.index_dir, args.corpus, dry_run=args.dry_run)
    todo = len(r.added) + len(r.changed)
    print(f"{len(records)} record(s) → {todo + len(r.unchanged)} chunk(s): {len(r.added)} added "
          f"{r.added or ''}, {len(r.changed)} changed {r.changed or ''}, {len(r.unchanged)} unchanged (skipped); "
          + (f"{todo} to embed (dry run)." if args.dry_run else
             f"{todo} embedded in {r.embed_s:.2f} s. Index: {r.count_before:,} → {r.count_after:,} vectors."))
    if not args.dry_run and (r.added or r.changed) and args.index_dir == REPO_ROOT / params["ingest"]["index_dir"]:
        print("The production index and corpus are DVC-tracked: run `dvc commit` and `dvc push` to record this update.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
