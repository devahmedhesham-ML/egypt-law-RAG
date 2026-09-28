"""Index the corpus: python -m rag.ingest [--yes] [--stop-on-warning] [--replicas N] [--cpu]

articles.json → checks (you decide on warnings) → one bilingual chunk per article → GPU embedding
with as many replicas as fit safely → Chroma at data/index/chroma → smoke queries → index_report.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from rag.corpus.build import print_issues, write_json
from rag.corpus.issues import Issue, warning
from rag.corpus.validate import validate_records
from rag.ingest.chunks import build_chunks, query_text
from rag.ingest.embed import GpuEmbedder, weights_mb
from rag.ingest.store import collection_name, search, write_index

REPO_ROOT = Path(__file__).resolve().parents[3]

# (question, article that should come back first); both languages
SMOKE_QUERIES = [
    ("At what age does a person reach legal majority?", 44),
    ("ما حكم هبة الأموال المستقبلة؟", 492),
    ("How does the Civil Code define a partnership?", 505),
    ("هل يجوز للزوج الرجوع في هبته لزوجته؟", 502),
]


def gate(n_warnings: int, *, yes: bool, stop_on_warning: bool, interactive: bool, ask=input) -> bool:
    """Decide whether to continue past warnings: --stop-on-warning > --yes > ask the person > refuse."""
    if not n_warnings:
        return True
    if stop_on_warning:
        return False
    if yes:
        return True
    if not interactive:
        return False
    return ask(f"{n_warnings} warnings. Continue with chunking and embedding? [y/N] ").strip().lower() in ("y", "yes")


def report_warnings(report_path: Path, articles_path: Path, produced_codes: set[str]) -> list[Issue]:
    """Build-time warnings (layout, markers) from corpus_report.json, if it describes this articles.json."""
    if not report_path.exists():
        return [warning("REPORT_MISSING", f"{report_path.name} not found: run python -m rag.corpus.build")]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("articles_sha256") != hashlib.sha256(articles_path.read_bytes()).hexdigest():
        return [warning("REPORT_STALE", f"{report_path.name} was built for a different articles.json: rebuild")]
    return [Issue(i["level"], i["code"], i["message"], i.get("article"), i.get("page"))
            for i in report["issues"] if i["level"] == "warning" and i["code"] not in produced_codes]


def main(argv: list[str] | None = None) -> int:
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    cfg, ccfg = params["ingest"], params["corpus"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="continue past warnings without asking (DVC, CI)")
    ap.add_argument("--stop-on-warning", action="store_true", help="stop if there is any warning")
    ap.add_argument("--replicas", type=int, help="replicas to use (capped at what fits safely); default: all that fit")
    ap.add_argument("--max-replicas", type=int, default=cfg["max_replicas"])
    ap.add_argument("--tokens-per-batch", type=int, default=cfg["tokens_per_batch"])
    ap.add_argument("--cpu", action="store_true", help="embed on CPU")
    args = ap.parse_args(argv)

    console = Console()
    started = time.perf_counter()
    articles_path = REPO_ROOT / ccfg["output"]
    if not articles_path.exists():
        console.print(f"[red]✗ {ccfg['output']} not found. Run python -m rag.corpus.build first.")
        return 2
    records = json.loads(articles_path.read_text(encoding="utf-8"))

    # 1) checks on the corpus, plus the build's layout warnings
    issues = validate_records(records, ccfg)
    issues += report_warnings(REPO_ROOT / ccfg["report"], articles_path, {i.code for i in issues})

    # 2) chunks, and whether any would be truncated by the model
    chunks = build_chunks(records, cfg["normalize_arabic"])
    if weights_mb(cfg["model"]) != 1500.0:  # weights cached locally: no Hub round-trips
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    tokens = [len(ids) for ids in tokenizer([c.embed_text for c in chunks])["input_ids"]]
    for c, t in zip(chunks, tokens):
        if t > cfg["max_chunk_tokens"]:
            issues.append(warning("CHUNK_TOO_LONG", f"{t:,} tokens > {cfg['max_chunk_tokens']:,}: the model would "
                                                    "cut the end of this article", article=c.article_number))
    t = Table(title="Chunks: one per article, Arabic + English + heading path", show_header=False, title_justify="left")
    t.add_row("Chunks", f"{len(chunks):,} ({sum(r['is_repealed'] for r in records)} repealed, indexed with their note)")
    t.add_row("Tokens", f"{sum(tokens):,} total · median {statistics.median(tokens):.0f} · max {max(tokens):,} "
                        f"(Article {chunks[tokens.index(max(tokens))].article_number})")
    t.add_row("Model", f"{cfg['model']} · Arabic normalization {'on' if cfg['normalize_arabic'] else 'off'}")
    console.print(t)

    # 3) the gate: warnings are shown, the person decides
    n_warnings = sum(i.level == "warning" for i in issues)
    print_issues(console, [i for i in issues if i.level == "warning"])
    if not gate(n_warnings, yes=args.yes, stop_on_warning=args.stop_on_warning, interactive=sys.stdin.isatty()):
        console.print(f"[yellow]Stopped before embedding ({n_warnings} warnings). Review them, then rerun "
                      "(add --yes to continue past them).")
        return 3

    # 4) embed
    embedder = GpuEmbedder(cfg["model"], tokens_per_batch=args.tokens_per_batch, max_replicas=args.max_replicas,
                           margin_gb=cfg["vram_margin_gb"], max_seq_len=cfg["max_chunk_tokens"],
                           replicas=args.replicas, force_cpu=args.cpu, log=console.print)
    smoke_texts = [query_text(q, cfg["normalize_arabic"]) for q, _ in SMOKE_QUERIES]
    vectors, query_vectors, stats = embedder.run([c.embed_text for c in chunks], tokens, smoke_texts)

    # 5) write Chroma
    index_dir = REPO_ROOT / cfg["index_dir"]
    name = collection_name(cfg["collection"], cfg["model"])
    corpus_sha = hashlib.sha256(articles_path.read_bytes()).hexdigest()
    count = write_index(index_dir, name, chunks, vectors, {
        "embedding_model": cfg["model"], "normalize_arabic": cfg["normalize_arabic"],
        "corpus_sha256": corpus_sha, "built_at": datetime.now(UTC).isoformat(timespec="seconds")})

    # 6) smoke queries: expected article on top, 10 distinct articles
    smoke = []
    st = Table(title="Smoke queries", title_justify="left")
    for col in ("query", "expected", "top 3 (score)", "rank", "distinct/10"):
        st.add_column(col)
    for (question, expected), hits in zip(SMOKE_QUERIES, search(index_dir, name, query_vectors, k=10)):
        numbers = [h["article_number"] for h in hits]
        rank = numbers.index(expected) + 1 if expected in numbers else None
        smoke.append({"query": question, "expected": expected, "rank": rank, "top": numbers,
                      "scores": [h["score"] for h in hits]})
        ok = "[green]" if rank == 1 else "[yellow]" if rank else "[red]"
        st.add_row(question, str(expected), ", ".join(f"{h['article_number']} ({h['score']:.2f})" for h in hits[:3]),
                   f"{ok}{rank or '—'}[/]", str(len(set(numbers))))
    console.print(st)

    # 7) report and summary
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": cfg["model"], "collection": name, "index_dir": cfg["index_dir"], "count": count,
        "dimensions": int(vectors.shape[1]), "normalize_arabic": cfg["normalize_arabic"],
        "corpus_articles_sha256": corpus_sha,
        "chunks": {"count": len(chunks), "tokens_total": sum(tokens), "tokens_max": max(tokens),
                   "tokens_median": statistics.median(tokens)},
        "embedding": stats.to_dict(),
        "smoke": smoke,
        "smoke_top1": sum(s["rank"] == 1 for s in smoke),
        "warnings": n_warnings,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    write_json(REPO_ROOT / cfg["report"], report)
    e = stats
    s = Table(title="Embedding", show_header=False, title_justify="left")
    s.add_row("Device", f"{e.device} {e.gpu or ''} · {e.replicas} replica{'s' if e.replicas != 1 else ''} "
                        f"· ~{e.per_replica_mb / 1024:.2f} GB each")
    s.add_row("Throughput", f"{e.texts_per_s:,.1f} chunks/s · {e.tokens_per_s:,.0f} tokens/s "
                            f"({e.batches} batches ≤{e.tokens_per_batch:,} tokens)")
    s.add_row("VRAM", f"peak {e.peak_used_mb / 1024:.1f} GB of {e.total_mb / 1024:.1f} GB "
                      f"(free before {e.free_before_mb / 1024:.1f} GB, margin {e.margin_mb / 1024:.1f} GB)")
    s.add_row("Time", f"model load {e.load_s:.1f} s · embedding {e.embed_s:.1f} s · total {report['elapsed_s']:.1f} s")
    console.print(s)
    console.print(f"[green]✔[/] Indexed {count:,} articles into {cfg['index_dir']} (collection {name}) in "
                  f"{report['elapsed_s']:.1f} s · smoke top-1 {report['smoke_top1']}/{len(smoke)}")
    if sys.stdout.isatty():
        print("\a", end="", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
