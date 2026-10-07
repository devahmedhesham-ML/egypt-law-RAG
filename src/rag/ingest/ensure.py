"""Make sure a usable index exists: local → dvc pull → build.

    python -m rag.ingest.ensure [--no-pull] [--no-build]

1. Local: data/index/chroma holds at least one vector per article in data/processed/articles.json → use it.
2. Pull: `dvc pull build_corpus index` from the public S3 remote (no credentials) → check again.
3. Build: the corpus first if it is missing (pull it, else build it from the PDF, pulled if needed), then
   `python -m rag.ingest --yes` (GPU when one is free, CPU otherwise).

Prints which source was used and exits 0, or 1 when every allowed step failed. CI and the quality gate run it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import yaml

from rag.ingest.store import collection_name, index_count

REPO_ROOT = Path(__file__).resolve().parents[3]


def _params() -> dict:
    return yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))


def index_status(root: Path = REPO_ROOT) -> tuple[bool, str]:
    """(usable, why) for the index under `root`."""
    params = _params()
    articles = root / params["corpus"]["output"]
    if not articles.exists():
        return False, f"{params['corpus']['output']} is missing"
    n = len(json.loads(articles.read_text(encoding="utf-8")))
    cfg = params["ingest"]
    count = index_count(root / cfg["index_dir"], collection_name(cfg["collection"], cfg["model"]))
    if count is None:
        return False, f"no index at {cfg['index_dir']}"
    if count < n:
        return False, f"index holds {count:,} vectors for {n:,} articles (incomplete or stale)"
    return True, f"{count:,} vectors for {n:,} articles"


def _run(cmd: list[str], log: Callable[[str], None]) -> bool:
    log(f"$ {' '.join(cmd)}")
    done = subprocess.run(cmd, cwd=REPO_ROOT)
    return done.returncode == 0


def ensure_index(*, pull: bool = True, build: bool = True, log: Callable[[str], None] = print,
                 status: Callable[[], tuple[bool, str]] = index_status,
                 run: Callable[[list[str]], bool] | None = None) -> str | None:
    """Returns "local", "pulled" or "built", or None if no allowed step produced a usable index."""
    run = run or (lambda cmd: _run(cmd, log))
    py = sys.executable
    ok, why = status()
    if ok:
        log(f"✔ local index: {why}")
        return "local"
    log(f"· local index not usable: {why}")
    if pull:
        if run([py, "-m", "dvc", "pull", "build_corpus", "index"]):
            ok, why = status()
            if ok:
                log(f"✔ pulled from the DVC remote: {why}")
                return "pulled"
        log(f"· pull did not give a usable index: {why if not ok else 'dvc pull failed'}")
    if not build:
        return None
    corpus = REPO_ROOT / _params()["corpus"]["output"]
    if not corpus.exists():
        log("· building: corpus missing, pulling it (or building it from the PDF)")
        if not (pull and run([py, "-m", "dvc", "pull", "build_corpus"]) and corpus.exists()):
            pdf_dvc = REPO_ROOT / (_params()["corpus"]["pdf"] + ".dvc")
            if pull:
                run([py, "-m", "dvc", "pull", str(pdf_dvc.relative_to(REPO_ROOT))])
            if not run([py, "-m", "rag.corpus.build"]):
                log("✗ could not build the corpus (is the source PDF available?)")
                return None
    log("· building the index: python -m rag.ingest --yes")
    if run([py, "-m", "rag.ingest", "--yes"]):
        ok, why = status()
        if ok:
            log(f"✔ built: {why}")
            return "built"
    log(f"✗ build did not give a usable index: {why}")
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Make sure a usable index exists: local → dvc pull → build")
    ap.add_argument("--no-pull", action="store_true", help="skip the DVC remote (go straight to building)")
    ap.add_argument("--no-build", action="store_true", help="never build (fail if neither local nor pull works)")
    args = ap.parse_args(argv)
    source = ensure_index(pull=not args.no_pull, build=not args.no_build)
    print(f"index source: {source or 'none'}")
    return 0 if source else 1


if __name__ == "__main__":
    sys.exit(main())
