"""Build the structured corpus: python -m rag.corpus.build [--pages 1-11] [--strict]

PDF → table rows (parallel) → articles + hierarchy → checks → data/processed/articles.json
and data/processed/corpus_report.json. Warnings never stop the build; review them before indexing.
Traced in Langfuse as one `build-corpus` trace (extract-pages, validate-records) with a build_warnings score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pymupdf
import yaml
from langfuse import propagate_attributes
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from rag import tracing
from rag.corpus.extract import iter_pages
from rag.corpus.issues import Issue, info, warning
from rag.corpus.parse import CorpusParser, to_records
from rag.corpus.validate import parse_ranges, validate_records

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))["corpus"]


def page_range(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(1, total + 1))
    a, _, b = spec.partition("-")
    return list(range(int(a), min(int(b or a), total) + 1))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def print_issues(console: Console, issues: list[Issue]) -> int:
    """Warnings table (with examples) and notes table; returns the number of warnings."""
    warn: dict[str, list[Issue]] = defaultdict(list)
    notes: dict[str, list[Issue]] = defaultdict(list)
    for i in issues:
        (warn if i.level == "warning" else notes)[i.code].append(i)
    if warn:
        wt = Table(title=f"[yellow]Warnings ({sum(map(len, warn.values()))}): review before indexing",
                   title_justify="left")
        wt.add_column("code", style="yellow")
        wt.add_column("n", justify="right")
        wt.add_column("examples")
        for code, items in sorted(warn.items(), key=lambda kv: -len(kv[1])):
            examples = "\n".join(
                f"{'Art. ' + str(i.article) if i.article else ''}{' p' + str(i.page) if i.page else ''}: {i.message}"[:150]
                for i in items[:3])
            wt.add_row(code, str(len(items)), examples + (f"\n… {len(items) - 3} more" if len(items) > 3 else ""))
        console.print(wt)
    if notes:
        nt = Table(title="Notes (expected quirks, recorded in the report)", title_justify="left")
        nt.add_column("code", style="cyan")
        nt.add_column("n", justify="right")
        nt.add_column("example")
        for code, items in sorted(notes.items(), key=lambda kv: -len(kv[1])):
            nt.add_row(code, str(len(items)), items[0].message[:110])
        console.print(nt)
    return sum(map(len, warn.values()))


def summarize(console: Console, report: dict, issues: list[Issue], out: Path) -> None:
    c = report["counts"]
    t = Table(title="Corpus", show_header=False, title_justify="left")
    t.add_row("Articles", f"{c['records']:,}  ({c['live']:,} live · {c['repealed']} repealed)")
    t.add_row("Pages", f"{c['pages']} · {c['tables']} tables · {c['page_split_articles']} articles span a page break")
    t.add_row("Hierarchy", f"{c['parts']} parts · {c['books']} books · {c['chapters']} chapters · "
                           f"{c['sections']} sections · {c['topics']} topics")
    t.add_row("Time", f"{report['elapsed_s']:.1f} s")
    console.print(t)
    n_warn = print_issues(console, issues)
    mark = "[green]✔[/]" if not n_warn else "[yellow]![/]"
    console.print(f"{mark} Wrote {out} ({c['records']:,} articles) in {report['elapsed_s']:.1f} s"
                  + (f" · [yellow]{n_warn} warnings to review[/]" if n_warn else " · no warnings"))


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", default=cfg["pdf"])
    ap.add_argument("--out", default=cfg["output"])
    ap.add_argument("--report", default=cfg["report"])
    ap.add_argument("--pages", help="only these pages, e.g. 1-11 (skips whole-corpus checks)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="processes for table detection")
    ap.add_argument("--strict", action="store_true", help="exit with status 1 if there are warnings")
    args = ap.parse_args(argv)

    lf = tracing.client()
    try:
        with lf.start_as_current_observation(
            as_type="span", name="build-corpus",
            input={"pdf": args.pdf, "pages": args.pages or "all", "workers": args.workers, "strict": args.strict},
        ) as root, propagate_attributes(trace_name="build-corpus", tags=["batch", "corpus"]):
            code = _build(args, cfg, lf, root)
            if code:
                root.update(level="ERROR", status_message=f"exit code {code}")
            return code
    finally:
        lf.flush()  # a batch job exits right after: send the trace first


def _build(args, cfg: dict, lf, root) -> int:
    console = Console()
    started = time.perf_counter()
    pdf = REPO_ROOT / args.pdf
    if not pdf.exists():
        console.print(f"[red]✗ {pdf} not found. Run `dvc pull` first.")
        return 2
    with pymupdf.open(pdf) as doc:
        total_pages = doc.page_count
    pages = page_range(args.pages, total_pages)

    parser = CorpusParser(parse_ranges(cfg.get("repealed", [])), cfg.get("part_corrections"))
    issues: list[Issue] = []
    tables = 0
    with lf.start_as_current_observation(as_type="span", name="extract-pages",
                                         input={"pages": [pages[0], pages[-1]], "workers": args.workers}) as span:
        bar = tqdm(iter_pages(str(pdf), pages, cfg["line_merge_pt"], args.workers), total=len(pages),
                   unit="page", desc="Extracting", dynamic_ncols=True)
        for result in bar:
            tables += result.tables
            if result.tables == 0:
                issues.append(warning("NO_TABLE", "no table found: this page's text is not in the corpus",
                                      page=result.page))
            if len(result.outside_text) > 3 and result.page != 1:  # page 1 opens with the promulgation law
                issues.append(info("TEXT_OUTSIDE_TABLE",
                                   f"{len(result.outside_text)} chars outside the table grid, read in position order: "
                                   f"'{result.outside_text[:60]}'", page=result.page))
            parser.feed_page(result)
            bar.set_postfix(articles=len(parser.articles), notes=len(parser.notes),
                            warnings=sum(i.level == "warning" for i in parser.issues + issues))
        bar.close()
        issues += parser.finish()
        records, record_issues = to_records(parser)
        issues += record_issues
        span.update(output={"pages": len(pages), "tables": tables, "records": len(records),
                            "repeal_notes": len(parser.notes)})

    with lf.start_as_current_observation(as_type="span", name="validate-records",
                                         input={"records": len(records), "full": args.pages is None}) as span:
        found = validate_records(records, cfg, full=args.pages is None)
        issues += found
        span.update(output=dict(Counter(f"{i.level}:{i.code}" for i in found)))

    out = REPO_ROOT / args.out
    write_json(out, records)
    live = [r for r in records if not r["is_repealed"]]
    levels = {lv: len({(r.get("part_number"), r.get("book_number"), r.get("chapter_number"), r.get("section_number"),
                        r.get("topic_number"))[: i + 1] for r in live if r.get(f"{lv}_number") is not None})
              for i, lv in enumerate(("part", "book", "chapter", "section", "topic"))}
    codes = Counter((i.level, i.code) for i in issues)
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pdf": args.pdf,
        "pdf_sha256": sha256(pdf),
        "articles_sha256": sha256(out),
        "pages_processed": [pages[0], pages[-1]],
        "counts": {
            "records": len(records), "live": len(live), "repealed": len(records) - len(live),
            "pages": len(pages), "tables": tables,
            "page_split_articles": sum(len(r["source_pages"]) > 1 for r in records),
            "parts": levels["part"], "books": levels["book"], "chapters": levels["chapter"],
            "sections": levels["section"], "topics": levels["topic"],
            "warnings": sum(i.level == "warning" for i in issues),
            "info": sum(i.level == "info" for i in issues),
        },
        "repealed_ranges": [f"{n.first}-{n.last}" for n in parser.notes],
        "issues_by_code": {code: {"level": level, "count": n} for (level, code), n in sorted(codes.items())},
        "issues": [i.to_dict() for i in issues],
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    write_json(REPO_ROOT / args.report, report)
    n_warn = report["counts"]["warnings"]
    lf.score_current_trace(name="build_warnings", value=n_warn, data_type="NUMERIC",
                           comment=", ".join(f"{code} {v['count']}" for code, v in report["issues_by_code"].items()
                                             if v["level"] == "warning") or None)
    root.update(output={"records": report["counts"]["records"], "live": report["counts"]["live"],
                        "repealed": report["counts"]["repealed"], "warnings": n_warn,
                        "elapsed_s": report["elapsed_s"]},
                **({"level": "WARNING", "status_message": f"{n_warn} warnings to review"} if n_warn else {}))
    summarize(console, report, issues, out.relative_to(REPO_ROOT))
    if sys.stdout.isatty():
        print("\a", end="", flush=True)
    return 1 if args.strict and report["counts"]["warnings"] else 0


if __name__ == "__main__":
    sys.exit(main())
