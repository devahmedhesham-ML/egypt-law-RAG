"""PDF pages → two reading-order line streams (English, Arabic), each line tagged with its table row.

Every page holds a two-column table: English in the left cell, Arabic in the right, one article
(or heading) per row. A few things sit outside the grid ("SECOND PART" on p. 113, "Article1022"
on p. 147), so outside text is kept too, in position order, with row = None.

Arabic is rebuilt from glyph positions because the PDF's text layer garbles it (lam-alef
ligatures, digit order, mirrored brackets).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import pymupdf

if hasattr(pymupdf, "no_recommend_layout"):
    pymupdf.no_recommend_layout()  # one-line tip printed by every worker otherwise

ALEFS = frozenset("اأإآ")
DIGIT = re.compile(r"[0-9٠-٩]")

# The Arabic font maps a few "…ه" ligature glyphs to private-use code points instead of letters
# (read from context across 25 occurrences: قرارا█م = قراراتهم, المر█ن = المرتهن, ا█دد = المهدد).
PRIVATE_LIGATURES = {"": "لمه", "": "به", "": "ته", "": "نه"}
_LIGATURE = "[" + "".join(PRIVATE_LIGATURES) + "]"


def fix_private_ligatures(text: str) -> str:
    # A word-initial ligature drawn before the word space: "يعلم ذا" → "يعلم بهذا".
    text = re.sub(rf"(\S)({_LIGATURE}) (?=\S)", lambda m: f"{m.group(1)} {PRIVATE_LIGATURES[m.group(2)]}", text)
    return re.sub(_LIGATURE, lambda m: PRIVATE_LIGATURES[m.group()], text)


@dataclass(frozen=True)
class Line:
    """One visual line of one column, in reading order."""

    text: str
    bold: bool
    x0: float  # left edge of the ink
    x1: float  # right edge of the ink
    y0: float
    page: int = 0  # 1-based
    row: int | None = None  # table row on the page; None = outside the table
    digit_alts: tuple[tuple[str, str], ...] = ()  # (as displayed, as stored) where the PDF disagrees


@dataclass(frozen=True)
class PageResult:
    page: int
    en: tuple[Line, ...]
    ar: tuple[Line, ...]
    tables: int
    rows: int
    outside_text: str  # text outside the table grid (the promulgation law on page 1)


def arabic_line(chars: list[dict]) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Rebuild one Arabic line: letters right→left by x, numbers left→right, lam-alef repaired.

    PyMuPDF emits each lam-alef ligature as a zero-width alef placed before the lam. Word wrote
    numbers inconsistently: some display reversed while their stored order is right, others the
    reverse. The displayed order is used; where the stored order differs, both are returned so the
    English translation can decide later.
    """
    cs = sorted(chars, key=lambda c: -(c["bbox"][0] + c["bbox"][2]) / 2)
    out: list[str] = []
    alts: list[tuple[str, str]] = []
    i = 0
    while i < len(cs):
        c = cs[i]
        zero_width = c["bbox"][2] - c["bbox"][0] < 0.01
        if c["c"] in ALEFS and zero_width and i + 1 < len(cs) and cs[i + 1]["c"] == "ل":
            out += ["ل", c["c"]]
            i += 2
            continue
        if DIGIT.match(c["c"]):
            j = i
            while j < len(cs) and (
                DIGIT.match(cs[j]["c"])
                or (cs[j]["c"] == " " and j + 1 < len(cs) and DIGIT.match(cs[j + 1]["c"]))
            ):
                j += 1
            digits = [d for d in cs[i:j] if DIGIT.match(d["c"])]
            shown = "".join(d["c"] for d in sorted(digits, key=lambda d: d["bbox"][0]))
            stored = "".join(d["c"] for d in sorted(digits, key=lambda d: d.get("k", 0)))
            if shown != stored:
                alts.append((shown, stored))
            out.append(shown)
            i = j
            continue
        out.append(c["c"])
        i += 1
    return "".join(out), tuple(alts)


def arabic_line_text(chars: list[dict]) -> str:
    return arabic_line(chars)[0]


def clean_ar(text: str) -> str:
    text = fix_private_ligatures(re.sub(r"\s+", " ", text).strip())
    text = re.sub(r"^[()]\s*([٠-٩]+|[ء-ي]ـ?)\s*[()]", r"(\1)", text)  # (١) (أ) (جـ)
    text = re.sub(r"^([٠-٩]+)\s*[()]\s*", r"\1) ", text)  # list item "١)" whose bracket the PDF mirrors to "١("
    text = re.sub(r"(?:^|\s)[()](?=\s|$)", " ", text)  # stray mirrored brackets
    text = re.sub(r"\s+([،.؛:؟!])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_en(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def cell_lines(chars: list[dict], lang: str, merge_pt: float, page: int = 0, row: int | None = None) -> list[Line]:
    """Group a cell's chars into visual lines (mirrored brackets sit ~2pt lower: merge them)."""
    groups: list[list[dict]] = []
    for c in sorted(chars, key=lambda c: c["bbox"][1]):
        if groups and abs(groups[-1][0]["bbox"][1] - c["bbox"][1]) < merge_pt:
            groups[-1].append(c)
        else:
            groups.append([c])
    lines = []
    for g in groups:
        alts: tuple[tuple[str, str], ...] = ()
        if lang == "ar":
            raw, alts = arabic_line(g)
            text = clean_ar(raw)
        else:
            text = clean_en("".join(c["c"] for c in sorted(g, key=lambda c: c["bbox"][0])))
        ink = [c for c in g if not c["c"].isspace()]
        if text and ink:
            lines.append(Line(
                text=text,
                bold=sum(c["bold"] for c in ink) > len(ink) / 2,
                x0=min(c["bbox"][0] for c in ink),
                x1=max(c["bbox"][2] for c in ink),
                y0=min(c["bbox"][1] for c in ink),
                page=page,
                row=row,
                digit_alts=alts,
            ))
    return lines


def page_chars(page: pymupdf.Page) -> list[dict]:
    out = []
    for block in page.get_text("rawdict", flags=0)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                bold = "Bold" in span["font"]
                out.extend({"c": c["c"], "bbox": c["bbox"], "bold": bold, "k": len(out) + i}
                           for i, c in enumerate(span["chars"]))
    return out


def _inside(c: dict, box) -> bool:
    x = (c["bbox"][0] + c["bbox"][2]) / 2
    y = (c["bbox"][1] + c["bbox"][3]) / 2
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def extract_page(page: pymupdf.Page, merge_pt: float) -> PageResult:
    page_no = page.number + 1
    chars = page_chars(page)
    mid = page.rect.width / 2
    is_ar = lambda c: (c["bbox"][0] + c["bbox"][2]) / 2 > mid  # noqa: E731
    tables = page.find_tables().tables
    en: list[Line] = []
    ar: list[Line] = []
    claimed: set[int] = set()
    n_rows = 0
    for table in tables:
        for trow in table.rows:
            cells = [c for c in trow.cells if c]
            if not cells:
                continue
            en_chars, ar_chars = [], []
            for i, c in enumerate(chars):
                if i in claimed:  # a char on a shared row boundary belongs to one row only
                    continue
                if len(cells) >= 2:
                    if _inside(c, cells[0]):
                        en_chars.append(c)
                    elif _inside(c, cells[-1]):
                        ar_chars.append(c)
                    else:
                        continue
                elif _inside(c, cells[0]):  # a merged full-width cell: split at the midline
                    (ar_chars if is_ar(c) else en_chars).append(c)
                else:
                    continue
                claimed.add(i)
            if en_chars or ar_chars:
                en += cell_lines(en_chars, "en", merge_pt, page_no, n_rows)
                ar += cell_lines(ar_chars, "ar", merge_pt, page_no, n_rows)
                n_rows += 1
    outside = [c for i, c in enumerate(chars) if i not in claimed]
    en += cell_lines([c for c in outside if not is_ar(c)], "en", merge_pt, page_no, None)
    ar += cell_lines([c for c in outside if is_ar(c)], "ar", merge_pt, page_no, None)
    en.sort(key=lambda ln: (ln.y0, ln.x0))
    ar.sort(key=lambda ln: (ln.y0, -ln.x1))
    outside_text = re.sub(r"\s+", " ", "".join(c["c"] for c in outside)).strip()
    return PageResult(page_no, tuple(en), tuple(ar), len(tables), n_rows, outside_text)


# --- parallel extraction (each worker process opens the PDF once) ---------------------

_DOC: pymupdf.Document | None = None
_MERGE_PT = 3.5


def _init_worker(pdf_path: str, merge_pt: float) -> None:
    global _DOC, _MERGE_PT
    _DOC = pymupdf.open(pdf_path)
    _MERGE_PT = merge_pt


def _extract_page_no(page_no: int) -> PageResult:
    return extract_page(_DOC[page_no - 1], _MERGE_PT)


def iter_pages(pdf_path: str, pages: Iterable[int], merge_pt: float, workers: int = 1) -> Iterator[PageResult]:
    """Yield PageResults in page order; table detection runs on `workers` processes."""
    pages = list(pages)
    if workers <= 1:
        _init_worker(pdf_path, merge_pt)
        yield from (_extract_page_no(p) for p in pages)
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(pdf_path, merge_pt)) as pool:
        yield from pool.map(_extract_page_no, pages, chunksize=1)
