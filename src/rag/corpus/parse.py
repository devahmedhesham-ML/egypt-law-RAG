"""Line streams → article drafts → corpus records.

Each column is read top to bottom across pages and split on its own markers: "Article N" in
English, "مادة (n)" in Arabic; the two sides are then joined by number. Table rows act as a
cross-check (an article's English and Arabic markers should share a row) and as a guard: a
marker in the middle of a row is accepted only when it is exactly the next number, so a
cross-reference that wraps onto its own line never splits an article.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.corpus.extract import Line, PageResult
from rag.corpus.hierarchy import Hierarchy
from rag.corpus.issues import Issue, info, warning

EN_MARK = re.compile(r"^A?rticle\s*(\d+)$")  # typos in the PDF: "rticle 452", "Article1022"
EN_MARK_GLUED = re.compile(r"^A?rticle\s*(\d+)\s+(?=[A-Z(])(.+)$")  # marker sharing a line with its text
AR_MARK = re.compile(r"^مادة\s*[()]?\s*([0-9٠-٩]+)\s*[()]?$")
AR_MARK_GLUED = re.compile(r"^مادة\s*[()]?\s*([0-9٠-٩]+)\s*[()]?\s+(.+)$")
REPEAL = re.compile(r"\bArticles?\s+(\d+)\s*[-–]\s*(\d+)\b.*\brepealed\b", re.I)
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

AR_ITEM = re.compile(r"^(?:\((?:[٠-٩]+|[ء-ي]ـ?)\)|(?:[٠-٩]+|[ء-ي]ـ?)\s*[-–)]\s)")  # (١) (أ) أ- جـ- ١- ١)
EN_ITEM = re.compile(r"^(?:\(?\d+[.)]|\(?[a-z]\))\s")  # 1. (a) a)
SENTENCE_END = re.compile(r"[.:;؛]$")


@dataclass
class Draft:
    number: int
    hierarchy: dict
    marker_at: tuple[int, int | None]  # (page, table row) of the marker line
    lines: list[Line] = field(default_factory=list)

    @property
    def pages(self) -> list[int]:
        return sorted({self.marker_at[0], *(ln.page for ln in self.lines)})


@dataclass
class RepealNote:
    first: int
    last: int
    page: int
    hierarchy: dict
    text_en: str
    text_ar: str = ""


def _match(lines_re: tuple[re.Pattern, re.Pattern], text: str):
    exact, glued = lines_re
    if m := exact.match(text):
        return int(m.group(1).translate(AR_DIGITS)), None
    if m := glued.match(text):
        return int(m.group(1).translate(AR_DIGITS)), m.group(2)
    return None


class CorpusParser:
    def __init__(self, expected_repealed: set[int], part_corrections: list[dict] | None = None) -> None:
        self.hierarchy = Hierarchy(part_corrections=list(part_corrections or []))
        self.expected_repealed = expected_repealed
        self.drafts: dict[str, dict[int, Draft]] = {"en": {}, "ar": {}}
        self.current: dict[str, Draft | None] = {"en": None, "ar": None}
        self.last: dict[str, int] = {"en": 0, "ar": 0}
        self.notes: list[RepealNote] = []
        self.issues: list[Issue] = []
        self.seen_table: dict[str, bool] = {"en": False, "ar": False}  # text outside the grid before it = preamble

    def _preamble(self, lang: str, ln: Line) -> bool:
        """The promulgation law above the first table on page 1 (it has its own "مادة ١")."""
        if ln.row is not None:
            self.seen_table[lang] = True
            return False
        if self.seen_table[lang]:
            return False
        self.issues.append(info("PREAMBLE", f"skipped text above the first table: '{ln.text[:50]}'", page=ln.page))
        return True

    @property
    def articles(self) -> dict[int, Draft]:  # progress display
        return self.drafts["en"]

    # --- markers -----------------------------------------------------------------------
    def _accept(self, lang: str, number: int, ln: Line, row_start: bool, page_top: bool) -> bool:
        last = self.last[lang]
        side = "English" if lang == "en" else "Arabic"
        if number <= last:
            if row_start:
                self.issues.append(info(
                    "CROSS_REFERENCE_SKIPPED",
                    f"{side} '{ln.text[:40]}' opens a row after Article {last}: treated as text",
                    article=last, page=ln.page))
            return False
        explained = self.expected_repealed | {n for note in self.notes for n in range(note.first, note.last + 1)}
        unexplained = sorted(set(range(last + 1, number)) - explained)
        if not row_start:
            # English cross-references look exactly like markers, so an English marker never starts an
            # article mid-row (a missed split shows up in the checks; a false one would be silent).
            # Arabic cross-references read "المادة", so a mid-row "مادة" may start the very next article.
            if lang == "en":
                self.issues.append(info("MARKER_MID_ROW_IGNORED", f"'{ln.text[:40]}' inside a row: treated as text",
                                        article=last, page=ln.page))
                return False
            if unexplained:
                return False
            self.issues.append(info(
                "MARKER_MID_ROW", f"{side} marker for Article {number} sits inside the previous row "
                                  "(broken table layout)", article=number, page=ln.page))
        elif unexplained:
            # An English cross-reference ("Article 12") can wrap onto the top of the next page. Arabic
            # cross-references read "المادة ١٢", so an Arabic line starting "مادة" is always a marker.
            if page_top and lang == "en":
                self.issues.append(warning(
                    "AMBIGUOUS_MARKER",
                    f"{side} '{ln.text[:40]}' at the top of page {ln.page} jumps from Article {last}: "
                    "treated as text, check the PDF", article=last, page=ln.page))
                return False
            self.issues.append(warning(
                "GAP_BEFORE_ARTICLE",
                f"{side}: Article {number} follows Article {last}; nothing found for {_ranges(unexplained)}",
                article=number, page=ln.page))
        if lang == "en" and not ln.text.startswith("Article "):
            self.issues.append(info("MARKER_TYPO", f"source prints '{ln.text.split()[0][:12]}…' for Article {number}",
                                    article=number, page=ln.page))
        self.last[lang] = number
        return True

    def _start(self, lang: str, number: int, glued: str | None, ln: Line, hierarchy: dict) -> None:
        draft = Draft(number, hierarchy, (ln.page, ln.row))
        if glued:
            draft.lines.append(Line(glued, ln.bold, ln.x0, ln.x1, ln.y0, ln.page, ln.row))
            self.issues.append(info("MARKER_GLUED", f"marker shared a line with its text: '{ln.text[:50]}'",
                                    article=number, page=ln.page))
        self.drafts[lang][number] = draft
        self.current[lang] = draft

    # --- pages -----------------------------------------------------------------------------
    @staticmethod
    def _row_starts(lines: tuple[Line, ...]) -> list[bool]:
        """True where a line is outside the table or preceded in its row only by bold (heading) lines."""
        seen_body: set[int] = set()
        out = []
        for ln in lines:
            out.append(ln.row is None or ln.row not in seen_body)
            if ln.row is not None and not ln.bold:
                seen_body.add(ln.row)
        return out

    def feed_page(self, page: PageResult) -> None:
        consumed_ar: set[int] = set()
        self._feed_english(page, consumed_ar)
        self._feed_arabic(page, consumed_ar)

    def _ar_lines_for(self, page: PageResult, row: int | None, y_from: float, y_to: float, *, bold_only: bool,
                      consumed: set[int]) -> list[str]:
        """Arabic lines belonging to an English heading/note block.

        Same row: every line above that row's Arabic article marker (some Arabic headings are not
        bold, e.g. "بيع التركة"). Outside the grid: bold lines level with the block.
        """
        marker_y = min((a.y0 for a in page.ar if row is not None and a.row == row
                        and _match((AR_MARK, AR_MARK_GLUED), a.text)), default=float("inf"))
        out = []
        for j, a in enumerate(page.ar):
            if j in consumed or _match((AR_MARK, AR_MARK_GLUED), a.text):
                continue
            same_row = row is not None and a.row == row and a.y0 < marker_y
            beside = a.row is None and (a.bold or not bold_only) and y_from - 4 <= a.y0 <= y_to + 4
            if same_row or beside:
                out.append(a.text)
                consumed.add(j)
        return out

    def _feed_english(self, page: PageResult, consumed_ar: set[int]) -> None:
        lines = page.en
        row_start = self._row_starts(lines)
        i = 0
        while i < len(lines):
            ln = lines[i]
            if self._preamble("en", ln):
                i += 1
                continue
            found = _match((EN_MARK, EN_MARK_GLUED), ln.text)
            if found and self._accept("en", found[0], ln, row_start[i], page_top=i == 0):
                self._start("en", found[0], found[1], ln, self.hierarchy.snapshot())
                i += 1
                continue
            if REPEAL.search(ln.text):  # "Articles 54-80 have been repealed …" (+ its wrapped tail)
                j = i + 1
                while j < len(lines) and lines[j].row == ln.row and not _match((EN_MARK, EN_MARK_GLUED), lines[j].text):
                    j += 1
                text_en = " ".join(x.text for x in lines[i:j]).lstrip("* ").strip()
                m = REPEAL.search(text_en)
                text_ar = " ".join(self._ar_lines_for(page, ln.row, ln.y0, lines[j - 1].y0, bold_only=False,
                                                      consumed=consumed_ar)).lstrip("* ").strip()
                note = RepealNote(int(m.group(1)), int(m.group(2)), ln.page, self.hierarchy.snapshot(), text_en, text_ar)
                self.notes.append(note)
                self.issues.append(info("REPEAL_NOTE", f"Articles {note.first}-{note.last}: '{text_en}'", page=ln.page))
                i = j
                continue
            if ln.bold and row_start[i]:  # heading block: consecutive bold lines of one row (or outside)
                j = i + 1
                while (j < len(lines) and lines[j].bold and lines[j].row == ln.row
                       and not _match((EN_MARK, EN_MARK_GLUED), lines[j].text) and not REPEAL.search(lines[j].text)):
                    j += 1
                block = lines[i:j]
                ar = self._ar_lines_for(page, ln.row, block[0].y0, block[-1].y0, bold_only=True, consumed=consumed_ar)
                self.hierarchy.feed([x.text for x in block], ar, ln.page)
                i = j
                continue
            cur = self.current["en"]
            if cur is None:
                self.issues.append((info if ln.row is None else warning)(
                    "TEXT_BEFORE_FIRST_ARTICLE", f"English text before Article 1: '{ln.text[:60]}'", page=ln.page))
            else:
                if ln.row is not None and row_start[i] and i > 0 and cur.marker_at != (ln.page, ln.row):
                    self.issues.append(info(
                        "ROW_WITHOUT_MARKER", f"a row without 'Article N' continues Article {cur.number}: "
                                              f"'{ln.text[:50]}'", article=cur.number, page=ln.page))
                cur.lines.append(ln)
            i += 1

    def _feed_arabic(self, page: PageResult, consumed: set[int]) -> None:
        lines = page.ar
        row_start = self._row_starts(lines)
        for j, ln in enumerate(lines):
            if j in consumed or self._preamble("ar", ln):
                continue
            found = _match((AR_MARK, AR_MARK_GLUED), ln.text)
            if found and self._accept("ar", found[0], ln, row_start[j], page_top=j == 0):
                en = self.drafts["en"].get(found[0])
                self._start("ar", found[0], found[1], ln, en.hierarchy if en else self.hierarchy.snapshot())
                continue
            if ln.bold and row_start[j]:
                self.issues.append(info("AR_HEADING_UNPAIRED",
                                        f"Arabic heading with no English heading beside it: '{ln.text[:50]}'",
                                        page=ln.page))
                continue
            cur = self.current["ar"]
            if cur is None:
                self.issues.append((info if ln.row is None else warning)(
                    "TEXT_BEFORE_FIRST_ARTICLE", f"Arabic text before مادة (١): '{ln.text[:60]}'", page=ln.page))
                continue
            cur.lines.append(ln)

    def finish(self) -> list[Issue]:
        en, ar = self.drafts["en"], self.drafts["ar"]
        for n in sorted(set(en) & set(ar)):
            e_at, a_at = en[n].marker_at, ar[n].marker_at
            if e_at != a_at:
                outside = e_at[1] is None or a_at[1] is None
                self.issues.append((info if outside else warning)(
                    "ROW_MISMATCH",
                    f"English marker at page {e_at[0]} row {e_at[1]}, Arabic at page {a_at[0]} row {a_at[1]}",
                    article=n, page=e_at[0]))
        for n in sorted(set(en) | set(ar)):
            pages = sorted({*(en[n].pages if n in en else []), *(ar[n].pages if n in ar else [])})
            for p in pages[1:]:
                self.issues.append(info("PAGE_BREAK", f"Article {n} continues on page {p}", article=n, page=p))
        return self.issues + self.hierarchy.events


# --- records --------------------------------------------------------------------------

def paragraphs(lines: list[Line], lang: str) -> str:
    """Join layout lines into paragraphs: a list marker, or a sentence ending a deliberately short line, starts one."""
    if not lines:
        return ""
    edge = max(ln.x1 for ln in lines) if lang == "en" else min(ln.x0 for ln in lines)  # Arabic lines end on the left
    item = AR_ITEM if lang == "ar" else EN_ITEM
    paras: list[str] = []
    prev: Line | None = None
    for ln in lines:
        if prev is None or item.match(ln.text):
            paras.append(ln.text)
        else:
            # Reflow test: if this line's first word would have fit after the previous line,
            # that line was broken on purpose, so a sentence ending there closes the paragraph.
            gap = (edge - prev.x1) if lang == "en" else (prev.x0 - edge)
            char_w = (ln.x1 - ln.x0) / max(len(ln.text), 1)
            first_word = ln.text.split()[0] if ln.text.split() else ""
            if SENTENCE_END.search(prev.text) and gap > (len(first_word) + 1) * char_w:
                paras.append(ln.text)
            elif lang == "en" and paras[-1].endswith("-") and ln.text[:1].islower():
                paras[-1] += ln.text  # "co-" + "owners" wrapped across lines
            else:
                paras[-1] += " " + ln.text
        prev = ln
    return "\n".join(paras)


def resolve_digits(text_ar: str, lines: list[Line], text_en: str) -> tuple[str, list[str]]:
    """Where the PDF displays and stores a number differently, keep the order the English confirms."""
    en_numbers = set(re.findall(r"\d+", text_en))
    changes = []
    for ln in lines:
        for shown, stored in ln.digit_alts:
            if stored.translate(AR_DIGITS) in en_numbers and shown.translate(AR_DIGITS) not in en_numbers:
                text_ar, n = re.subn(rf"(?<![0-9٠-٩]){shown}(?![0-9٠-٩])", stored, text_ar, count=1)
                if n:
                    changes.append(f"{shown}→{stored}")
    return text_ar, changes


def to_records(parser: CorpusParser) -> tuple[list[dict], list[Issue]]:
    en, ar = parser.drafts["en"], parser.drafts["ar"]
    noted = {n: note for note in parser.notes for n in range(note.first, note.last + 1)}
    records = []
    issues: list[Issue] = []
    last_hierarchy: dict = {}
    for n in sorted(set(en) | set(ar) | set(noted)):
        e, a, note = en.get(n), ar.get(n), noted.get(n)
        hierarchy = e.hierarchy if e else a.hierarchy if a else note.hierarchy if note else last_hierarchy
        last_hierarchy = hierarchy
        text_en = paragraphs(e.lines, "en") if e else ""
        text_ar = paragraphs(a.lines, "ar") if a else ""
        if a and text_en:
            text_ar, changes = resolve_digits(text_ar, a.lines, text_en)
            if changes:
                issues.append(info("DIGITS_FROM_TEXT_LAYER",
                                   f"PDF displays {', '.join(c.split('→')[0] for c in changes)} reversed; the English "
                                   f"confirms the stored order ({', '.join(changes)})", article=n))
        note_ar = (note.text_ar or None) if note else None
        if note and text_ar and not note_ar:  # "Article 54": its Arabic cell holds the repeal note
            note_ar, text_ar = text_ar, ""
        pages = sorted({*(e.pages if e else []), *(a.pages if a else [])}) or [note.page]
        records.append({
            "article_number": n,
            **hierarchy,
            "text_ar": text_ar,
            "text_en": text_en,
            "is_repealed": note is not None,
            "repeal_note": note.text_en if note else None,
            "repeal_note_ar": note_ar,
            "source_pages": pages,
            "ar_number": n if a else None,
            "citation": f"Egyptian Civil Code, Article {n}",
        })
    return records, issues


def _ranges(numbers: list[int]) -> str:
    out, start = [], None
    for i, n in enumerate(numbers):
        start = n if start is None else start
        if i + 1 == len(numbers) or numbers[i + 1] != n + 1:
            out.append(f"{start}" if start == n else f"{start}-{n}")
            start = None
    return ", ".join(out)
