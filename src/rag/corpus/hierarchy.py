"""Heading rows → numbered, bilingual hierarchy: part > book > chapter > section > topic > subtopic.

Rules come from dumping every heading row of the PDF:
- Articles 1–88 (preliminary chapter) have sections only; Part 1 starts after Article 88.
- A keyword may sit alone with its title on the next line/row, or share one line with it.
- A new level resets every level below it.
- Topics are numbered in English ("1.", "2-", "1.Sale") or, when the English is not, in Arabic ("٣-", "أولا").
- Unnumbered headings are subtopics; two in one row are joined with " — ".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.corpus.issues import Issue

LEVELS = ("part", "book", "chapter", "section", "topic", "subtopic")
KEYWORD_LEVELS = ("part", "book", "chapter", "section")

KEYWORDS = (
    ("part", re.compile(r"^(FIRST|SECOND|THIRD|FOURTH)\s+PART\b\.?\s*(.*)$", re.I)),
    ("book", re.compile(r"^BOOK\s+([IVXL]+)\b\.?\s*(.*)$", re.I)),
    ("chapter", re.compile(r"^CHAPTER\s+([IVXL]+)\b\.?\s*(.*)$", re.I)),
    ("section", re.compile(r"^SECTION\s+([IVXL]+)\b\.?\s*(.*)$", re.I)),
)
AR_KEYWORD = re.compile(r"^(القسم|الكتاب|الباب|الفصل)\s+(\S+?)\.?(?:\s+(.+))?$")
EN_NUMBERED = re.compile(r"^(\d+)\s*[.\-–]\s*(.+)$")
AR_NUMBERED = re.compile(r"^([٠-٩0-9]+)\s*[-–.]\s*(.+)$")
AR_ORDINAL_PREFIX = re.compile(r"^(أولا|ثانيا|ثالثا|رابعا|خامسا|سادسا)\s*[:：]?\s*-?\s*(.+)$")
TASHKEEL = re.compile(r"[ً-ْٰ]")

ORDINAL_EN = {"FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4}
ORDINAL_AR = {
    "الأول": 1, "الاول": 1, "أولا": 1, "الثاني": 2, "الثانى": 2, "ثانيا": 2, "الثالث": 3, "ثالثا": 3,
    "الرابع": 4, "رابعا": 4, "الخامس": 5, "خامسا": 5, "السادس": 6, "سادسا": 6, "السابع": 7,
    "الثامن": 8, "التاسع": 9, "العاشر": 10,
}
AR_ORDINAL_WORD = {1: "الأول", 2: "الثاني", 3: "الثالث", 4: "الرابع", 5: "الخامس", 6: "السادس",
                   7: "السابع", 8: "الثامن", 9: "التاسع", 10: "العاشر"}
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50}
SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}


def roman_to_int(token: str) -> int:
    total, prev = 0, 0
    for ch in reversed(token.upper()):
        value = ROMAN[ch]
        total += -value if value < prev else value
        prev = max(prev, value)
    return total


def smart_title(text: str) -> str:
    """Title-case ALL-CAPS headings ("OBLIGATIONS OR PERSONAL RIGHTS" → "Obligations or Personal Rights")."""
    if not text.isupper():
        return text
    words = text.lower().split()
    return " ".join(w if i and w in SMALL_WORDS else w.capitalize() for i, w in enumerate(words))


def clean_title_en(text: str) -> str:
    return smart_title(text.strip().rstrip(":. ").strip())


def clean_title_ar(text: str | None) -> str | None:
    if not text:
        return None
    text = re.sub(r"^[()]+|[()]+$", "", text.strip()).strip().rstrip(":.").strip()
    return text or None


def ar_number(text: str) -> tuple[int | None, str]:
    """('١- الشخص الطبيعي') → (1, 'الشخص الطبيعي'); ('أولا:- أركان العقد') → (1, 'أركان العقد')."""
    plain = TASHKEEL.sub("", text).strip()
    if m := AR_NUMBERED.match(plain):
        return int(m.group(1).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))), m.group(2).strip()
    if m := AR_ORDINAL_PREFIX.match(plain):
        return ORDINAL_AR[m.group(1)], m.group(2).strip()
    return None, plain


@dataclass
class Level:
    number: int | None
    title_en: str | None
    title_ar: str | None
    corrected: bool = False


@dataclass
class Hierarchy:
    part_corrections: list[dict] = field(default_factory=list)
    levels: dict[str, Level] = field(default_factory=dict)
    events: list[Issue] = field(default_factory=list)
    _pending: str | None = None  # keyword level still waiting for its title
    _pending_ar_partial: bool = False  # its Arabic title started on the keyword line and continues
    _corrections_applied: set = field(default_factory=set)

    # --- state ------------------------------------------------------------------
    def _set(self, name: str, level: Level, page: int | None) -> None:
        previous = self.levels.get(name)
        if (
            name in KEYWORD_LEVELS and previous and previous.number is not None and level.number is not None
            and level.number <= previous.number and not level.corrected
        ):
            self.events.append(Issue(
                "warning", "HEADING_ORDER",
                f"{name} {level.number} follows {name} {previous.number} under the same parent", page=page))
        self.levels[name] = level
        for lower in LEVELS[LEVELS.index(name) + 1:]:
            self.levels.pop(lower, None)

    def snapshot(self) -> dict:
        out: dict = {}
        for name in LEVELS:
            lv = self.levels.get(name)
            if name != "subtopic":
                out[f"{name}_number"] = lv.number if lv else None
            out[f"{name}_title_en"] = lv.title_en if lv else None
            out[f"{name}_title_ar"] = lv.title_ar if lv else None
        return out

    # --- feeding heading rows -----------------------------------------------------
    @staticmethod
    def _align(en: list[str], ar: list[str]) -> tuple[list[str], list[str]]:
        """Pair EN and AR heading lines. English wraps more often; the Arabic line count decides."""
        en = [e for e in en if e]
        merged: list[str] = []
        for e in en:  # lowercase or "(" starts continue the previous line
            if merged and (e[:1].islower() or e.startswith("(")):
                merged[-1] += " " + e
            else:
                merged.append(e)
        en = merged
        ar = [a for a in ar if a]
        while len(en) > max(len(ar), 1):
            idx = next(
                (i for i in range(1, len(en))
                 if len(en[i].split()) <= 2 and not EN_NUMBERED.match(en[i])
                 and not any(rx.match(en[i]) for _, rx in KEYWORDS)),
                len(en) - 1,
            )
            en[idx - 1: idx + 1] = [f"{en[idx - 1]} {en[idx]}"]
        while len(ar) > len(en) and len(ar) > 1:
            ar[-2:] = [f"{ar[-2]} {ar[-1]}"]
        return en, ar + [""] * (len(en) - len(ar))

    def _apply_part_correction(self, book_number: int, page: int | None) -> None:
        """Official structure for parts: fill a part title the PDF leaves out, or add a part it skips."""
        for fix in self.part_corrections:
            key = (fix["starts_at_book"], fix["number"])
            current = self.levels.get("part")
            if book_number < fix["starts_at_book"]:
                continue
            if current is not None and current.number == fix["number"]:
                filled = [lang for lang in ("en", "ar") if not getattr(current, f"title_{lang}")]
                for lang in filled:
                    setattr(current, f"title_{lang}", fix[f"title_{lang}"])
                if filled and key not in self._corrections_applied:
                    self._corrections_applied.add(key)
                    self.events.append(Issue(
                        "info", "PART_TITLE_FROM_OFFICIAL",
                        f"Part {fix['number']}: {'/'.join(filled)} title not in the PDF, taken from the official "
                        f"structure ({fix.get('source', 'n/a')})", page=page))
                continue
            if current is None or (current.number or 0) < fix["number"]:
                self._set("part", Level(fix["number"], fix["title_en"], fix["title_ar"], corrected=True), page)
                if key not in self._corrections_applied:
                    self._corrections_applied.add(key)
                    self.events.append(Issue(
                        "info", "PART_CORRECTION",
                        f"Part {fix['number']} '{fix['title_en']}' added from the official structure at Book "
                        f"{book_number} (not printed in the PDF; source: {fix.get('source', 'n/a')})", page=page))

    def feed(self, en_lines: list[str], ar_lines: list[str], page: int | None = None) -> None:
        en, ar = self._align(en_lines, ar_lines)
        row_has_topic = False
        subtopic_en: list[str] = []
        subtopic_ar: list[str] = []
        touched: list[str] = []
        for e, a in zip(en, ar):
            e, a = e.strip(), a.strip()
            keyword = next(((name, m) for name, rx in KEYWORDS if (m := rx.match(e))), None)
            if keyword:
                name, m = keyword
                number = ORDINAL_EN[m.group(1).upper()] if name == "part" else roman_to_int(m.group(1))
                rest_en = m.group(2).strip()
                am = AR_KEYWORD.match(a)
                rest_ar = clean_title_ar(am.group(3)) if am else None
                if am and ORDINAL_AR.get(am.group(2)) not in (None, number):
                    self.events.append(Issue(
                        "warning", "HEADING_NUMBER_MISMATCH",
                        f"English '{e}' vs Arabic '{a}': numbers differ", page=page))
                if name == "book":
                    self._apply_part_correction(number, page)
                self._set(name, Level(number, clean_title_en(rest_en) if rest_en else None, rest_ar), page)
                self._pending = name if not (rest_en and rest_ar) else None
                self._pending_ar_partial = bool(rest_ar) and not rest_en
                touched.append(name)
                continue
            if self._pending:  # title line(s) of the keyword above
                lv = self.levels[self._pending]
                if not lv.title_en:
                    lv.title_en = clean_title_en(e)
                if not lv.title_ar:
                    lv.title_ar = clean_title_ar(a)
                elif self._pending_ar_partial and a:
                    lv.title_ar = clean_title_ar(f"{lv.title_ar} {a}")
                touched.append(self._pending)
                self._pending, self._pending_ar_partial = None, False
                continue
            en_num = EN_NUMBERED.match(e)
            num_ar, title_ar = ar_number(a) if a else (None, "")
            if row_has_topic:
                is_topic = False
            elif en_num:
                is_topic = True
            elif e.endswith(":"):
                is_topic = False
            else:
                is_topic = num_ar is not None
            if is_topic:
                number = int(en_num.group(1)) if en_num else num_ar
                if en_num and num_ar is not None and num_ar != number:
                    self.events.append(Issue(
                        "info", "TOPIC_NUMBER_MISMATCH", f"topic '{e}' vs Arabic '{a}'", page=page))
                self._set("topic", Level(number, clean_title_en(en_num.group(2) if en_num else e),
                                         clean_title_ar(title_ar)), page)
                row_has_topic = True
                touched.append("topic")
            else:
                subtopic_en.append(clean_title_en(e))
                if a:
                    subtopic_ar.append(clean_title_ar(title_ar) or "")
        if subtopic_en:
            ar_title = " — ".join(t for t in subtopic_ar if t) or None
            self._set("subtopic", Level(None, " — ".join(subtopic_en), ar_title), page)
            touched.append("subtopic")
        for name in dict.fromkeys(touched):
            lv = self.levels.get(name)
            if lv and lv.title_en and not lv.title_ar and name != self._pending:
                self.events.append(Issue(
                    "info", "HEADING_NO_ARABIC", f"{name} '{lv.title_en}' has no Arabic title", page=page))
