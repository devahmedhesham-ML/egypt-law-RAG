"""Record-level corpus checks. They warn; the person indexing decides whether to continue."""

from __future__ import annotations

import re
from collections import Counter

from rag.corpus.golden import GOLDEN
from rag.corpus.issues import Issue, info, warning

NUMBERED_LEVELS = ("part", "book", "chapter", "section", "topic")

TERMINAL_EN = re.compile(r"[.;:)\]\"'’”]$")
TERMINAL_AR = re.compile(r"[.؛:)\]»”\"]$")
PRESENTATION_FORMS = re.compile(r"[ﭐ-﷿ﹰ-﻿]")
PRIVATE_USE = re.compile(r"[-]")
LIGATURE_BUG = re.compile(r"األ|اإل|اآل|إال|أال|(?:^|\s)ال(?=[\s،.؛:]|$)")
STRAY_BRACKETS = re.compile(r"\(\s*\(|\)\s*\)|\)\(|(?:^|\s)[()](?:\s|$)")
LATIN = re.compile(r"[A-Za-z]+")
ARABIC_LETTER = re.compile(r"[ء-ي]")
AR_PARA_NUMBERS = re.compile(r"^\(([٠-٩]+)\)", re.M)
AR_LETTER_ITEMS = re.compile(r"^\(?([ء-ي])ـ?\s*(?:\)|[-–]\s)", re.M)  # (أ) or أ-
EN_LETTER_ITEMS = re.compile(r"^\(?([a-h])\)", re.M)  # i), v), x) are roman numerals, not letters
AR_NUMBER_ITEMS = re.compile(r"^([٠-٩]+)\s*[-–)]\s", re.M)  # ١- ١)
EN_NUMBER_ITEMS = re.compile(r"^(\d+)[.)]\s", re.M)  # 1. 2.
ABJAD = "أبجدهوزحطيكلمنسعفصقرشتثخذضظغ"
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def parse_ranges(ranges: list[str]) -> set[int]:
    out: set[int] = set()
    for r in ranges:
        a, _, b = str(r).partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def _tail(text: str, n: int = 40) -> str:
    return ("…" + text[-n:]) if len(text) > n else text


def _squash(text: str | None) -> str:
    return re.sub(r"\s+", "", text or "")


def check_numbering(records: list[dict], cfg: dict) -> list[Issue]:
    issues: list[Issue] = []
    numbers = [r["article_number"] for r in records]
    expected = set(range(cfg["first_article"], cfg["last_article"] + 1))
    for n, c in Counter(numbers).items():
        if c > 1:
            issues.append(warning("DUPLICATE_ARTICLE", f"Article {n} appears {c} times", article=n))
    missing = sorted(expected - set(numbers))
    if missing:
        issues.append(warning("MISSING_ARTICLES", f"{len(missing)} numbers missing: {_ranges(missing)}"))
    extra = sorted(set(numbers) - expected)
    if extra:
        issues.append(warning("UNEXPECTED_ARTICLES", f"outside {min(expected)}-{max(expected)}: {extra[:20]}"))
    repealed = {r["article_number"] for r in records if r["is_repealed"]}
    want = parse_ranges(cfg.get("repealed", []))
    if repealed != want:
        issues.append(warning(
            "REPEALED_MISMATCH",
            f"repealed in the PDF but not expected: {_ranges(sorted(repealed - want)) or 'none'}; "
            f"expected but not found: {_ranges(sorted(want - repealed)) or 'none'}"))
    return issues


def _sequence_ok(values: list[int]) -> bool:
    return values == list(range(1, len(values) + 1))


def check_record(r: dict, cfg: dict) -> list[Issue]:
    n = r["article_number"]
    issues: list[Issue] = []
    page = (r.get("source_pages") or [None])[0]
    w = lambda code, msg: issues.append(warning(code, msg, article=n, page=page))  # noqa: E731

    if r["is_repealed"]:
        if r["text_en"] or r["text_ar"]:
            w("REPEALED_HAS_TEXT", "repealed article still carries text")
        if not r.get("repeal_note"):
            w("REPEALED_NO_NOTE", "repealed article has no repeal note")
        return issues

    ar, en = r["text_ar"], r["text_en"]
    # completeness: both languages present, same number, nothing cut off
    if not ar:
        w("EMPTY_TEXT_AR", "no Arabic text")
    if not en:
        w("EMPTY_TEXT_EN", "no English text")
    if r.get("ar_number") is None:
        w("AR_NUMBER_MISSING", "no 'مادة (n)' found in the Arabic cell")
    elif r["ar_number"] != n:
        w("AR_NUMBER_MISMATCH", f"Arabic cell says مادة {r['ar_number']}, English says Article {n}")
    # A missing final period on a one-page article is source style (the text is all there); across
    # a page break, or ending mid-clause on a comma, it can mean lost lines.
    multi_page = len(r.get("source_pages") or []) > 1
    for lang, text, terminal, commas in (("EN", en, TERMINAL_EN, ",-"), ("AR", ar, TERMINAL_AR, "،,-")):
        if text and not terminal.search(text):
            if multi_page or text[-1] in commas:
                w(f"CUT_OFF_{lang}", f"{'spans a page break and ' if multi_page else ''}does not end a sentence: "
                                     f"'{_tail(text)}'")
            else:
                issues.append(info(f"NO_FINAL_PERIOD_{lang}", f"no final period (source style): '{_tail(text)}'",
                                   article=n, page=page))
    total = len(ar) + len(en)
    if ar and en and min(len(ar), len(en)) < cfg["min_chars"]:
        w("TOO_SHORT", f"only {min(len(ar), len(en))} characters in one language")
    if total > cfg["max_chars"]:
        w("TOO_LONG", f"{total} characters: possibly two articles merged")
    lo, hi = cfg["ar_en_length_ratio"]
    if ar and en and not lo <= len(ar) / len(en) <= hi:
        w("LENGTH_RATIO", f"Arabic/English length ratio {len(ar) / len(en):.2f} outside [{lo}, {hi}]: "
                          "one side may be incomplete")

    # structure: numbered paragraphs and lettered items must run 1, 2, 3 … / a, b, c …
    paras = [int(p.translate(AR_DIGITS)) for p in AR_PARA_NUMBERS.findall(ar)]
    if paras and not _sequence_ok(paras):
        w("PARAGRAPH_SEQUENCE", f"Arabic paragraphs numbered {paras}: one may be missing or out of order")
    ar_items = [ABJAD.find(x) + 1 for x in AR_LETTER_ITEMS.findall(ar)]
    if ar_items and not _sequence_ok(ar_items):
        w("LIST_SEQUENCE_AR", f"Arabic list items out of order: {AR_LETTER_ITEMS.findall(ar)}")
    en_items = [ord(x) - ord("a") + 1 for x in EN_LETTER_ITEMS.findall(en)]
    if en_items and not _sequence_ok(en_items):
        w("LIST_SEQUENCE_EN", f"English list items out of order: {EN_LETTER_ITEMS.findall(en)}")
    ar_num_items = [int(x.translate(AR_DIGITS)) for x in AR_NUMBER_ITEMS.findall(ar)]
    en_num_items = [int(x) for x in EN_NUMBER_ITEMS.findall(en)]
    ar_total, en_total = len(ar_items) + len(ar_num_items), len(en_items) + len(en_num_items)
    if ar_total and en_total and ar_total != en_total:  # both sides list items, but not the same number
        w("LIST_COUNT_MISMATCH", f"{ar_total} Arabic list items vs {en_total} English: an item may be missing")
    elif ar_total != en_total:
        issues.append(info("LIST_FORMAT_DIFFERS", f"list items: {ar_total} in Arabic, {en_total} in English "
                                                  "(one side runs them inline)", article=n, page=page))

    # numbers: a reversed Arabic number (٦٦٤ for 466) means the digit-order fix failed
    body_ar = AR_PARA_NUMBERS.sub("", ar)
    ar_nums = {int(d.translate(AR_DIGITS)) for d in re.findall(r"[٠-٩]+", body_ar)}
    en_nums = {int(d) for d in re.findall(r"\d+", en)}
    for x in sorted(ar_nums - en_nums):
        rev = int(str(x)[::-1])
        if x >= 10 and rev != x and rev in en_nums:
            w("NUMBER_REVERSED", f"Arabic has {x} where the English has {rev}")

    # extraction artifacts
    if PRESENTATION_FORMS.search(ar):
        w("ARTIFACT_PRESENTATION_FORMS", "Arabic presentation-form glyphs left in the text")
    if m := PRIVATE_USE.search(ar + en):
        w("ARTIFACT_PRIVATE_GLYPH", f"private-use glyph U+{ord(m.group()):04X} (a font ligature with no letters): "
                                    "add it to PRIVATE_LIGATURES in rag/corpus/extract.py")
    if m := LIGATURE_BUG.search(ar):
        w("ARTIFACT_LIGATURE", f"garbled lam-alef near '{ar[max(0, m.start() - 12): m.end() + 12]}'")
    if m := STRAY_BRACKETS.search(ar):
        w("ARTIFACT_BRACKETS", f"stray bracket near '{ar[max(0, m.start() - 12): m.end() + 12]}'")
    if m := LATIN.search(ar):
        w("LATIN_IN_ARABIC", f"Latin text in the Arabic: '{m.group(0)}'")
    if ARABIC_LETTER.search(en):
        w("ARABIC_IN_ENGLISH", "Arabic letters in the English text")
    if en.count("(") != en.count(")"):
        issues.append(info("PARENS_UNBALANCED_EN", "unbalanced parentheses in the English", article=n, page=page))

    # hierarchy
    if r.get("section_number") is None and r.get("chapter_number") is None:
        w("NO_HIERARCHY", "article has neither a chapter nor a section")
    for level in NUMBERED_LEVELS:
        if r.get(f"{level}_number") is not None and not r.get(f"{level}_title_en"):
            w("HEADING_NO_TITLE", f"{level} {r[f'{level}_number']} has no English title")
    return issues


def check_golden(records: list[dict]) -> list[Issue]:
    by_number = {r["article_number"]: r for r in records}
    issues: list[Issue] = []
    for n, expected in GOLDEN.items():
        r = by_number.get(n)
        if r is None:
            continue
        for key, want in expected.items():
            got = r.get(key)
            if key.startswith("text_"):
                if _squash(got) != _squash(want):
                    issues.append(warning("GOLDEN_MISMATCH", f"{key} differs from the reference transcription", n))
                elif got.count("\n") != want.count("\n"):
                    issues.append(warning(
                        "GOLDEN_PARAGRAPHS",
                        f"{key}: {got.count(chr(10)) + 1} paragraphs, reference has {want.count(chr(10)) + 1}", n))
            elif got != want:
                issues.append(warning("GOLDEN_MISMATCH", f"{key} = {got!r}, reference {want!r}", n))
    return issues


def validate_records(records: list[dict], cfg: dict, *, full: bool = True) -> list[Issue]:
    """All record checks; `full=False` skips whole-corpus checks (for partial page ranges)."""
    issues = check_numbering(records, cfg) if full else []
    for r in records:
        issues.extend(check_record(r, cfg))
    issues.extend(check_golden(records))
    return issues


def _ranges(numbers: list[int]) -> str:
    out, start = [], None
    for i, x in enumerate(numbers):
        start = x if start is None else start
        if i + 1 == len(numbers) or numbers[i + 1] != x + 1:
            out.append(f"{start}" if start == x else f"{start}-{x}")
            start = None
    return ", ".join(out)
