"""Corpus extraction, hierarchy, parsing and checks on synthetic input (no PDF needed)."""

from rag.corpus.extract import Line, PageResult, arabic_line, clean_ar
from rag.corpus.hierarchy import Hierarchy
from rag.corpus.parse import AR_MARK, EN_MARK, EN_MARK_GLUED, CorpusParser, paragraphs, resolve_digits, to_records
from rag.corpus.validate import check_numbering, check_record, validate_records

CFG = {"first_article": 1, "last_article": 5, "repealed": ["3-4"], "min_chars": 15, "max_chars": 6000,
       "ar_en_length_ratio": [0.35, 1.6]}


def ch(c, x0, x1, k=0, y=100.0):
    return {"c": c, "bbox": (x0, y, x1, y + 10), "bold": False, "k": k}


# --- Arabic rebuilt from glyph positions ------------------------------------------------

def test_lam_alef_ligature_is_repaired():
    # "الأم" as the PDF draws it: the hamza-alef is a zero-width glyph placed before the lam
    chars = [ch("ا", 528.78, 531.06), ch("أ", 528.78, 528.78), ch("ل", 522.77, 528.78), ch("م", 518.42, 522.77)]
    assert arabic_line(chars)[0] == "الأم"


def test_digits_follow_display_order_and_keep_the_stored_order():
    word = [ch("م", 540, 545, 0), ch("ا", 537, 540, 1), ch("د", 533, 537, 2), ch("ة", 529, 533, 3), ch(" ", 526, 529, 4)]
    # 492 drawn left to right, but stored in the text layer as ٢٩٤
    digits = [ch("٢", 510, 515, 5), ch("٩", 505, 510, 6), ch("٤", 500, 505, 7)]
    text, alts = arabic_line(word + digits)
    assert text == "مادة ٤٩٢"
    assert alts == (("٤٩٢", "٢٩٤"),)


def test_english_confirms_which_digit_order_is_right():
    lines = [Line("x", False, 0, 1, 0, digit_alts=(("٥٣٢", "٢٣٥"),))]
    fixed, changes = resolve_digits("وفقا للمواد من ٥٣٢ الي ٢٤٣", lines, "in accordance with Articles 235 to 243")
    assert fixed == "وفقا للمواد من ٢٣٥ الي ٢٤٣" and changes == ["٥٣٢→٢٣٥"]
    same, none = resolve_digits("المادة ٤٩٢", [Line("x", False, 0, 1, 0, digit_alts=(("٤٩٢", "٢٩٤"),))], "Article 492")
    assert same == "المادة ٤٩٢" and none == []


def test_private_use_ligatures_become_letters():
    assert clean_ar("يعلوا قرارام إلى") == "يعلوا قراراتهم إلى"
    assert clean_ar("لم يصدر حكم ائي") == "لم يصدر حكم نهائي"
    assert clean_ar("أن يقوم ذه الإزالة") == "أن يقوم بهذه الإزالة"  # glyph drawn before the word space
    assert clean_ar("المتعلقة لاك الشيء") == "المتعلقة بهلاك الشيء"


def test_mirrored_brackets_are_normalized():
    assert clean_ar("(١ ( تسرى النصوص") == "(١) تسرى النصوص"
    assert clean_ar("١( الدولة") == "١) الدولة"
    assert AR_MARK.match(clean_ar("مادة ( ١ ("))
    assert clean_ar("العقلية ، ولم") == "العقلية، ولم"


def test_markers_versus_cross_references():
    for marker in ("Article 44", "rticle 452", "Article1022"):
        assert EN_MARK.match(marker)
    for cross_ref in ("Article 444.", "Article 901 has been delivered to him."):
        assert not EN_MARK.match(cross_ref) and not EN_MARK_GLUED.match(cross_ref)
    assert EN_MARK_GLUED.match("Article 277 If the option belongs to the debtor").groups() == (
        "277", "If the option belongs to the debtor")
    assert AR_MARK.match("مادة (٤٤)") and AR_MARK.match("مادة ٤٤") and not AR_MARK.match("المادة ٤٤")


def test_paragraphs_break_on_markers_and_deliberately_short_lines():
    en = [Line("A special domicile may be elected for the performance of", False, 36, 290, 0),
          Line("a specific legal act.", False, 36, 130, 12),
          Line("The election of domicile must be evidenced by writing.", False, 36, 275, 24)]
    assert paragraphs(en, "en").split("\n") == [
        "A special domicile may be elected for the performance of a specific legal act.",
        "The election of domicile must be evidenced by writing."]
    ar = [Line("(١) يجوز اتخاذ موطن مختار.", False, 400, 559, 0), Line("(٢) ولا يجوز إثبات.", False, 420, 559, 12)]
    assert paragraphs(ar, "ar").count("\n") == 1
    wrapped = [Line("to the co-", False, 36, 290, 0), Line("owners jointly.", False, 36, 120, 12)]
    assert paragraphs(wrapped, "en") == "to the co-owners jointly."


# --- hierarchy ------------------------------------------------------------------------

def test_keyword_title_on_the_next_row():
    h = Hierarchy()
    h.feed(["Section II"], ["الفصل الثاني"])
    h.feed(["Methods of Extinction"], ["انقضاء الالتزام"])
    s = h.snapshot()
    assert (s["section_number"], s["section_title_en"], s["section_title_ar"]) == (2, "Methods of Extinction",
                                                                                  "انقضاء الالتزام")


def test_topic_numbered_only_in_arabic_and_topic_plus_subtopic():
    h = Hierarchy()
    h.feed(["Associations"], ["٣- الجمعيات"])
    assert (h.snapshot()["topic_number"], h.snapshot()["topic_title_ar"]) == (3, "الجمعيات")
    h.feed(["1. Elements of Contracts", "Consent:"], ["أولا:- أركان العقد", "١- الرضاء"])
    s = h.snapshot()
    assert (s["topic_number"], s["topic_title_en"], s["topic_title_ar"]) == (1, "Elements of Contracts", "أركان العقد")
    assert (s["subtopic_title_en"], s["subtopic_title_ar"]) == ("Consent", "الرضاء")


def test_new_level_resets_lower_levels_and_titles_are_bilingual():
    h = Hierarchy()
    h.feed(["FIRST PART", "OBLIGATIONS OR PERSONAL RIGHTS"], ["القسم الأول", "الالتزامات أو الحقوق الشخصية"])
    h.feed(["1. Elements"], ["١- أركان"])
    h.feed(["Chapter III", "Contracts for the hire of services"], ["الباب الثالث العقود", "الواردة على العمل"])
    s = h.snapshot()
    assert s["part_title_en"] == "Obligations or Personal Rights"
    assert (s["chapter_number"], s["chapter_title_ar"]) == (3, "العقود الواردة على العمل")
    assert s["topic_number"] is None


def test_part_title_missing_in_arabic_is_filled_from_the_official_structure():
    fix = {"starts_at_book": 3, "number": 2, "title_en": "Real Rights", "title_ar": "الحقوق العينية", "source": "x"}
    h = Hierarchy(part_corrections=[fix])
    h.feed(["SECOND PART"], [])
    h.feed(["REAL RIGHTS"], [])
    h.feed(["BOOK III", "The Principal Real Rights"], ["الكتاب الثالث", "الحقوق العينية الأصلية"])
    s = h.snapshot()
    assert (s["part_number"], s["part_title_en"], s["part_title_ar"]) == (2, "Real Rights", "الحقوق العينية")
    assert any(e.code == "PART_TITLE_FROM_OFFICIAL" for e in h.events)


def test_heading_out_of_order_is_flagged():
    h = Hierarchy()
    h.feed(["Section III", "Gifts"], ["الفصل الثالث", "الهبة"])
    h.feed(["Section II", "Exchange"], ["الفصل الثاني", "المقايضة"])
    assert any(e.code == "HEADING_ORDER" for e in h.events)


# --- parser -----------------------------------------------------------------------------

def L(text, row, y, page=1, bold=False, x0=36.0, x1=290.0):
    return Line(text, bold, x0, x1, y, page, row)


def A(text, row, y, page=1, bold=False):
    return Line(text, bold, 300.0, 559.0, y, page, row)


def page(no, en, ar):
    return PageResult(no, tuple(en), tuple(ar), 1, 3, "")


def parse(*pages, repealed=frozenset()):
    p = CorpusParser(set(repealed))
    for pg in pages:
        p.feed_page(pg)
    issues = p.finish()
    records, more = to_records(p)
    return records, issues + more


def test_articles_headings_page_breaks_and_preamble():
    p1 = page(1,
              en=[L("SECTION I", 0, 10, bold=True), L("Laws", 0, 22, bold=True),
                  L("Article 1", 1, 40), L("First article text", 1, 52)],
              ar=[A("مادة ١", None, 2), A("الفصل الأول", 0, 10, bold=True), A("القانون", 0, 22, bold=True),
                  A("مادة (١)", 1, 40), A("نص المادة الأولى", 1, 52)])
    p2 = page(2, en=[L("continues here.", 0, 10, page=2), L("Article 2", 1, 30, page=2), L("Second.", 1, 42, page=2)],
              ar=[A("تتمة.", 0, 10, page=2), A("مادة (٢)", 1, 30, page=2), A("الثانية.", 1, 42, page=2)])
    records, issues = parse(p1, p2)
    first, second = records
    assert (first["article_number"], first["section_number"], first["section_title_ar"]) == (1, 1, "القانون")
    assert first["text_en"] == "First article text continues here." and first["source_pages"] == [1, 2]
    assert first["text_ar"] == "نص المادة الأولى تتمة." and first["ar_number"] == 1
    assert second["text_en"] == "Second." and second["text_ar"] == "الثانية."
    codes = {i.code for i in issues}
    assert "PREAMBLE" in codes and "PAGE_BREAK" in codes  # the promulgation "مادة ١" above the table is skipped


def test_repeal_note_creates_flagged_stubs():
    pg = page(1,
              en=[L("Article 2", 0, 10), L("* Articles 2-4 have been repealed by Presidential", 0, 22), L("Decree.", 0, 34),
                  L("Article 5", 1, 50), L("Text.", 1, 62)],
              ar=[A("مادة (٢)", 0, 10), A("* المواد من ٢ إلى ٤ ملغاة", 0, 22), A("مادة (٥)", 1, 50), A("نص.", 1, 62)])
    p = CorpusParser(set())
    p.last = {"en": 1, "ar": 1}
    p.feed_page(pg)
    records, _ = to_records(p)
    by = {r["article_number"]: r for r in records}
    assert [n for n, r in by.items() if r["is_repealed"]] == [2, 3, 4]
    assert by[2]["repeal_note"] == "Articles 2-4 have been repealed by Presidential Decree."
    assert by[2]["repeal_note_ar"] and by[2]["text_ar"] == "" and by[2]["text_en"] == ""
    assert not by[5]["is_repealed"] and by[5]["text_en"] == "Text."


def test_cross_reference_at_row_start_is_text_and_mid_row_marker_must_be_next():
    pg = page(1,
              en=[L("Article 1", 0, 10), L("see", 0, 22), L("Article 2", 0, 34), L("Article 1", 1, 50), L("again.", 1, 62),
                  L("Article2", None, 80), L("Second article.", None, 92)],
              ar=[A("مادة (١)", 0, 10), A("نص.", 0, 22), A("مادة (٢)", 0, 80), A("الثانية.", 0, 92)])
    records, issues = parse(pg)
    by = {r["article_number"]: r for r in records}
    # "Article 2" mid-row is a wrapped cross-reference; "Article 1" opening row 1 repeats a number: both stay text
    assert by[1]["text_en"] == "see Article 2 Article 1 again."
    # the real "Article2" sits outside the grid; the Arabic "مادة (٢)" sits mid-row and is the next number
    assert by[2]["text_en"] == "Second article." and by[2]["text_ar"] == "الثانية."
    assert by[1]["text_ar"] == "نص."
    assert {"CROSS_REFERENCE_SKIPPED", "MARKER_MID_ROW_IGNORED", "MARKER_MID_ROW"} <= {i.code for i in issues}


def test_page_top_jump_rejected_in_english_accepted_in_arabic():
    p = CorpusParser(set())
    p.last = {"en": 10, "ar": 10}
    p.current = {"en": None, "ar": None}
    p.feed_page(page(3, en=[L("Article 12", 0, 5, page=3)], ar=[A("مادة ١٢", 0, 5, page=3)]))
    codes = [i.code for i in p.issues]
    assert "AMBIGUOUS_MARKER" in codes and p.last["en"] == 10
    assert "GAP_BEFORE_ARTICLE" in codes and p.last["ar"] == 12


# --- checks -------------------------------------------------------------------------------

def rec(n=1, **kw):
    base = {"article_number": n, "text_ar": "نص عربي كامل للمادة الأولى هنا.", "text_en": "A complete English article text.",
            "is_repealed": False, "repeal_note": None, "source_pages": [1], "ar_number": n,
            "section_number": 1, "section_title_en": "S", "chapter_number": None}
    return {**base, **kw}


def codes(r):
    return {i.code for i in check_record(r, CFG)}


def test_checks_flag_incomplete_or_misaligned_articles():
    assert codes(rec()) == set()
    assert "AR_NUMBER_MISMATCH" in codes(rec(ar_number=7))
    assert "AR_NUMBER_MISSING" in codes(rec(ar_number=None))
    assert "EMPTY_TEXT_AR" in codes(rec(text_ar=""))
    assert "CUT_OFF_EN" in codes(rec(text_en="A sentence that stops", source_pages=[1, 2]))
    assert "NO_FINAL_PERIOD_EN" in codes(rec(text_en="A sentence that just lacks its period"))
    assert "CUT_OFF_AR" in codes(rec(text_ar="نص ينتهي بفاصلة وهذا مريب،"))
    assert "PARAGRAPH_SEQUENCE" in codes(rec(text_ar="(١) أول فقرة هنا.\n(٣) ثالث فقرة هنا."))
    assert "LIST_COUNT_MISMATCH" in codes(rec(text_ar="أ- أول بند هنا.\nب- ثاني بند هنا.", text_en="a) one item\nb) two items\nc) three."))
    assert "NUMBER_REVERSED" in codes(rec(text_ar="وفقا لأحكام المادة ٣٥٢ من هذا القانون.", text_en="Under Article 253 of this law."))
    assert "ARTIFACT_LIGATURE" in codes(rec(text_ar="تقع هبة األموال المستقبلة باطلة."))
    assert "LATIN_IN_ARABIC" in codes(rec(text_ar="نص فيه كلمة English هنا."))
    assert "NO_HIERARCHY" in codes(rec(section_number=None))


def test_numbering_checks():
    records = [rec(1), rec(2), rec(3, is_repealed=True, repeal_note="x", text_ar="", text_en=""),
               rec(4, is_repealed=True, repeal_note="x", text_ar="", text_en="")]
    issues = {i.code: i for i in check_numbering(records, CFG)}
    assert "MISSING_ARTICLES" in issues and "5" in issues["MISSING_ARTICLES"].message
    assert "REPEALED_MISMATCH" not in issues
    assert "REPEALED_MISMATCH" in {i.code for i in check_numbering(records[:3], CFG)}


def test_golden_records_are_checked():
    good = rec(492, text_ar="تقع هبة الأموال المستقبلة باطلة.", text_en="A gift of future property is void.",
               part_number=1, book_number=2, section_number=3, section_title_en="Gifts")
    assert not {i.code for i in validate_records([good], CFG, full=False)} & {"GOLDEN_MISMATCH", "GOLDEN_PARAGRAPHS"}
    bad = {**good, "text_en": "A gift of future property is valid."}
    assert "GOLDEN_MISMATCH" in {i.code for i in validate_records([bad], CFG, full=False)}
