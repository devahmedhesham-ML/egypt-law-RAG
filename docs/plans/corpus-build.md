# Plan: Civil Code PDF → one record and one chunk per article

Status: **implemented** (`src/rag/corpus`, `src/rag/ingest`, `dvc.yaml`). The plan below was revised after testing on PDF pages 1–11; the "As built" section records where the full PDF changed it.

## As built
**Result.** `dvc repro` builds 1,149 records (1,093 live, 56 repealed) in ~3.5 s and indexes them in ~30 s. All four smoke queries (two Arabic, two English) return the expected article first, with 10 distinct articles in every top 10.

**Extraction became a hybrid.** "One table row = one article" holds almost everywhere but breaks in a few places:
- "SECOND PART" (p. 113) sits outside the table grid;
- "Article1022" (p. 147) sits outside the grid, and its English continues outside it at the top of p. 148;
- p. 147's table splits into one row per line.

So each column is read top to bottom, including text outside the grid in position order, and split on its own markers ("Article N", "مادة (n)"). The two sides are joined by article number. Table rows are kept as a cross-check (an article's two markers should share a row) and as a guard:
- An English "Article N" never starts an article mid-row, because English cross-references look identical.
- An Arabic "مادة" may start the very next article mid-row, because Arabic cross-references read "المادة".
- Text outside the grid before the first table is skipped (the promulgation law, which has its own "مادة ١").

**Part 2 correction (supersedes the rule below).** The PDF *does* print "SECOND PART" (outside the grid) and "REAL RIGHTS" (in the table). Only the Arabic title is missing, and that alone is filled from the official structure.

**More Arabic repairs found on the full PDF:**
- **Digit order.** Word stored some numbers correctly but displays them reversed (p. 34 shows "٥٣٢" for Article 235), and others the other way round (article markers). Where the displayed and stored orders differ, the one the English text of the same article confirms wins; this happened 3 times.
- **Private-use ligatures.** The Arabic font maps four "…ه" ligatures to private-use code points: U+E811 لمه, U+E812 به, U+E814 ته, U+E815 نه. That's 25 occurrences, all mapped back to letters.
- **Unbolded headings.** Some Arabic headings are not bold ("بيع التركة"), so heading titles are taken from the whole row above that row's Arabic marker.

**Checks are warnings, never failures.** The 11 warnings left are all defects in the source PDF, reviewed by hand:

| Article | Warning | Cause in the PDF |
|---|---|---|
| 1022 | no Arabic text, no Arabic marker | no "مادة ١٠٢٢" printed; its Arabic sits inside Article 1021's cell (hence 1021's length ratio and the gap before 1023) |
| 519 | length ratio 0.32 | the Arabic omits a clause the English has |
| 970 | length ratio 3.73 | the Arabic carries a later amendment the English lacks |
| 1060 | length ratio 0.33 | Arabic much shorter than the English |
| 260, 936, 1115 | cut off at a page break | no final period in the Arabic across the page break |
| 813 | ends on a comma | the Arabic ends mid-clause |

A single-page article without a final period is a note, not a warning (source style, 33 cases). Reference articles 43, 44, 88 and 89 (transcribed from screenshots) and 492 match word for word.

**Indexing.** The chunk is heading path + "Article N | مادة N" + Arabic + English, as planned. On an RTX 4070 Ti SUPER, one embedding replica with 4,096-token batches is fastest (8.8 s, 33.9k tokens/s, 4.7 GB). Four replicas are slower (10.1 s, 11.9 GB), because one already keeps the GPU ~91% busy. Auto mode therefore adds replicas only when they fit in memory **and** the GPU has idle compute.

## Context
Handbook Step 0 for Project 2: the PDF is raw input, not the corpus. Every later stage depends on a structured, citable corpus: retrieval, `/ask` citations, RAGAS, and the test console (which switches from its 19-article sample to `data/processed/articles.json` automatically).

Two requirements drive this revision:
1. **One chunk per article, Arabic and English together.** The top 10 retrieved results must be 10 different articles, not 5 articles each returned twice (once per language).
2. **The PDF is a table.** Each row holds one article: English in the left cell, Arabic in the right, with ruled lines between rows. The row is the article boundary.

## Extraction: table rows, not text markers
`page.find_tables()` finds one 2-column table per page (column split at x≈298, rows bounded by horizontal rules). Each row is one of:

| Row kind | How it's recognized | What happens |
|---|---|---|
| Article | English cell starts with a line that is exactly `Article N` | new record |
| Heading | all lines bold (`FIRST PART`, `BOOK I`, `CHAPTER I`, `SECTION II`, `1. Elements of Contracts`, `Consent:`) | updates the hierarchy |
| Continuation | first row on a page, no `Article N`, not bold | appended to the previous article (page break) |
| Repeal note | `Articles 54-80 have been repealed…` / `Articles 389-417 repealed` | flagged stub records for the whole range |

The promulgation law on page 1 sits above the table (y < 277), so it's excluded without any special rule.

### Tested on pages 1–11 (prototype, in memory)
- 112 records (Articles 1–112): 85 live, 27 repealed (54–80), **no gaps**.
- Arabic article number agrees with the English one on **all 85** live articles.
- **All 7 page-split articles stitched correctly**: 6 (p1→2), 20, 42, 83, 91, 98, 104 (p10→11).
- Articles 43 and 44 match the PDF word for word, including Arabic paragraph markers `(١) (٢) (٣)`.
- Repeal stubs carry the note ("…by Presidential Decree.", two-line note joined) and the section they sit in.

### PDF pitfalls and fixes (found by probing the whole file)
| Pitfall | Evidence | Fix |
|---|---|---|
| Lam-alef ligatures garbled | `األموال` for `الأموال`; the alef is a **zero-width** char before the lam | Swap zero-width alef-variant + `ل` → `ل` + alef (geometric, so the definite article `ال` is untouched) |
| Digit order inconsistent | `٢٩٤` for 492; numbers split across spans (`٠١ ٦` = 601) | Rebuild each Arabic line from char **x-positions**: letters right→left, digit runs left→right |
| Mirrored, offset brackets | `مادة ( ١` + `(` 2pt lower | Merge chars within 3.5pt vertically; normalize `(n)`, `(أ)` markers; strip stray brackets |
| Cross-references look like headings | `Article 444.` wrapped onto its own line | Heading must be exactly `Article N`, no trailing text or period |
| Source typos | `rticle 452`, `Article1022` | `^A?rticle\s*(\d+)$` |
| Source text typos | `٥١ أكتوبر` (should be 15) | Kept faithful; listed in the build report |
| Repealed ranges given as notes | p7, p53 | Stub records with `is_repealed: true`, never deleted |

## Hierarchy: numbers + titles in both languages
Heading rows hold the keyword + number and the title in both cells (EN left, AR right). Every level is stored as a **number** plus an **English and Arabic title**.

| Level | English keyword | Arabic keyword | Number from |
|---|---|---|---|
| part | `FIRST PART` | القسم | ordinal word (FIRST → 1) |
| book | `BOOK I` | الكتاب | Roman numeral |
| chapter | `CHAPTER I` / `Chapter IV` | الباب | Roman numeral |
| section | `SECTION I` / `Section II.` | الفصل | Roman numeral |
| topic | `1.` `2-` `1.Sale…` | `١-` or `أولا` | EN number, else AR number (EN sometimes unnumbered: "Associations" = `٣- الجمعيات`) |
| subtopic | unnumbered (`Consent:`, `Obligations of the Vendor`) | unnumbered | title only |

Rules found by dumping all 221 heading rows:
- **Articles 1–88 (Preliminary Chapter)** have **no part, book or chapter**, only section 1–3 (+ topic). Part 1 starts after Article 88.
- A keyword may sit alone with its title in the next row (`Section II` p46 → title p47) or on the same line (`Section I The Right of Ownership in General`).
- A new level resets all levels below it.
- Wrapped English titles ("…without an / Owner") are joined when the Arabic cell has fewer lines; otherwise the second line is the next level (topic + subtopic in one row, e.g. `1. Elements of Contracts / Consent:`).
- **Part 2** (corrected in "As built" above: the PDF prints "SECOND PART / REAL RIGHTS", outside the table grid, so table-row extraction missed it). The official text puts Books III–IV under **القسم الثاني: الحقوق العينية** (Real Rights), per [qadaya.net part 3](https://qadaya.net/?p=6646) and [part 4](https://qadaya.net/?p=6648). Only the Arabic title "الحقوق العينية" is taken from there; the build report records it.

Prototype over the whole PDF: 1,087 articles tagged; no level ends up with a number but no title. Articles 89, 147, 418, 492 and 1149 come out as expected.

## Record schema (`data/processed/articles.json`)
```json
{
  "article_number": 89,
  "part_number": 1, "part_title_en": "Obligations or Personal Rights", "part_title_ar": "الالتزامات أو الحقوق الشخصية",
  "book_number": 1, "book_title_en": "Obligations Generally", "book_title_ar": "الالتزامات بوجه عام",
  "chapter_number": 1, "chapter_title_en": "Sources of Obligations", "chapter_title_ar": "مصادر الالتزام",
  "section_number": 1, "section_title_en": "Contracts", "section_title_ar": "العقد",
  "topic_number": 1, "topic_title_en": "Elements of Contracts", "topic_title_ar": "أركان العقد",
  "subtopic_title_en": "Consent", "subtopic_title_ar": "الرضاء",
  "…": "text_ar, text_en, is_repealed, repeal_note, source_pages, citation as below"
}
```
Example of a Preliminary-Chapter article (no part/book/chapter):
```json
{
  "article_number": 44,
  "part_number": null, "book_number": null, "chapter_number": null,
  "section_number": 2, "section_title_en": "Persons", "section_title_ar": "الأشخاص",
  "topic_number": 1, "topic_title_en": "Individuals", "topic_title_ar": "الشخص الطبيعي",
  "text_ar": "(١) كل شخص بلغ سن الرشد متمتعا بقواه العقلية، ولم يحجر عليه، يكون كامل الأهلية لمباشرة حقوقه المدنية.\n(٢) وسن الرشد هى إحدى وعشرون سنة ميلادية كاملة.",
  "text_en": "All persons attaining majority in possession of their mental faculties and not under legal disability, have full legal capacity to exercise their civil rights. The majority of a person is fixed at twenty one years completed in accordance with the Gregorian calendar.",
  "is_repealed": false, "repeal_note": null,
  "source_pages": [6],
  "citation": "Egyptian Civil Code, Article 44"
}
```
- `text_ar` keeps the source's paragraph markers, one paragraph per line.
- English has no paragraph markers in the PDF, so `text_en` is one block.
- All-caps headings are title-cased.
- `source_pages` lists every page an article spans.

## Chunking: one bilingual chunk per article
Chunk text (built at ingestion, not stored in the corpus):
```
Section 2 Persons | الفصل الثاني الأشخاص > Topic 1 Individuals | الشخص الطبيعي
Article 44 | مادة 44
<text_ar>
<text_en>
```
Metadata: `article_number`, every `*_number` and `*_title_en`/`*_title_ar`, `is_repealed`, `source_pages`.

One vector per article means the top 10 are always 10 distinct articles. Repealed articles are indexed with their note, so "What does Article 60 say?" retrieves "repealed" instead of a hallucination.

### Experiment (pages 1–10: 77 articles, 32 questions = 16 EN + 16 AR, Qwen3-Embedding-0.6B)
Questions were paraphrased, not copied from the text, each with one known correct article.

| Strategy | Distinct articles in top 10 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| S1 separate AR and EN chunks | **7.2** | 94% | 97% | 97% | 0.952 |
| S2 one bilingual chunk | 10 | 91% | 97% | 97% | 0.938 |
| **S3 one bilingual chunk + heading path (chosen)** | **10** | **94%** | **97%** | **100%** | **0.961** |
| S4 separate chunks, grouped by article | 10 | 94% | 97% | 97% | 0.953 |
| S5 query-language chunk only | 10 | 94% | 97% | 100% | 0.954 |

- Separate chunks waste about 28% of the top-10 slots on the same article's other language.
- The heading path lifts the plain bilingual chunk (Arabic Recall@1 88% → 94%).
- A custom legal instruction for the query did not help; keep Qwen3's default query prompt.
- The eval is small: one question is about 3 points. **Re-run S3 vs S4 on the full corpus** (about 1,093 live articles, more distractors) as the first MLflow experiment before committing to S3 for good.
- If long articles ever have to be split by paragraph, group hits by `article_number` (S4-style) so the top 10 stay 10 distinct articles.

## Files
```
src/rag/corpus/
  extract.py     table rows → cell lines (char-level Arabic rebuild)
  parse.py       rows → articles: markers, headings, continuations, repeal notes
  normalize.py   Arabic search normalization (same function applied to queries)
  validate.py    corpus checks (used by build and by pytest)
  build.py       CLI: python -m rag.corpus.build → articles.json + corpus_report.json
tests/test_corpus_extract.py   unit tests on synthetic chars (no PDF; CI-safe)
tests/test_corpus.py           validation of the built corpus (skips if not built)
dvc.yaml                       build_corpus stage
params.yaml                    corpus: section
```
Reused:
- `pymupdf` and `pyarabic` (already in `requirements.txt`);
- the DVC-tracked PDF;
- `rag.ui.data.load_articles()` and the console's status check, which already look for `data/processed/articles.json`.

`arabic-reshaper` / `python-bidi` turn out to be unnecessary; they stay in `requirements.txt` until a separate cleanup.

## Core code (from the tested prototype)

### extract.py
```python
def table_rows(doc, pages):
    """Yield (page_no, en_lines, ar_lines) for each table row, in reading order."""
    for pno in pages:
        page = doc[pno]
        chars = page_chars(page)                         # rawdict chars with a bold flag
        for table in page.find_tables().tables:
            for row in table.rows:
                left, right = row.cells[0], row.cells[-1]
                en = cell_lines([c for c in chars if left and inside(c, left)], "en")
                ar = cell_lines([c for c in chars if right and inside(c, right)], "ar")
                if en or ar:
                    yield pno + 1, en, ar


def arabic_line_text(chars):
    """Letters right-to-left by x, numbers left-to-right by x, lam-alef ligature repaired."""
    cs = sorted(chars, key=lambda c: -(c["bbox"][0] + c["bbox"][2]) / 2)
    out, i = [], 0
    while i < len(cs):
        c = cs[i]
        zero_width = c["bbox"][2] - c["bbox"][0] < 0.01
        if c["c"] in ALEFS and zero_width and i + 1 < len(cs) and cs[i + 1]["c"] == "ل":
            out += ["ل", c["c"]]; i += 2; continue
        if DIGIT.match(c["c"]):
            j = i
            while j < len(cs) and (DIGIT.match(cs[j]["c"]) or cs[j]["c"] == " " and j + 1 < len(cs) and DIGIT.match(cs[j + 1]["c"])):
                j += 1
            digits = [d for d in cs[i:j] if DIGIT.match(d["c"])]
            out += [d["c"] for d in sorted(digits, key=lambda d: d["bbox"][0])]
            i = j; continue
        out.append(c["c"]); i += 1
    return "".join(out)
```

### parse.py (row loop, condensed)
```python
for page_no, en, ar in table_rows(doc, pages):
    starts_article = False
    for ln in en:
        if (m := EN_MARK.match(ln.text)):                  # ^A?rticle\s*(\d+)$
            cur = articles[int(m[1])] = Article(int(m[1]), page_no, snapshot(head))
            starts_article = True; continue
        if (m := REPEAL.search(ln.text)):                  # Articles 54-80 … repealed
            note = RepealNote(ln.text, page_no, snapshot(head))
            for n in range(int(m[1]), int(m[2]) + 1): repeals[n] = note
            continue
        if ln.bold and not starts_article:                 # heading row
            update_hierarchy(head, ln.text); continue
        if cur and cur.number in repeals and not cur.en:   # "Decree." tail of the note
            repeals[cur.number].text += " " + ln.text; continue
        cur.en.append(ln.text); cur.pages.add(page_no)     # body or page-break continuation
    for ln in ar:
        if (m := AR_MARK.match(ln.text)):                  # مادة (n) → cross-check number
            cur.ar_number = to_int(m[1]); continue
        if not ln.bold: cur.ar.append(ln.text)
```

### dvc.yaml
```yaml
stages:
  build_corpus:
    cmd: python -m rag.corpus.build
    deps:
      - data/raw/egyptian_civil_code.pdf
      - src/rag/corpus
    params:
      - corpus
    outs:
      - data/processed/articles.json
    metrics:
      - data/processed/corpus_report.json:
          cache: false
```
```yaml
# params.yaml
corpus:
  first_article: 1
  last_article: 1149
  repealed: ["54-80", "389-417"]
  line_merge_pt: 3.5
  max_chars: 6000
```

## Validation (build fails if any check fails)
- Numbers are exactly 1..1149, each once. The repealed set is exactly {54..80, 389..417}.
- The Arabic `مادة (n)` agrees with the English `Article N` for every live article.
- Every live record has non-empty `text_ar` and `text_en`, and a non-null `part`.
- Every page-break continuation row was attached to an article (none dropped).
- No record over `max_chars` (6000) and none under 15 chars unless repealed.
- No extraction artifacts in `text_ar`:
  - Arabic presentation forms;
  - `األ` / `إال` ligature bugs;
  - stray or doubled brackets;
  - Latin letters.
- No Arabic letters in `text_en`.
- Golden records match exactly: 43 and 44 (from the PDF, above) and 492.

`corpus_report.json` records:
- counts;
- page-split articles;
- repealed ranges;
- tolerated source typos;
- warnings.

This is the "what you hit" section for the handbook report.

## Verification
1. `pytest tests/test_corpus_extract.py`: ligature swap, digit order, markers vs cross-references, bracket cleanup, continuation rows.
2. `dvc repro` builds and validates; `corpus_report.json` shows 1,149 records, 56 repealed, 0 failures.
3. `pytest tests/test_corpus.py` passes on the built file.
4. Eyeball 20 random articles against the PDF pages listed in `source_pages` (handbook requirement).
5. The test console shows "Structured corpus: Working, 1149 articles"; the Corpus view lists every article.
6. `dvc push`, then commit `dvc.yaml`, `dvc.lock`, `params.yaml`, code and tests.
7. First retrieval experiment: S3 vs S4 on the full corpus, logged to MLflow.
