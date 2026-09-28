# Plan: Civil Code PDF → one record and one chunk per article

Status: proposed, not implemented. Revised after testing on PDF pages 1–11.

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

## Record schema (`data/processed/articles.json`)
```json
{
  "article_number": 44,
  "part": "Preliminary Chapter", "book": null, "chapter": null,
  "section": "Persons", "topic": "Individuals", "subtopic": null,
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
Preliminary Chapter > Persons > Individuals
Article 44 | مادة 44
<text_ar>
<text_en>
```
Metadata: `article_number`, `part` … `subtopic`, `is_repealed`, `source_pages`.

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
