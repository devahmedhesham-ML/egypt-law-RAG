# Plan: Civil Code PDF → one JSON record per article (DVC stage + validation)

Status: proposed, not implemented.

## Context
Handbook Step 0 for Project 2: the PDF is raw input, not the corpus. Every later stage depends on a structured, citable corpus: retrieval, the `/ask` citations, RAGAS, and the test console (which switches from its 19-article sample to `data/processed/articles.json` automatically). To ground the design, the pipeline was prototyped **in memory against the real PDF** (read-only, nothing written).

**Prototype result:** 1,149/1,149 article numbers accounted for (1,088 EN + 1,091 AR + 56 repealed), no gaps. Article 492 comes out verbatim. Article 147 matches the handbook's example schema.

## What the PDF actually does (evidence from probing)
| Pitfall | Evidence | Fix |
|---|---|---|
| Two columns (EN left, AR right), but some text blocks span both | blocks with x0=36 → x1=559 | Assign each **span** by its x-centre vs page midline, never by block |
| Fonts separate the languages | EN = Calibri, AR = ArialMT; headings `*-Bold` | Bold = heading signal; font = cross-check |
| Lam-alef ligatures garbled by PyMuPDF | `األموال` for `الأموال`, `إال` for `إلا`; the alef is a **zero-width** char placed before the lam | Swap zero-width alef-variant + `ل` → `ل` + alef (geometric rule, so the definite article `ال` is untouched) |
| Digit order inconsistent (Word writes some numbers visually, some logically) | `٢٩٤` for 492; `٥١ أكتوبر` for 15 | Rebuild Arabic text from char **x-positions**: letters right→left, digit runs left→right, over the whole visual line (numbers can be split across spans: `مادة ٠١ ٦` = 601) |
| Mirrored and offset brackets | `مادة ( ١` + `(` 2pt lower; `(١ …) (` | Merge spans within 3.5pt vertically; normalize `(n)` markers; strip stray brackets |
| Cross-references look like headings | `Article 444.` wrapped onto its own line | Article heading = a span that is **exactly** `Article N` (no trailing text or period), detected before line merging |
| Source typos | `rticle 452`, `Article1022` | `^A?rticle\s*(\d+)$` |
| Promulgation law on p.1 has its own "مادة ١/٢" | untranslated, above `نصوص القانون المدنى` | Arabic stream starts after that heading |
| Repealed ranges given as notes, not headings | p7 `* Articles 54-80 have been repealed by Presidential Decree.`; p53 `Articles 389-417 repealed` | Parse notes → flagged stub records (don't delete); note lines never enter article text |
| Heading hierarchy with inconsistent case | `FIRST PART`, `BOOK I`, `CHAPTER I`, `SECTION II`/`Section I`, `1. Elements of Contracts`, `Consent:` | State machine: part → book → chapter → section → topic → subtopic; before `FIRST PART` = "Preliminary Chapter" |

Remaining defects the prototype showed:
- `(١)(` bracket leftovers;
- Article 54 body = "Decree." (tail of a two-line note);
- repealed stubs lack hierarchy;
- 9 EN/AR mismatches (explained above: split digits, merged heading lines).

All are covered by the fixes in the table. Validation fails the build if any recur.

## Record schema (handbook schema, extended)
```json
{
  "article_number": 492,
  "part": "Obligations or Personal Rights", "book": "Specific Contracts",
  "chapter": "Contracts as Regards Ownership", "section": "Gifts",
  "topic": "Elements of a Gift", "subtopic": null,
  "text_ar": "تقع هبة الأموال المستقبلة باطلة.",
  "text_en": "A gift of future property is void.",
  "text_ar_normalized": "تقع هبة الاموال المستقبلة باطلة.",
  "is_repealed": false, "repeal_note": null,
  "source_page": 64,
  "citation": "Egyptian Civil Code, Article 492"
}
```
- Paragraphs are joined with `\n`. A new paragraph starts at an Arabic `(n)` marker, or at a vertical gap larger than 1.6× the line spacing, so long articles can later be split by paragraph (handbook step 4).
- All-caps headings are title-cased.
- `text_ar` stays faithful to the source; `text_ar_normalized` is for search only (see below).

## Files
```
src/rag/corpus/
  extract.py     PDF → visual lines per column (char-level Arabic rebuild, span markers)
  parse.py       lines → articles: markers, hierarchy, repeal notes, paragraphs
  normalize.py   Arabic search normalization + punctuation spacing
  validate.py    corpus checks (used by build and by pytest)
  build.py       CLI: python -m rag.corpus.build → articles.json + corpus_report.json
tests/test_corpus_extract.py   unit tests on synthetic chars (no PDF; CI-safe)
tests/test_corpus.py           validation of the built corpus (skips if not built)
dvc.yaml                       build_corpus stage
params.yaml                    corpus: section
```
Reused:
- `pymupdf` and `pyarabic` (already in `requirements.txt`);
- `data/raw/egyptian_civil_code.pdf` (DVC-tracked);
- the console's `rag.ui.data.load_articles()`, which already prefers `data/processed/articles.json`;
- the `rag.ui.status._corpus()` stage, which flips to "working" when the file exists.

`arabic-reshaper` / `python-bidi` turn out to be unnecessary: position-based rebuild replaces them. They stay in `requirements.txt` until a separate cleanup decides otherwise.

## Core code
The extraction logic and the regexes below ran in the in-memory prototype. The `parse.py` excerpt is a condensed sketch of the prototype's loop, with the fixes for the defects it exposed folded in.

### extract.py: faithful Arabic from glyph positions
```python
ALEFS, DIGIT = set("اأإآ"), re.compile(r"[0-9٠-٩]")

def arabic_line_text(chars: list[dict]) -> str:
    """Rebuild one visual Arabic line: letters right→left, numbers left→right, fix lam-alef."""
    cs = sorted(chars, key=lambda c: -(c["bbox"][0] + c["bbox"][2]) / 2)
    out, i = [], 0
    while i < len(cs):
        c = cs[i]
        zero_width = c["bbox"][2] - c["bbox"][0] < 0.01
        if c["c"] in ALEFS and zero_width and i + 1 < len(cs) and cs[i + 1]["c"] == "ل":
            out += ["ل", c["c"]]; i += 2; continue          # ligature: lam first
        if DIGIT.match(c["c"]):
            j = i
            while j < len(cs) and (DIGIT.match(cs[j]["c"]) or cs[j]["c"] == " " and j + 1 < len(cs) and DIGIT.match(cs[j + 1]["c"])):
                j += 1
            digits = [d for d in cs[i:j] if DIGIT.match(d["c"])]
            out += [d["c"] for d in sorted(digits, key=lambda d: d["bbox"][0])]  # numbers read LTR
            i = j; continue
        out.append(c["c"]); i += 1
    return "".join(out)
```
Lines are built per (page, column): collect spans, assign by x-centre, group spans within 3.5pt vertically. For Arabic, rebuild at **char level across the whole line**. For English, join spans left→right. Each line keeps `page`, `y`, `bold`, and `is_marker`, where the marker test runs on the raw span before merging:
```python
EN_MARK = re.compile(r"^A?rticle\s*(\d+)$")                   # 'rticle 452', 'Article1022'
AR_MARK = re.compile(r"^مادة\s*[()]?\s*([0-9٠-٩]+)\s*[()]?$")  # 'مادة (١)' with mirrored brackets
REPEAL  = re.compile(r"Articles?\s+(\d+)\s*-\s*(\d+)\b.*repealed", re.I)
```

### parse.py: hierarchy state machine + repeal notes
```python
LEVELS = ["part", "book", "chapter", "section", "topic", "subtopic"]
KEYWORDS = [("part", r"^(FIRST|SECOND|THIRD|FOURTH) PART$"), ("book", r"^BOOK [IVXLC]+$"),
            ("chapter", r"^CHAPTER [IVXLC]+$"), ("section", r"^SECTION [IVXLC]+$")]

def parse_english(lines):
    head, pending, arts, cur, repeals, in_note = {"part": "Preliminary Chapter"}, None, {}, None, {}, False
    for ln in lines:
        if ln.marker is not None:                      # "Article N" heading
            cur, in_note = ln.marker, False
            arts[cur] = Draft(page=ln.page, **{k: head.get(k) for k in LEVELS}); continue
        if (m := REPEAL.search(ln.text)):              # "* Articles 54-80 have been repealed …"
            note = RepealNote(text=ln.text, page=ln.page, hierarchy=dict(head))  # stubs inherit hierarchy
            for n in range(int(m[1]), int(m[2]) + 1):
                repeals[n] = note
            in_note = True; continue
        if in_note and not ln.bold:                    # "Decree." continuation stays in the note
            note.text += " " + ln.text; continue
        if ln.bold:
            level = keyword_level(ln.text)
            if level: pending = level; continue       # next bold line is its title
            level = pending or ("topic" if re.match(r"^\d+\s*\.", ln.text) else "subtopic")
            head[level] = smart_title(re.sub(r"^\d+\s*\.\s*", "", ln.text).rstrip(":"))
            for lower in LEVELS[LEVELS.index(level) + 1:]: head.pop(lower, None)
            pending = None; continue
        if cur is not None: arts[cur].add_line(ln)     # paragraph break on large y-gap
    return arts, repeals
```
`parse_arabic` skips lines until `نصوص القانون`, splits on `AR_MARK`, and drops bold lines (Arabic headings). A new paragraph starts at a `(n)` or `(أ)` marker.

`merge()` joins by article number:
- 1..1149 from EN ∪ AR ∪ repeal notes;
- a repealed record takes `repeal_note` and the inherited hierarchy;
- every EN/AR disagreement goes into the report.

### normalize.py
```python
def normalize_ar(text: str) -> str:
    """For embeddings/search only (same function must be applied to queries)."""
    text = araby.strip_tashkeel(araby.strip_tatweel(text))
    text = re.sub("[أإآٱ]", "ا", text).replace("ى", "ي")
    return re.sub(r"\s+([،.؛:؟])", r"\1", text)
```

### validate.py (handbook step 3: validate before you embed)
Returns a list of failures. `build.py` exits non-zero if any exist, so `dvc repro` can't produce a broken corpus.
- Numbers are exactly 1..1149 (from `params.corpus`), each once.
- The repealed set is exactly {54..80, 389..417}.
- Every non-repealed record has non-empty `text_ar` and `text_en`, and a non-null `part`.
- Length: no record over `max_chars` (6000; the longest real one is 1143 at 3,564) and none under 15 chars unless repealed.
- No extraction artifacts in `text_ar`:
  - Arabic presentation forms (U+FB50–FDFF, U+FE70–FEFF);
  - ligature bugs (`\bا[أإآ]ل`, `إال`);
  - stray `(`/`)` or `)(`;
  - Latin letters.
- No Arabic letters in `text_en`.
- Golden records match exactly: 492 (full text), 505 (first sentence), 147 (first sentence + hierarchy).

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
`corpus_report.json` records:
- counts;
- repealed ranges;
- EN/AR mismatches;
- source typos that were tolerated (`rticle 452`, `Article1022`, `٥١ أكتوبر`);
- warnings.

It's the "what you hit" section for the handbook report. After `dvc repro`, run `dvc push` so the JSON is pullable from the public S3 remote.

## Verification
1. `pytest tests/test_corpus_extract.py`: ligature swap, digit ordering (visual and logical input), markers vs cross-refs, bracket cleanup, on synthetic char dicts.
2. `dvc repro` → builds and validates; `dvc status` clean; `data/processed/corpus_report.json` shows 1,149 records, 56 repealed, 0 failures.
3. `pytest tests/test_corpus.py` passes on the built file.
4. Eyeball check (handbook: "eyeball 20 random articles"): a script prints 20 random records side by side with the PDF page number, compared by hand.
5. The console at `http://localhost:7860`:
   - Status shows "Structured corpus: Working, 1149 articles";
   - the Corpus view lists every article;
   - Ask works with any article as context.
6. Commit `dvc.yaml`, `dvc.lock`, `params.yaml`, code and tests; `dvc push`.
