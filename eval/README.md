# Evaluation question set

`questions.jsonl` has 62 questions: 28 topics from all four books of the Civil Code, each asked once in Arabic and
once in English, plus 2 that name an article by number and 4 out-of-scope questions the system should decline. It
feeds the chunking experiments (MLflow), and later RAGAS and the CI gate. How it was built, how it is checked and
its limits: [docs/evaluation-dataset.md](../docs/evaluation-dataset.md).

| Field | Meaning |
|---|---|
| `id`, `pair` | `q07-ar` / `q07-en` share `pair: q07`: same question in both languages |
| `type` | `factual` (the answer is stated in one article), `reasoning` (apply an article to a situation), `by_number`, `out_of_scope` |
| `relevant_articles` | the article(s) a correct answer rests on; empty for out-of-scope |
| `reference` | a short correct answer in the question's language, citing its articles |
| `status` | `draft` until reviewed |

Edit the list in [build_questions.py](build_questions.py), then run `python eval/build_questions.py`;
`tests/test_eval_set.py` checks the file stays consistent.

## Review checklist (status: draft)

The questions were written from the article texts and paraphrase them, so retrieval is tested on meaning rather than
shared words. Before trusting the scores, a legal reader should check each question:

1. Is `relevant_articles` complete? A question may also rest on another article (e.g. an exception elsewhere).
2. Is the reference answer correct and not overstated?
3. Is the Arabic natural, the way a person would actually ask it?
4. Are any questions too easy (reusing the article's own words) or ambiguous?

When a question is reviewed, set its `status` to `reviewed` in `build_questions.py`.
