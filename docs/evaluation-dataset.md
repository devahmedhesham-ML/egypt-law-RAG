# How the evaluation dataset was built

The evaluation set, [eval/questions.jsonl](../eval/questions.jsonl), holds 62 questions on the Egyptian Civil Code.
Each one comes with the article(s) a correct answer rests on and a short reference answer. It feeds the chunking
experiments ([reports/chunking_experiments.md](../reports/chunking_experiments.md)) and will feed RAGAS and the CI
quality gate. This document explains what is in it, how it was made, how it is checked, and its limits.

**Status: draft.** All 62 questions were written by Claude (the AI coding assistant working on this repository) from
the text of the articles, and none has been reviewed by a legal reader yet. Every score computed from the set is
provisional until that review ([eval/README.md](../eval/README.md) has the checklist).

## What the set is for

The set has to answer two questions about the system:

1. **Retrieval**: does the right article reach the model? Each question names its relevant articles, so a ranked
   list of retrieved articles can be scored (hit@1, recall@5, MRR, nDCG).
2. **Answers**: is the answer grounded in the articles and correct? Each question has a reference answer for RAGAS
   metrics that compare against one (answer correctness, context recall), and faithfulness uses the retrieved
   articles.

That purpose drove five design choices:

- **Both languages, same questions.** Every topic is asked once in Arabic and once in English. Users ask in either
  language, and a legal question should find the same article whichever language it is asked in; pairing makes that
  measurable (`ar_en_top1_agreement`).
- **Paraphrase, don't copy.** Questions are worded the way a person would ask ("My tenant died. Does the lease end
  automatically?"), not with the article's own sentences. Copying the article's words would make retrieval look better
  than it is, because matching shared words is easy.
- **Two kinds of question.** *Factual* questions ask for something an article states ("What is a lease?").
  *Reasoning* questions describe a situation and need an article applied to it ("Can a husband take back a gift he
  gave to his wife?" → a gift between spouses is an obstacle to revoking it, Article 502(d)).
- **Questions the system must decline.** Four questions come from other bodies of law (criminal, labour,
  constitutional). No article answers them, so a good system says the provided articles do not cover the question.
- **Article numbers.** Two questions name an article directly, testing the lookup that puts named articles first,
  including a repealed one (Article 60), where the right answer is "repealed".

## What it contains

| | Arabic | English | Total |
|---|---|---|---|
| Factual (13 topics) | 13 | 13 | 26 |
| Reasoning (15 topics) | 15 | 15 | 30 |
| Names an article by number | 1 | 1 | 2 |
| Out of scope | 2 | 2 | 4 |
| **Total** | **31** | **31** | **62** |

The 58 in-scope questions rest on 31 distinct articles (30 in force, plus the repealed Article 60), spread over the
code's four books:

| Part of the code | Topics | Articles |
|---|---|---|
| Preliminary provisions (laws, persons) | 3 + 1 by number | 1, 29, 44; 60 (repealed) |
| Book 1: Obligations generally | 12 | 89, 120, 125, 129, 147, 157, 163, 172, 179, 226, 227, 374 |
| Book 2: Specific contracts | 9 + 1 by number | 418, 447, 452, 492, 502, 558, 601, 739, 772; 505 |
| Book 3: Principal real rights | 2 | 802; 968 and 969 |
| Book 4: Accessory rights (real securities) | 2 | 1030, 1130 |

All topics, with the English version of each question (the Arabic version asks the same thing):

| Pair | Type | Relevant | Where in the code | Question (English) |
|---|---|---|---|---|
| q01 | reasoning | 1 | Preliminary → Laws and their application | If no statute covers a dispute, what should an Egyptian judge base the decision on? |
| q02 | factual | 29 | Preliminary → Persons | When does a person's legal personality start and when does it end? |
| q03 | factual | 44 | Preliminary → Persons | How old must someone be to have full capacity to exercise their civil rights? |
| q04 | factual | 89 | Obligations → Contracts | At what point is a contract concluded between two people? |
| q05 | reasoning | 120 | Obligations → Contracts | I signed a contract while mistaken about something essential. Can I have it annulled? |
| q06 | reasoning | 125 | Obligations → Contracts | Does deliberately keeping quiet about an important fact during negotiations count as fraud that lets the other side void the contract? |
| q07 | reasoning | 129 | Obligations → Contracts | Someone took advantage of my obvious recklessness to make me sign a grossly one-sided contract. What can I do, and how long do I have? |
| q08 | reasoning | 147 | Obligations → Contracts | An unforeseeable general crisis made my contract ruinously expensive to perform, though not impossible. Can a court help? |
| q09 | reasoning | 157 | Obligations → Contracts | The other party to my bilateral contract has not performed. What are my options? |
| q10 | factual | 163 | Obligations → Unlawful acts | Is a person who harms someone else through their own fault obliged to compensate them? |
| q11 | factual | 172 | Obligations → Unlawful acts | How long does a victim have to sue for compensation for an unlawful act? |
| q12 | reasoning | 179 | Obligations → Enrichment without just cause | Someone gained money at my expense without any legal justification. Must they pay me back? |
| q13 | factual | 226 | Obligations → Compensation in lieu of performance | What interest does a debtor owe for being late in paying a known sum of money? |
| q14 | reasoning | 227 | Obligations → Compensation in lieu of performance | Can a lender and borrower agree on ten percent annual interest? |
| q15 | factual | 374 | Obligations → Extinction without payment | What is the ordinary limitation period for obligations? |
| q16 | factual | 418 | Specific contracts → Sale | How does the Civil Code define a contract of sale? |
| q17 | reasoning | 447 | Specific contracts → Sale | Is a seller liable for a hidden defect in the goods even if he did not know about it? |
| q18 | reasoning | 452 | Specific contracts → Sale | I found a defect in something I bought two years after it was delivered. Can I still claim under the warranty? |
| q19 | factual | 492 | Specific contracts → Gifts | Can a person give away, as a gift, property they will only own in the future? |
| q20 | reasoning | 502 | Specific contracts → Gifts | Can a husband take back a gift he gave to his wife? |
| q21 | factual | 558 | Specific contracts → Leases | What is a lease under the Civil Code? |
| q22 | reasoning | 601 | Specific contracts → Leases | My tenant died. Does the lease end automatically? |
| q23 | reasoning | 739 | Specific contracts → Gaming and betting | I lost money on a bet. Is the bet enforceable, and can I get the money back? |
| q24 | factual | 772 | Specific contracts → Suretyship | What does a guarantor promise under a contract of suretyship? |
| q25 | factual | 802 | Real rights → Ownership | What rights does the owner of a thing have over it? |
| q26 | reasoning | 968, 969 | Real rights → Acquisition of ownership | How long must someone possess land to become its owner by prescription? |
| q27 | factual | 1030 | Real securities → Mortgages | What does an official mortgage give the creditor? |
| q28 | reasoning | 1130 | Real securities → Privileged rights | Can a creditor and debtor create a privileged right simply by agreeing on it? |
| q29 | by number | 60 | Preliminary → Persons (repealed) | (Arabic only) ماذا تنص المادة 60 من القانون المدني؟ |
| q30 | by number | 505 | Specific contracts → Partnership | (English only) What does Article 505 say? |
| q31–q34 | out of scope | — | outside the Civil Code | Penalty for theft; paid annual leave; (Arabic) penalty for forging official documents; (Arabic) conditions for running for parliament |

## How it was made

1. **Map the code.** Every section of `data/processed/articles.json` (the corpus built from the official bilingual
   PDF) was listed with its article range, 55 sections in all, so the questions could cover the whole code rather
   than the parts that come to mind first.
2. **Shortlist articles.** About 40 candidates were picked across the sections, favouring provisions a person would
   plausibly ask about: definitions of the main contracts, capacity, how contracts are formed and annulled, liability,
   interest, limitation periods, warranties, gifts, leases, gambling, guarantees, ownership, prescription, mortgages
   and privileges.
3. **Read before writing.** The full English text of every candidate was read, with the Arabic text of several, so each
   question and reference answer matches what the article actually says, and the Arabic questions avoid reusing the
   article's own phrasing.
4. **Balance.** The shortlist was cut to 28 topics. Over-represented sections were thinned (four gift articles became
   two) and the freed places went to the last two books (ownership, Article 802; privileges, Article 1130), which the
   shortlist barely covered.
5. **Write each topic twice.** The English and Arabic questions were each written directly in their language, as
   natural questions, not translated word for word. Many use everyday situations ("I lost money on a bet") instead of
   legal terms.
6. **Label.** Each question got its type (factual or reasoning), its relevant article(s) and a one- or two-sentence
   reference answer in the same language, citing the article as the system is asked to (`[Article 502(d)]`,
   `[المادة 502 (د)]`). One topic needs two articles: acquisitive prescription takes 15 years under Article 968, or 5
   years with good faith and a registered title under Article 969.
7. **Add the special cases.** The two by-number questions (Article 60 in Arabic, Article 505 in English) and four
   out-of-scope questions (two per language) were added last.

The set is kept as code: the list lives in [eval/build_questions.py](../eval/build_questions.py), and
`python eval/build_questions.py` writes `questions.jsonl`. Each line looks like this:

```json
{"id": "q20-en", "pair": "q20", "lang": "en", "type": "reasoning",
 "question": "Can a husband take back a gift he gave to his wife?", "relevant_articles": [502],
 "reference": "No, a gift between spouses is one of the obstacles that prevent revoking a gift [Article 502(d)].",
 "status": "draft"}
```

## How it is checked

[tests/test_eval_set.py](../tests/test_eval_set.py) runs with the rest of the test suite and fails if the set breaks
its own rules:

- at least 50 questions, unique ids, every field filled, known types;
- out-of-scope questions, and only those, have no relevant articles;
- Arabic and English are balanced, and every factual or reasoning topic appears in both languages;
- every reference answer cites all of its relevant articles;
- every relevant article exists in the corpus.

Every MLflow run records which version of the set it used (`eval_sha256`, the first 12 characters of the file's
SHA-256) and how many questions were still drafts (`eval_draft`), so scores from different versions are never mixed
up silently.

## Limits

- **Not reviewed yet.** The relevant articles and reference answers are a careful first draft, not a legal reader's
  judgement. The review checklist is in [eval/README.md](../eval/README.md).
- **Written by the builder of the system.** The same assistant that built the retrieval wrote the questions.
  Paraphrasing limits the effect, but questions written by the people who will use the system (the feedback log is a
  good source) would be a fairer test.
- **Mostly one relevant article per question.** Real questions often rest on a rule plus an exception elsewhere. A
  retriever that returns a closely related article first (Article 447 for the warranty-period question q18, or Article
  500 for the spouse's gift, q20) is counted wrong at rank 1 even when that article helps. The review may widen some
  `relevant_articles`.
- **Small and uneven.** 58 in-scope questions: one question moves hit@1 by 0.017. 21 of the 28 topics come from the
  first two books; the real-rights and securities books have 2 topics each, and only 30 of the 1,093 live articles are
  covered.
- **Formal Arabic only.** No Egyptian colloquial Arabic, spelling variants, typos or mixed Arabic-English questions,
  all of which real users write.
- **Few declines.** Four out-of-scope questions are enough to see whether the system declines, not to set a reliable
  "no relevant article" threshold.
- **One question fails everywhere.** q28 (can a privilege be created by agreement? → Article 1130) is missed by all 8
  chunking runs in both languages. It is the first question to look at in the review: the wording may steer retrieval
  away, or another article may deserve to count as relevant.

## Next steps

1. Review all 62 (checklist in [eval/README.md](../eval/README.md)) and mark reviewed questions `"status": "reviewed"`.
2. Grow the set past 100 with the gaps above in mind: questions needing two articles, colloquial Arabic, the
   under-covered books, and more out-of-scope questions.
3. Add real questions from the test console's feedback log once testers use it.
