# How reliable is the faithfulness judge?

The quality gate is judged by the production model itself (Qwen2.5-7B-Instruct-AWQ on vLLM). To see how far its
scores can be trusted, the same 20 gate answers (identical text, generated once by Qwen2.5) were re-judged by a much
larger model, Bedrock `openai.gpt-oss-120b`, on 2026-10-07:

| Judge | Mean faithfulness | Passes 0.75 | Same score as the other judge |
|---|---|---|---|
| Qwen2.5-7B (default, `python -m rag.evaluation.gate`) | 0.787 | yes | 10 of 20 questions |
| gpt-oss-120b (`--judge-backend bedrock`) | 0.834 | yes | 10 of 20 questions |

The judges disagree on half the questions, in both directions:

| Question | Qwen judge | gpt-oss judge | Answer (start) |
|---|---|---|---|
| q19-ar | **0.00** | 1.00 | هبة مال لن يملكه الواهب إلا في المستقبل باطلة حسب [Article 492]. *(correct and grounded)* |
| q11-en | **1.00** | 0.50 | A victim has up to three years from the date they knew of the injury… |
| q26-en | **1.00** | 0.50 | To become the owner of land through acquisitive prescription, someone… |
| q27-ar / q27-en | 0.50 / 0.67 | 1.00 / 1.00 | الرهن الرسمي يمنح الدائن حقاً عيناً… / An official mortgage gives the creditor a real right… |
| q04-ar, q08-ar, q17-ar, q17-en, q26-ar | within ±0.2 of each other | | |

**Reading it:** the 7B judge is noisy per question, both too harsh (q19-ar: 0 for a correct answer) and too lenient,
but its mean lands close to the larger judge's on this set, so it works as a gate on the average. Do not read single
questions' scores from it. Faithfulness also measures grounding, not legal correctness: a separate check of two
Qwen2.5 answers found one wrong conclusion (a spouse's gift "can" be revoked; Article 502(d) says it cannot) and one
answer that drifted into Chinese mid-sentence, both judged faithful by the 7B judge.

Reproduce: `python -m rag.evaluation.gate --judge-backend bedrock --no-mlflow --report /tmp/gate_bedrock.md`.
