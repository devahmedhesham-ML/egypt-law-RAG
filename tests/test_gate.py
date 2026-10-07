"""Quality gate: question selection, pass/fail rule and report, with the model calls faked."""

from pathlib import Path

import pytest

from rag.evaluation import gate


def test_ci_questions_are_the_flagged_in_scope_ones():
    qs = gate.load_questions(Path(gate.REPO_ROOT) / "eval" / "questions.jsonl", "ci")
    assert len(qs) == 20 and all(q["ci"] and q["relevant_articles"] for q in qs)
    assert len(gate.load_questions(Path(gate.REPO_ROOT) / "eval" / "questions.jsonl", "all")) == 58


@pytest.mark.parametrize("scores,expected", [([0.9, 0.8, 0.7, 1.0], 0), ([0.5, 0.6, 0.9, 0.7], 1),
                                             ([1.0, float("nan"), float("nan"), 1.0], 1)])
def test_gate_passes_only_above_threshold_with_enough_judged(monkeypatch, tmp_path, scores, expected):
    qs = [{"id": f"q{i}", "lang": "en", "question": f"q{i}?", "relevant_articles": [1], "ci": True} for i in range(4)]
    monkeypatch.setattr(gate, "load_questions", lambda path, subset: qs)

    def fake_run(questions, **kw):
        rows = [{"id": q["id"], "lang": "en", "question": q["question"], "relevant": [1], "retrieved": [1, 2],
                 "answer": "ok [Article 1]", "faithfulness": s} for q, s in zip(questions, scores)]
        return {"rows": rows, "metrics": gate.summarize(rows), "answer_model": "qwen", "judge_model": "qwen",
                "elapsed_s": 1.0}

    monkeypatch.setattr(gate, "run_gate", fake_run)
    report = tmp_path / "gate.md"
    assert gate.main(["--report", str(report), "--no-mlflow"]) == expected
    assert ("PASSED" in report.read_text(encoding="utf-8")) == (expected == 0)
