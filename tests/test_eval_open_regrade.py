# Tests for eval_open/regrade.py
# This script re-grades an existing answers CSV with a (possibly different)
# judge, under both rubrics, writing a resumable scores CSV and a final
# Wilcoxon summary. The output path is a hardcoded relative string built
# from --tag, so these tests chdir into a tmp_path with a matching layout.

import pandas as pd
import pytest
from scipy.stats import wilcoxon

import eval_open.regrade as regrade


def _write_answers(tmp_path, rows):
    path = tmp_path / "answers.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_main_grades_missing_pairs_and_resumes_done_ones(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "open").mkdir(parents=True)
    answers = _write_answers(tmp_path, {
        "index": [1, 2], "question": ["Q1", "Q2"], "gt": ["G1", "G2"], "answer": ["A1", "A2"],
    })

    out = tmp_path / "results" / "open" / "regrade_mytag_scores.csv"
    # index=1/original is already done; everything else should still run
    pd.DataFrame([{"index": 1, "rubric": "original", "score": 0.7, "judge_raw": "old"}]).to_csv(out, index=False)

    monkeypatch.setattr(regrade, "build_grading_prompt", lambda q, gt, ans, rubric: f"{q}|{gt}|{ans}|{rubric}")

    grade_calls = []

    def fake_grade(prompt, judge=None, model=None, delay=None):
        grade_calls.append((prompt, judge, model, delay))
        return 0.5, "raw-reply"

    monkeypatch.setattr(regrade, "grade", fake_grade)

    monkeypatch.setattr(
        "sys.argv",
        ["regrade.py", "--answers", str(answers), "--tag", "mytag",
         "--judge", "claude", "--delay", "0"],
    )

    regrade.main()

    # 4 total pairs (2 answers x 2 rubrics) minus the 1 already-done pair = 3 calls
    assert len(grade_calls) == 3
    prompts_graded = {c[0] for c in grade_calls}
    assert prompts_graded == {"Q1|G1|A1|rephrased", "Q2|G2|A2|original", "Q2|G2|A2|rephrased"}
    for _, judge, model, delay in grade_calls:
        assert judge == "claude" and model is None and delay == 0.0

    final = pd.read_csv(out)
    assert len(final) == 4
    assert set(zip(final["index"], final["rubric"])) == {
        (1, "original"), (1, "rephrased"), (2, "original"), (2, "rephrased"),
    }
    # the resumed row keeps its original score, untouched
    assert final.loc[(final["index"] == 1) & (final["rubric"] == "original"), "score"].iloc[0] == 0.7

    printed = capsys.readouterr().out
    assert "re-grading 2 answers x 2 rubrics with judge=claude model=default" in printed


def test_main_raises_when_results_open_directory_is_missing(tmp_path, monkeypatch):
    # CURRENT BEHAVIOR (looks like a bug): main() never creates results/open/
    # before writing the output CSV, so if the directory doesn't already
    # exist the very first write blows up instead of being created.
    monkeypatch.chdir(tmp_path)
    answers = _write_answers(tmp_path, {
        "index": [1], "question": ["Q1"], "gt": ["G1"], "answer": ["A1"],
    })
    monkeypatch.setattr(regrade, "build_grading_prompt", lambda q, gt, ans, rubric: "prompt")
    monkeypatch.setattr(regrade, "grade", lambda prompt, judge=None, model=None, delay=None: (0.5, "raw"))
    monkeypatch.setattr("sys.argv", ["regrade.py", "--answers", str(answers), "--tag", "notmade"])

    with pytest.raises(OSError):
        regrade.main()


def test_main_prints_wilcoxon_summary_matching_real_scipy_call(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "open").mkdir(parents=True)
    answers = _write_answers(tmp_path, {
        "index": [1, 2], "question": ["Q1", "Q2"], "gt": ["G1", "G2"], "answer": ["A1", "A2"],
    })
    monkeypatch.setattr(regrade, "build_grading_prompt", lambda q, gt, ans, rubric: f"{q}-{rubric}")

    # deterministic scores: index1 original=0.2/rephrased=0.8, index2 original=0.4/rephrased=0.4
    score_map = {"Q1-original": 0.2, "Q1-rephrased": 0.8, "Q2-original": 0.4, "Q2-rephrased": 0.4}
    monkeypatch.setattr(regrade, "grade", lambda prompt, judge=None, model=None, delay=None: (score_map[prompt], "raw"))
    monkeypatch.setattr("sys.argv", ["regrade.py", "--answers", str(answers), "--tag", "sum", "--judge", "gemini"])

    regrade.main()

    # recompute the expected summary the same way regrade.py does, from the
    # same real scipy call, instead of hand-guessing a p-value
    d = pd.Series([0.8 - 0.2, 0.4 - 0.4])
    expected_p = wilcoxon(d, zero_method="zsplit").pvalue if d.abs().sum() else 1.0

    printed = capsys.readouterr().out
    assert "judge=gemini(default)" in printed
    assert f"n={2}" in printed
    assert "original :  30.0%" in printed
    assert "rephrased:  60.0%" in printed
    assert f"delta    : +30.0 pts   Wilcoxon p={expected_p:.3g}" in printed


def test_main_falls_back_to_p_1_when_wilcoxon_raises_valueerror(tmp_path, monkeypatch, capsys):
    # the try/except around wilcoxon() is defensive code that real inputs
    # rarely trigger, so force it directly to pin down the fallback branch
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "open").mkdir(parents=True)
    answers = _write_answers(tmp_path, {
        "index": [1, 2], "question": ["Q1", "Q2"], "gt": ["G1", "G2"], "answer": ["A1", "A2"],
    })
    monkeypatch.setattr(regrade, "build_grading_prompt", lambda q, gt, ans, rubric: f"{q}-{rubric}")
    score_map = {"Q1-original": 0.2, "Q1-rephrased": 0.8, "Q2-original": 0.4, "Q2-rephrased": 0.4}
    monkeypatch.setattr(regrade, "grade", lambda prompt, judge=None, model=None, delay=None: (score_map[prompt], "raw"))

    def raising_wilcoxon(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(regrade, "wilcoxon", raising_wilcoxon)
    monkeypatch.setattr("sys.argv", ["regrade.py", "--answers", str(answers), "--tag", "err"])

    regrade.main()

    printed = capsys.readouterr().out
    assert "Wilcoxon p=1" in printed
