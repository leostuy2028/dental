# Tests for eval_open/run_reproduce.py
# Reproduces the paper's GPT-4o open-ended result: GPT-4o answers each prose_ref
# item, then GPT-4o grades it under two rubrics (original, rephrased). main() takes
# --n/--seed only.

import os

import pandas as pd

import eval_open.run_reproduce as rr


# ---------- prose_ref_indices ----------

def test_prose_ref_indices_samples_deterministically(tmp_path, monkeypatch):
    preds_path = tmp_path / "preds.parquet"
    pd.DataFrame({"index": [1, 2, 3, 4, 5], "ref_type": ["prose_ref"] * 5}).to_parquet(preds_path)
    monkeypatch.setattr(rr, "PREDS", str(preds_path))

    assert rr.prose_ref_indices(3, 0) == [1, 2, 3]


def test_prose_ref_indices_ignores_non_prose_ref_rows(tmp_path, monkeypatch):
    preds_path = tmp_path / "preds.parquet"
    pd.DataFrame({"index": [1, 2, 3, 4], "ref_type": ["coord_ref", "prose_ref", "prose_ref", "coord_ref"]}
                ).to_parquet(preds_path)
    monkeypatch.setattr(rr, "PREDS", str(preds_path))

    assert rr.prose_ref_indices(2, 0) == [2, 3]


def test_prose_ref_indices_n_larger_than_pool_returns_the_whole_pool(tmp_path, monkeypatch):
    # unlike run_frontier.prose_ref_indices, this one takes n/seed as params and
    # clamps with min(n, len(p)), so it does NOT raise when n exceeds the pool
    preds_path = tmp_path / "preds.parquet"
    pd.DataFrame({"index": [1, 2, 3, 4, 5], "ref_type": ["prose_ref"] * 5}).to_parquet(preds_path)
    monkeypatch.setattr(rr, "PREDS", str(preds_path))

    assert rr.prose_ref_indices(100, 0) == [1, 2, 3, 4, 5]


# ---------- phase1_answers ----------

def test_phase1_answers_writes_expected_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("results/open")
    items = pd.DataFrame({"index": [1, 2], "image": ["b64_1", "b64_2"],
                          "question": ["Q1?", "Q2?"], "answer": ["GT1", "GT2"]})

    calls = []

    def fake_answer(image_b64, question, model=None):
        calls.append((image_b64, question, model))
        return "a real answer"

    monkeypatch.setattr(rr, "answer_question_openai", fake_answer)
    ans_out = "results/open/answers.csv"

    out = rr.phase1_answers(items, ans_out)

    assert list(out.columns) == ["index", "question", "gt", "answer"]
    assert (out["answer"] == "a real answer").all()
    assert calls == [("b64_1", "Q1?", "gpt-4o"), ("b64_2", "Q2?", "gpt-4o")]
    assert os.path.exists(ans_out)


def test_phase1_answers_resumes_and_skips_done_indices(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("results/open")
    items = pd.DataFrame({"index": [1, 2], "image": ["b64_1", "b64_2"],
                          "question": ["Q1?", "Q2?"], "answer": ["GT1", "GT2"]})
    ans_out = "results/open/answers.csv"
    pd.DataFrame({"index": [1], "question": ["Q1?"], "gt": ["GT1"], "answer": ["done"]}).to_csv(
        ans_out, index=False)

    calls = []

    def fake_answer(image_b64, question, model=None):
        calls.append(1)
        return "fresh"

    monkeypatch.setattr(rr, "answer_question_openai", fake_answer)
    out = rr.phase1_answers(items, ans_out)

    assert len(calls) == 1
    assert sorted(out["answer"]) == ["done", "fresh"]


# ---------- phase2_grade ----------

def test_phase2_grade_grades_both_rubrics_for_every_answer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("results/open")
    answers = pd.DataFrame({"index": [1, 2], "question": ["Q1?", "Q2?"],
                            "gt": ["GT1", "GT2"], "answer": ["A1", "A2"]})

    monkeypatch.setattr(rr, "grade", lambda prompt, judge=None, model=None, delay=None: (0.5, "raw"))
    score_out = "results/open/scores.csv"

    out = rr.phase2_grade(answers, score_out)

    assert list(out.columns) == ["index", "rubric", "score", "judge_raw"]
    assert sorted(out["rubric"]) == ["original", "original", "rephrased", "rephrased"]
    assert len(out) == 4  # 2 answers x 2 rubrics
    assert os.path.exists(score_out)


def test_phase2_grade_resumes_and_skips_index_rubric_pairs_already_done(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("results/open")
    answers = pd.DataFrame({"index": [1, 2], "question": ["Q1?", "Q2?"],
                            "gt": ["GT1", "GT2"], "answer": ["A1", "A2"]})
    score_out = "results/open/scores.csv"
    pd.DataFrame({"index": [1], "rubric": ["original"], "score": [1.0], "judge_raw": ["x"]}).to_csv(
        score_out, index=False)

    calls = []

    def fake_grade(prompt, judge=None, model=None, delay=None):
        calls.append(1)
        return 0.3, "raw2"

    monkeypatch.setattr(rr, "grade", fake_grade)
    out = rr.phase2_grade(answers, score_out)

    # 4 total pairs minus the 1 already done = 3 fresh grade() calls
    assert len(calls) == 3
    assert len(out) == 4


# ---------- main() ----------

def test_main_writes_answers_and_scores_and_prints_the_pivoted_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "open_ended.parquet"
    pd.DataFrame({"index": [1, 2], "image": ["b64_1", "b64_2"],
                 "question": ["Q1?", "Q2?"], "answer": ["GT1", "GT2"]}).to_parquet(data_path)
    preds_path = tmp_path / "predictions.parquet"
    pd.DataFrame({"index": [1, 2], "ref_type": ["prose_ref", "prose_ref"]}).to_parquet(preds_path)

    monkeypatch.setattr(rr, "DATA", str(data_path))
    monkeypatch.setattr(rr, "PREDS", str(preds_path))
    monkeypatch.setattr(rr, "answer_question_openai", lambda *a, **kw: "a real answer")
    monkeypatch.setattr(rr, "grade", lambda prompt, judge=None, model=None, delay=None: (0.5, "raw"))

    monkeypatch.setattr("sys.argv", ["run_reproduce.py", "--n", "2", "--seed", "0"])
    rr.main()

    answers = pd.read_csv("results/open/reproduce_gpt4o_prose2_answers.csv")
    scores = pd.read_csv("results/open/reproduce_gpt4o_prose2_scores.csv")
    assert len(answers) == 2
    assert len(scores) == 4  # 2 items x 2 rubrics
    assert (scores["score"] == 0.5).all()

    out = capsys.readouterr().out
    assert "ORIGINAL rubric (paper protocol): 50.0%" in out
    assert "rephrased rubric               : 50.0%" in out
