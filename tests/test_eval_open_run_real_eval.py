# Tests for eval_open/run_real_eval.py
# E-open-1 (REAL, faithful prompt): a model answers a fixed sample of items
# with the verbatim inference prompt, then a judge grades each real answer
# under both rubrics. Two resumable phases (answers, scores) plus a summary.

import base64
from io import BytesIO

import pandas as pd
import pytest
from PIL import Image

import eval_open.run_real_eval as run_real_eval


def make_b64_image(w=5, h=5):
    img = Image.new("RGB", (w, h), (10, 20, 30))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------- get_item_indices ----------

def test_get_item_indices_filters_by_ref_type_and_clamps_n(tmp_path, monkeypatch):
    preds = pd.DataFrame({
        "index": [1, 2, 3, 4, 5],
        "ref_type": ["coord_ref", "coord_ref", "coord_ref", "prose_ref", "prose_ref"],
    })
    path = tmp_path / "predictions.parquet"
    preds.to_parquet(path)
    monkeypatch.setattr(run_real_eval, "PREDS", str(path))

    idx = run_real_eval.get_item_indices("coord_ref", 100, seed=0)
    assert idx == [1, 2, 3]  # n larger than the pool clamps to the whole pool

    idx_all = run_real_eval.get_item_indices("all", 2, seed=0)
    expected = sorted(preds.sample(2, random_state=0)["index"].tolist())
    assert idx_all == expected


def test_get_item_indices_result_is_sorted(tmp_path, monkeypatch):
    preds = pd.DataFrame({"index": [5, 1, 3], "ref_type": ["coord_ref"] * 3})
    path = tmp_path / "predictions.parquet"
    preds.to_parquet(path)
    monkeypatch.setattr(run_real_eval, "PREDS", str(path))

    idx = run_real_eval.get_item_indices("coord_ref", 3, seed=0)
    assert idx == sorted(idx)


# ---------- phase1_answers ----------

def test_phase1_answers_calls_answerer_and_resumes_done_indices(tmp_path, monkeypatch):
    ans_out = tmp_path / "answers.csv"
    monkeypatch.setattr(run_real_eval, "ANS_OUT", str(ans_out))
    pd.DataFrame([{"index": 1, "question": "Q1", "gt": "G1", "answer": "OLD"}]).to_csv(ans_out, index=False)

    b64 = make_b64_image()
    items = pd.DataFrame([
        {"index": 1, "image": b64, "question": "Q1", "answer": "G1"},
        {"index": 2, "image": b64, "question": "Q2", "answer": "G2"},
    ])

    calls = []

    def fake_answer_question(img, question, model=None):
        calls.append((question, model))
        return f"ANSWER-{question}"

    monkeypatch.setattr(run_real_eval, "answer_question", fake_answer_question)

    result = run_real_eval.phase1_answers(items)

    assert calls == [("Q2", run_real_eval.ANSWER_MODEL)]  # index 1 was already done
    assert len(result) == 2
    written = pd.read_csv(ans_out)
    assert len(written) == 2
    assert written.loc[written["index"] == 2, "answer"].iloc[0] == "ANSWER-Q2"
    assert written.loc[written["index"] == 1, "answer"].iloc[0] == "OLD"  # untouched


def test_phase1_answers_writes_gt_as_a_string(tmp_path, monkeypatch):
    ans_out = tmp_path / "answers.csv"
    monkeypatch.setattr(run_real_eval, "ANS_OUT", str(ans_out))
    b64 = make_b64_image()
    items = pd.DataFrame([{"index": 1, "image": b64, "question": "Q1", "answer": 42}])  # non-string gt
    monkeypatch.setattr(run_real_eval, "answer_question", lambda img, q, model=None: "A")

    result = run_real_eval.phase1_answers(items)
    assert result.loc[0, "gt"] == "42"


# ---------- phase2_grade ----------

def test_phase2_grade_grades_each_rubric_and_resumes_done_pairs(tmp_path, monkeypatch):
    score_out = tmp_path / "scores.csv"
    monkeypatch.setattr(run_real_eval, "SCORE_OUT", str(score_out))
    pd.DataFrame([{"index": 1, "rubric": "original", "score": 0.9, "judge_raw": "old"}]).to_csv(score_out, index=False)

    answers = pd.DataFrame([
        {"index": 1, "question": "Q1", "gt": "G1", "answer": "A1"},
        {"index": 2, "question": "Q2", "gt": "G2", "answer": "A2"},
    ])

    calls = []

    def fake_grade(prompt, judge=None, model=None):
        calls.append((judge, model))
        return 0.5, "R"

    monkeypatch.setattr(run_real_eval, "grade", fake_grade)

    result = run_real_eval.phase2_grade(answers)

    # 2 answers x 2 rubrics = 4 pairs, minus 1 already-done = 3 calls
    assert len(calls) == 3
    assert all(j == run_real_eval.JUDGE[0] and m == run_real_eval.JUDGE[1] for j, m in calls)
    final = pd.read_csv(score_out)
    assert len(final) == 4
    assert final.loc[(final["index"] == 1) & (final["rubric"] == "original"), "score"].iloc[0] == 0.9


# ---------- report ----------

def test_report_reads_score_out_and_answerer_model_from_module_globals(monkeypatch, capsys):
    # report() reads ANS_OUT/SCORE_OUT/ANSWER_MODEL straight from module globals,
    # not from a parameter, so calling it standalone requires setting SCORE_OUT first
    monkeypatch.setattr(run_real_eval, "SCORE_OUT", "results/open/real_x_scores.csv")
    scores = pd.DataFrame([
        {"index": 1, "rubric": "original", "score": 0.2},
        {"index": 1, "rubric": "rephrased", "score": 0.8},
        {"index": 2, "rubric": "original", "score": 0.4},
        {"index": 2, "rubric": "rephrased", "score": 0.4},
    ])

    run_real_eval.report(scores)

    printed = capsys.readouterr().out
    assert "REAL 30-item result" in printed
    assert f"answerer = judge = {run_real_eval.ANSWER_MODEL};  n=2" in printed
    assert "original rubric :  30.0%" in printed
    assert "rephrased rubric:  60.0%" in printed
    assert "delta (new-orig): +30.0 pts" in printed
    assert "results/open/real_x_scores.csv" in printed


def test_report_falls_back_to_p_1_when_wilcoxon_raises_valueerror(monkeypatch, capsys):
    # report() does `from scipy.stats import wilcoxon` INSIDE the function body,
    # so (unlike a top-level import) patching the real scipy.stats module takes
    # effect here: it is re-imported fresh on every call.
    import scipy.stats

    monkeypatch.setattr(run_real_eval, "SCORE_OUT", "results/open/real_x_scores.csv")
    monkeypatch.setattr(scipy.stats, "wilcoxon", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    scores = pd.DataFrame([
        {"index": 1, "rubric": "original", "score": 0.2},
        {"index": 1, "rubric": "rephrased", "score": 0.8},
    ])

    run_real_eval.report(scores)

    printed = capsys.readouterr().out
    assert "Wilcoxon p=1" in printed


# ---------- main (end to end) ----------

def test_main_end_to_end_writes_answers_scores_and_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    preds = pd.DataFrame({"index": [1, 2], "ref_type": ["coord_ref", "coord_ref"]})
    preds_path = tmp_path / "predictions.parquet"
    preds.to_parquet(preds_path)
    monkeypatch.setattr(run_real_eval, "PREDS", str(preds_path))

    b64 = make_b64_image()
    data = pd.DataFrame([
        {"index": 1, "image": b64, "question": "Q1", "answer": "G1"},
        {"index": 2, "image": b64, "question": "Q2", "answer": "G2"},
    ])
    data_path = tmp_path / "open_ended.parquet"
    data.to_parquet(data_path)
    monkeypatch.setattr(run_real_eval, "DATA", str(data_path))

    monkeypatch.setattr(run_real_eval, "answer_question", lambda img, q, model=None: f"ANS-{q}")
    monkeypatch.setattr(run_real_eval, "grade", lambda prompt, judge=None, model=None: (0.6, "R"))

    monkeypatch.setattr(
        "sys.argv",
        ["run_real_eval.py", "--ref-type", "coord_ref", "--n", "2", "--tag", "mytag"],
    )

    run_real_eval.main()

    ans_out = tmp_path / "results" / "open" / "real_mytag_answers.csv"
    score_out = tmp_path / "results" / "open" / "real_mytag_scores.csv"
    assert ans_out.exists() and score_out.exists()

    answers = pd.read_csv(ans_out)
    assert len(answers) == 2
    assert set(answers["answer"]) == {"ANS-Q1", "ANS-Q2"}

    scores = pd.read_csv(score_out)
    assert len(scores) == 4  # 2 items x 2 rubrics
    assert (scores["score"] == 0.6).all()

    printed = capsys.readouterr().out
    assert "ref_type=coord_ref  n=2  seed=0  ->  results/open/real_mytag_answers.csv" in printed
    assert "REAL 30-item result" in printed
    assert "original rubric :  60.0%" in printed


def test_main_default_tag_combines_ref_type_and_n(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    preds = pd.DataFrame({"index": [1], "ref_type": ["prose_ref"]})
    preds_path = tmp_path / "predictions.parquet"
    preds.to_parquet(preds_path)
    monkeypatch.setattr(run_real_eval, "PREDS", str(preds_path))

    b64 = make_b64_image()
    data = pd.DataFrame([{"index": 1, "image": b64, "question": "Q1", "answer": "G1"}])
    data_path = tmp_path / "open_ended.parquet"
    data.to_parquet(data_path)
    monkeypatch.setattr(run_real_eval, "DATA", str(data_path))

    monkeypatch.setattr(run_real_eval, "answer_question", lambda img, q, model=None: "A")
    monkeypatch.setattr(run_real_eval, "grade", lambda prompt, judge=None, model=None: (1.0, "R"))
    monkeypatch.setattr("sys.argv", ["run_real_eval.py", "--ref-type", "prose_ref", "--n", "1"])

    run_real_eval.main()

    # no --tag given -> tag defaults to f"{ref_type}{n}"
    assert (tmp_path / "results" / "open" / "real_prose_ref1_answers.csv").exists()
    assert (tmp_path / "results" / "open" / "real_prose_ref1_scores.csv").exists()
