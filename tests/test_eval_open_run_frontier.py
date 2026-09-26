# Tests for eval_open/run_frontier.py
# Compares two Gemini "thinking" configs (think_off/think_on) head-to-head on the
# same prose_ref items, judged by GPT-4o. main() takes no CLI args; it loops over
# the module-level CONFIGS dict, answering then grading each config.

import base64
import os
from io import BytesIO

import pandas as pd
from PIL import Image

import eval_open.run_frontier as rf


def _tiny_jpeg_b64():
    img = Image.new("RGB", (20, 10))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------- prose_ref_indices ----------

def test_prose_ref_indices_samples_deterministically(tmp_path, monkeypatch):
    preds_path = tmp_path / "preds.parquet"
    pd.DataFrame({"index": [1, 2, 3, 4, 5], "ref_type": ["prose_ref"] * 5}).to_parquet(preds_path)
    monkeypatch.setattr(rf, "PREDS", str(preds_path))
    monkeypatch.setattr(rf, "N", 3)
    monkeypatch.setattr(rf, "SEED", 0)

    assert rf.prose_ref_indices() == [1, 2, 3]


def test_prose_ref_indices_ignores_non_prose_ref_rows(tmp_path, monkeypatch):
    preds_path = tmp_path / "preds.parquet"
    pd.DataFrame({"index": [1, 2, 3, 4], "ref_type": ["coord_ref", "prose_ref", "prose_ref", "coord_ref"]}
                ).to_parquet(preds_path)
    monkeypatch.setattr(rf, "PREDS", str(preds_path))
    monkeypatch.setattr(rf, "N", 2)
    monkeypatch.setattr(rf, "SEED", 0)

    assert rf.prose_ref_indices() == [2, 3]


def test_prose_ref_indices_raises_when_n_exceeds_the_pool(tmp_path, monkeypatch):
    # CURRENT BEHAVIOR (looks like a bug): unlike run_coord_arms.select_items and
    # run_reproduce.prose_ref_indices, this one does NOT clamp N to len(pool), so a
    # pool smaller than the fixed module constant N (100) raises instead of coping
    preds_path = tmp_path / "preds.parquet"
    pd.DataFrame({"index": [1, 2, 3], "ref_type": ["prose_ref"] * 3}).to_parquet(preds_path)
    monkeypatch.setattr(rf, "PREDS", str(preds_path))
    monkeypatch.setattr(rf, "N", 100)
    monkeypatch.setattr(rf, "SEED", 0)

    try:
        rf.prose_ref_indices()
        assert False, "expected a ValueError"
    except ValueError:
        pass


# ---------- is_complete ----------

def test_is_complete_true_for_each_terminal_punctuation_mark():
    for ch in ['.', '!', '?', ')', '"', '*']:
        assert rf.is_complete(f"some text{ch}") is True


def test_is_complete_false_for_other_endings():
    assert rf.is_complete("some text ending mid-word") is False


def test_is_complete_false_for_empty_string():
    assert rf.is_complete("") is False


def test_is_complete_false_for_none():
    assert rf.is_complete(None) is False


def test_is_complete_whitespace_only_string_is_true():
    # CURRENT BEHAVIOR (looks like a bug): "   " is truthy, so it takes the
    # a.rstrip()[-1:] path; rstrip() empties it, and "" is considered "in" the
    # punctuation string (empty substring check), so a blank answer counts complete
    assert rf.is_complete("   ") is True


# ---------- phase1_answers ----------

def test_phase1_answers_writes_expected_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("results/open")
    img_b64 = _tiny_jpeg_b64()
    items = pd.DataFrame({"index": [1, 2], "image": [img_b64, img_b64],
                          "question": ["Q1?", "Q2?"], "answer": ["GT1", "GT2"]})

    calls = []

    def fake_answer_question(img, question, model=None, thinking_budget=None, max_output_tokens=None):
        calls.append((question, thinking_budget, max_output_tokens))
        return "A complete answer."

    monkeypatch.setattr(rf, "answer_question", fake_answer_question)

    out = rf.phase1_answers(items, "think_off", rf.CONFIGS["think_off"])

    assert list(out.columns) == ["index", "question", "gt", "answer", "complete"]
    assert (out["complete"] == True).all()
    assert len(calls) == 2
    assert all(tb == 0 and mo == 12288 for _, tb, mo in calls)
    assert os.path.exists("results/open/frontier_think_off_answers.csv")


def test_phase1_answers_resumes_and_skips_done_indices(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("results/open")
    img_b64 = _tiny_jpeg_b64()
    items = pd.DataFrame({"index": [1, 2], "image": [img_b64, img_b64],
                          "question": ["Q1?", "Q2?"], "answer": ["GT1", "GT2"]})
    pd.DataFrame({"index": [1], "question": ["Q1?"], "gt": ["GT1"],
                 "answer": ["done"], "complete": [True]}).to_csv(
        "results/open/frontier_think_off_answers.csv", index=False)

    calls = []

    def fake_answer_question(img, question, model=None, thinking_budget=None, max_output_tokens=None):
        calls.append(1)
        return "ok."

    monkeypatch.setattr(rf, "answer_question", fake_answer_question)
    out = rf.phase1_answers(items, "think_off", rf.CONFIGS["think_off"])

    assert len(calls) == 1
    assert sorted(out["answer"]) == ["done", "ok."]


# ---------- phase2_grade ----------

def test_phase2_grade_writes_a_score_per_answer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("results/open")
    answers = pd.DataFrame({"index": [1, 2], "question": ["Q1?", "Q2?"],
                            "gt": ["GT1", "GT2"], "answer": ["A1", "A2"]})

    monkeypatch.setattr(rf, "grade", lambda prompt, judge=None, model=None, delay=None: (0.5, "raw"))
    out = rf.phase2_grade(answers, "think_off")

    assert list(out.columns) == ["index", "score", "judge_raw"]
    assert (out["score"] == 0.5).all()
    assert os.path.exists("results/open/frontier_think_off_scores.csv")


def test_phase2_grade_resumes_and_skips_done_indices(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("results/open")
    answers = pd.DataFrame({"index": [1, 2], "question": ["Q1?", "Q2?"],
                            "gt": ["GT1", "GT2"], "answer": ["A1", "A2"]})
    pd.DataFrame({"index": [1], "score": [1.0], "judge_raw": ["x"]}).to_csv(
        "results/open/frontier_think_off_scores.csv", index=False)

    calls = []

    def fake_grade(prompt, judge=None, model=None, delay=None):
        calls.append(1)
        return 0.2, "raw2"

    monkeypatch.setattr(rf, "grade", fake_grade)
    out = rf.phase2_grade(answers, "think_off")

    assert len(calls) == 1
    assert sorted(out["score"].tolist()) == [0.2, 1.0]


# ---------- main() ----------

def test_main_runs_both_configs_and_writes_all_four_csvs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    img_b64 = _tiny_jpeg_b64()

    data_path = tmp_path / "open_ended.parquet"
    pd.DataFrame({"index": [1, 2], "image": [img_b64, img_b64],
                 "question": ["Q1?", "Q2?"], "answer": ["GT1", "GT2"]}).to_parquet(data_path)
    preds_path = tmp_path / "predictions.parquet"
    pd.DataFrame({"index": [1, 2], "ref_type": ["prose_ref", "prose_ref"]}).to_parquet(preds_path)

    monkeypatch.setattr(rf, "DATA", str(data_path))
    monkeypatch.setattr(rf, "PREDS", str(preds_path))
    monkeypatch.setattr(rf, "N", 2)
    monkeypatch.setattr(rf, "SEED", 0)

    calls = []

    def fake_answer_question(img, question, model=None, thinking_budget=None, max_output_tokens=None):
        calls.append((thinking_budget, max_output_tokens))
        return "A complete answer."

    monkeypatch.setattr(rf, "answer_question", fake_answer_question)
    monkeypatch.setattr(rf, "grade", lambda prompt, judge=None, model=None, delay=None: (0.5, "raw"))

    rf.main()

    files = sorted(os.listdir("results/open"))
    assert files == [
        "frontier_think_off_answers.csv", "frontier_think_off_scores.csv",
        "frontier_think_on_answers.csv", "frontier_think_on_scores.csv",
    ]
    # both CONFIGS entries were used, each with its own thinking_budget/max_output_tokens
    assert set(calls) == {(0, 12288), (-1, 12288)}
    out = capsys.readouterr().out
    assert "GEMINI-3.5 OPEN-ENDED FRONTIER" in out
