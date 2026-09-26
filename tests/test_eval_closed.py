# Tests for eval_closed.py
# The original (oldest, simplest) closed-ended MCQ harness for Claude: no
# resume-path options, no few-shot, no meta sidecar — just "run every question,
# write results/closed_results.csv, resume if it already exists."

import pandas as pd
import pytest

import eval_closed
from clients.errors import APICallFailed


def make_df(n):
    return pd.DataFrame({
        "index": list(range(n)),
        "file_name": [f"img{i}.jpg" for i in range(n)],
        "category": ["Teeth" if i % 2 == 0 else "Patho" for i in range(n)],
        "question": [f"Q{i}?" for i in range(n)],
        "answer": ["A" if i % 2 == 0 else "B" for i in range(n)],
    })


def fake_build_prompt(row):
    return "system", [{"role": "user", "content": row["question"]}]


# ---------- print_summary ----------

def test_print_summary_prints_the_real_accuracy(capsys):
    df = pd.DataFrame({"category": ["Teeth", "Teeth", "Patho"], "correct": [True, False, True]})
    eval_closed.print_summary(df)
    out = capsys.readouterr().out
    expected_acc = df["correct"].mean() * 100
    assert f"{expected_acc:.2f}%" in out
    assert "Teeth" in out and "Patho" in out


def test_print_summary_lists_categories_best_first(capsys):
    df = pd.DataFrame({"category": ["Low", "Low", "High"], "correct": [False, False, True]})
    eval_closed.print_summary(df)
    out = capsys.readouterr().out
    assert out.index("High") < out.index("Low")


# ---------- run(): happy path ----------

def test_run_scores_every_row_and_writes_the_results_csv(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    df = make_df(3)
    monkeypatch.setattr(eval_closed, "load_closed", lambda: df)
    monkeypatch.setattr(eval_closed, "build_prompt", fake_build_prompt)
    # answering "A" for every row means indices 0 and 2 (answer "A") are correct
    # and index 1 (answer "B") is wrong, so the summary output is non-trivial
    monkeypatch.setattr(eval_closed, "call", lambda system, messages: ("A", "raw text"))
    monkeypatch.setattr(eval_closed, "looks_like_refusal", lambda raw: False)

    eval_closed.run()

    out_df = pd.read_csv(eval_closed.RESULTS_PATH)
    assert list(out_df["index"]) == [0, 1, 2]
    assert list(out_df["predicted"]) == ["A", "A", "A"]
    assert list(out_df["correct"]) == [True, False, True]
    assert "Overall accuracy" in capsys.readouterr().out


def test_run_respects_the_limit_argument(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_df(5)
    monkeypatch.setattr(eval_closed, "load_closed", lambda: df)
    monkeypatch.setattr(eval_closed, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(eval_closed, "call", lambda system, messages: ("A", "raw"))
    monkeypatch.setattr(eval_closed, "looks_like_refusal", lambda raw: False)

    eval_closed.run(limit=2)

    out_df = pd.read_csv(eval_closed.RESULTS_PATH)
    assert list(out_df["index"]) == [0, 1]


def test_run_marks_refusals_using_looks_like_refusal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_df(1)
    monkeypatch.setattr(eval_closed, "load_closed", lambda: df)
    monkeypatch.setattr(eval_closed, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(eval_closed, "call", lambda system, messages: (None, "I cannot help"))
    monkeypatch.setattr(eval_closed, "looks_like_refusal", lambda raw: True)

    eval_closed.run()

    out_df = pd.read_csv(eval_closed.RESULTS_PATH)
    assert bool(out_df["refused"].iloc[0]) is True
    assert bool(out_df["correct"].iloc[0]) is False  # predicted None != answer "A"


# ---------- run(): resuming ----------

def test_run_skips_rows_already_present_in_the_results_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_df(3)
    monkeypatch.setattr(eval_closed, "load_closed", lambda: df)
    monkeypatch.setattr(eval_closed, "build_prompt", fake_build_prompt)

    calls = []

    def fake_call(system, messages):
        calls.append(messages)
        return "A", "raw"

    monkeypatch.setattr(eval_closed, "call", fake_call)
    monkeypatch.setattr(eval_closed, "looks_like_refusal", lambda raw: False)

    import os
    os.makedirs("results", exist_ok=True)
    pd.DataFrame([{"index": 0, "file_name": "img0.jpg", "category": "Teeth", "question": "Q0?",
                  "answer": "A", "predicted": "A", "raw_response": "raw", "correct": True,
                  "refused": False}]).to_csv(eval_closed.RESULTS_PATH, index=False)

    eval_closed.run()

    assert len(calls) == 2  # only indices 1 and 2 were called
    out_df = pd.read_csv(eval_closed.RESULTS_PATH)
    assert sorted(out_df["index"]) == [0, 1, 2]


# ---------- run(): API failures are skipped, not recorded ----------

def test_run_skips_an_item_whose_call_raises_api_call_failed(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    df = make_df(2)
    monkeypatch.setattr(eval_closed, "load_closed", lambda: df)
    monkeypatch.setattr(eval_closed, "build_prompt", fake_build_prompt)

    def fake_call(system, messages):
        if "Q0?" in str(messages):
            raise APICallFailed("boom")
        return "A", "raw"

    monkeypatch.setattr(eval_closed, "call", fake_call)
    monkeypatch.setattr(eval_closed, "looks_like_refusal", lambda raw: False)

    eval_closed.run()

    out_df = pd.read_csv(eval_closed.RESULTS_PATH)
    # the failed item (index 0) is left out entirely, not recorded with an error string
    assert list(out_df["index"]) == [1]
    assert "[SKIP index 0]" in capsys.readouterr().out


# ---------- run(): writes a partial CSV every 10 results ----------

def test_run_writes_a_partial_csv_after_every_ten_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_df(11)
    monkeypatch.setattr(eval_closed, "load_closed", lambda: df)
    monkeypatch.setattr(eval_closed, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(eval_closed, "looks_like_refusal", lambda raw: False)

    def fake_call(system, messages):
        if "Q10?" in str(messages):
            raise RuntimeError("simulated crash on the 11th item")
        return "A", "raw"

    monkeypatch.setattr(eval_closed, "call", fake_call)

    with pytest.raises(RuntimeError, match="simulated crash"):
        eval_closed.run()

    # the crash happened AFTER the 10-results checkpoint write, so the partial
    # file on disk already has the first 10 rows even though run() never returned
    out_df = pd.read_csv(eval_closed.RESULTS_PATH)
    assert len(out_df) == 10
