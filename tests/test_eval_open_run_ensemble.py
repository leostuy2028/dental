# Tests for eval_open/run_ensemble.py
# Deliberation probe: gemini looks again at a concrete question with two draft
# answers (its own + gpt-5-mini's, order swapped per item) and writes a final
# answer, which is then graded by GPT-4o. This file only characterizes the
# CSV-in/CSV-out plumbing (answer -> grade_parallel -> regrade_sequential) --
# every model call is faked.
import random

import pandas as pd
import pytest

import eval_open.run_batched as rb
import eval_open.run_ensemble as run_ensemble


# ---------- build_user ----------

def test_build_user_exact_text():
    got = run_ensemble.build_user("How many teeth?", "30 teeth", "29 teeth")
    assert got == (
        "Question: How many teeth?\n\n"
        "Two draft answers were proposed by different readers.\n\n"
        "Answer A:\n30 teeth\n\n"
        "Answer B:\n29 teeth\n\n"
        "Examine the panoramic X-ray yourself and determine the most accurate answer to the question. "
        "You may agree with A, agree with B, or write a corrected answer if both are wrong or incomplete. "
        "Reply with ONLY your single best final answer to the question."
    )


# ---------- answer() ----------

def _write_answer_fixtures(tmp_path):
    data = pd.DataFrame([
        {"index": 0, "question": "How many teeth are visible?", "image_name": "img1.jpg",
         "answer": "30", "image": "B64_IMG1"},
        {"index": 1, "question": "How many teeth are missing?", "image_name": "img2.jpg",
         "answer": "2", "image": "B64_IMG2"},
        {"index": 2, "question": "Caption the whole image.", "image_name": "img3.jpg",
         "answer": "a panoramic X-ray", "image": "B64_IMG3"},
    ])
    data_path = tmp_path / "open_ended.parquet"
    data.to_parquet(data_path)

    gem_path = tmp_path / "gem_answers.csv"
    pd.DataFrame([{"index": 0, "answer": "gem says 30"},
                  {"index": 1, "answer": "gem says 2"}]).to_csv(gem_path, index=False)

    gpt_path = tmp_path / "gpt_answers.csv"
    pd.DataFrame([{"index": 0, "answer": "gpt says 30"},
                  {"index": 1, "answer": "gpt says 2"}]).to_csv(gpt_path, index=False)

    return data_path, gem_path, gpt_path


def test_answer_only_processes_concrete_bucket_and_swaps_order(tmp_path, monkeypatch, capsys):
    data_path, gem_path, gpt_path = _write_answer_fixtures(tmp_path)
    ans_path = tmp_path / "ans.csv"
    monkeypatch.setattr(run_ensemble, "DATA", str(data_path))
    monkeypatch.setattr(run_ensemble, "GEM_ANSWERS", str(gem_path))
    monkeypatch.setattr(run_ensemble, "GPT_ANSWERS", str(gpt_path))
    monkeypatch.setattr(run_ensemble, "ANS", str(ans_path))

    def fake_gemini(image_b64, system, user, model):
        if image_b64 == "B64_IMG2":
            raise RuntimeError("gemini boom")
        return user   # echo the built prompt so the test can check a1/a2 order

    monkeypatch.setattr(rb, "_gemini", fake_gemini)

    run_ensemble.answer(workers=2)

    out = capsys.readouterr().out
    assert "concrete=2  done=0  todo=2" in out
    assert "[skip 1] gemini boom" in out   # index 1 raised inside work() and was skipped

    written = pd.read_csv(ans_path)
    assert sorted(written["index"].tolist()) == [0]   # only index 0 survives (1 skipped, 2 not concrete)

    row = written[written["index"] == 0].iloc[0]
    # random.Random(0).random() == 0.844... >= 0.5, so gem_is_A is False for index 0,
    # meaning a1=gpt answer, a2=gem answer (see the swap logic in work()).
    assert random.Random(0).random() >= 0.5
    assert bool(row["gem_is_A"]) is False
    expected_prompt = run_ensemble.build_user("How many teeth are visible?", "gpt says 30", "gem says 30")
    assert row["final"] == expected_prompt


def test_answer_resume_crashes_when_combining_old_and_new_rows(tmp_path, monkeypatch, capsys):
    # CURRENT BEHAVIOR (looks like a bug): when ANS already has rows, `done.values()`
    # holds pandas Series (from `pd.read_csv(...).iterrows()`), but freshly-answered
    # rows are plain dicts (from work()). `rec` ends up a list mixing Series and dicts,
    # and pd.DataFrame(rec) raises AttributeError -- so resuming with at least one new
    # item to answer crashes instead of writing the merged CSV.
    data_path, gem_path, gpt_path = _write_answer_fixtures(tmp_path)
    ans_path = tmp_path / "ans.csv"
    pd.DataFrame([{"index": 0, "image_name": "img1.jpg", "question": "How many teeth are visible?",
                   "gt": "30", "gem_is_A": False, "final": "already answered"}]).to_csv(ans_path, index=False)
    monkeypatch.setattr(run_ensemble, "DATA", str(data_path))
    monkeypatch.setattr(run_ensemble, "GEM_ANSWERS", str(gem_path))
    monkeypatch.setattr(run_ensemble, "GPT_ANSWERS", str(gpt_path))
    monkeypatch.setattr(run_ensemble, "ANS", str(ans_path))

    calls = []

    def fake_gemini(image_b64, system, user, model):
        calls.append(image_b64)
        return "final answer"

    monkeypatch.setattr(rb, "_gemini", fake_gemini)

    with pytest.raises(AttributeError, match="dtype"):
        run_ensemble.answer(workers=2)

    out = capsys.readouterr().out
    assert "concrete=2  done=1  todo=1" in out
    assert calls == ["B64_IMG2"]   # only the not-yet-done concrete row (index 1) is (re)requested


# ---------- grade_parallel() ----------

def test_grade_parallel_skips_already_scored_and_writes_mean(tmp_path, monkeypatch, capsys):
    ans_path = tmp_path / "ans.csv"
    sco_path = tmp_path / "sco.csv"
    pd.DataFrame([
        {"index": 0, "question": "Q0", "gt": "gt0", "final": "final0"},
        {"index": 1, "question": "Q1", "gt": "gt1", "final": "final1"},
    ]).to_csv(ans_path, index=False)
    pd.DataFrame([{"index": 0, "score": 0.9}]).to_csv(sco_path, index=False)
    monkeypatch.setattr(run_ensemble, "ANS", str(ans_path))
    monkeypatch.setattr(run_ensemble, "SCO", str(sco_path))

    graded_indices = []

    def fake_grade(prompt, judge=None):
        graded_indices.append(prompt)
        return 0.5, "raw"

    monkeypatch.setattr(run_ensemble, "grade", fake_grade)

    run_ensemble.grade_parallel(workers=2)

    assert len(graded_indices) == 1
    assert "Q1" in graded_indices[0]   # only index 1 (not already scored) was graded

    out_scores = pd.read_csv(sco_path).set_index("index")["score"].to_dict()
    assert out_scores == {0: 0.9, 1: 0.5}

    out = capsys.readouterr().out
    assert "ENSEMBLE concrete score: 70.0%" in out   # mean(0.9, 0.5) * 100


# ---------- regrade_sequential() ----------

def test_regrade_sequential_checkpoints_and_overwrites_sco(tmp_path, monkeypatch, capsys):
    ans_path = tmp_path / "ans.csv"
    sco_path = tmp_path / "sco.csv"
    pd.DataFrame([
        {"index": 0, "question": "Q0", "gt": "gt0", "final": "final0"},
        {"index": 1, "question": "Q1", "gt": "gt1", "final": "final1"},
        {"index": 2, "question": "Q2", "gt": "gt2", "final": "final2"},
    ]).to_csv(ans_path, index=False)
    pd.DataFrame([{"index": 0, "score": 1.0}]).to_csv(str(sco_path) + ".clean", index=False)
    monkeypatch.setattr(run_ensemble, "ANS", str(ans_path))
    monkeypatch.setattr(run_ensemble, "SCO", str(sco_path))

    sleep_calls = []
    monkeypatch.setattr(run_ensemble.time, "sleep", lambda s: sleep_calls.append(s))

    grade_calls = []

    def fake_grade(prompt, judge=None, delay=None):
        grade_calls.append((prompt, delay))
        return 0.4, "raw"

    monkeypatch.setattr(run_ensemble, "grade", fake_grade)

    run_ensemble.regrade_sequential(delay=1.2)

    assert len(grade_calls) == 2   # index 0 already done, so only 1 and 2 are (re)graded
    assert all(d == 1.2 for _, d in grade_calls)
    assert sleep_calls == [1.2, 1.2]

    clean = pd.read_csv(str(sco_path) + ".clean").set_index("index")["score"].to_dict()
    final = pd.read_csv(sco_path).set_index("index")["score"].to_dict()
    assert clean == {0: 1.0, 1: 0.4, 2: 0.4}
    assert final == clean   # SCO is overwritten with the same rows as the .clean checkpoint

    out = capsys.readouterr().out
    assert "regrading 2 of 3 (sequential)" in out
    assert "done: clean ensemble concrete score = 60.0%  (n=3)" in out   # mean(1.0, 0.4, 0.4) * 100


# ---------- main() ----------

def test_main_normal_path_calls_answer_then_grade(monkeypatch):
    calls = []
    monkeypatch.setattr(run_ensemble, "answer", lambda workers: calls.append(("answer", workers)))
    monkeypatch.setattr(run_ensemble, "grade_parallel", lambda workers: calls.append(("grade_parallel", workers)))
    monkeypatch.setattr(run_ensemble, "regrade_sequential", lambda delay: calls.append(("regrade", delay)))
    monkeypatch.setattr("sys.argv", ["run_ensemble"])

    run_ensemble.main()

    assert calls == [("answer", 8), ("grade_parallel", 8)]   # default --workers=8, no --regrade
    assert rb.THINKING_BUDGET == run_ensemble.THINKING


def test_main_regrade_path_only_calls_regrade_sequential(monkeypatch):
    calls = []
    monkeypatch.setattr(run_ensemble, "answer", lambda workers: calls.append(("answer", workers)))
    monkeypatch.setattr(run_ensemble, "grade_parallel", lambda workers: calls.append(("grade_parallel", workers)))
    monkeypatch.setattr(run_ensemble, "regrade_sequential", lambda delay: calls.append(("regrade", delay)))
    monkeypatch.setattr("sys.argv", ["run_ensemble", "--regrade", "--delay", "0.01"])

    run_ensemble.main()

    assert calls == [("regrade", 0.01)]
