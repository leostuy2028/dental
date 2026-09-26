# Tests for eval_open/finish_inject.py
# This one-shot script re-runs a handful of leftover images through the
# "detector chart injection" experiment, grades them, and merges the result
# into an existing results CSV. Paths are hardcoded relative strings, so
# these tests chdir into a tmp_path with the same folder layout.

import json

import pandas as pd
import pytest

import eval_open.finish_inject as finish_inject


def _setup_files(tmp_path, op_rows, tmap):
    (tmp_path / "data").mkdir()
    pd.DataFrame(op_rows).to_parquet(tmp_path / "data" / "open_ended.parquet")
    (tmp_path / "reference").mkdir()
    (tmp_path / "reference" / "mmoral_map.json").write_text(json.dumps(tmap), encoding="utf-8")
    (tmp_path / "results" / "open").mkdir(parents=True)


def _write_old_csv(tmp_path, rows):
    path = tmp_path / "results" / "open" / "old_scores.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _fake_select_targets(index_dtype_score):
    def fake(op, wrong_only=False):
        assert wrong_only is False  # finish_inject.py always calls it this way
        return pd.DataFrame(index_dtype_score)
    return fake


def test_main_runs_to_completion_and_merges_new_rows(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    op_rows = {
        "image_name": ["im1.jpg", "im1.jpg", "im1.jpg"],
        "index": [10, 11, 12],
        "question": ["Q1", "Q2", "Q3"],
        "answer": ["GT1", "GT2", "GT3"],
        "image": ["raw1", "raw1", "raw1"],
    }
    _setup_files(tmp_path, op_rows, {"im1.jpg": {"count": 3, "teeth": []}})
    old_csv = _write_old_csv(tmp_path, {
        "image": ["oldimg.jpg"], "idx": [1], "dtype": ["count"], "base": [0.5], "B": [0.7],
    })

    monkeypatch.setattr(finish_inject, "CSV", str(old_csv))
    monkeypatch.setattr(finish_inject, "REMAIN", ["im1.jpg"])

    # only index 10 and 11 are "targets"; index 12 must be skipped
    monkeypatch.setattr(finish_inject.T, "select_targets", _fake_select_targets(
        {"index": [10, 11], "dtype": ["count", "missing"], "score": [0.3, 0.4]}))
    monkeypatch.setattr(finish_inject.T, "strip_b64", lambda s: "STRIPPED:" + s)
    monkeypatch.setattr(finish_inject.T, "build_chart", lambda entry: "CHART")

    def fake_run_image(provider, model, b64, qs, chart):
        assert provider == "gemini" and model == "gemini-3.5-flash"
        assert b64 == "STRIPPED:raw1"
        assert chart == "CHART"
        return ["A10", "A11", "A12"]

    monkeypatch.setattr(finish_inject.T, "run_image", fake_run_image)
    monkeypatch.setattr(finish_inject, "build_grading_prompt",
                        lambda q, gt, pred, rubric: f"{q}|{gt}|{pred}|{rubric}")

    grade_calls = []

    def fake_grade(prompt):
        grade_calls.append(prompt)
        return {"Q1|GT1|A10|original": 0.9, "Q2|GT2|A11|original": 0.4}[prompt], "raw"

    monkeypatch.setattr(finish_inject, "grade", fake_grade)

    finish_inject.main()

    # index 12 was not a target, so only 2 grade calls happened
    assert grade_calls == ["Q1|GT1|A10|original", "Q2|GT2|A11|original"]

    full = pd.read_csv(old_csv)
    assert len(full) == 3  # 1 old + 2 new
    new_rows = full[full.image == "im1.jpg"].sort_values("idx")
    assert new_rows["idx"].tolist() == [10, 11]
    assert new_rows["dtype"].tolist() == ["count", "missing"]
    assert new_rows["base"].tolist() == [0.3, 0.4]
    assert new_rows["B"].tolist() == [0.9, 0.4]

    printed = capsys.readouterr().out
    assert "=== 3 of 148 questions (2 new this run) ===" in printed
    assert "STOPPED" not in printed


def test_main_exception_mid_loop_still_writes_completed_rows(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    op_rows = {
        "image_name": ["im1.jpg", "im1.jpg", "im2.jpg", "im2.jpg"],
        "index": [10, 11, 20, 21],
        "question": ["Q1", "Q2", "Q3", "Q4"],
        "answer": ["GT1", "GT2", "GT3", "GT4"],
        "image": ["raw1", "raw1", "raw2", "raw2"],
    }
    _setup_files(tmp_path, op_rows, {"im1.jpg": {}, "im2.jpg": {}})
    old_csv = _write_old_csv(tmp_path, {
        "image": ["oldimg.jpg"], "idx": [1], "dtype": ["count"], "base": [0.5], "B": [0.7],
    })

    monkeypatch.setattr(finish_inject, "CSV", str(old_csv))
    monkeypatch.setattr(finish_inject, "REMAIN", ["im1.jpg", "im2.jpg"])

    monkeypatch.setattr(finish_inject.T, "select_targets", _fake_select_targets(
        {"index": [10, 11, 20, 21], "dtype": ["count", "missing", "count", "missing"],
         "score": [0.3, 0.4, 0.1, 0.2]}))
    monkeypatch.setattr(finish_inject.T, "strip_b64", lambda s: s)
    monkeypatch.setattr(finish_inject.T, "build_chart", lambda entry: "CHART")
    monkeypatch.setattr(finish_inject, "build_grading_prompt",
                        lambda q, gt, pred, rubric: f"{q}::{pred}")
    monkeypatch.setattr(finish_inject, "grade", lambda prompt: (0.9, "raw"))

    calls = {"n": 0}

    def fake_run_image(provider, model, b64, qs, chart):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("429 rate limited")
        return ["A10", "A11"]

    monkeypatch.setattr(finish_inject.T, "run_image", fake_run_image)

    finish_inject.main()

    full = pd.read_csv(old_csv)
    # 1 old row + 2 new rows from im1.jpg only; im2.jpg's exception happened
    # before any of its rows were appended
    assert len(full) == 3
    assert set(full["image"]) == {"oldimg.jpg", "im1.jpg"}

    printed = capsys.readouterr().out
    assert "STOPPED (Gemini still unavailable?): 429 rate limited" in printed
    assert "=== 3 of 148 questions (2 new this run) ===" in printed


def test_main_no_new_rows_when_first_image_raises(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    op_rows = {
        "image_name": ["im1.jpg"], "index": [10], "question": ["Q1"],
        "answer": ["GT1"], "image": ["raw1"],
    }
    _setup_files(tmp_path, op_rows, {"im1.jpg": {}})
    old_csv = _write_old_csv(tmp_path, {
        "image": ["oldimg.jpg"], "idx": [1], "dtype": ["count"], "base": [0.5], "B": [0.7],
    })

    monkeypatch.setattr(finish_inject, "CSV", str(old_csv))
    monkeypatch.setattr(finish_inject, "REMAIN", ["im1.jpg"])
    monkeypatch.setattr(finish_inject.T, "select_targets", _fake_select_targets(
        {"index": [10], "dtype": ["count"], "score": [0.3]}))
    monkeypatch.setattr(finish_inject.T, "strip_b64", lambda s: s)
    monkeypatch.setattr(finish_inject.T, "build_chart", lambda entry: "CHART")

    def raising_run_image(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(finish_inject.T, "run_image", raising_run_image)

    finish_inject.main()

    # `full` falls back to exactly `old` (no concat) when `new` is empty
    full = pd.read_csv(old_csv)
    assert len(full) == 1
    assert full["image"].tolist() == ["oldimg.jpg"]

    printed = capsys.readouterr().out
    assert "STOPPED (Gemini still unavailable?): boom" in printed
    assert "(0 new this run)" in printed
