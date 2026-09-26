# Tests for eval_open/run_inject_other.py
# Measures the detector-chart injection on the 139 "other-concrete" open-ended
# questions (the ones NOT already covered by count/missing/whichtooth/cond#N).
# main() uses hardcoded relative paths, so tests monkeypatch.chdir into a tmp_path
# built with the same folder layout, and fake every model/judge call.
import json

import pandas as pd
import pytest

import eval_open.run_inject_other as run_inject_other


def _build_fixture(tmp_path):
    """2 images, 6 questions total; only 2 of them land in the 'other' bucket
    this script targets (one tooth-mentioning, one not)."""
    (tmp_path / "data").mkdir()
    (tmp_path / "results" / "open").mkdir(parents=True)
    (tmp_path / "reference").mkdir()

    rows = [
        {"index": 0, "image_name": "imga.jpg", "question": "How many teeth are visible?",
         "answer": "30", "image": "B64_A"},
        {"index": 1, "image_name": "imga.jpg", "question": "Are any teeth missing or absent from the arch?",
         "answer": "none", "image": "B64_A"},
        {"index": 2, "image_name": "imga.jpg", "question": "Identify the areas of bone loss near the molar.",
         "answer": "distal to #36", "image": "B64_A"},
        {"index": 3, "image_name": "imgb.jpg", "question": "Which tooth has a filling?",
         "answer": "#14", "image": "B64_B"},
        {"index": 4, "image_name": "imgb.jpg", "question": "What is the condition of #26?",
         "answer": "caries", "image": "B64_B"},
        {"index": 5, "image_name": "imgb.jpg", "question": "Identify the areas of bone loss near the mandibular canal.",
         "answer": "left ramus", "image": "B64_B"},
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "data" / "open_ended.parquet")

    pd.DataFrame([{"index": i, "score": s} for i, s in
                  [(0, 0.9), (1, 0.9), (2, 0.3), (3, 0.9), (4, 0.9), (5, 0.5)]]).to_csv(
        tmp_path / "results" / "open" / "batched_gemini35_plain578_scores.csv", index=False)

    tmap = {
        "imga.jpg": {"count": 30, "teeth": [{"fdi": "36", "box_2d": [0, 0, 1, 1], "conf": 0.9}]},
        "imgb.jpg": {"count": 28, "teeth": [{"fdi": "14", "box_2d": [0, 0, 1, 1], "conf": 0.8}]},
    }
    with open(tmp_path / "reference" / "mmoral_map.json", "w", encoding="utf-8") as f:
        json.dump(tmap, f)


def _fake_grade_by_keyword(prompt, judge=None):
    if "molar" in prompt:
        return 0.2, "raw"
    if "mandibular canal" in prompt:
        return 0.6, "raw"
    raise AssertionError(f"unexpected grading prompt: {prompt[:80]}")


def test_main_clean_run_writes_csv_and_prints_summary(tmp_path, monkeypatch, capsys):
    _build_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    out_path = tmp_path / "results" / "open" / "detector_inject_other139.csv"
    monkeypatch.setattr(run_inject_other, "OUT", str(out_path))

    def fake_run_image(provider, model, b64, questions, chart):
        assert provider == "gemini" and model == "gemini-3.5-flash"
        return [f"ans{i}" for i in range(len(questions))]

    monkeypatch.setattr(run_inject_other.T, "run_image", fake_run_image)
    monkeypatch.setattr(run_inject_other, "grade", _fake_grade_by_keyword)

    run_inject_other.main()

    d = pd.read_csv(out_path)
    assert list(d.columns) == ["image", "index", "group", "base", "B"]
    got = d.set_index("index")[["image", "group", "base", "B"]].to_dict("index")
    assert got == {
        2: {"image": "imga.jpg", "group": "tooth", "base": 0.3, "B": 0.2},
        5: {"image": "imgb.jpg", "group": "nontooth", "base": 0.5, "B": 0.6},
    }

    out = capsys.readouterr().out
    assert "2 other-concrete questions across 2 images (tooth 1, non-tooth 1)" in out
    assert "=== 2 of 2 other-concrete graded ===" in out
    assert "baseline 40.0%  ->  B (+chart) 40.0%   NET +0.0 pts" in out
    assert "nontooth n=  1  50.0% -> 60.0%  (+10.0)" in out
    assert "tooth    n=  1  30.0% -> 20.0%  (-10.0)" in out


def test_main_stops_partway_but_keeps_partial_rows(tmp_path, monkeypatch, capsys):
    _build_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    out_path = tmp_path / "results" / "open" / "detector_inject_other139.csv"
    monkeypatch.setattr(run_inject_other, "OUT", str(out_path))

    def fake_run_image(provider, model, b64, questions, chart):
        if b64 == "B64_B":
            raise RuntimeError("429 rate limited")
        return [f"ans{i}" for i in range(len(questions))]

    monkeypatch.setattr(run_inject_other.T, "run_image", fake_run_image)
    monkeypatch.setattr(run_inject_other, "grade", _fake_grade_by_keyword)

    run_inject_other.main()   # must not raise -- the except Exception catches it

    d = pd.read_csv(out_path)
    assert len(d) == 1   # only imga (processed first) made it before the exception
    assert d.iloc[0]["index"] == 2

    out = capsys.readouterr().out
    assert "STOPPED (Gemini cap?): 429 rate limited" in out
    assert "=== 1 of 2 other-concrete graded ===" in out
    assert "baseline 30.0%  ->  B (+chart) 20.0%   NET -10.0 pts" in out


def test_main_stops_before_any_row_prints_nan_percent(tmp_path, monkeypatch, capsys):
    _build_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    out_path = tmp_path / "results" / "open" / "detector_inject_other139.csv"
    monkeypatch.setattr(run_inject_other, "OUT", str(out_path))

    def fake_run_image(provider, model, b64, questions, chart):
        raise RuntimeError("cap hit immediately")

    monkeypatch.setattr(run_inject_other.T, "run_image", fake_run_image)
    monkeypatch.setattr(run_inject_other, "grade", _fake_grade_by_keyword)

    run_inject_other.main()   # still must not raise

    d = pd.read_csv(out_path)
    assert len(d) == 0
    assert list(d.columns) == ["image", "index", "group", "base", "B"]

    out = capsys.readouterr().out
    assert "STOPPED (Gemini cap?): cap hit immediately" in out
    assert "=== 0 of 2 other-concrete graded ===" in out
    # CURRENT BEHAVIOR (looks like a bug, but does not crash): mean of an empty
    # column is NaN, and the f-string happily prints "nan%" / "+nan pts".
    assert "baseline nan%  ->  B (+chart) nan%   NET +nan pts" in out


def test_main_missing_baseline_score_crashes_the_print_but_is_swallowed(tmp_path, monkeypatch, capsys):
    # CURRENT BEHAVIOR (looks like a bug): the per-item print line does
    # `base.get(idx):.1f` with NO default, unlike the CSV row's `base.get(idx, "")`.
    # If a target index is missing from the scores CSV, formatting None raises
    # TypeError -- but that happens INSIDE the try block, so the broad
    # `except Exception` treats it exactly like a Gemini-cap error: it prints
    # "STOPPED", and every image not yet processed (here, imgb) is silently
    # dropped, even though nothing was actually wrong with the API.
    _build_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    sc = pd.read_csv(tmp_path / "results" / "open" / "batched_gemini35_plain578_scores.csv")
    sc = sc[sc["index"] != 2]   # drop index 2's baseline score entirely
    sc.to_csv(tmp_path / "results" / "open" / "batched_gemini35_plain578_scores.csv", index=False)
    out_path = tmp_path / "results" / "open" / "detector_inject_other139.csv"
    monkeypatch.setattr(run_inject_other, "OUT", str(out_path))

    def fake_run_image(provider, model, b64, questions, chart):
        return [f"ans{i}" for i in range(len(questions))]

    monkeypatch.setattr(run_inject_other.T, "run_image", fake_run_image)
    monkeypatch.setattr(run_inject_other, "grade", _fake_grade_by_keyword)

    run_inject_other.main()   # does not raise -- the TypeError is caught, not propagated

    d = pd.read_csv(out_path)
    assert len(d) == 1   # the writerow for index 2 happens BEFORE the crashing print
    assert d.iloc[0]["index"] == 2
    assert pd.isna(d.iloc[0]["base"])   # base.get(idx, "") wrote "", read back as NaN

    out = capsys.readouterr().out
    assert "STOPPED (Gemini cap?): unsupported format string passed to NoneType.__format__" in out
    assert "=== 1 of 2 other-concrete graded ===" in out
    assert "baseline nan%  ->  B (+chart) 20.0%   NET +nan pts" in out
