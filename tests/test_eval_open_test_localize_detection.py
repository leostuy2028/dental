# Tests for eval_open/test_localize_detection.py
# (an experiment SCRIPT, not a pytest file -- see pyproject.toml's testpaths).
# Compares a "which tooth has X" control answer (no detection map) against the
# same question re-answered with a detector tooth-chart injected, batched per
# image, then grades both arms and reports the paired Wilcoxon delta.
import json

import pandas as pd

import eval_open.run_batched as rb
import eval_open.test_localize_detection as L


# ---------- localize_idx ----------

def test_localize_idx_filters_teeth_category_and_which_tooth_question():
    df = pd.DataFrame([
        {"index": 0, "category": "Teeth Findings", "question": "Which tooth has a filling?"},
        {"index": 1, "category": "Teeth Findings", "question": "WHICH TEETH are missing?"},   # case-insensitive
        {"index": 2, "category": "Teeth Findings", "question": "How many teeth are visible?"},  # wrong question
        {"index": 3, "category": "General", "question": "Which tooth has a filling?"},          # wrong category
    ])
    assert L.localize_idx(df) == {0, 1}


def test_localize_idx_handles_non_string_category():
    df = pd.DataFrame([{"index": 0, "category": None, "question": "Which tooth is affected?"}])
    assert L.localize_idx(df) == set()   # str(None) == "None", does not contain "Teeth"


# ---------- teeth_set ----------

def test_teeth_set_extracts_fdi_codes():
    assert L.teeth_set("tooth #18 and 26 are affected") == {"18", "26"}


def test_teeth_set_returns_empty_set_with_no_match():
    assert L.teeth_set("no teeth mentioned here") == set()


def test_teeth_set_handles_none():
    assert L.teeth_set(None) == set()


# ---------- main ----------

def _write_main_fixture(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "reference").mkdir()

    rows = [
        {"index": 0, "image_name": "imgL.jpg", "category": "Teeth Findings",
         "question": "Which tooth has a filling?", "answer": "#14", "image": "B64_L"},
        {"index": 1, "image_name": "imgL.jpg", "category": "Teeth Findings",
         "question": "Which teeth are missing?", "answer": "none", "image": "B64_L"},
        {"index": 2, "image_name": "imgOther.jpg", "category": "General",
         "question": "What is the overall condition?", "answer": "fine", "image": "B64_O"},
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "data" / "open_ended.parquet")
    with open(tmp_path / "reference" / "opg_primer.txt", "w", encoding="utf-8") as f:
        f.write("tiny primer")


def _patch_constants(tmp_path, monkeypatch, det_map):
    monkeypatch.setattr(L, "DATA", str(tmp_path / "data" / "open_ended.parquet"))
    monkeypatch.setattr(L, "PRIMER", str(tmp_path / "reference" / "opg_primer.txt"))
    det_path = tmp_path / "reference" / "teeth_detections.json"
    with open(det_path, "w", encoding="utf-8") as f:
        json.dump(det_map, f)
    monkeypatch.setattr(L, "DET", str(det_path))


def test_main_aborts_when_a_localize_image_is_missing_from_detection_map(tmp_path, monkeypatch, capsys):
    _write_main_fixture(tmp_path)
    _patch_constants(tmp_path, monkeypatch, det_map={})   # imgL.jpg NOT in the map
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["test_localize_detection"])

    L.main()

    out = capsys.readouterr().out
    assert "ABORT: 1/1 LOCALIZE images still lack a detection map." in out
    assert not (tmp_path / "results" / "open" / "localize_detection.csv").exists()


def test_main_full_success_path_writes_csv_and_prints_wilcoxon(tmp_path, monkeypatch, capsys):
    _write_main_fixture(tmp_path)
    _patch_constants(tmp_path, monkeypatch, det_map={"imgL.jpg": "CHART FOR imgL"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["test_localize_detection"])

    def fake_answer_image(b64, questions, primer, system, provider, model, detection_text=None):
        tag = "ctrl" if detection_text is None else "det"
        return [f"{tag}_ans for {q}" for q in questions], f"{tag}_raw"

    monkeypatch.setattr(rb, "answer_image", fake_answer_image)

    def fake_grade(prompt, judge=None):
        if "ctrl_ans" in prompt:
            return 0.4, "raw"
        if "det_ans" in prompt:
            return 0.8, "raw"
        raise AssertionError(f"unexpected grading prompt: {prompt[:80]}")

    monkeypatch.setattr(L, "grade", fake_grade)

    L.main()

    out_csv = tmp_path / "results" / "open" / "localize_detection.csv"
    d = pd.read_csv(out_csv)
    assert sorted(d["index"].tolist()) == [0, 1]   # only the 2 LOCALIZE rows, imgOther excluded
    assert (d["score_ctrl"] == 0.4).all() and (d["score_det"] == 0.8).all()
    assert d["ans_ctrl"].str.startswith("ctrl_ans").all()
    assert d["ans_det"].str.startswith("det_ans").all()

    out = capsys.readouterr().out
    assert "gpt-5-mini: 2 LOCALIZE items across 1 images; BOTH arms (batched)..." in out
    assert "LOCALIZE: baseline (no det, batched) vs +detection map, gpt-5-mini minimal, n=2" in out
    assert "control (baseline)  : 40.0%" in out
    assert "+ detection map     : 80.0%" in out
    assert "paired delta        : +40.0 pts  (Wilcoxon p=0.5; up 2, down 0, same 0)" in out
