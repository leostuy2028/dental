# Tests for eval_open/test_detector_inject.py
# (an experiment SCRIPT, not a pytest file -- pyproject.toml restricts pytest
# collection to tests/, so importing it here as ordinary source is safe).
# It injects the trained YOLO tooth map into the prompt for concrete questions
# the VLM currently gets wrong, and measures the effect vs a no-chart control.
#
# NOTE on PRIMER: `PRIMER = open("reference/opg_primer.txt", ...).read()` runs
# at module IMPORT time. reference/opg_primer.txt already exists for real in
# this repo (checked before writing these tests), and pytest's cwd is the repo
# root, so the import below succeeds against the real file -- no chdir dance
# needed just to import the module. PRIMER is still an ordinary module
# attribute afterward, so tests that care about its content monkeypatch it.
import json

import pandas as pd
import pytest

import eval_open.test_detector_inject as T


# ---------- anat ----------

def test_anat_combines_quadrant_and_tooth_name():
    assert T.anat("18") == "upper-right third molar (wisdom)"
    assert T.anat("26") == "upper-left first molar"


def test_anat_raises_keyerror_for_unknown_quadrant_digit():
    # QUAD only has keys "1".."4"; a malformed code like "59" blows up.
    with pytest.raises(KeyError):
        T.anat("59")


# ---------- build_chart ----------

def test_build_chart_zero_teeth():
    got = T.build_chart({"count": 5, "teeth": []})
    assert got == (
        "Total teeth detected on this X-ray: 5.\n"
        "Of these, 0 were confidently numbered (FDI code = position):\n"
        "\n"
        "(The other detected teeth were located but not confidently numbered.)"
    )


def test_build_chart_one_tooth():
    got = T.build_chart({"count": 5, "teeth": [{"fdi": "18", "box_2d": [0, 0, 1, 1], "conf": 0.9}]})
    assert got == (
        "Total teeth detected on this X-ray: 5.\n"
        "Of these, 1 were confidently numbered (FDI code = position):\n"
        "  #18 = upper-right third molar (wisdom)\n"
        "(The other detected teeth were located but not confidently numbered.)"
    )


def test_build_chart_sorts_teeth_by_fdi_as_a_string_not_a_number():
    # entry.teeth given out of order; build_chart sorts by t["fdi"], which is a
    # plain string sort (`sorted(..., key=lambda t: t["fdi"])`). For real 2-digit
    # FDI codes this happens to agree with numeric order (all same length), which
    # this test also confirms with a full-string match.
    entry = {"count": 30, "teeth": [{"fdi": "26"}, {"fdi": "11"}, {"fdi": "18"}]}
    got = T.build_chart(entry)
    assert got == (
        "Total teeth detected on this X-ray: 30.\n"
        "Of these, 3 were confidently numbered (FDI code = position):\n"
        "  #11 = upper-right central incisor\n"
        "  #18 = upper-right third molar (wisdom)\n"
        "  #26 = upper-left first molar\n"
        "(The other detected teeth were located but not confidently numbered.)"
    )
    # a direct demonstration that the sort key is a STRING sort: "8" (one char)
    # sorts AFTER "18" and "26" because string comparison looks at the first
    # character first ("1" < "2" < "8"), unlike a numeric sort where 8 < 18 < 26.
    assert sorted([{"fdi": "18"}, {"fdi": "8"}, {"fdi": "26"}], key=lambda t: t["fdi"]) == \
        [{"fdi": "18"}, {"fdi": "26"}, {"fdi": "8"}]


def test_build_chart_crashes_on_a_malformed_single_digit_fdi():
    # CURRENT BEHAVIOR (looks like a bug): build_chart assumes every fdi is a
    # well-formed 2-character FDI code. A malformed single-digit code sorts fine
    # (it's still just a string), but anat() indexes f[1] and KeyErrors/IndexErrors.
    with pytest.raises((KeyError, IndexError)):
        T.build_chart({"count": 1, "teeth": [{"fdi": "8"}]})


# ---------- strip_b64 ----------

def test_strip_b64_removes_jpeg_and_png_data_uri_prefix():
    assert T.strip_b64("data:image/jpeg;base64,ABC123") == "ABC123"
    assert T.strip_b64("data:image/png;base64,XYZ") == "XYZ"


def test_strip_b64_leaves_a_bare_string_unchanged():
    assert T.strip_b64("ABC123") == "ABC123"


def test_strip_b64_stringifies_non_string_input():
    assert T.strip_b64(None) == "None"


# ---------- select_targets ----------

def _select_targets_op():
    return pd.DataFrame([
        {"index": 0, "image_name": "imgA.jpg", "question": "How many teeth are visible?"},
        {"index": 1, "image_name": "imgA.jpg", "question": "Which tooth has a filling?"},
        {"index": 2, "image_name": "imgB.jpg", "question": "What is the condition of #26?"},
        {"index": 3, "image_name": "imgB.jpg", "question": "Are any teeth missing or absent from the arch?"},
        {"index": 4, "image_name": "imgB.jpg", "question": "Describe the overall dental health."},
    ])


def _write_scores(tmp_path, monkeypatch, rows):
    p = tmp_path / "scores.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    monkeypatch.setattr(T, "SCORES", str(p))


def test_select_targets_wrong_only_keeps_low_scoring_relevant_dtypes(tmp_path, monkeypatch):
    _write_scores(tmp_path, monkeypatch, [
        {"index": 0, "score": 0.2}, {"index": 1, "score": 0.6},
        {"index": 2, "score": 0.3}, {"index": 3, "score": 0.7}, {"index": 4, "score": 0.1},
    ])
    got = T.select_targets(_select_targets_op(), wrong_only=True)
    assert sorted(got["index"].tolist()) == [0, 2]   # dtype in {count,missing,whichtooth,cond#N} AND score<=0.4
    assert dict(zip(got["index"], got["dtype"])) == {0: "count", 2: "cond#N"}


def test_select_targets_not_wrong_only_keeps_all_relevant_dtypes(tmp_path, monkeypatch):
    _write_scores(tmp_path, monkeypatch, [
        {"index": 0, "score": 0.2}, {"index": 1, "score": 0.6},
        {"index": 2, "score": 0.3}, {"index": 3, "score": 0.7}, {"index": 4, "score": 0.1},
    ])
    got = T.select_targets(_select_targets_op(), wrong_only=False)
    assert sorted(got["index"].tolist()) == [0, 1, 2, 3]   # index 4 excluded: bucket is Vague, not Concrete
    assert dict(zip(got["index"], got["dtype"])) == {0: "count", 1: "whichtooth", 2: "cond#N", 3: "missing"}


# ---------- run_image ----------

def test_run_image_without_chart_does_not_inject_tooth_chart_text(monkeypatch):
    seen = {}

    def fake_provider(b64, system, user, model, exemplars=None):
        seen["user"] = user
        return "1. answer one\n2. answer two"

    monkeypatch.setitem(T.PROVIDERS, "gemini", fake_provider)

    got = T.run_image("gemini", "gemini-3.5-flash", "B64IMG", ["q1?", "q2?"], None)

    assert "TOOTH CHART" not in seen["user"]
    assert got == T.parse_numbered("1. answer one\n2. answer two", 2)


def test_run_image_with_chart_injects_tooth_chart_text(monkeypatch):
    seen = {}

    def fake_provider(b64, system, user, model, exemplars=None):
        seen["user"] = user
        return "1. answer one\n2. answer two"

    monkeypatch.setitem(T.PROVIDERS, "gemini", fake_provider)

    got = T.run_image("gemini", "gemini-3.5-flash", "B64IMG", ["q1?", "q2?"], "MY CHART TEXT")

    assert "TOOTH CHART" in seen["user"]
    assert "MY CHART TEXT" in seen["user"]
    assert got == ["answer one", "answer two"]


# ---------- main ----------

def _main_fixture(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "reference").mkdir()
    (tmp_path / "results" / "open").mkdir(parents=True)

    rows = [
        {"index": 0, "image_name": "imgA.jpg", "question": "How many teeth are visible?",
         "answer": "30", "image": "B64_A"},
        {"index": 1, "image_name": "imgA.jpg", "question": "Which tooth has a filling?",
         "answer": "#14", "image": "B64_A"},
        {"index": 2, "image_name": "imgB.jpg", "question": "What is the condition of #26?",
         "answer": "caries", "image": "B64_B"},
        {"index": 3, "image_name": "imgB.jpg", "question": "Are any teeth missing or absent from the arch?",
         "answer": "none", "image": "B64_B"},
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "data" / "open_ended.parquet")

    tmap = {
        "imgA.jpg": {"count": 30, "teeth": [{"fdi": "18", "box_2d": [0, 0, 1, 1], "conf": 0.9}]},
        "imgB.jpg": {"count": 28, "teeth": [{"fdi": "26", "box_2d": [0, 0, 1, 1], "conf": 0.8}]},
    }
    with open(tmp_path / "reference" / "mmoral_map.json", "w", encoding="utf-8") as f:
        json.dump(tmap, f)

    pd.DataFrame([{"index": 0, "score": 0.2}, {"index": 1, "score": 0.6},
                  {"index": 2, "score": 0.3}, {"index": 3, "score": 0.7}]).to_csv(
        tmp_path / "results" / "open" / "batched_gemini35_plain578_scores.csv", index=False)


def _patch_main_constants(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "DATA", str(tmp_path / "data" / "open_ended.parquet"))
    monkeypatch.setattr(T, "MAP", str(tmp_path / "reference" / "mmoral_map.json"))
    monkeypatch.setattr(T, "SCORES", str(tmp_path / "results" / "open" / "batched_gemini35_plain578_scores.csv"))
    monkeypatch.setattr(T, "PRIMER", "tiny primer text")


def _fake_provider_numbered(b64, system, user, model, exemplars=None):
    return "1. one\n2. two\n3. three\n4. four"


def test_main_all_and_both_arms_writes_csv_and_prints_summary(tmp_path, monkeypatch, capsys):
    _main_fixture(tmp_path)
    _patch_main_constants(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(T.PROVIDERS, "gemini", _fake_provider_numbered)
    monkeypatch.setattr(T, "grade", lambda prompt, judge=None: (0.55, "raw"))
    monkeypatch.setattr("sys.argv", ["test_detector_inject", "--all"])

    T.main()

    out_csv = tmp_path / "results" / "open" / "detector_inject_gemini35flash_all.csv"
    d = pd.read_csv(out_csv)
    assert sorted(d["index"].tolist()) == [0, 1, 2, 3]
    assert list(d.columns) == ["image", "index", "dtype", "q", "base", "B_chart", "ansB", "A_nochart", "ansA"]
    assert (d["B_chart"] == 0.55).all() and (d["A_nochart"] == 0.55).all()

    out = capsys.readouterr().out
    assert "4 target questions across 2 images; running 2 images." in out
    assert "arms: A=no chart, B=+chart | model=gemini-3.5-flash think=8192 effort=high" in out
    assert "=== 4 target questions ===" in out
    assert "baseline (committed plain578):         45.0%" in out
    assert "A (no chart, think 8192):         55.0%" in out
    assert "B (+ detector chart, think 8192):  55.0%" in out
    assert "NET (B - baseline): +10.0 pts" in out
    # both the rescue (base<=0.4) and already-right (base>0.4) groups are non-empty
    assert "rescue        (base<=0.4, n= 2):   25% ->   55%" in out
    assert "already-right (base >0.4, n= 2):   65% ->   55%   <- regression check" in out
    assert "wrote results/open/detector_inject_gemini35flash_all.csv" in out


def test_main_chart_only_default_wrong_only_skips_a_arm(tmp_path, monkeypatch, capsys):
    _main_fixture(tmp_path)
    _patch_main_constants(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(T.PROVIDERS, "gemini", _fake_provider_numbered)
    monkeypatch.setattr(T, "grade", lambda prompt, judge=None: (0.55, "raw"))
    monkeypatch.setattr("sys.argv", ["test_detector_inject", "--chart-only"])

    T.main()

    out_csv = tmp_path / "results" / "open" / "detector_inject_gemini35flash.csv"
    d = pd.read_csv(out_csv)
    # default (no --all) is wrong_only=True: only the base<=0.4 rows (index 0, 2)
    assert sorted(d["index"].tolist()) == [0, 2]
    assert "A_nochart" not in d.columns   # --chart-only skips the no-chart arm entirely

    out = capsys.readouterr().out
    assert "arms: B=+chart only" in out
    assert "A (no chart" not in out
    assert "already-right" not in out   # no row has base>0.4 in the wrong_only subset, so `reg` is empty
