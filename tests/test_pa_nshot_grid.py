# Tests for paper_analysis/nshot_grid.py
# This script re-derives the predicted letter from each n-shot run's raw_response
# (never the stored "predicted" column, which an old harness mis-parsed) and
# reports accuracy/CI, %A, and a chi-square-vs-uniform per model x mode x k cell.

import os
import json

import pandas as pd

import nshot_grid as ng


# ---------- small pure helpers ----------

def test_path_for_direct_has_no_cot_suffix():
    assert ng.path_for("2.5-flash", "direct", 0) == \
        "results/nshot/closed_gemini-2.5-flash_k0_cleanshuf_think0.csv"


def test_path_for_cot_adds_suffix():
    assert ng.path_for("3.5-flash", "cot", 5) == \
        "results/nshot/closed_gemini-3.5-flash_k5_cleanshuf_think0_cot.csv"


def test_wilson_zero_n():
    assert ng.wilson(0, 0) == (0.0, 0.0)


def test_wilson_rounds_to_whole_percent():
    # verified by running ng.wilson(3, 10)
    assert ng.wilson(3, 10) == (11, 60)


def test_chi2_uniform_zero_n_is_zero():
    assert ng.chi2_uniform([0, 0, 0, 0]) == 0.0


def test_chi2_uniform_even_split_is_zero():
    assert ng.chi2_uniform([25, 25, 25, 25]) == 0.0


def test_chi2_uniform_lopsided_matches_hand_count():
    # verified against ng.cell() on the same counts below
    assert round(ng.chi2_uniform([7, 1, 0, 0]), 1) == 17.0


# ---------- cell() ----------

def _write_direct_fixture(tmp_path):
    # 8 rows: one is a stored API-failure sentinel (excluded from n), one has no
    # letter and no cue but DOES contain the standalone word "A" (extract_letter's
    # last-resort standalone-token rule catches it).
    df = pd.DataFrame({
        "raw_response": ["A", "B", "(C)", "max retries exceeded",
                          "The correct answer is D", "not a letter at all", "A", "B"],
        "answer":       ["A", "A", "C",   "B",
                          "D",                        "C",                  "A", "A"],
    })
    path = tmp_path / "results" / "nshot" / "closed_gemini-2.5-flash_k0_cleanshuf_think0.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_cell_direct_mode_excludes_api_failures(tmp_path, monkeypatch):
    _write_direct_fixture(tmp_path)
    monkeypatch.setattr(ng, "REPO", str(tmp_path))

    c = ng.cell("2.5-flash", "direct", 0)

    assert c["n"] == 7  # 8 rows minus 1 api-failure sentinel
    assert c["api_failures_excluded"] == 1
    assert c["acc"] == 57.1
    assert c["ci"] == [25, 84]
    assert c["counts"] == {"A": 3, "B": 2, "C": 1, "D": 1}
    assert c["pctA"] == 43
    assert c["chi2"] == 1.6
    assert c["sig"] is False
    assert c["unparseable"] == 0


def _write_cot_fixture(tmp_path):
    # 9 rows, cot mode: 7 clean "Answer: A" lines, 1 "Answer: B", and 1 reply with
    # no answer line at all -> None (a real "no pick" reply, scored wrong, counted
    # as unparseable, and NOT scavenged from the reasoning text).
    rows_resp = ["Answer: A"] * 7 + ["Answer: B", "no cue here at all"]
    rows_ans = ["A"] * 7 + ["A", "A"]
    df = pd.DataFrame({"raw_response": rows_resp, "answer": rows_ans})
    path = tmp_path / "results" / "nshot" / "closed_gemini-3.5-flash_k0_cleanshuf_think0_cot.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_cell_cot_mode_is_lopsided_and_significant(tmp_path, monkeypatch):
    _write_cot_fixture(tmp_path)
    monkeypatch.setattr(ng, "REPO", str(tmp_path))

    c = ng.cell("3.5-flash", "cot", 0)

    assert c["n"] == 9
    assert c["api_failures_excluded"] == 0
    assert c["acc"] == 77.8
    assert c["ci"] == [45, 94]
    assert c["counts"] == {"A": 7, "B": 1, "C": 0, "D": 0}
    assert c["pctA"] == 78
    assert c["chi2"] == 17.0
    assert c["sig"] is True  # 17.0 > CHI2_CRIT_05 (7.815)
    assert c["unparseable"] == 1


# ---------- _row / _table formatting ----------

def test_row_bolds_the_best_accuracy_and_flags_significance(tmp_path, monkeypatch):
    _write_direct_fixture(tmp_path)
    _write_cot_fixture(tmp_path)
    monkeypatch.setattr(ng, "REPO", str(tmp_path))
    low = ng.cell("2.5-flash", "direct", 0)
    high = ng.cell("3.5-flash", "cot", 0)

    best = max(low["acc"], high["acc"])
    row_low = ng._row(low, best)
    row_high = ng._row(high, best)

    assert row_low == "| 2.5-flash | direct | 0 | 57.1 [25–84] | 43 | 1.6 |"
    assert row_high == "| 3.5-flash | CoT | 0 | **77.8 [45–94]** | **78** | 17.0\\* |"

    table = ng._table([low, high], "<!-- prov -->")
    assert table.splitlines()[0] == "<!-- prov -->"
    assert row_low in table
    assert row_high in table


# ---------- main() end to end ----------

def test_main_writes_tables_and_values_json(tmp_path, monkeypatch, capsys):
    _write_direct_fixture(tmp_path)
    out_dir = tmp_path / "_generated"
    monkeypatch.setattr(ng, "REPO", str(tmp_path))
    monkeypatch.setattr(ng, "OUT_DIR", str(out_dir))
    monkeypatch.setattr(ng, "MODELS", ["2.5-flash"])
    monkeypatch.setattr(ng, "MODES", ["direct"])
    monkeypatch.setattr(ng, "KS", [0])
    monkeypatch.setattr(ng, "CONDENSED", [("2.5-flash", "direct", 0)])

    ng.main()
    out = capsys.readouterr().out

    assert "| 2.5-flash | direct | 0 | **57.1 [25–84]** | 43 | 1.6 |" in out
    assert "full 16-cell grid + condensed + values.json written to" in out
    assert "WARN: 2.5-flash direct k0 EXCLUDED 1 API-failure row(s); scored on n=7." in out
    # this cell has zero unparseable replies, so no "note:" line for it
    assert "note:" not in out

    full_path = out_dir / "nshot_grid_table.md"
    cond_path = out_dir / "nshot_grid_condensed.md"
    values_path = out_dir / "nshot_grid.values.json"
    assert full_path.exists() and cond_path.exists() and values_path.exists()

    vals = json.loads(values_path.read_text(encoding="utf-8"))
    assert vals["2.5-flash|direct|k0"]["n"] == 7
    assert vals["_generator"] == "paper_analysis/nshot_grid.py"
    assert "raw_response" in vals["_note"]


def test_main_prints_a_note_for_unparseable_replies(tmp_path, monkeypatch, capsys):
    _write_cot_fixture(tmp_path)
    monkeypatch.setattr(ng, "REPO", str(tmp_path))
    monkeypatch.setattr(ng, "OUT_DIR", str(tmp_path / "_generated"))
    monkeypatch.setattr(ng, "MODELS", ["3.5-flash"])
    monkeypatch.setattr(ng, "MODES", ["cot"])
    monkeypatch.setattr(ng, "KS", [0])
    monkeypatch.setattr(ng, "CONDENSED", [("3.5-flash", "cot", 0)])

    ng.main()
    out = capsys.readouterr().out

    assert "note: 3.5-flash cot k0 has 1 unparseable reply(ies)" in out
