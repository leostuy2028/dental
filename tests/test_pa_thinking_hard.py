# Tests for paper_analysis/thinking_hard.py
# This script reads a sweep of "thinking budget" result CSVs (0, 512, 2048, 8192,
# dynamic) for the same 50 hard items and reports accuracy per budget (with a Wilson
# CI), the letter distribution per budget, and a paired McNemar test between budget 0
# and budget 8192. Writes a markdown table + a JSON of the numbers.

import json
import os

import pandas as pd
import pytest

import thinking_hard as th


# ---------- wilson ----------

def test_wilson_zero_n_gives_zero_interval():
    assert th.wilson(0, 0) == (0.0, 0.0)


def test_wilson_matches_hand_checked_interval():
    lo, hi = th.wilson(3, 5)
    assert round(lo, 1) == 23.1
    assert round(hi, 1) == 88.2


# ---------- mcnemar_exact ----------

def test_mcnemar_exact_no_discordant_pairs_gives_p_1():
    assert th.mcnemar_exact(0, 0) == 1.0


def test_mcnemar_exact_symmetric_split_gives_p_1():
    assert th.mcnemar_exact(1, 1) == 1.0


def test_mcnemar_exact_one_sided_gives_small_p():
    p = th.mcnemar_exact(0, 8)
    assert p < 0.05


# ---------- load ----------

def test_load_returns_none_when_file_missing(tmp_path):
    assert th.load(str(tmp_path), "9999") is None


def test_load_reads_the_matching_pattern_file(tmp_path):
    cot_dir = tmp_path / "results/closed_ended/cot_length"
    cot_dir.mkdir(parents=True)
    df = pd.DataFrame({"index": [1, 2], "correct": [True, False], "predicted": ["A", "B"]})
    df.to_csv(cot_dir / "gemini-3.5-flash__direct-think0__hard50-shuffled__n50.csv", index=False)
    out = th.load(str(tmp_path), "0")
    assert list(out["index"]) == [1, 2]
    assert list(out.index) == [1, 2]  # drop=False keeps "index" as a column AND the index


# ---------- main() : no sweep files at all ----------

def test_main_raises_systemexit_when_no_csvs_found(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "DIR", str(tmp_path / "empty"))
    with pytest.raises(SystemExit, match="no sweep CSVs found"):
        th.main()


# ---------- main() end to end ----------

def build_fixture(tmp_path):
    cot_dir = tmp_path / "cotdir"
    cot_dir.mkdir()
    b0 = pd.DataFrame({
        "index": [1, 2, 3, 4, 5],
        "correct": [True, False, True, False, True],
        "predicted": ["A", "B", "A", "C", "D"],
    })
    b0.to_csv(cot_dir / "gemini-3.5-flash__direct-think0__hard50-shuffled__n50.csv", index=False)

    b512 = pd.DataFrame({
        "index": [1, 2, 3, 4],
        "correct": [True, True, False, False],
        "predicted": ["A", "A", "B", "C"],
    })
    b512.to_csv(cot_dir / "gemini-3.5-flash__direct-think512__hard50-shuffled__n50.csv", index=False)

    b8192 = pd.DataFrame({
        "index": [1, 2, 3, 4, 5],
        "correct": [True, True, True, False, False],
        "predicted": ["A", "A", "A", "C", "B"],
    })
    b8192.to_csv(cot_dir / "gemini-3.5-flash__direct-think8192__hard50-shuffled__n50.csv", index=False)
    # budgets "2048" and "-1" (dynamic) are intentionally left out -> pending rows
    return cot_dir


def test_main_prints_table_with_pending_rows_and_paired_stats(tmp_path, monkeypatch, capsys):
    cot_dir = build_fixture(tmp_path)
    out_dir = tmp_path / "outdir"
    monkeypatch.setattr(th, "DIR", str(cot_dir))
    monkeypatch.setattr(th, "OUT_DIR", str(out_dir))

    th.main()
    out = capsys.readouterr().out

    assert "| off | 60.0% [23-88] | 2/1/1/1 | 5 |" in out
    assert "| 512 | 50.0% [15-85] | 2/1/1/0 | 4 |" in out
    assert "| 2048 | *[pending]* | | |" in out
    assert "| 8192 | 60.0% [23-88] | 3/1/1/0 | 5 |" in out
    assert "| dynamic | *[pending]* | | |" in out
    assert ("**Paired budget 0 -> 8192 (same 5 items):** 60.0% -> 60.0%; "
            "thinking rescued 1, broke 1 (net +0); McNemar exact p = 1.0.") in out


def test_main_writes_table_and_json_files(tmp_path, monkeypatch):
    cot_dir = build_fixture(tmp_path)
    out_dir = tmp_path / "outdir"
    monkeypatch.setattr(th, "DIR", str(cot_dir))
    monkeypatch.setattr(th, "OUT_DIR", str(out_dir))

    th.main()

    table_path = out_dir / "thinking_hard_table.md"
    json_path = out_dir / "thinking_hard.values.json"
    assert table_path.exists()
    assert json_path.exists()
    assert "GEN: thinking_hard" in table_path.read_text()

    vals = json.loads(json_path.read_text())
    assert vals["_generator"] == "paper_analysis/thinking_hard.py"
    assert set(vals["per_budget"]) == {"0", "512", "8192"}
    assert vals["paired_0_vs_8192"]["rescued"] == 1
    assert vals["paired_0_vs_8192"]["broke"] == 1


def test_main_skips_paired_stats_when_budget_8192_missing(tmp_path, monkeypatch, capsys):
    cot_dir = tmp_path / "cotdir_only0"
    cot_dir.mkdir()
    df = pd.DataFrame({"index": [1, 2], "correct": [True, False], "predicted": ["A", "B"]})
    df.to_csv(cot_dir / "gemini-3.5-flash__direct-think0__hard50-shuffled__n50.csv", index=False)
    monkeypatch.setattr(th, "DIR", str(cot_dir))
    monkeypatch.setattr(th, "OUT_DIR", str(tmp_path / "outdir_only0"))

    th.main()
    out = capsys.readouterr().out
    assert "Paired budget 0 -> 8192" not in out
