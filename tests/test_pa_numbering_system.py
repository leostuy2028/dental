# Tests for paper_analysis/numbering_system.py
# This script scans the released questions (no API calls) for tooth codes written
# as "#NN" and buckets each question by whether that code is ambiguous between the
# FDI and US-Universal numbering systems, then reports gemini-3.5's accuracy per
# bucket. It also has a "probe" subcommand that just prints a pointer to results
# recorded elsewhere (the actual model probes are not re-run here).

import sys

import pandas as pd

import numbering_system as nsy


# ---------- fdi_valid ----------

def test_fdi_valid_true_cases():
    assert nsy.fdi_valid(11) is True    # smallest valid code
    assert nsy.fdi_valid(48) is True    # largest valid code (quadrant 4, tooth 8)
    assert nsy.fdi_valid(32) is True
    assert nsy.fdi_valid(33) is True


def test_fdi_valid_false_cases():
    assert nsy.fdi_valid(8) is False    # below 11
    assert nsy.fdi_valid(99) is False   # quadrant 9 doesn't exist
    assert nsy.fdi_valid(19) is False   # tooth 9 doesn't exist in a quadrant


# ---------- blast_radius_closed ----------

def _write_closed_fixture(base):
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cl = pd.DataFrame({
        "index": [1, 2, 3, 4, 5],
        "question": [
            "Which tooth is #12?",              # ambiguous only -> HIGH_RISK
            "Compare #12 and #34.",              # ambiguous + fdi-only -> DISAMBIGUATED
            "Identify tooth #34.",               # fdi-only only -> SAFE_FDI
            "Identify tooth #34 too.",           # fdi-only only -> SAFE_FDI
            "No codes mentioned here.",          # -> NO_CODES
        ],
        "option1": ["a"] * 5, "option2": ["b"] * 5, "option3": ["c"] * 5, "option4": ["d"] * 5,
    })
    cl.to_parquet(data_dir / "closed_ended.parquet")

    baseline_dir = base / "results" / "closed_ended" / "position_bias"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    base_df = pd.DataFrame({
        "index": [1, 2, 3, 4, 5],
        "correct": [True, False, True, True, False],
    })
    base_df.to_csv(baseline_dir / "gemini-3.5-flash__coax-direct-k0__shuffled__n491.csv", index=False)


def test_blast_radius_closed_buckets_by_ambiguity(tmp_path, capsys):
    _write_closed_fixture(tmp_path)

    nsy.blast_radius_closed(str(tmp_path))
    out = capsys.readouterr().out

    assert "CLOSED (5 questions):" in out
    assert "HIGH_RISK         1 (20%)   gemini-3.5 acc 100.0%" in out
    assert "DISAMBIGUATED     1 (20%)   gemini-3.5 acc 0.0%" in out
    assert "SAFE_FDI          2 (40%)   gemini-3.5 acc 100.0%" in out
    assert "NO_CODES          1 (20%)   gemini-3.5 acc 0.0%" in out


# ---------- blast_radius_open ----------

def _write_open_fixture(base):
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    op = pd.DataFrame({
        "answer": [
            "The tooth is #12.",           # ambiguous code, not fdi-only
            "Refer to tooth #34 (FDI).",    # fdi-only code
            '{"tooth_id": "21"}',           # tooth_id pattern, ambiguous code
            "No code mentioned.",           # no code at all
        ]
    })
    op.to_parquet(data_dir / "open_ended.parquet")


def test_blast_radius_open_reports_code_coverage(tmp_path, capsys):
    _write_open_fixture(tmp_path)

    nsy.blast_radius_open(str(tmp_path))
    out = capsys.readouterr().out

    assert "OPEN (4 questions):" in out
    assert "reference contains a tooth code:           3 (75%)" in out
    assert "reference has an FDI-only (>32) code:      1 (25%)" in out


# ---------- main() ----------

def test_main_runs_both_scans(tmp_path, monkeypatch, capsys):
    _write_closed_fixture(tmp_path)
    _write_open_fixture(tmp_path)
    # main() computes its repo path from the module's own __file__, anchored two
    # directories up -- point that at a fake path under tmp_path so it resolves
    # to our fixture tree without touching cwd.
    fake_file = tmp_path / "paper_analysis" / "numbering_system.py"
    monkeypatch.setattr(nsy, "__file__", str(fake_file))
    monkeypatch.setattr(sys, "argv", ["numbering_system.py"])

    nsy.main()
    out = capsys.readouterr().out

    assert "CLOSED (5 questions):" in out
    assert "OPEN (4 questions):" in out


def test_main_probe_arg_just_prints_a_pointer(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["numbering_system.py", "probe"])

    nsy.main()
    out = capsys.readouterr().out

    assert "API probes are recorded" in out
    # the probe branch returns before scanning any data
    assert "CLOSED (" not in out
