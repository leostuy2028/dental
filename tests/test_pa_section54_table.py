# Tests for paper_analysis/section54_table.py
# This script rebuilds the paper's §5.4 "does prior knowledge help" table from three
# committed result CSVs (no primer / + OPG primer / + primer + exemplars): overall
# accuracy, a paired McNemar test between arms, per-dimension accuracy, and a
# targeted-vs-control split.

import os
import pandas as pd

import section54_table as s54


def write_csv(path, rows):
    df = pd.DataFrame(rows, columns=["index", "category", "predicted", "correct"])
    df.to_csv(path, index=False)


# ---------- mcnemar_exact ----------

def test_mcnemar_exact_counts_rescued_and_broke():
    before = [True, False, True, False]
    after = [True, True, False, False]
    # index1: T->T (same), index2: F->T (rescued), index3: T->F (broke), index4: F->F (same)
    rescued, broke, p = s54.mcnemar_exact(before, after)
    assert rescued == 1
    assert broke == 1
    assert p == 1.0


def test_mcnemar_exact_no_discordant_pairs_gives_p_1():
    rescued, broke, p = s54.mcnemar_exact([True, True], [True, True])
    assert (rescued, broke, p) == (0, 0, 1.0)


def test_mcnemar_exact_all_rescued_no_broke():
    rescued, broke, p = s54.mcnemar_exact([False, False, False], [True, True, True])
    assert rescued == 3
    assert broke == 0
    assert 0 < p <= 1.0


# ---------- load ----------

def test_load_keeps_only_the_four_columns_and_sets_index(tmp_path):
    path = tmp_path / "one.csv"
    write_csv(path, [(1, "Teeth", "A", True), (2, "Jaw", "B", False)])
    df = s54.load(str(path))
    assert list(df.index) == [1, 2]
    assert list(df.columns) == ["category", "predicted", "correct"]
    assert df.loc[1, "predicted"] == "A"
    assert bool(df.loc[2, "correct"]) is False


# ---------- main() : missing file ----------

def test_main_prints_missing_and_returns_when_a_csv_is_absent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(s54, "ARMS", [
        ("no primer", str(tmp_path / "does_not_exist.csv")),
        ("+ OPG primer", str(tmp_path / "also_missing.csv")),
        ("+ primer + v2 exemplars", str(tmp_path / "missing_too.csv")),
    ])
    s54.main()
    out = capsys.readouterr().out
    assert "MISSING: " in out
    assert "does_not_exist.csv" in out
    # main returned early: none of the later section headers were printed
    assert "Section 5.4" not in out


# ---------- main() : full run against a tiny fixture ----------

ROWS_A0 = [
    (1, "Teeth", "A", True),
    (2, "Patho", "B", False),
    (3, "HisT", "C", False),
    (4, "Jaw", "D", False),
    (5, "SumRec", "A", True),
    (6, "Teeth,HisT", "B", False),
    (7, "Jaw,Patho", "C", True),
    (8, "SumRec,Teeth", "D", False),
]
ROWS_A1 = [
    (1, "Teeth", "A", True),
    (2, "Patho", "B", False),
    (3, "HisT", "A", True),
    (4, "Jaw", "D", False),
    (5, "SumRec", "A", True),
    (6, "Teeth,HisT", "A", True),
    (7, "Jaw,Patho", "D", False),
    (8, "SumRec,Teeth", "D", False),
]
ROWS_A2 = [
    (1, "Teeth", "A", True),
    (2, "Patho", "A", True),
    (3, "HisT", "A", True),
    (4, "Jaw", "A", True),
    (5, "SumRec", "A", True),
    (6, "Teeth,HisT", "A", True),
    (7, "Jaw,Patho", "D", False),
    (8, "SumRec,Teeth", "A", True),
]


def setup_arms(tmp_path, monkeypatch):
    paths = []
    for name, rows in [("a0", ROWS_A0), ("a1", ROWS_A1), ("a2", ROWS_A2)]:
        p = tmp_path / f"{name}.csv"
        write_csv(p, rows)
        paths.append(str(p))
    monkeypatch.setattr(s54, "ARMS", [
        ("no primer", paths[0]),
        ("+ OPG primer", paths[1]),
        ("+ primer + v2 exemplars", paths[2]),
    ])


def test_main_full_run_matches_hand_verified_output(tmp_path, monkeypatch, capsys):
    setup_arms(tmp_path, monkeypatch)
    s54.main()
    out = capsys.readouterr().out

    # header line per arm: n=8, unparseable=0 (this fixture has no NaN predictions)
    assert "[no primer                 ] n=8  unparseable=0" in out
    assert "[+ OPG primer              ] n=8  unparseable=0" in out
    assert "[+ primer + v2 exemplars   ] n=8  unparseable=0" in out

    # overall accuracy: 3/8, 4/8, 7/8 -- run and confirmed against the real code
    assert "no primer                   37.50%" in out
    assert "+ OPG primer                50.00%" in out
    assert "+ primer + v2 exemplars     87.50%" in out

    # paired mcnemar between the three arm pairs
    assert "no primer                  -> + OPG primer                rescued   2 / broke   1  net   +1  p=1" in out
    assert "+ OPG primer               -> + primer + v2 exemplars     rescued   3 / broke   0  net   +3  p=0.25" in out
    assert "no primer                  -> + primer + v2 exemplars     rescued   5 / broke   1  net   +4  p=0.2188" in out

    # per-dimension accuracy table
    assert "Teeth             33.3%           66.7%          100.0%    (n=3)" in out
    assert "Patho             50.0%            0.0%           50.0%    (n=2)" in out
    assert "HisT               0.0%          100.0%          100.0%    (n=2)" in out
    assert "Jaw               50.0%            0.0%           50.0%    (n=2)" in out
    assert "SumRec            50.0%           50.0%          100.0%    (n=2)" in out

    # targeted (HisT or Jaw) vs control specificity check
    assert "HisT OR Jaw (targeted)      25.0%    50.0%    75.0%   (n=4)" in out
    assert "neither (control)           50.0%    50.0%   100.0%   (n=4)" in out


def test_main_raises_on_index_mismatch_between_arms(tmp_path, monkeypatch):
    # CURRENT BEHAVIOR (looks like a bug): a real index mismatch would crash the whole
    # script with a bare AssertionError instead of a helpful message about which rows
    # differ -- the f-string names the arm but not the missing/extra indices.
    p0 = tmp_path / "a0.csv"
    p1 = tmp_path / "a1.csv"
    p2 = tmp_path / "a2.csv"
    write_csv(p0, ROWS_A0)
    write_csv(p1, [(99, "Teeth", "A", True)])  # different index set entirely
    write_csv(p2, ROWS_A2)
    monkeypatch.setattr(s54, "ARMS", [
        ("no primer", str(p0)),
        ("+ OPG primer", str(p1)),
        ("+ primer + v2 exemplars", str(p2)),
    ])
    try:
        s54.main()
        assert False, "expected an AssertionError"
    except AssertionError as e:
        assert "index mismatch in + OPG primer" in str(e)
