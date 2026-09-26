# Tests for paper_analysis/region_crop.py
# P13 region-crop check: does splitting the panoramic into left/right crops help the
# model, paired against the same shuffled items? Prints overall + paired flips + by
# dimension + by question type. No files are written (stdout only).
import os

import pandas as pd

import region_crop as rc


# ---------- mcnemar ----------

def test_mcnemar_is_one_when_no_discordant_pairs():
    assert rc.mcnemar(0, 0) == 1.0


def test_mcnemar_matches_hand_counted_two_sided_binomial():
    # n=2, k=min(1,1)=1: sum(comb(2,0..1)) = 1+2 = 3 -> 2*3/4 = 1.5 -> capped at 1.0
    assert rc.mcnemar(1, 1) == 1.0
    # n=3, k=min(0,3)=0: sum(comb(3,0..0)) = 1 -> 2*1/8 = 0.25
    assert rc.mcnemar(0, 3) == 0.25


# ---------- flips ----------

def test_flips_counts_rescued_and_broken_items():
    a = pd.DataFrame({"correct": [True, False, True, False]})
    b = pd.DataFrame({"correct": [True, True, False, False]})
    mask = pd.Series([True, True, True, True])
    n, fa, ca, resc, broke = rc.flips(a, b, mask)
    assert n == 4
    assert fa == 50.0   # 2/4 correct in a
    assert ca == 50.0   # 2/4 correct in b
    assert resc == 1    # idx1: wrong in a, right in b
    assert broke == 1   # idx2: right in a, wrong in b


def test_flips_respects_the_mask():
    a = pd.DataFrame({"correct": [True, False, True, False]})
    b = pd.DataFrame({"correct": [True, True, False, False]})
    mask = pd.Series([True, True, False, False])
    n, fa, ca, resc, broke = rc.flips(a, b, mask)
    assert n == 2
    assert fa == 50.0
    assert ca == 100.0
    assert resc == 1
    assert broke == 0


# ---------- main ----------

def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_main_prints_overall_by_dimension_and_by_question_type(tmp_path, monkeypatch, capsys):
    full_csv = tmp_path / "full.csv"
    crop_csv = tmp_path / "crop.csv"
    rows_full = [
        {"index": 0, "question": "which tooth is affected?", "category": "Teeth", "correct": 0},
        {"index": 1, "question": "is there any pathology present?", "category": "Patho", "correct": 1},
        {"index": 2, "question": "what historical treatment is shown?", "category": "HisT", "correct": 1},
        {"index": 3, "question": "describe the jaw region", "category": "Jaw", "correct": 0},
        {"index": 4, "question": "how many teeth are missing overall?", "category": "SumRec", "correct": 1},
        {"index": 5, "question": "which teeth are impacted?", "category": "Teeth", "correct": 0},
    ]
    rows_crop = [
        {"index": 0, "question": "which tooth is affected?", "category": "Teeth", "correct": 1},
        {"index": 1, "question": "is there any pathology present?", "category": "Patho", "correct": 1},
        {"index": 2, "question": "what historical treatment is shown?", "category": "HisT", "correct": 0},
        {"index": 3, "question": "describe the jaw region", "category": "Jaw", "correct": 0},
        {"index": 4, "question": "how many teeth are missing overall?", "category": "SumRec", "correct": 1},
        {"index": 5, "question": "which teeth are impacted?", "category": "Teeth", "correct": 0},
    ]
    _write_csv(full_csv, rows_full)
    _write_csv(crop_csv, rows_crop)

    # FULL/CROP are joined onto `repo` with os.path.join(repo, FULL); an absolute
    # replacement makes os.path.join discard `repo` entirely, so we don't need to
    # fake up the whole repo layout or touch the real one.
    monkeypatch.setattr(rc, "FULL", str(full_csv))
    monkeypatch.setattr(rc, "CROP", str(crop_csv))

    rc.main()

    out = capsys.readouterr().out
    assert "P13 region crops" in out
    assert "6 items (paired)" in out
    # overall: full 3/6=50.0%, crop 3/6=50.0%
    assert "overall: full 50.0%  ->  +crops 50.0%   (Δ +0.0)" in out
    # paired flips: idx0 rescued (0->1), idx2 broke (1->0) -> net 0
    assert "paired flips: crops rescued 1, broke 1 (net +0)" in out
    assert "McNemar p=1.000" in out
    assert "by DIMENSION" in out
    for dim in rc.DIMS:
        assert dim in out
    assert "by QUESTION TYPE" in out
    assert "localization/counting (should help)" in out
    assert "other (should not hurt)" in out
