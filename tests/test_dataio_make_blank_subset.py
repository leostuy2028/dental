# Tests for dataio/make_blank_subset.py
# Carves the fixed 38 blank-option items out of the canonical closed-ended parquet. The
# 38 indices (BLANK_CORRECT / BLANK_DISTRACTOR) are hardcoded in the source, and main()
# asserts len(sub) == 38, so the test's tiny canonical file must contain exactly those 38
# indices (built with a loop, not by hand) for the success path.

import pandas as pd
import pytest

import dataio.make_blank_subset as m


def _make_canonical_df():
    """38 rows matching BLANK_INDICES. BLANK_CORRECT rows key the None option (A);
    BLANK_DISTRACTOR rows have a None option that is NOT the answer (None is at B)."""
    rows = []
    for idx in sorted(m.BLANK_CORRECT):
        rows.append({"index": idx, "category": "Cat", "question": f"Q{idx}", "answer": "A",
                     "option1": "None", "option2": "foo", "option3": "bar", "option4": "baz"})
    for idx in sorted(m.BLANK_DISTRACTOR):
        rows.append({"index": idx, "category": "Cat", "question": f"Q{idx}", "answer": "A",
                     "option1": "real", "option2": "None", "option3": "bar", "option4": "baz"})
    return pd.DataFrame(rows)


def test_main_splits_none_as_correct_vs_none_as_distractor(tmp_path, monkeypatch, capsys):
    df = _make_canonical_df()
    canon = tmp_path / "canon.parquet"
    df.to_parquet(canon, index=False)
    out_parquet = tmp_path / "out.parquet"
    out_manifest = tmp_path / "man.csv"

    monkeypatch.setattr(m, "CANONICAL", str(canon))
    monkeypatch.setattr(m, "OUT_PARQUET", str(out_parquet))
    monkeypatch.setattr(m, "OUT_MANIFEST", str(out_manifest))

    m.main()

    sub = pd.read_parquet(out_parquet)
    assert len(sub) == 38

    man = pd.read_csv(out_manifest).set_index("index")
    for idx in m.BLANK_CORRECT:
        assert man.loc[idx, "correct_is_none"] == True
        assert man.loc[idx, "none_position"] == "A"
    for idx in m.BLANK_DISTRACTOR:
        assert man.loc[idx, "correct_is_none"] == False
        assert man.loc[idx, "none_position"] == "B"

    printed = capsys.readouterr().out
    assert "wrote" in printed and "38 items" in printed
    assert "correct answer is the 'None' option: 32" in printed
    assert "'None' is a distractor only:         6" in printed


def test_main_rejects_canonical_source_with_nan(tmp_path, monkeypatch):
    df = _make_canonical_df()
    df.loc[df["index"] == sorted(m.BLANK_CORRECT)[0], "option1"] = None
    canon = tmp_path / "canon.parquet"
    df.to_parquet(canon, index=False)

    monkeypatch.setattr(m, "CANONICAL", str(canon))
    monkeypatch.setattr(m, "OUT_PARQUET", str(tmp_path / "out.parquet"))
    monkeypatch.setattr(m, "OUT_MANIFEST", str(tmp_path / "man.csv"))

    with pytest.raises(AssertionError, match="NaN"):
        m.main()


def test_main_self_check_catches_a_blank_item_missing_its_none_option(tmp_path, monkeypatch):
    df = _make_canonical_df()
    bad_idx = sorted(m.BLANK_CORRECT)[0]
    df.loc[df["index"] == bad_idx, "option1"] = "not none"  # no "None" anywhere in this row
    canon = tmp_path / "canon.parquet"
    df.to_parquet(canon, index=False)

    monkeypatch.setattr(m, "CANONICAL", str(canon))
    monkeypatch.setattr(m, "OUT_PARQUET", str(tmp_path / "out.parquet"))
    monkeypatch.setattr(m, "OUT_MANIFEST", str(tmp_path / "man.csv"))

    with pytest.raises(AssertionError, match="without a 'None' option"):
        m.main()
