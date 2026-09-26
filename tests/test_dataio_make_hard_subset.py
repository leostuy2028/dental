# Tests for dataio/make_hard_subset.py
# Builds the "hard" subset: items BOTH gpt-4o AND gemini-2.5-flash got wrong (on the
# shuffled key), sampled deterministically with a seeded random.Random.

import pandas as pd
import pytest

import dataio.make_hard_subset as m


def _write_inputs(tmp_path):
    shuf_p = tmp_path / "shuf.parquet"
    gpt_p = tmp_path / "gpt.csv"
    gem_p = tmp_path / "gem.csv"

    sh = pd.DataFrame({
        "index": [1, 2, 3, 4, 5],
        "option1": ["a"] * 5, "option2": ["b"] * 5, "option3": ["c"] * 5, "option4": ["d"] * 5,
        "answer": ["A", "B", "C", "D", "A"],
        "category": ["C1", "C2", "C3", "C4", "C5"],
        "question": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    })
    sh.to_parquet(shuf_p, index=False)

    # gpt-4o gets 1, 2, 4 wrong; gemini-2.5 gets 1, 4 wrong -> intersection {1, 4}
    pd.DataFrame({"index": [1, 2, 3, 4, 5], "correct": [False, False, True, False, True]}).to_csv(gpt_p, index=False)
    pd.DataFrame({"index": [1, 2, 4, 5], "correct": [False, True, False, True]}).to_csv(gem_p, index=False)
    return shuf_p, gpt_p, gem_p


def test_wrong_indices_reads_the_indices_marked_incorrect(tmp_path):
    _, gpt_p, _ = _write_inputs(tmp_path)
    assert m.wrong_indices(str(gpt_p), "") == {1, 2, 4}


def test_main_selects_items_wrong_for_both_models(tmp_path, monkeypatch, capsys):
    shuf_p, gpt_p, gem_p = _write_inputs(tmp_path)
    out_parquet = tmp_path / "out.parquet"
    out_manifest = tmp_path / "man.csv"

    monkeypatch.setattr(m, "SHUFFLED", str(shuf_p))
    monkeypatch.setattr(m, "GPT4O", str(gpt_p))
    monkeypatch.setattr(m, "GEM25", str(gem_p))
    monkeypatch.setattr(m, "OUT_PARQUET", str(out_parquet))
    monkeypatch.setattr(m, "OUT_MANIFEST", str(out_manifest))
    monkeypatch.setattr(m, "N", 2)

    m.main()

    man = pd.read_csv(out_manifest)
    assert sorted(man["index"].tolist()) == [1, 4]
    assert list(man.columns) == ["index", "category", "answer_key_shuffled", "question"]

    sub = pd.read_parquet(out_parquet)
    assert sorted(sub["index"].tolist()) == [1, 4]
    assert sub.isna().sum().sum() == 0

    printed = capsys.readouterr().out
    assert "gpt-4o wrong 3 | gemini-2.5 wrong 2 | intersection(hard) 2" in printed
    assert "wrote" in printed and "2 items (seed 20260707)" in printed


def test_main_raises_when_not_enough_hard_items(tmp_path, monkeypatch):
    shuf_p, gpt_p, gem_p = _write_inputs(tmp_path)
    monkeypatch.setattr(m, "SHUFFLED", str(shuf_p))
    monkeypatch.setattr(m, "GPT4O", str(gpt_p))
    monkeypatch.setattr(m, "GEM25", str(gem_p))
    monkeypatch.setattr(m, "OUT_PARQUET", str(tmp_path / "out2.parquet"))
    monkeypatch.setattr(m, "OUT_MANIFEST", str(tmp_path / "man2.csv"))
    monkeypatch.setattr(m, "N", 3)   # only 2 hard items exist

    with pytest.raises(AssertionError, match="only 2 hard items, need 3"):
        m.main()


def test_main_rejects_a_shuffled_source_with_nan_options(tmp_path, monkeypatch):
    shuf_p = tmp_path / "shuf_nan.parquet"
    sh = pd.DataFrame({
        "index": [1, 2], "option1": [None, "a"], "option2": ["b", "b"],
        "option3": ["c", "c"], "option4": ["d", "d"], "answer": ["A", "B"],
        "category": ["C1", "C2"], "question": ["Q1", "Q2"],
    })
    sh.to_parquet(shuf_p, index=False)
    monkeypatch.setattr(m, "SHUFFLED", str(shuf_p))

    with pytest.raises(AssertionError, match="NaN"):
        m.main()
