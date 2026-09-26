# Tests for dataio/prepare_datasets.py
# shuffle_row() permutes the four options of one row (seeded by index) and re-points the
# answer letter to follow the correct option's new position. main() applies this to the
# whole canonical parquet and writes the shuffled derivative.

import pandas as pd
import pytest

import dataio.prepare_datasets as m


def test_shuffle_row_permutes_options_and_keeps_answer_pointing_at_the_same_text():
    row = {"option1": "a", "option2": "b", "option3": "c", "option4": "d", "answer": "B", "index": 7}

    out = m.shuffle_row(row, seed=7)

    # the answer text originally at option2 ("b") must still be findable at the new answer letter
    new_letter = out["answer"]
    assert out[m.OPTS[m.L2I[new_letter]]] == "b"
    # every original option value is still present, just reordered
    assert sorted(out[c] for c in m.OPTS) == ["a", "b", "c", "d"]


def test_shuffle_row_is_deterministic_for_a_given_seed():
    row = {"option1": "a", "option2": "b", "option3": "c", "option4": "d", "answer": "B", "index": 7}

    out1 = m.shuffle_row(row, seed=7)
    out2 = m.shuffle_row(row, seed=7)

    assert out1 == out2


def test_main_writes_a_shuffled_parquet_with_same_answer_text(tmp_path, monkeypatch, capsys):
    canon = tmp_path / "canon.parquet"
    shuffled = tmp_path / "shuf.parquet"
    df = pd.DataFrame({
        "index": [0, 1, 2],
        "option1": ["a", "e", "i"], "option2": ["b", "f", "j"],
        "option3": ["c", "g", "k"], "option4": ["d", "h", "l"],
        "answer": ["A", "B", "C"],
    })
    df.to_parquet(canon, index=False)
    monkeypatch.setattr(m, "CANONICAL", str(canon))
    monkeypatch.setattr(m, "SHUFFLED", str(shuffled))

    m.main()

    out = pd.read_parquet(shuffled)
    assert len(out) == 3
    assert out.isna().sum().sum() == 0
    # the correct option's TEXT must be unchanged for every row
    orig_text = [df.iloc[i][m.OPTS[m.L2I[df.iloc[i]["answer"]]]] for i in range(3)]
    new_text = [out.iloc[i][m.OPTS[m.L2I[out.iloc[i]["answer"]]]] for i in range(3)]
    assert orig_text == new_text

    printed = capsys.readouterr().out
    assert "wrote" in printed and "3 rows, 0 NaN" in printed
    assert "flatter" in printed


def test_main_rejects_a_canonical_source_with_nan_options(tmp_path, monkeypatch):
    canon = tmp_path / "canon.parquet"
    df = pd.DataFrame({
        "index": [0, 1],
        "option1": ["a", None], "option2": ["b", "f"],
        "option3": ["c", "g"], "option4": ["d", "h"],
        "answer": ["A", "B"],
    })
    df.to_parquet(canon, index=False)
    monkeypatch.setattr(m, "CANONICAL", str(canon))
    monkeypatch.setattr(m, "SHUFFLED", str(tmp_path / "shuf.parquet"))

    with pytest.raises(AssertionError, match="NaN"):
        m.main()
