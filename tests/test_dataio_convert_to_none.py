# Tests for dataio/convert_to_none.py
# One-time conversion script: reads the raw parquet (blank options = NaN) and writes a
# canonical parquet where blank options are the string "None". main() uses module-level
# path constants (RAW, CANONICAL); we monkeypatch those to absolute tmp_path files so the
# os.path.join(repo, RAW) call in main() just returns our absolute path unchanged.

import pandas as pd
import pytest

import dataio.convert_to_none as m


def test_main_fills_nan_options_with_the_string_none(tmp_path, monkeypatch, capsys):
    raw_path = tmp_path / "raw.parquet"
    canon_path = tmp_path / "canon.parquet"
    df = pd.DataFrame({
        "option1": ["a", None],
        "option2": ["c", "d"],
        "option3": [None, "f"],
        "option4": ["g", "h"],
    })
    df.to_parquet(raw_path)

    monkeypatch.setattr(m, "RAW", str(raw_path))
    monkeypatch.setattr(m, "CANONICAL", str(canon_path))

    m.main()

    out = pd.read_parquet(canon_path)
    assert out["option1"].tolist() == ["a", "None"]
    assert out["option3"].tolist() == ["None", "f"]
    assert out.isna().sum().sum() == 0

    printed = capsys.readouterr().out
    assert "2 NaN option cells" in printed
    assert "0 NaN, 2 'None' option cells" in printed
    assert "done: canonical is None-normalized" in printed


def test_main_raises_when_raw_file_is_missing(tmp_path, monkeypatch):
    missing = tmp_path / "nope.parquet"
    monkeypatch.setattr(m, "RAW", str(missing))

    with pytest.raises(SystemExit) as exc:
        m.main()

    assert "raw file not found" in str(exc.value)


def test_main_asserts_all_nan_gone_after_fillna(tmp_path, monkeypatch):
    # a file with no NaN at all should still convert cleanly (n_nan_before == 0)
    raw_path = tmp_path / "raw_clean.parquet"
    canon_path = tmp_path / "canon_clean.parquet"
    df = pd.DataFrame({
        "option1": ["a", "b"], "option2": ["c", "d"],
        "option3": ["e", "f"], "option4": ["g", "h"],
    })
    df.to_parquet(raw_path)
    monkeypatch.setattr(m, "RAW", str(raw_path))
    monkeypatch.setattr(m, "CANONICAL", str(canon_path))

    m.main()

    out = pd.read_parquet(canon_path)
    assert out.equals(df)
