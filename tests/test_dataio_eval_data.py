# Tests for dataio/eval_data.py
# read_closed() is the one place closed-ended parquets get loaded. It refuses (SystemExit)
# to hand back a dataframe that still has NaN in the option columns, so a run can never
# silently use the raw un-normalized data.

import pandas as pd
import pytest

from dataio.eval_data import read_closed, OPTION_COLUMNS


def test_reads_a_clean_file_with_no_nan(tmp_path):
    df = pd.DataFrame({
        "option1": ["a", "b"], "option2": ["c", "d"],
        "option3": ["e", "f"], "option4": ["g", "h"],
    })
    path = tmp_path / "clean.parquet"
    df.to_parquet(path)

    out = read_closed(str(path))

    assert out.shape == (2, 4)


def test_raises_when_an_option_cell_is_nan(tmp_path):
    df = pd.DataFrame({
        "option1": ["a", None], "option2": ["c", "d"],
        "option3": ["e", "f"], "option4": ["g", "h"],
    })
    path = tmp_path / "dirty.parquet"
    df.to_parquet(path)

    with pytest.raises(SystemExit) as exc:
        read_closed(str(path))

    assert "1 NaN option cell" in str(exc.value)
    assert "dataio/convert_to_none.py" in str(exc.value)


def test_file_with_no_option_columns_is_returned_unchanged(tmp_path):
    # CURRENT BEHAVIOR: if none of the OPTION_COLUMNS are present, the NaN check is
    # skipped entirely and the dataframe is returned as-is.
    df = pd.DataFrame({"foo": [1, 2]})
    path = tmp_path / "no_options.parquet"
    df.to_parquet(path)

    out = read_closed(str(path))

    assert out.shape == (2, 1)
    assert list(out.columns) == ["foo"]


def test_option_columns_constant_is_the_expected_four():
    assert OPTION_COLUMNS == ["option1", "option2", "option3", "option4"]
