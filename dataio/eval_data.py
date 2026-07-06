"""
Single entry point for loading a closed-ended eval parquet, with a hard guard against the
raw NaN dataset. Every closed-ended harness reads through here so a run can never accidentally
use `data/closed_ended_raw.parquet` (or any un-normalized file): the model would then see
"nan" for a blank option instead of the benchmark's "None". Use the None-normalized canonical
`data/closed_ended.parquet` (and its derivatives) built by dataio/convert_to_none.py.
"""
import pandas as pd

OPTION_COLUMNS = ["option1", "option2", "option3", "option4"]


def read_closed(path):
    df = pd.read_parquet(path)
    cols = [c for c in OPTION_COLUMNS if c in df.columns]
    if cols:
        n_nan = int(df[cols].isna().sum().sum())
        if n_nan:
            raise SystemExit(
                f"{path} has {n_nan} NaN option cell(s). Closed-ended runs must use the "
                f"None-normalized canonical set (data/closed_ended.parquet or a derivative), "
                f"never the raw NaN file. Regenerate with: python dataio/convert_to_none.py")
    return df
