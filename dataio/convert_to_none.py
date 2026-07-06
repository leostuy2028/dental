"""
ONE-TIME conversion: raw MMOral closed-ended data (blank options as NaN)  ->  the
canonical working set (blank options as the string "None").

This is the ONLY file in the codebase that is allowed to read the raw NaN dataset. Every
other script, harness, and derivative reads `data/closed_ended.parquet` (the None-normalized
canonical produced here) and must NEVER touch the raw again.

Why "None": the benchmark's own dataset class fills blank options this way at load time —
`MMOral_OPG_CLOSED.post_build` runs `self.data[col].fillna('None')` on the option columns
(MMOral-Bench-EvalKit/vlmeval/dataset/mmoral.py). Both the prompt it shows the model and the
answer parser's `index2ans` then read "None". Storing NaN and letting `str(NaN)` become "nan"
was a reproduction bug on our side; this conversion removes NaN at the source, once.

  data/closed_ended_raw.parquet   (FROZEN, raw HF download, blanks = NaN)  --fillna('None')-->
  data/closed_ended.parquet       (canonical, blanks = "None", the source of truth)

Run once:  python dataio/convert_to_none.py
"""
import os
import pandas as pd

RAW = "data/closed_ended_raw.parquet"          # the ONLY reference to the raw NaN file
CANONICAL = "data/closed_ended.parquet"        # canonical None-normalized working set
OPTION_COLUMNS = ["option1", "option2", "option3", "option4"]


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(repo, RAW)
    if not os.path.exists(raw_path):
        raise SystemExit(f"raw file not found: {RAW}. It must be the frozen HF download.")

    df = pd.read_parquet(raw_path)
    n_nan_before = int(df[OPTION_COLUMNS].isna().sum().sum())
    for col in OPTION_COLUMNS:                  # mirrors MMOral_OPG_CLOSED.post_build
        df[col] = df[col].fillna("None")
    n_nan_after = int(df[OPTION_COLUMNS].isna().sum().sum())

    assert n_nan_after == 0, f"still {n_nan_after} NaN option cells after fillna"
    df.to_parquet(os.path.join(repo, CANONICAL), index=False)

    n_none = int((df[OPTION_COLUMNS] == "None").sum().sum())
    print(f"raw   : {RAW}  ({len(df)} rows, {n_nan_before} NaN option cells)")
    print(f"canon : {CANONICAL}  ({len(df)} rows, 0 NaN, {n_none} 'None' option cells)")
    print("done: canonical is None-normalized. All derivatives must build from this file.")


if __name__ == "__main__":
    main()
