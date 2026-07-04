"""
Prepare the closed-ended eval sets.

Removes the 38 malformed rows that carry a blank (NaN) option — 32 of which
have the *correct* option blank (unanswerable; see RESEARCH_PLAN §3.7) — to
produce a CLEAN set, and emits position-debiased (shuffled-key) versions.
Naming lets any run cite exactly which set it used:

  closed_ended.parquet                 491  original   whole set (as-is)
  closed_ended_shuffled.parquet        491  shuffled   whole, debiased  [pre-existing]
  closed_ended_clean.parquet           453  original   blank-option rows removed
  closed_ended_clean_shuffled.parquet  453  shuffled   clean + debiased  <-- primary eval set

Run from the repo root:  python prepare_datasets.py
"""
import random
import pandas as pd

# ---------------------------------------------------------------------------
# DATA PROVENANCE (recorded 2026-07-04)
# Source: HuggingFace dataset  OralGPT/MMOral-OPG-Bench  (MIT-licensed)
#   https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench
# `closed_ended.parquet` and `open_ended.parquet` were built from that dataset via
# `datasets.load_dataset` (download cached at data/OralGPT___mm_oral-opg-bench).
# Because HuggingFace may not host it forever, the two large ORIGINAL parquets are committed
# to this repo via **Git LFS** (see .gitattributes), so the original, unshuffled data is
# preserved alongside the code and does not depend on the dataset staying online.
# The derived sets written below (clean / clean-shuffled) are committed as normal git blobs
# and are fully regenerable by running this script against closed_ended.parquet.
# ---------------------------------------------------------------------------

SRC = "data/closed_ended.parquet"
CLEAN = "data/closed_ended_clean.parquet"
CLEAN_SHUF = "data/closed_ended_clean_shuffled.parquet"

OPTS = ["option1", "option2", "option3", "option4"]
L2I = {"A": 0, "B": 1, "C": 2, "D": 3}
I2L = {0: "A", 1: "B", 2: "C", 3: "D"}


def shuffle_row(row, seed):
    """Permute the four options (seeded per row) and re-point the answer.
    NaN-safe: tracks the correct option by position, not by value, so a blank
    distractor cannot break it. (The clean set has no blanks; this keeps the
    function correct if ever run on the whole set.)"""
    options = [row["option1"], row["option2"], row["option3"], row["option4"]]
    correct_pos = L2I[row["answer"]]
    order = list(range(4))
    random.Random(seed).shuffle(order)
    new_options = [options[i] for i in order]
    new_answer = I2L[order.index(correct_pos)]
    return dict(zip(OPTS, new_options), answer=new_answer)


def main():
    df = pd.read_parquet(SRC)
    blank = df[OPTS].isna().any(axis=1)
    corr_blank = df.apply(lambda r: pd.isna(r[OPTS[L2I[r["answer"]]]]), axis=1)
    dropped = df[blank]["index"].tolist()

    print(f"whole set: {len(df)} rows")
    print(f"  blank-option rows:        {int(blank.sum())}  -> DROPPED")
    print(f"    of which correct blank: {int(corr_blank.sum())} (unanswerable)")
    print(f"    blank distractor only:  {int((blank & ~corr_blank).sum())}")
    print(f"  dropped indices: {dropped}")

    clean = df[~blank].reset_index(drop=True).copy()
    clean.to_parquet(CLEAN, index=False)
    print(f"\nwrote {CLEAN}: {len(clean)} rows")

    # shuffle the clean set (seed per row by its benchmark index -> reproducible)
    shuf = clean.copy()
    updates = [shuffle_row(r, seed=int(r["index"])) for _, r in clean.iterrows()]
    for col in OPTS + ["answer"]:
        shuf[col] = [u[col] for u in updates]

    # verify: the correct option TEXT is unchanged by the shuffle
    for i in range(len(clean)):
        a = clean.iloc[i][OPTS[L2I[clean.iloc[i]["answer"]]]]
        b = shuf.iloc[i][OPTS[L2I[shuf.iloc[i]["answer"]]]]
        assert a == b, f"answer text changed at row {i}: {a!r} != {b!r}"
    shuf.to_parquet(CLEAN_SHUF, index=False)
    print(f"wrote {CLEAN_SHUF}: {len(shuf)} rows")

    print("\nanswer-key distribution (%):")
    def dist(s):
        vc = s.value_counts(normalize=True).reindex(list("ABCD")).fillna(0) * 100
        return "  ".join(f"{k}={v:4.1f}" for k, v in vc.items())
    print(f"  whole   original : {dist(df['answer'])}")
    print(f"  clean   original : {dist(clean['answer'])}")
    print(f"  clean   shuffled : {dist(shuf['answer'])}   <- should be ~25 each")


if __name__ == "__main__":
    main()
