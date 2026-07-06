"""
Build the derivative closed-ended eval sets, STRICTLY from the canonical None-normalized
source `data/closed_ended.parquet` (produced once by dataio/convert_to_none.py). This script
must never read the raw NaN file.

Produces:
  data/closed_ended_shuffled.parquet   491 rows, options permuted per-row (seeded by index)
                                        -> position-debiased key; correct-option TEXT preserved.

The canonical source already has blanks as "None" (no NaN), so the shuffle is a plain,
position-tracked permutation with no NaN special-casing.

Run:  python dataio/prepare_datasets.py
"""
import os
import random
import pandas as pd

CANONICAL = "data/closed_ended.parquet"          # None-normalized source of truth
SHUFFLED = "data/closed_ended_shuffled.parquet"
OPTS = ["option1", "option2", "option3", "option4"]
L2I = {"A": 0, "B": 1, "C": 2, "D": 3}
I2L = {0: "A", 1: "B", 2: "C", 3: "D"}


def shuffle_row(row, seed):
    """Permute the four options (seeded per row) and re-point the answer by POSITION."""
    options = [row[c] for c in OPTS]
    correct_pos = L2I[row["answer"]]
    order = list(range(4))
    random.Random(seed).shuffle(order)
    new_options = [options[i] for i in order]
    new_answer = I2L[order.index(correct_pos)]
    return dict(zip(OPTS, new_options), answer=new_answer)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_parquet(os.path.join(repo, CANONICAL))
    assert int(df[OPTS].isna().sum().sum()) == 0, \
        "canonical source has NaN options — run dataio/convert_to_none.py first"

    shuf = df.copy()
    updates = [shuffle_row(r, seed=int(r["index"])) for _, r in df.iterrows()]
    for col in OPTS + ["answer"]:
        shuf[col] = [u[col] for u in updates]

    # verify the correct option TEXT is unchanged by the shuffle
    for i in range(len(df)):
        a = df.iloc[i][OPTS[L2I[df.iloc[i]["answer"]]]]
        b = shuf.iloc[i][OPTS[L2I[shuf.iloc[i]["answer"]]]]
        assert a == b, f"answer text changed at row {i}: {a!r} != {b!r}"
    assert int(shuf[OPTS].isna().sum().sum()) == 0, "shuffled set has NaN"
    shuf.to_parquet(os.path.join(repo, SHUFFLED), index=False)

    def dist(s):
        vc = s.value_counts(normalize=True).reindex(list("ABCD")).fillna(0) * 100
        return "  ".join(f"{k}={v:4.1f}" for k, v in vc.items())
    print(f"wrote {SHUFFLED}: {len(shuf)} rows, 0 NaN")
    print(f"  key (canonical) : {dist(df['answer'])}")
    print(f"  key (shuffled)  : {dist(shuf['answer'])}   <- flatter")


if __name__ == "__main__":
    main()
