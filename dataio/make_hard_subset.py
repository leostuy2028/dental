"""
Build the "hard-50" subset for the §5.4 thinking-budget sweep.

Goal: a set of genuinely hard items on which to ask "does more (hidden) thinking
help on multiple choice?" — chosen so the sweep is not circular. We define "hard"
from models that are NOT in the sweep: an item is hard if BOTH gpt-4o AND
gemini-2.5-flash answered it WRONG (revised/coax prompt, shuffled key). The sweep
model, gemini-3.5-flash, is not used in selection, so its budget-0 score on this
set is an honest baseline (not conditioned to be all-wrong).

Everything is on the SHUFFLED (position-debiased) key, so the B-skew freebie is
gone and any accuracy gain from thinking is genuine reading, not letter drift.

Selection is deterministic: fixed source CSVs + fixed seed. Difficulty is a
property of the image/question, so the intersection is ~188 items; we sample 50.

Run:   python dataio/make_hard_subset.py
Reads: data/closed_ended_shuffled.parquet  (frozen, position-debiased)
       results/closed_ended/position_bias/gpt-4o-2024-11-20__coax-direct-k0__shuffled__n491.csv
       results/closed_ended/position_bias/gemini-2.5-flash__coax-direct-k0__shuffled__n491.csv
Writes:data/closed_ended_hard50_shuffled.parquet
       results/closed_ended/cot_length/hard50_manifest.csv
"""
import os
import random
import pandas as pd

SHUFFLED = "data/closed_ended_shuffled.parquet"
GPT4O = "results/closed_ended/position_bias/gpt-4o-2024-11-20__coax-direct-k0__shuffled__n491.csv"
GEM25 = "results/closed_ended/position_bias/gemini-2.5-flash__coax-direct-k0__shuffled__n491.csv"
OUT_PARQUET = "data/closed_ended_hard50_shuffled.parquet"
OUT_MANIFEST = "results/closed_ended/cot_length/hard50_manifest.csv"
OPTS = ["option1", "option2", "option3", "option4"]
SEED = 20260707
N = 50


def wrong_indices(csv_path, repo):
    d = pd.read_csv(os.path.join(repo, csv_path))
    return set(d.loc[~d["correct"].astype(bool), "index"].astype(int))


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_parquet(os.path.join(repo, SHUFFLED)).set_index("index", drop=False)
    assert int(df[OPTS].isna().sum().sum()) == 0, "shuffled source has NaN options"

    w4 = wrong_indices(GPT4O, repo)
    w25 = wrong_indices(GEM25, repo)
    hard = sorted(w4 & w25)  # wrong for BOTH → genuinely hard, 3.5-flash uninvolved
    print(f"gpt-4o wrong {len(w4)} | gemini-2.5 wrong {len(w25)} | intersection(hard) {len(hard)}")
    assert len(hard) >= N, f"only {len(hard)} hard items, need {N}"

    picked = sorted(random.Random(SEED).sample(hard, N))
    sub = df.loc[picked].reset_index(drop=True).copy()
    assert len(sub) == N and int(sub[OPTS].isna().sum().sum()) == 0

    os.makedirs(os.path.join(repo, os.path.dirname(OUT_PARQUET)) or ".", exist_ok=True)
    sub.to_parquet(os.path.join(repo, OUT_PARQUET), index=False)

    man = sub[["index", "category", "answer", "question"]].copy()
    man = man.rename(columns={"answer": "answer_key_shuffled"})
    os.makedirs(os.path.join(repo, os.path.dirname(OUT_MANIFEST)), exist_ok=True)
    man.sort_values("index").to_csv(os.path.join(repo, OUT_MANIFEST), index=False)

    print(f"wrote {OUT_PARQUET}: {N} items (seed {SEED})")
    print(f"wrote {OUT_MANIFEST}")
    print(f"  indices: {picked}")
    print(f"  key dist on the 50: {sub['answer'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
