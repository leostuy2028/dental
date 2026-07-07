"""
Build the enriched selection for the §5.5 / E11 knowledge-context screen.

Rationale (see RESEARCH_PLAN / DENTIST_AUDIT-style notes): a random slice wastes
calls on items with no headroom. We instead enrich for items gemini-3.5-flash
currently MISSES (on the shuffled key), in the dimensions the primer addresses
(Teeth/Patho/HisT/Jaw, skip SumRec), plus a small currently-CORRECT control
stratum to measure the answer-churn / break rate. FDI-numbering / counting
questions (whose answers are literally in the primer) are tagged as the "canary".

Selection is deterministic (fixed baseline CSV + seed). The day-old baseline is a
selection heuristic ONLY; the paired comparison re-runs no-context today.

Run:   python dataio/make_e11_subset.py
Reads: results/closed_ended/position_bias/gemini-3.5-flash__coax-direct-k0__shuffled__n491.csv
       data/closed_ended_shuffled.parquet
Writes:results/closed_ended/knowledge_context/e11_selection.csv  (indices + labels; committed)
       data/closed_ended_e11_sel.parquet                          (subset for the harness; gitignored, regenerable)
"""
import os
import random
import pandas as pd

BASE = "results/closed_ended/position_bias/gemini-3.5-flash__coax-direct-k0__shuffled__n491.csv"
SHUF = "data/closed_ended_shuffled.parquet"
OUT_SEL = "results/closed_ended/knowledge_context/e11_selection.csv"
OUT_PARQUET = "data/closed_ended_e11_sel.parquet"
SEED = 20260707
DIMS = ("Teeth", "Patho", "HisT", "Jaw")
CANARY = "how many|number of|numbering|which tooth|which teeth|fdi"
N_MISS_CANARY, N_MISS_OTHER, N_CORRECT = 20, 80, 30


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = pd.read_csv(os.path.join(repo, BASE)).set_index("index")
    sh = pd.read_parquet(os.path.join(repo, SHUF)).set_index("index", drop=False)
    base["rel"] = base["category"].map(lambda c: any(d in str(c) for d in DIMS))
    canary_idx = set(sh.index[sh["question"].str.lower().str.contains(CANARY)])

    rng = random.Random(SEED)
    miss = list(base.index[(~base["correct"].astype(bool)) & base["rel"]])
    corr = list(base.index[(base["correct"].astype(bool)) & base["rel"]])
    miss_canary = [i for i in miss if i in canary_idx]
    miss_other = [i for i in miss if i not in canary_idx]
    sel_miss = set(rng.sample(miss_canary, min(N_MISS_CANARY, len(miss_canary)))) \
        | set(rng.sample(miss_other, min(N_MISS_OTHER, len(miss_other))))
    sel_corr = set(rng.sample(corr, N_CORRECT))
    sel = sorted(sel_miss | sel_corr)

    man = pd.DataFrame({"index": sel})
    man["category"] = man["index"].map(base["category"])
    man["heuristic_correct"] = man["index"].map(base["correct"].astype(bool))  # selection label only
    man["is_canary"] = man["index"].isin(canary_idx)
    os.makedirs(os.path.join(repo, os.path.dirname(OUT_SEL)), exist_ok=True)
    man.to_csv(os.path.join(repo, OUT_SEL), index=False)

    sh.loc[sel].reset_index(drop=True).to_parquet(os.path.join(repo, OUT_PARQUET), index=False)
    print(f"selected {len(sel)}: misses {len(sel_miss)} + correct {len(sel_corr)} | canary {int(man['is_canary'].sum())}")
    print(f"wrote {OUT_SEL} + {OUT_PARQUET}")


if __name__ == "__main__":
    main()
