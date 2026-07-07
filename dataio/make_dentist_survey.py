"""
Select items for the dentist ground-truth PROBE (§7: is the benchmark's key
actually correct?). This is a learning-maximizing probe, not an unbiased base-rate
sample: it concentrates the dentist's ~50 minutes on the items most likely to be
mislabeled, and spans an information gradient so we can measure the key-error rate
per bucket and test whether models predict key errors.

Buckets are defined by how the three frontier models (gpt-4o, gemini-2.5-flash,
gemini-3.5-flash) answer, on the SHUFFLED (position-debiased) key. Using the
shuffled key is deliberate: on the original B-skewed key, "models disagree with the
key" is contaminated by position bias (models all avoiding B looks like a key-error
flag but is not). On the shuffled key the correct answer is spread across positions,
and because all three models saw the identical shuffled options, models agreeing on
a letter means they agree on the same option CONTENT — the clean, bias-free signal.

  T1  all 3 models agree on a NON-key answer, in BOTH orderings (order-invariant)
      -> strongest "the key may be wrong" flag. Reviewed in full (all 25).
  T2  2 of 3 agree on a non-key answer (shuffled)          -> moderate suspects.
  T3  models split among themselves (shuffled)             -> genuinely ambiguous/hard.
  C   all 3 agree WITH the key (shuffled)                  -> near-certainly correct (control).

Blank/"None" items get NO special handling: the earlier "blank-answer flaw" was a
retracted reproduction bug (RESEARCH_PLAN §3.7, cut from the paper in v0.9), so they
are ordinary items here and appear only if a bucket happens to draw them.

Run:   python dataio/make_dentist_survey.py
Writes:results/dentist_audit/survey_manifest.csv  (our records: buckets + keys)
"""
import os
import re
import random
import pandas as pd
from collections import Counter

CLOSED = "data/closed_ended.parquet"
OPEN = "data/open_ended.parquet"
SHUF = {
    "gpt4o": "results/closed_ended/position_bias/gpt-4o-2024-11-20__coax-direct-k0__shuffled__n491.csv",
    "g25": "results/closed_ended/position_bias/gemini-2.5-flash__coax-direct-k0__shuffled__n491.csv",
    "g35": "results/closed_ended/position_bias/gemini-3.5-flash__coax-direct-k0__shuffled__n491.csv",
}
ORIG = {
    "gpt4o": "results/closed_ended/gpt-4o-2024-11-20__coax-direct-k0__whole__n491.csv",
    "g25": "results/closed_ended/position_bias/gemini-2.5-flash__coax-direct-k0__whole__n491.csv",
    "g35": "results/closed_ended/position_bias/gemini-3.5-flash__coax-direct-k0__whole__n491.csv",
}
OUT = "results/dentist_audit/survey_manifest.csv"
SEED = 20260707
N_T1, N_T2, N_T3, N_C = 25, 15, 10, 10          # closed = 60
N_OPEN_PROSE, N_OPEN_COORD = 15, 0               # open = 15 (coordinate items dropped: a
#                                                  dentist can't judge pixel boxes by eye)


def is_coord(s):
    s = str(s)
    return ("box_2d" in s) or bool(re.search(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]", s)) \
        or s.strip().startswith(("[{", '{"', "{'"))


def load(p, repo):
    return pd.read_csv(os.path.join(repo, p)).set_index("index")


def buckets(dd, idx):
    kk = dd["gpt4o"]["answer"]
    out = {}
    for i in idx:
        k = kk[i]
        preds = [dd["gpt4o"].loc[i, "predicted"], dd["g25"].loc[i, "predicted"], dd["g35"].loc[i, "predicted"]]
        top, topn = Counter(preds).most_common(1)[0]
        nkey = sum(p == k for p in preds)
        if topn == 3 and top != k:
            out[i] = "T1"
        elif topn == 2 and top != k and nkey < 2:
            out[i] = "T2"
        elif topn == 3 and top == k:
            out[i] = "C"
        elif nkey >= 2:
            out[i] = "B"
        else:
            out[i] = "T3"
    return pd.Series(out)


def take(rng, pool, n):
    pool = sorted(pool)
    return set(rng.sample(pool, min(n, len(pool))))


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rng = random.Random(SEED)
    closed = pd.read_parquet(os.path.join(repo, CLOSED)).set_index("index", drop=False)
    dsh = {k: load(v, repo) for k, v in SHUF.items()}
    dor = {k: load(v, repo) for k, v in ORIG.items()}
    idx = dsh["gpt4o"].index

    # Exclude items with a "None" option: they are the retracted §3.7 blank-answer
    # template (no special handling), and a literal "None" option would collide with
    # the viewer's "None of these options is correct" choice.
    opt_cols = ["option1", "option2", "option3", "option4"]
    none_opt = set(closed.index[closed[opt_cols].eq("None").any(axis=1)])
    valid = set(idx) - none_opt

    tsh = buckets(dsh, idx)
    tor = buckets(dor, idx)
    t1 = take(rng, (set(tsh[tsh == "T1"].index) & set(tor[tor == "T1"].index)) & valid, N_T1)  # order-invariant
    # keep 60 closed total: if T1 has fewer than N_T1 valid items, T2 backfills the gap
    t2 = take(rng, (set(tsh[tsh == "T2"].index) & valid) - t1, 60 - len(t1) - N_T3 - N_C)
    t3 = take(rng, (set(tsh[tsh == "T3"].index) & valid) - t1 - t2, N_T3)
    cc = take(rng, (set(tsh[tsh == "C"].index) & valid) - t1 - t2 - t3, N_C)

    rows = []
    for bucket, idxs in [("T1", t1), ("T2", t2), ("T3", t3), ("C", cc)]:
        for i in sorted(idxs):
            r = closed.loc[i]
            rows.append({"task_type": "closed", "bucket": bucket, "index": int(i),
                         "category": r["category"], "question": r["question"],
                         "A": r["option1"], "B": r["option2"], "C": r["option3"], "D": r["option4"],
                         "answer_key": r["answer"], "reference_answer": ""})

    op = pd.read_parquet(os.path.join(repo, OPEN)).set_index("index", drop=False)
    op["coord"] = op["answer"].map(is_coord)
    op = op[op["category"] != "Report"]  # Report is a task-type bucket, not a diagnostic dimension
    prose = op[~op["coord"]]
    prose_patho = set(prose.index[prose["category"].str.contains("Patho")])
    op_p = take(rng, prose_patho, 6) | take(rng, set(prose.index) - prose_patho, N_OPEN_PROSE - 6)
    op_c = take(rng, set(op.index[op["coord"]]), N_OPEN_COORD)
    for bucket, idxs in [("prose", op_p), ("coord", op_c)]:
        for i in sorted(idxs):
            r = op.loc[i]
            rows.append({"task_type": "open", "bucket": bucket, "index": int(i),
                         "category": r["category"], "question": r["question"],
                         "A": "", "B": "", "C": "", "D": "",
                         "answer_key": "", "reference_answer": r["answer"]})

    man = pd.DataFrame(rows)
    man["_blk"] = (man["task_type"] == "open").astype(int)
    man = man.sample(frac=1, random_state=SEED).sort_values("_blk", kind="stable").drop(columns="_blk")
    man.insert(0, "survey_order", range(1, len(man) + 1))
    # unique id: closed and open datasets each number from 0, so index alone is ambiguous
    man.insert(1, "item_id", man["task_type"].str[0] + man["index"].astype(str))

    os.makedirs(os.path.join(repo, os.path.dirname(OUT)), exist_ok=True)
    man.to_csv(os.path.join(repo, OUT), index=False)
    print(f"wrote {OUT}: {len(man)} items")
    print("  closed buckets:", man[man.task_type == "closed"]["bucket"].value_counts().to_dict())
    print("  open buckets:  ", man[man.task_type == "open"]["bucket"].value_counts().to_dict())
    print("  unique item_id check (should be 0):", int(man["item_id"].duplicated().sum()))
    print("  within-closed dup index (should be 0):",
          int(man[man.task_type == "closed"]["index"].duplicated().sum()))


if __name__ == "__main__":
    main()
