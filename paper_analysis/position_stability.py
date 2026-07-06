"""
GENERATOR for: PAPER_DRAFT.md §5.3 — content-stable / letter-stable / neither.

For each model, pair its answers on the full 491 (original key) against the 491-shuffled set
(same items, options reordered) and classify every item:
  content-stable : picked the same OPTION TEXT (tooth) in both -> position-robust
  letter-stable  : picked the same LETTER (now a different tooth) -> answered by position
  neither        : different letter AND different tooth -> inconsistent

Same items, temperature 0, so a flip is a real response to the reordering, not noise. Reads
run CSVs (predicted letter per item) + the clean & clean-shuffled parquets (to map a letter
to the option text it points at). No API calls.

Run:   python paper_analysis/position_stability.py
Writes: paper_analysis/_generated/position_stability_table.md, position_stability.values.json
"""
import os
import sys
import json
import datetime
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from clients.parsing import is_api_failure  # noqa: E402
OUT_DIR = os.path.join(HERE, "_generated")
L2OPT = {"A": "option1", "B": "option2", "C": "option3", "D": "option4"}

CE = "results/closed_ended"
PB = f"{CE}/position_bias"
# model -> (original-key run CSV, SHUFFLED-key run CSV), both on the frozen 491, revised
# (coax) prompt, direct, thinking off, temperature 0 — the same paired runs as Table 5.4.
MODELS = {
    "GPT-4o": (f"{CE}/gpt-4o-2024-11-20__coax-direct-k0__whole__n491.csv",
               f"{PB}/gpt-4o-2024-11-20__coax-direct-k0__shuffled__n491.csv"),
    "Gemini-2.5-flash": (f"{PB}/gemini-2.5-flash__coax-direct-k0__whole__n491.csv",
                         f"{PB}/gemini-2.5-flash__coax-direct-k0__shuffled__n491.csv"),
    "Gemini-3.5-flash": (f"{PB}/gemini-3.5-flash__coax-direct-k0__whole__n491.csv",
                         f"{PB}/gemini-3.5-flash__coax-direct-k0__shuffled__n491.csv"),
}

# Prompt-axis check (GPT-4o only): does the revised vs original prompt change the measured
# stability? Both now on the full 491. Not the main figure; run with --compare-prompts.
PROMPT_COMPARE = {
    "GPT-4o (revised)": (f"{CE}/gpt-4o-2024-11-20__coax-direct-k0__whole__n491.csv",
                         f"{PB}/gpt-4o-2024-11-20__coax-direct-k0__shuffled__n491.csv"),
    "GPT-4o (original)": (f"{CE}/gpt-4o-2024-11-20__faithful-direct-k0__whole__n491.csv",
                          f"{PB}/gpt-4o-2024-11-20__faithful-direct-k0__shuffled__n491.csv"),
}


def _rd(p):
    return pd.read_csv(os.path.join(REPO, *p.split("/")))


def _is_failed(raw):
    """A row with no real model output: empty/NaN reply or the API-failure sentinel.
    Such a row's `predicted` is meaningless (the faithful parser random-guesses on an
    empty reply), so it must be excluded from a content-stability count."""
    if is_api_failure(raw):
        return True
    return (not isinstance(raw, str)) or raw.strip() in ("", "nan")


def classify(clean_csv, shuf_csv, clean_opts, shuf_opts):
    cl_df = _rd(clean_csv).set_index("index")
    sh_df = _rd(shuf_csv).set_index("index")
    cl = cl_df["predicted"]
    sh = sh_df["predicted"]
    idx = [i for i in shuf_opts.index if i in cl.index and i in sh.index]
    # exclude items where either run has no real model output (empty/failed API call) —
    # otherwise a random-guessed fallback letter pollutes the stability count.
    valid = [i for i in idx
             if not _is_failed(cl_df.loc[i, "raw_response"])
             and not _is_failed(sh_df.loc[i, "raw_response"])]
    excluded = len(idx) - len(valid)
    if excluded:
        print(f"  [{os.path.basename(shuf_csv)}] excluded {excluded} item(s) with a "
              f"failed/empty reply; scoring on n={len(valid)}")
    idx = valid
    # accuracy on the SAME paired items, clean key vs shuffled key (the coarse view)
    acc_clean = round(100 * float(cl_df.loc[idx, "correct"].mean()), 1)
    acc_shuf = round(100 * float(sh_df.loc[idx, "correct"].mean()), 1)
    buckets = {"content": 0, "letter": 0, "neither": 0}
    for i in idx:
        lo, ls = cl.get(i), sh.get(i)
        if not (isinstance(lo, str) and isinstance(ls, str) and lo in L2OPT and ls in L2OPT):
            buckets["neither"] += 1
            continue
        otext = str(clean_opts.loc[i, L2OPT[lo]])
        stext = str(shuf_opts.loc[i, L2OPT[ls]])
        if otext == stext:
            buckets["content"] += 1
        elif lo == ls:
            buckets["letter"] += 1
        else:
            buckets["neither"] += 1
    n = len(idx)
    pct = {k: round(100 * v / n, 1) for k, v in buckets.items()}
    return {"n": n, "excluded_failed": excluded, "count": buckets, "pct": pct,
            "acc_clean": acc_clean, "acc_shuffled": acc_shuf,
            "acc_drop": round(acc_clean - acc_shuf, 1)}


def main():
    import sys
    models = PROMPT_COMPARE if "--compare-prompts" in sys.argv else MODELS
    tag = "prompt_compare" if "--compare-prompts" in sys.argv else "position_stability"
    # map letters -> option text from the canonical (None) set + its shuffle. These contain
    # every run's indices; on the shared indices the options are identical to the retired
    # clean sets, so results are unchanged.
    clean_opts = pd.read_parquet(os.path.join(REPO, "data/closed_ended.parquet")).set_index("index")
    shuf_opts = pd.read_parquet(os.path.join(REPO, "data/closed_ended_shuffled.parquet")).set_index("index")
    vals = {}
    for name, (cc, sc) in models.items():
        if not os.path.exists(os.path.join(REPO, *sc.split("/"))):
            print(f"  [skip {name}] shuffled run not found yet: {sc}")
            continue
        vals[name] = classify(cc, sc, clean_opts, shuf_opts)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    prov = (f"<!-- GENERATED by paper_analysis/position_stability.py on {stamp}. "
            f"Do not hand-edit; run `python paper_analysis/position_stability.py`. -->")
    lines = [prov,
             "| Model | Acc orig | Acc shuffled | Acc drop | Content-stable | Letter-stable | Neither |",
             "|:--|--:|--:|--:|--:|--:|--:|"]
    for name, v in vals.items():
        p = v["pct"]
        lines.append(f"| {name} | {v['acc_clean']}% | {v['acc_shuffled']}% | {v['acc_drop']} | "
                     f"{p['content']}% | {p['letter']}% | {p['neither']}% |")
    table = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, f"{tag}_table.md"), "w", encoding="utf-8") as f:
        f.write(table)
    out = dict(vals)
    out["_generator"] = "paper_analysis/position_stability.py"
    out["_generated_utc"] = stamp
    with open(os.path.join(OUT_DIR, f"{tag}.values.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(table)
    for name, v in vals.items():
        print(f"  {name}: n={v['n']}  content {v['pct']['content']}%  "
              f"letter {v['pct']['letter']}%  neither {v['pct']['neither']}%")
    print(f"\nwrote {OUT_DIR}/{tag}_table.md + .values.json")


if __name__ == "__main__":
    main()
