"""
§5.4 — Does more (hidden) thinking help on hard multiple-choice items?

Reads the gemini-3.5-flash thinking-budget sweep on the 50 hard items
(gpt-4o AND gemini-2.5 both wrong, shuffled/debiased key) and reports, with zero
new API calls:
  - accuracy vs thinking budget, each with a 95% Wilson interval
  - the paired budget-0 -> budget-8192 comparison (McNemar): rescued vs broken items
  - the predicted-letter distribution per budget (does any gain drift toward a letter?)

Because the sweep is on the shuffled key, the always-one-letter freebie is gone, so
a gain has to be genuine reading. Selection used only gpt-4o and gemini-2.5, so
gemini-3.5's budget-0 score is an honest baseline.

Run:   python paper_analysis/thinking_hard.py
Writes:paper_analysis/_generated/thinking_hard_table.md
       paper_analysis/_generated/thinking_hard.values.json
"""
import os
import glob
import json
import datetime
import pandas as pd

DIR = "results/closed_ended/cot_length"
PATTERN = "gemini-3.5-flash__direct-think{b}__hard50-shuffled__n50.csv"
BUDGETS = [("0", "off"), ("512", "512"), ("2048", "2048"), ("8192", "8192"), ("-1", "dynamic")]
OUT_DIR = "paper_analysis/_generated"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (100 * (c - h) / d, 100 * (c + h) / d)


def mcnemar_exact(b, c):
    """Two-sided exact binomial p on the discordant pairs (b vs c)."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load(repo, b):
    path = os.path.join(repo, DIR, PATTERN.format(b=b))
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).set_index("index", drop=False)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs = {b: load(repo, b) for b, _ in BUDGETS}
    have = {b: d for b, d in runs.items() if d is not None}
    if not have:
        raise SystemExit(f"no sweep CSVs found under {DIR}")

    rows = []
    per_budget = {}
    for b, label in BUDGETS:
        d = runs.get(b)
        if d is None:
            rows.append(f"| {label} | *[pending]* | | |")
            continue
        n = len(d)
        k = int(d["correct"].astype(bool).sum())
        lo, hi = wilson(k, n)
        acc = 100 * k / n
        letters = {L: int((d["predicted"] == L).sum()) for L in "ABCD"}
        ld = "/".join(str(letters[L]) for L in "ABCD")
        rows.append(f"| {label} | {acc:.1f}% [{lo:.0f}-{hi:.0f}] | {ld} | {n} |")
        per_budget[b] = {"label": label, "n": n, "correct": k, "acc": round(acc, 1),
                         "ci": [round(lo, 1), round(hi, 1)], "letters": letters}

    # paired budget-0 vs budget-8192 (the amount-axis endpoints), same items
    paired = None
    if runs.get("0") is not None and runs.get("8192") is not None:
        a, z = runs["0"], runs["8192"]
        common = sorted(set(a["index"]) & set(z["index"]))
        a, z = a.loc[common], z.loc[common]
        ac, zc = a["correct"].astype(bool), z["correct"].astype(bool)
        rescued = int((~ac & zc).sum())   # wrong at 0, right at 8K
        broke = int((ac & ~zc).sum())     # right at 0, wrong at 8K
        p = mcnemar_exact(rescued, broke)
        paired = {"n": len(common), "acc0": round(100 * ac.mean(), 1),
                  "acc8192": round(100 * zc.mean(), 1),
                  "rescued": rescued, "broke": broke, "net": rescued - broke,
                  "mcnemar_p": round(p, 4)}

    os.makedirs(os.path.join(repo, OUT_DIR), exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    md = [f"<!-- GEN: thinking_hard (generator: paper_analysis/thinking_hard.py; "
          f"source: {DIR}/gemini-3.5-flash__direct-think*__hard50-shuffled__n50.csv; "
          f"regenerate: python paper_analysis/thinking_hard.py) -->",
          "| Thinking budget | Accuracy [95% CI] | A/B/C/D | n |",
          "|:--|--:|:--|--:|", *rows, "<!-- /GEN: thinking_hard -->"]
    if paired:
        md += ["", f"**Paired budget 0 -> 8192 (same {paired['n']} items):** "
               f"{paired['acc0']}% -> {paired['acc8192']}%; "
               f"thinking rescued {paired['rescued']}, broke {paired['broke']} "
               f"(net {paired['net']:+d}); McNemar exact p = {paired['mcnemar_p']}."]
    with open(os.path.join(repo, OUT_DIR, "thinking_hard_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(repo, OUT_DIR, "thinking_hard.values.json"), "w") as f:
        json.dump({"_generator": "paper_analysis/thinking_hard.py",
                   "_source_dir": DIR, "_generated_utc": stamp,
                   "per_budget": per_budget, "paired_0_vs_8192": paired}, f, indent=2)

    print("\n".join(md))
    print(f"\nwrote {OUT_DIR}/thinking_hard_table.md + .values.json")


if __name__ == "__main__":
    main()
