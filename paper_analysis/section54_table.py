"""
Section 5.4 — "Does giving the model prior knowledge help?"
Regenerates the full-benchmark results table + paired stats from the three stored
CSVs. No API calls: reads only committed result files, so the table is reproducible
on demand.

Three arms, all gemini-3.5-flash / coax / direct / think=0 / temp=0 / full-res, on
the shuffled key (data/closed_ended_shuffled.parquet, n=491). Only the context varies:
  A0  no primer            (position_bias/...__coax-direct-k0__shuffled__n491.csv)
  A1  + OPG text primer    (reference/opg_primer.txt)
  A2  + primer + 12 visual exemplars (reference/exemplars_v2.json: DENTEX + Zenodo)

Run:  python paper_analysis/section54_table.py
"""
import os
import sys
from math import comb
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
KC = "results/closed_ended/knowledge_context"
PB = "results/closed_ended/position_bias"

ARMS = [
    ("no primer",        f"{PB}/gemini-3.5-flash__coax-direct-k0__shuffled__n491.csv"),
    ("+ OPG primer",     f"{KC}/gemini-3.5-flash__coax-direct-ctx-opgprimer__shuffled__n491.csv"),
    ("+ primer + v2 exemplars", f"{KC}/gemini-3.5-flash__coax-direct-primerv1+visualex-v2__shuffled__n491.csv"),
]
DIMS = ["Teeth", "Patho", "HisT", "Jaw", "SumRec"]


def load(path):
    df = pd.read_csv(path)[["index", "category", "predicted", "correct"]]
    return df.set_index("index")


def mcnemar_exact(before, after):
    """Two-sided exact (binomial) McNemar on the discordant pairs of two paired
    boolean-correct series aligned by index. Returns (rescued, broke, p)."""
    j = pd.DataFrame({"b": before, "a": after})
    rescued = int(((~j.b) & (j.a)).sum())   # 0 -> 1
    broke = int(((j.b) & (~j.a)).sum())     # 1 -> 0
    n = rescued + broke
    if n == 0:
        return rescued, broke, 1.0
    k = min(rescued, broke)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)
    return rescued, broke, p


def main():
    data = {}
    for name, path in ARMS:
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            return
        data[name] = load(path)

    # sanity: same 491 items, aligned
    idx0 = set(data[ARMS[0][0]].index)
    for name, _ in ARMS:
        assert set(data[name].index) == idx0, f"index mismatch in {name}"
        unp = int(data[name]["predicted"].isna().sum())
        print(f"[{name:26s}] n={len(data[name])}  unparseable={unp}")
    print()

    print("=== Section 5.4: overall accuracy (shuffled key, n=491) ===")
    for name, _ in ARMS:
        print(f"  {name:26s} {data[name]['correct'].mean()*100:6.2f}%")

    print("\n=== Paired McNemar (exact, 2-sided) ===")
    pairs = [(ARMS[0][0], ARMS[1][0]), (ARMS[1][0], ARMS[2][0]), (ARMS[0][0], ARMS[2][0])]
    for a, b in pairs:
        r, br, p = mcnemar_exact(data[a]["correct"], data[b]["correct"])
        print(f"  {a:26s} -> {b:26s}  rescued {r:3d} / broke {br:3d}  net {r-br:+4d}  p={p:.4g}")

    print("\n=== Per-dimension accuracy (item may carry multiple tags) ===")
    hdr = "  {:8s}" + "".join(f"{n:>16s}" for n, _ in ARMS)
    print(hdr.format("dim", ""))
    base = data[ARMS[0][0]]["category"]
    for dim in DIMS:
        mask = base.str.contains(dim)
        row = f"  {dim:8s}"
        for name, _ in ARMS:
            sub = data[name][mask.values]
            row += f"{sub['correct'].mean()*100:14.1f}% "
        row += f"   (n={int(mask.sum())})"
        print(row)

    # targeted (HisT or Jaw) vs control — the specificity check for the v2 exemplars
    print("\n=== Specificity check: HisT/Jaw (exemplar-covered) vs neither ===")
    tag = base
    targ = tag.str.contains("HisT") | tag.str.contains("Jaw")
    for label, mask in [("HisT OR Jaw (targeted)", targ), ("neither (control)", ~targ)]:
        row = f"  {label:24s}"
        for name, _ in ARMS:
            sub = data[name][mask.values]
            row += f"{sub['correct'].mean()*100:8.1f}%"
        row += f"   (n={int(mask.sum())})"
        print(row)


if __name__ == "__main__":
    main()
