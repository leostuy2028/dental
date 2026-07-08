"""
P13 region-crop test (§5.4): does splitting the wide panoramic into left/right half
crops help the model? Paired full-image vs full+crops on the same shuffled items.

The whole point is the STRATIFIED read: crops should help the localization/counting
questions (Ceiling 2, fixable) but not subtle pathology (Ceiling 1, modality floor),
and must not hurt the rest. Prints overall + paired flips + by-dimension + by
question-type (localization vs not). No API.

Usage: python paper_analysis/region_crop.py
"""
import os
import re
import sys
import pandas as pd
from math import comb

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIR = "results/closed_ended/region_crop"
FULL = f"{DIR}/gemini-3.5-flash__coax-direct-full__shuffled__idx0-149.csv"
CROP = f"{DIR}/gemini-3.5-flash__coax-direct-crops__shuffled__idx0-149.csv"
DIMS = ("Teeth", "Patho", "HisT", "Jaw", "SumRec")
# localization / counting questions = the ones region crops should most help
LOC = r"which tooth|which teeth|how many|number of|numbering|which site|which region|which of the following (tooth|teeth|wisdom)"


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def flips(a, b, mask):
    ac = a.loc[mask, "correct"].astype(bool)
    bc = b.loc[mask, "correct"].astype(bool)
    resc = int((~ac & bc).sum())
    broke = int((ac & ~bc).sum())
    return len(ac), 100 * ac.mean(), 100 * bc.mean(), resc, broke


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    a = pd.read_csv(os.path.join(repo, FULL)).set_index("index")
    b = pd.read_csv(os.path.join(repo, CROP)).set_index("index")
    idx = sorted(set(a.index) & set(b.index))
    a, b = a.loc[idx], b.loc[idx]
    q = a["question"].astype(str).str.lower()
    cat = a["category"].astype(str)

    n, fa, ca, resc, broke = flips(a, b, a.index.to_series().astype(bool) | True)
    print("=" * 62)
    print(f"P13 region crops — gemini-3.5-flash, shuffled, {n} items (paired)")
    print("=" * 62)
    print(f"overall: full {fa:.1f}%  ->  +crops {ca:.1f}%   (Δ {ca-fa:+.1f})")
    print(f"paired flips: crops rescued {resc}, broke {broke} (net {resc-broke:+d}), McNemar p={mcnemar(resc,broke):.3f}")

    print("\nby DIMENSION (does it help where info is present, not where modality is the floor?):")
    print(f"  {'dim':7}{'n':>4}{'full':>7}{'crops':>7}{'resc':>6}{'broke':>6}")
    for d in DIMS:
        m = cat.str.contains(d)
        if m.sum():
            n_, f_, c_, r_, br_ = flips(a, b, m)
            print(f"  {d:7}{n_:>4}{f_:>6.0f}%{c_:>6.0f}%{r_:>6}{br_:>6}")

    print("\nby QUESTION TYPE:")
    loc = q.str.contains(LOC, regex=True)
    for name, m in [("localization/counting (should help)", loc), ("other (should not hurt)", ~loc)]:
        n_, f_, c_, r_, br_ = flips(a, b, m)
        print(f"  {name:38} n={n_:3}  full {f_:.0f}% -> crops {c_:.0f}%  (rescued {r_}, broke {br_})")


if __name__ == "__main__":
    main()
