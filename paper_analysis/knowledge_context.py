"""
§5.5 / E11 — does an in-context OPG reading primer help? (go/no-go screen)

Reads the paired no-context vs +primer runs on the enriched selection and reports
the decisive signal: on items the model gets WRONG without context, how often does
the primer RESCUE them (wrong->right), versus how often it BREAKS correct items
(right->wrong). Rescue >> break = real knowledge transfer; rescue ~= break = just
answer-churn (a 4-option perturbation lands right ~1/3 of the time by chance).

Decomposition is by TODAY's no-context result (the drift-safe paired baseline), not
the day-old heuristic used only to enrich the sample. Also breaks out by dimension
and the FDI/counting "canary" (questions whose answer is literally in the primer).

Usage: python paper_analysis/knowledge_context.py
"""
import os
import sys
import pandas as pd
from math import comb

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIR = "results/closed_ended/knowledge_context"
NOCTX = f"{DIR}/gemini-3.5-flash__coax-direct-noctx__e11sel__n130.csv"
CTX = f"{DIR}/gemini-3.5-flash__coax-direct-ctx-opgprimer__e11sel__n130.csv"
SEL = f"{DIR}/e11_selection.csv"
DIMS = ("Teeth", "Patho", "HisT", "Jaw")


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (100 * (c - h) / d, 100 * (c + h) / d)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    a = pd.read_csv(os.path.join(repo, NOCTX)).set_index("index")
    b = pd.read_csv(os.path.join(repo, CTX)).set_index("index")
    sel = pd.read_csv(os.path.join(repo, SEL)).set_index("index")
    idx = sorted(set(a.index) & set(b.index))
    a, b, sel = a.loc[idx], b.loc[idx], sel.loc[idx]
    ac, bc = a["correct"].astype(bool), b["correct"].astype(bool)

    print("=" * 64)
    print(f"E11 knowledge-context screen — gemini-3.5-flash, {len(idx)} items (paired)")
    print("=" * 64)
    print(f"overall acc: no-context {100*ac.mean():.1f}%  ->  +primer {100*bc.mean():.1f}%")
    rescued = int((~ac & bc).sum())   # wrong w/o context, right with it
    broke = int((ac & ~bc).sum())     # right w/o context, wrong with it
    print(f"paired flips: rescued {rescued}, broke {broke}, net {rescued-broke:+d}, "
          f"McNemar p={mcnemar(rescued, broke):.3f}")

    # THE decisive rates, decomposed by today's no-context correctness
    nwrong, nright = int((~ac).sum()), int(ac.sum())
    rescue_rate = rescued / nwrong if nwrong else 0
    break_rate = broke / nright if nright else 0
    rlo, rhi = wilson(rescued, nwrong)
    blo, bhi = wilson(broke, nright)
    print(f"\n>>> RESCUE rate (of {nwrong} misses): {100*rescue_rate:.0f}% [{rlo:.0f}-{rhi:.0f}]")
    print(f">>> BREAK  rate (of {nright} correct): {100*break_rate:.0f}% [{blo:.0f}-{bhi:.0f}]")
    print(f">>> SIGNAL = rescue - break = {100*(rescue_rate-break_rate):+.0f} pts "
          f"(churn null ~= 0; clearly positive => dig deeper)")

    # by dimension (on the misses)
    print("\nrescue by dimension (misses only):")
    for d in DIMS:
        m = sel["category"].str.contains(d) & ~ac
        n = int(m.sum())
        r = int((m & bc).sum())
        if n:
            print(f"  {d:6} rescued {r}/{n} ({100*r/n:.0f}%)")

    # canary: FDI/counting questions (answer is literally in the primer)
    can = sel["is_canary"].astype(bool)
    cw = can & ~ac
    print(f"\ncanary (FDI/count Qs whose answer is in the primer): "
          f"misses {int(cw.sum())}, rescued {int((cw & bc).sum())} "
          f"-> if this is ~0, the model isn't using the context at all")


if __name__ == "__main__":
    main()
