"""
Analyze the E-open-1 isolation run.

Headline: the coordinate bonus = score(coords) - score(prose), paired per item,
reported per judge x rubric x ref_type, with a Wilcoxon signed-rank test.

  python eval_open/analyze.py [results/open/isolation.csv]
"""
import sys
import pandas as pd
from scipy.stats import wilcoxon

IN = sys.argv[1] if len(sys.argv) > 1 else "results/open/isolation.csv"


def main():
    df = pd.read_csv(IN)
    # pivot variants side by side, paired on (index, judge, rubric)
    wide = df.pivot_table(index=["index", "ref_type", "judge", "rubric"],
                          columns="variant", values="score").reset_index()
    wide = wide.dropna(subset=["prose", "coords"])
    wide["bonus"] = wide["coords"] - wide["prose"]

    print(f"paired items: {len(wide)}  (from {IN})\n")
    print(f"{'judge':8s} {'rubric':10s} {'ref_type':10s} {'n':>4s} "
          f"{'prose':>6s} {'coords':>6s} {'bonus':>7s} {'p(wilcox)':>10s}")
    print("-" * 72)
    for judge in sorted(wide.judge.unique()):
        for ref in ["coord_ref", "prose_ref"]:
            for rubric in ["original", "rephrased"]:
                sub = wide[(wide.judge == judge) & (wide.ref_type == ref)
                           & (wide.rubric == rubric)]
                if sub.empty:
                    continue
                bonus = sub["bonus"]
                try:
                    p = wilcoxon(bonus, zero_method="zsplit").pvalue if bonus.abs().sum() else 1.0
                except ValueError:
                    p = 1.0
                print(f"{judge:8s} {rubric:10s} {ref:10s} {len(sub):4d} "
                      f"{sub.prose.mean():6.3f} {sub.coords.mean():6.3f} "
                      f"{bonus.mean():+7.3f} {p:10.3g}")
        print()

    # headline: coord_ref bonus, original vs rephrased, averaged across judges
    print("=== HEADLINE: coordinate bonus on coord_ref items (mean over judges) ===")
    cr = wide[wide.ref_type == "coord_ref"]
    for rubric in ["original", "rephrased"]:
        b = cr[cr.rubric == rubric]["bonus"]
        print(f"  {rubric:10s}: mean bonus = {b.mean():+.3f}  (n={len(b)})")
    orig = cr[cr.rubric == "original"]["bonus"].mean()
    reph = cr[cr.rubric == "rephrased"]["bonus"].mean()
    print(f"  bias removed by rephrase: {orig - reph:+.3f} "
          f"({(1 - reph/orig)*100:.0f}% of the original bonus)" if orig else "")


if __name__ == "__main__":
    main()
