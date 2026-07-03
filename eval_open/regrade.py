"""
Re-grade EXISTING cached model answers with a different judge.

Reads an answers CSV (index, question, gt, answer) produced by run_real_eval.py and
grades each answer under both rubrics with the chosen judge. The answers are NOT
regenerated — only the judge changes, so any score difference is attributable to
the judge alone.

  python -m eval_open.regrade --answers results/open/real30_natural_answers.csv \
      --judge gpt-4o --model gpt-4o --tag coord30_gpt4o
"""
import argparse
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

from eval_open.judges import grade
from eval_open.rubrics import build_grading_prompt

RUBRICS = ["original", "rephrased"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, help="answers CSV to re-grade")
    ap.add_argument("--judge", default="gpt-4o", choices=["gpt-4o", "gemini", "claude"])
    ap.add_argument("--model", default=None, help="judge model id override")
    ap.add_argument("--tag", required=True, help="output tag -> results/open/regrade_<tag>_scores.csv")
    ap.add_argument("--delay", type=float, default=0.0, help="pause (s) between grades to stay under rate limits")
    args = ap.parse_args()

    ans = pd.read_csv(args.answers)
    out = f"results/open/regrade_{args.tag}_scores.csv"
    records, done = [], set()
    if Path(out).exists():
        records = pd.read_csv(out).to_dict("records")
        done = {(int(r["index"]), r["rubric"]) for r in records}

    print(f"re-grading {len(ans)} answers x {len(RUBRICS)} rubrics with "
          f"judge={args.judge} model={args.model or 'default'} -> {out}")
    for _, a in ans.iterrows():
        for rubric in RUBRICS:
            if (int(a["index"]), rubric) in done:
                continue
            prompt = build_grading_prompt(a["question"], a["gt"], a["answer"], rubric)
            score, raw = grade(prompt, judge=args.judge, model=args.model, delay=args.delay)
            records.append({"index": int(a["index"]), "rubric": rubric,
                            "score": score, "judge_raw": str(raw).replace("\n", " ")[:60]})
            print(f"  idx={a['index']}/{rubric} -> {score}")
            pd.DataFrame(records).to_csv(out, index=False)

    piv = pd.DataFrame(records).pivot_table(index="index", columns="rubric", values="score").dropna()
    o, r = piv["original"], piv["rephrased"]
    d = r - o
    try:
        p = wilcoxon(d, zero_method="zsplit").pvalue if d.abs().sum() else 1.0
    except ValueError:
        p = 1.0
    print("\n" + "=" * 56)
    print(f"judge={args.judge}({args.model or 'default'})  source={args.answers}  n={len(piv)}")
    print(f"  original : {o.mean()*100:5.1f}%")
    print(f"  rephrased: {r.mean()*100:5.1f}%")
    print(f"  delta    : {d.mean()*100:+.1f} pts   Wilcoxon p={p:.3g}")
    print("=" * 56)


if __name__ == "__main__":
    main()
