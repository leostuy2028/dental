"""
E-open-1 (REAL, faithful prompt): 30 coordinate-sensitive items, graded under both rubrics.

Pipeline (no fabricated data, no forced coordinates):
  Phase 1  the model ANSWERS each item's X-ray using the VERBATIM VLMEvalKit
           open-ended inference prompt (answerer.py) — natural output only.
           -> results/open/real30_natural_answers.csv  (full answer text saved)
  Phase 2  the judge grades each real answer under {original, rephrased}.
           -> results/open/real30_natural_scores.csv

Both phases are resumable. The 30 items are the deterministic coord_ref sample
(coord_ref rows of predictions.parquet, .sample(30, random_state=0)).

  python -m eval_open.run_real_eval
"""
import argparse
import pandas as pd
from pathlib import Path

from data_loader import decode_image
from eval_open.answerer import answer_question, INFERENCE_PROMPT
from eval_open.judges import grade
from eval_open.rubrics import build_grading_prompt

DATA = "data/open_ended.parquet"
PREDS = "eval_open/predictions.parquet"      # used to reproduce a deterministic sample
ANSWER_MODEL = "gemini-3.5-flash"
JUDGE = ("gemini", "gemini-3.5-flash")       # (judge name, model) — per "just use gemini 3.5"
RUBRICS = ["original", "rephrased"]

# set by main() from CLI
ANS_OUT = SCORE_OUT = None


def get_item_indices(ref_type, n, seed):
    """Deterministic sample of `n` items of the given ref_type ('coord_ref',
    'prose_ref', or 'all')."""
    preds = pd.read_parquet(PREDS)
    if ref_type != "all":
        preds = preds[preds.ref_type == ref_type]
    n = min(n, len(preds))
    return sorted(preds.sample(n, random_state=seed)["index"].tolist())


def phase1_answers(items):
    records, done = [], set()
    if Path(ANS_OUT).exists():
        records = pd.read_csv(ANS_OUT).to_dict("records")
        done = {int(r["index"]) for r in records}
    for _, it in items.iterrows():
        if int(it["index"]) in done:
            continue
        img = decode_image(it["image"])
        ans = answer_question(img, it["question"], model=ANSWER_MODEL)
        records.append({"index": int(it["index"]), "question": it["question"],
                        "gt": str(it["answer"]), "answer": ans})
        print(f"  [answer] idx={it['index']}: {ans[:70].replace(chr(10),' ')}...")
        pd.DataFrame(records).to_csv(ANS_OUT, index=False)
    return pd.DataFrame(records)


def phase2_grade(answers):
    records, done = [], set()
    if Path(SCORE_OUT).exists():
        records = pd.read_csv(SCORE_OUT).to_dict("records")
        done = {(int(r["index"]), r["rubric"]) for r in records}
    jname, jmodel = JUDGE
    for _, a in answers.iterrows():
        for rubric in RUBRICS:
            if (int(a["index"]), rubric) in done:
                continue
            prompt = build_grading_prompt(a["question"], a["gt"], a["answer"], rubric)
            score, raw = grade(prompt, judge=jname, model=jmodel)
            records.append({"index": int(a["index"]), "rubric": rubric,
                            "score": score, "judge_raw": str(raw).replace("\n", " ")[:60]})
            print(f"  [grade] idx={a['index']}/{rubric} -> {score}")
            pd.DataFrame(records).to_csv(SCORE_OUT, index=False)
    return pd.DataFrame(records)


def report(scores):
    from scipy.stats import wilcoxon
    piv = scores.pivot_table(index="index", columns="rubric", values="score")
    o, r = piv["original"], piv["rephrased"]
    d = r - o
    try:
        p = wilcoxon(d, zero_method="zsplit").pvalue if d.abs().sum() else 1.0
    except ValueError:
        p = 1.0
    print("\n" + "=" * 60)
    print("REAL 30-item result — faithful prompt, natural answers")
    print(f"answerer = judge = {ANSWER_MODEL};  n={len(piv)}")
    print("=" * 60)
    print(f"  original rubric : {o.mean()*100:5.1f}%")
    print(f"  rephrased rubric: {r.mean()*100:5.1f}%")
    print(f"  delta (new-orig): {d.mean()*100:+.1f} pts   (Wilcoxon p={p:.3g})")
    print("  (mean judge score x100; per-item deltas in", SCORE_OUT, ")")


def main():
    global ANS_OUT, SCORE_OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-type", default="coord_ref",
                    choices=["coord_ref", "prose_ref", "all"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="output filename tag (default: <ref-type><n>)")
    args = ap.parse_args()
    tag = args.tag or f"{args.ref_type}{args.n}"
    ANS_OUT = f"results/open/real_{tag}_answers.csv"
    SCORE_OUT = f"results/open/real_{tag}_scores.csv"

    idx = get_item_indices(args.ref_type, args.n, args.seed)
    df = pd.read_parquet(DATA)
    items = df[df["index"].isin(idx)].sort_values("index")
    Path("results/open").mkdir(parents=True, exist_ok=True)
    print(f"ref_type={args.ref_type}  n={len(items)}  seed={args.seed}  ->  {ANS_OUT}")
    print("Inference prompt (verbatim VLMEvalKit):")
    print("  " + INFERENCE_PROMPT.replace("\n", "\n  ") + "\n")
    print(f"Phase 1: answering {len(items)} items ...")
    answers = phase1_answers(items)
    print(f"Phase 2: grading {len(answers)} answers x {len(RUBRICS)} rubrics ...")
    scores = phase2_grade(answers)
    report(scores)


if __name__ == "__main__":
    main()
