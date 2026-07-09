"""
§6.2 coordinate-elicitation study — four prompt arms, ONE model, the benchmark's
UNCHANGED grader (rubric='original', verbatim VLMEvalKit).

Question: the benchmark's grader rewards coordinates (§6.1), but its inference
prompt never asks for them. If we change only the PROMPT (never the grader) to
elicit coordinates, how many points appear? And are they real localization or
just format? The four arms isolate each lever:

  plain  ->  coax  ->  coax_primer  ->  coax_primer_coords
   |         |            |                |
   |         |            |                +-- + coordinate JSON (format lever)
   |         |            +------------------- + OPG reading primer (knowledge lever)
   |         +-------------------------------- + persona/never-refuse (refusal lever)
   +------------------------------------------ faithful VLMEvalKit prompt (baseline)

Data integrity: every answer is REAL model output (answer_openai_custom); nothing
is hand-authored or coordinate-forced with fabricated numbers. Coordinates the
model emits are its own; their accuracy is checked separately (IoU) so we can tell
"grader rewards presence" from "model localizes correctly".

  python -m eval_open.run_coord_arms --n 30 --model gpt-4o --judge gpt-4o
"""
import argparse
import re
import pandas as pd
from pathlib import Path

from dataio.data_loader import decode_image
from eval_open.answerer import answer_openai_custom
from eval_open.judges import grade
from eval_open.rubrics import build_grading_prompt
from eval_open.prompts_open import build_arm, ARMS
from utils.results_io import write_results

DATA = "data/open_ended.parquet"
PREDS = "eval_open/predictions.parquet"
PRIMER = "reference/opg_primer.txt"
RUBRIC = "original"                       # the benchmark's own grader, UNCHANGED
DETAIL = "high"                           # paper-faithful image detail
MAX_TOKENS = {"coax_primer_coords": 2048}  # coordinate JSON needs headroom; others 1024 (both non-binding)

_REFUSAL_PAT = re.compile(
    r"\b(i'?m unable|i am unable|cannot (see|view|interpret|analyze|provide)|"
    r"unable to (see|view|interpret|analyze|provide)|i can'?t (see|view|help)|"
    r"as an ai|i cannot assist)\b", re.I)


def looks_refused(ans):
    a = str(ans).strip()
    return (not a) or bool(_REFUSAL_PAT.search(a[:200]))


def select_items(ref_type, n, seed):
    """Deterministic sample of `n` items of the given ref_type ('coord_ref',
    'prose_ref', or 'all'). n larger than the pool selects the whole pool."""
    preds = pd.read_parquet(PREDS)
    if ref_type != "all":
        preds = preds[preds.ref_type == ref_type]
    n = min(n, len(preds))
    return sorted(preds.sample(n, random_state=seed)["index"].tolist())


def phase1_answer(items, arms, model, ans_out):
    records, done = [], set()
    if Path(ans_out).exists():
        records = pd.read_csv(ans_out).to_dict("records")
        done = {(int(r["index"]), r["arm"]) for r in records}
    primer = open(PRIMER, encoding="utf-8").read()
    for _, it in items.iterrows():
        img_b64 = it["image"]
        W, H = decode_image(img_b64).size
        for arm in arms:
            if (int(it["index"]), arm) in done:
                continue
            system, user = build_arm(arm, it["question"], primer, W, H)
            ans = answer_openai_custom(img_b64, system, user, model=model,
                                       max_tokens=MAX_TOKENS.get(arm, 1024),
                                       detail=DETAIL)
            records.append({"index": int(it["index"]), "arm": arm,
                            "question": it["question"], "gt": str(it["answer"]),
                            "answer": ans, "refused": int(looks_refused(ans))})
            print(f"  [answer] idx={it['index']:>3} {arm:<20} "
                  f"{'REFUSED' if looks_refused(ans) else ans[:50].replace(chr(10),' ')}")
            pd.DataFrame(records).to_csv(ans_out, index=False)
    return pd.DataFrame(records)


def phase2_grade(answers, judge, judge_model, score_out):
    records, done = [], set()
    if Path(score_out).exists():
        records = pd.read_csv(score_out).to_dict("records")
        done = {(int(r["index"]), r["arm"]) for r in records}
    for _, a in answers.iterrows():
        key = (int(a["index"]), a["arm"])
        if key in done:
            continue
        prompt = build_grading_prompt(a["question"], a["gt"], a["answer"], RUBRIC)
        score, raw = grade(prompt, judge=judge, model=judge_model)
        records.append({"index": int(a["index"]), "arm": a["arm"],
                        "score": score, "judge_raw": str(raw).replace("\n", " ")[:40]})
        print(f"  [grade]  idx={a['index']:>3} {a['arm']:<20} -> {score}")
        pd.DataFrame(records).to_csv(score_out, index=False)
    return pd.DataFrame(records)


def report(answers, scores, arms):
    from scipy.stats import wilcoxon
    print("\n" + "=" * 68)
    print(f"COORD-ARMS  n={answers['index'].nunique()}  grader=original (unchanged)")
    print("=" * 68)
    piv = scores.pivot_table(index="index", columns="arm", values="score")
    ref = answers.groupby("arm")["refused"].mean() * 100
    print(f"{'arm':<22}{'score':>8}{'refuse%':>9}")
    for arm in arms:
        s = piv[arm].mean() * 100 if arm in piv else float("nan")
        print(f"{arm:<22}{s:>7.1f}%{ref.get(arm, 0):>8.1f}%")
    print("-" * 68)
    for a, b in zip(arms, arms[1:]):
        if a in piv and b in piv:
            d = (piv[b] - piv[a]).dropna()
            try:
                p = wilcoxon(d, zero_method="zsplit").pvalue if d.abs().sum() else 1.0
            except ValueError:
                p = 1.0
            print(f"  {b} - {a}: {d.mean()*100:+.1f} pts  (Wilcoxon p={p:.3g})")
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--ref-type", default="coord_ref",
                    choices=["coord_ref", "prose_ref", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="gpt-4o", help="answerer model")
    ap.add_argument("--judge", default="gpt-4o", choices=["gpt-4o", "gemini", "claude"])
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--arms", default=",".join(ARMS), help="comma-separated subset of arms")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    tag = args.tag or f"{args.model}_n{args.n}"
    ans_out = f"results/open/coordarms_{tag}_answers.csv"
    score_out = f"results/open/coordarms_{tag}_scores.csv"
    Path("results/open").mkdir(parents=True, exist_ok=True)

    idx = select_items(args.ref_type, args.n, args.seed)
    df = pd.read_parquet(DATA)
    items = df[df["index"].isin(idx)].sort_values("index")
    print(f"ref_type={args.ref_type} n={len(items)} seed={args.seed} arms={arms}")
    print(f"answerer={args.model}  judge={args.judge}  grader={RUBRIC} (unchanged)")
    print(f"-> {ans_out}\n-> {score_out}\n")

    print(f"Phase 1: {len(items)} items x {len(arms)} arms = {len(items)*len(arms)} answers ...")
    answers = phase1_answer(items, arms, args.model, ans_out)
    print(f"\nPhase 2: grading {len(answers)} answers under '{RUBRIC}' with {args.judge} ...")
    scores = phase2_grade(answers, args.judge, args.judge_model, score_out)
    report(answers, scores, arms)

    # --- reproducibility sidecars (utils/results_io) --------------------------
    cmd = (f"python -m eval_open.run_coord_arms --n {args.n} --ref-type {args.ref_type} "
           f"--seed {args.seed} --model {args.model} --judge {args.judge} "
           f"--arms {args.arms} --tag {tag}")
    base_meta = {
        "experiment": "E-open-coord (coordinate-elicitation arms)",
        "paper_section": "§6.1/§6.2",
        "model": args.model, "judge": f"{args.judge}/{args.judge_model or 'default'}",
        "config": (f"temperature=0.0, image_detail={DETAIL}, "
                   f"max_tokens=1024 (coords 2048; verified non-binding, no truncation); "
                   f"grader=rubric='original' (verbatim VLMEvalKit, UNCHANGED)"),
        "dataset": f"data/open_ended.parquet (ref_type={args.ref_type})",
        "slice": f"{args.ref_type} sample n={len(items)} seed={args.seed}",
        "arms": arms, "n": int(answers["index"].nunique()), "command": cmd,
    }
    write_results(pd.read_csv(ans_out), ans_out,
                  {**base_meta, "description": "REAL model answers, 4 prompt arms (plain/coax/+primer/+coords)"})
    write_results(pd.read_csv(score_out), score_out,
                  {**base_meta, "description": "judge scores under the benchmark's own unchanged grader"})
    print(f"\nsidecars written: {ans_out}.meta.json, {score_out}.meta.json")


if __name__ == "__main__":
    main()
