"""
Reproduce the paper's GPT-4o open-ended result on prose_ref items.

GPT-4o ANSWERS the X-ray (verbatim VLMEvalKit prompt), then GPT-4o JUDGES each
answer under the ORIGINAL rubric (the paper's rubric). Paper GPT-4o open-ended = 37.5%
over ALL items; prose_ref-only excludes the hard coordinate items so expect >= that.
Deviations from the paper (documented, not hidden): judge is GPT-4o not GPT-4-turbo
(no turbo access on this account); answerer == judge (both GPT-4o); prose_ref subset only.

  python -m eval_open.run_reproduce --n 100 --seed 0
"""
import argparse
import pandas as pd
from pathlib import Path

from eval_open.answerer import answer_question_openai, INFERENCE_PROMPT
from eval_open.judges import grade
from eval_open.rubrics import build_grading_prompt

DATA = "data/open_ended.parquet"
PREDS = "eval_open/predictions.parquet"
ANSWER_MODEL = "gpt-4o"
JUDGE_MODEL = "gpt-4o"
RUBRICS = ["original", "rephrased"]   # original = paper reproduction; rephrased = bonus


def prose_ref_indices(n, seed):
    p = pd.read_parquet(PREDS)
    p = p[p.ref_type == "prose_ref"]
    return sorted(p.sample(min(n, len(p)), random_state=seed)["index"].tolist())


def phase1_answers(items, ans_out):
    records, done = [], set()
    if Path(ans_out).exists():
        records = pd.read_csv(ans_out).to_dict("records")
        done = {int(r["index"]) for r in records}
    for _, it in items.iterrows():
        if int(it["index"]) in done:
            continue
        ans = answer_question_openai(it["image"], it["question"], model=ANSWER_MODEL)
        records.append({"index": int(it["index"]), "question": it["question"],
                        "gt": str(it["answer"]), "answer": ans})
        print(f"  [gpt-4o answer] idx={it['index']}: {ans[:60].replace(chr(10),' ')}...")
        pd.DataFrame(records).to_csv(ans_out, index=False)
    return pd.DataFrame(records)


def phase2_grade(answers, score_out):
    records, done = [], set()
    if Path(score_out).exists():
        records = pd.read_csv(score_out).to_dict("records")
        done = {(int(r["index"]), r["rubric"]) for r in records}
    for _, a in answers.iterrows():
        for rubric in RUBRICS:
            if (int(a["index"]), rubric) in done:
                continue
            prompt = build_grading_prompt(a["question"], a["gt"], a["answer"], rubric)
            score, raw = grade(prompt, judge="gpt-4o", model=JUDGE_MODEL, delay=0.2)
            records.append({"index": int(a["index"]), "rubric": rubric,
                            "score": score, "judge_raw": str(raw).replace("\n", " ")[:40]})
            pd.DataFrame(records).to_csv(score_out, index=False)
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tag = f"gpt4o_prose{args.n}"
    ans_out = f"results/open/reproduce_{tag}_answers.csv"
    score_out = f"results/open/reproduce_{tag}_scores.csv"

    idx = prose_ref_indices(args.n, args.seed)
    df = pd.read_parquet(DATA)
    items = df[df["index"].isin(idx)].sort_values("index")
    Path("results/open").mkdir(parents=True, exist_ok=True)
    print(f"Reproduction: GPT-4o answerer + judge, prose_ref, n={len(items)}, seed={args.seed}")
    print("Prompt (verbatim):", repr(INFERENCE_PROMPT))
    answers = phase1_answers(items, ans_out)
    scores = phase2_grade(answers, score_out)

    piv = scores.pivot_table(index="index", columns="rubric", values="score").dropna()
    print("\n" + "=" * 56)
    print(f"GPT-4o answerer + GPT-4o judge, prose_ref, n={len(piv)}")
    print(f"  ORIGINAL rubric (paper protocol): {piv['original'].mean()*100:.1f}%")
    print(f"  rephrased rubric               : {piv['rephrased'].mean()*100:.1f}%")
    print(f"  paper GPT-4o open-ended (all items, GPT-4-turbo judge): 37.5%")
    print("  NB: prose_ref-only excludes coord items -> expect >= 37.5%")
    print("=" * 56)


if __name__ == "__main__":
    main()
