"""
Gemini open-ended accuracy frontier on the fair (prose_ref) subset.

Two answerer configs, compared head-to-head on the SAME 100 prose_ref items that
GPT-4o answered (seed 0), judged by GPT-4o under the ORIGINAL rubric:

  think_off : thinking_budget=0   (concise, completes)
  think_on  : thinking_budget=-1  (native reasoning, 12K token budget so it completes)

Everything is real model output (verbatim benchmark prompt, no CoT scaffold). We also
report the completion rate to confirm the truncation bug is gone.

  python -m eval_open.run_frontier
"""
import pandas as pd
from pathlib import Path

from data_loader import decode_image
from eval_open.answerer import answer_question, INFERENCE_PROMPT
from eval_open.judges import grade
from eval_open.rubrics import build_grading_prompt

DATA = "data/open_ended.parquet"
PREDS = "eval_open/predictions.parquet"
ANSWER_MODEL = "gemini-3.5-flash"
JUDGE = ("gpt-4o", "gpt-4o")
N, SEED = 100, 0
RUBRIC = "original"   # frontier = accuracy on the benchmark's own rubric
CONFIGS = {
    "think_off": dict(thinking_budget=0,  max_output_tokens=12288),
    "think_on":  dict(thinking_budget=-1, max_output_tokens=12288),
}


def prose_ref_indices():
    p = pd.read_parquet(PREDS)
    p = p[p.ref_type == "prose_ref"]
    return sorted(p.sample(N, random_state=SEED)["index"].tolist())


def is_complete(a):
    return a.rstrip()[-1:] in '.!?)"*' if a else False


def phase1_answers(items, cfg_name, cfg):
    out = f"results/open/frontier_{cfg_name}_answers.csv"
    records, done = [], set()
    if Path(out).exists():
        records = pd.read_csv(out).to_dict("records")
        done = {int(r["index"]) for r in records}
    for _, it in items.iterrows():
        if int(it["index"]) in done:
            continue
        img = decode_image(it["image"])
        ans = answer_question(img, it["question"], model=ANSWER_MODEL, **cfg)
        records.append({"index": int(it["index"]), "question": it["question"],
                        "gt": str(it["answer"]), "answer": ans,
                        "complete": is_complete(ans)})
        print(f"  [{cfg_name}] idx={it['index']} len={len(ans)} complete={is_complete(ans)}")
        pd.DataFrame(records).to_csv(out, index=False)
    return pd.DataFrame(records)


def phase2_grade(answers, cfg_name):
    out = f"results/open/frontier_{cfg_name}_scores.csv"
    records, done = [], set()
    if Path(out).exists():
        records = pd.read_csv(out).to_dict("records")
        done = {int(r["index"]) for r in records}
    jname, jmodel = JUDGE
    for _, a in answers.iterrows():
        if int(a["index"]) in done:
            continue
        prompt = build_grading_prompt(a["question"], a["gt"], a["answer"], RUBRIC)
        score, raw = grade(prompt, judge=jname, model=jmodel, delay=0.2)
        records.append({"index": int(a["index"]), "score": score,
                        "judge_raw": str(raw).replace("\n", " ")[:30]})
        pd.DataFrame(records).to_csv(out, index=False)
    return pd.DataFrame(records)


def main():
    idx = prose_ref_indices()
    df = pd.read_parquet(DATA)
    items = df[df["index"].isin(idx)].sort_values("index")
    Path("results/open").mkdir(parents=True, exist_ok=True)
    print(f"Frontier: gemini-3.5-flash x {list(CONFIGS)} on {len(items)} prose_ref items, "
          f"judge=GPT-4o, rubric={RUBRIC}\nPrompt: {INFERENCE_PROMPT!r}\n")

    summary = {}
    for cfg_name, cfg in CONFIGS.items():
        print(f"--- {cfg_name} {cfg}: answering ---")
        ans = phase1_answers(items, cfg_name, cfg)
        print(f"--- {cfg_name}: grading with GPT-4o ---")
        sc = phase2_grade(ans, cfg_name)
        merged = ans.merge(sc, on="index")
        summary[cfg_name] = (merged["score"].mean() * 100,
                             merged["complete"].mean() * 100, len(merged))

    print("\n" + "=" * 60)
    print("GEMINI-3.5 OPEN-ENDED FRONTIER (prose_ref, GPT-4o judge, original rubric)")
    print("=" * 60)
    print(f"{'config':10s} {'accuracy':>9s} {'complete':>9s} {'n':>4s}")
    for cfg_name, (acc, comp, n) in summary.items():
        print(f"{cfg_name:10s} {acc:8.1f}% {comp:8.0f}% {n:4d}")
    print("\ncompare: GPT-4o self-answered the same 100 prose_ref items at 29.8% "
          "overall (42.1% on the 68 it didn't refuse).")


if __name__ == "__main__":
    main()
