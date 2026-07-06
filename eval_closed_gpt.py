"""
GPT closed-ended MCQ harness for MMOral-Bench — mirrors eval_closed_gemini.py.

Adds two GPT-specific things:
  --prompt {house,faithful,antirefuse}  select the prompt mode (prompts/gpt.py)
  a `refused` column + a refusal-rate line in the summary (GPT declines on some
  medical images; see RESEARCH_PLAN §3.10). Refusals are logged, not hidden.

Slicing matches the Gemini harness:
  k==0 -> score the whole dataset (0-shot); use --start/--limit for a sub-slice.
  For the full 453-item E1 run: --k 0 --start 0  (no --limit).
"""
import os
import sys
import random
import argparse
import pandas as pd

from prompts.gpt import build_prompt
from clients import gpt_client
from clients.errors import APICallFailed
from utils.vlmeval_parse import faithful_predict
from utils import results_io

POOL_SIZE = 50


def extract(mode, raw, row, cot):
    """Return (predicted_letter|None, used_fallback, refused) for the given mode."""
    if mode == "faithful":
        index2ans = {"A": row["option1"], "B": row["option2"],
                     "C": row["option3"], "D": row["option4"]}
        # seed the random fallback per item for reproducibility (see vlmeval_parse)
        pred, used_fallback = faithful_predict(raw, index2ans, seed=int(row["index"]))
        return pred, used_fallback, gpt_client.looks_like_refusal(raw)
    # coax: strict letter extraction, honest refusal flag (no random rescue)
    pred = gpt_client.extract_answer(raw, cot=cot)
    return pred, False, gpt_client.looks_like_refusal(raw)


def get_examples(pool_df, test_row, k, seed=None):
    if k == 0:
        return []
    same_cat = pool_df[pool_df["category"] == test_row["category"]]
    overlap = set(same_cat["index"]) & {test_row["index"]}
    assert len(overlap) == 0, f"DATA CONTAMINATION: index {overlap} in both pool and test"
    if len(same_cat) == 0:
        return []
    rng = random.Random(seed)
    n = min(k, len(same_cat))
    return same_cat.sample(n=n, random_state=rng.randint(0, 2**31)).to_dict("records")


def print_summary(df, model, k, mode):
    acc = df["correct"].mean() * 100
    refusal = df["refused"].mean() * 100
    fallback = df["used_fallback"].mean() * 100
    print(f"\n{'='*55}")
    print(f"Model    : {model}   prompt={mode}   {k}-shot   n={len(df)}")
    print(f"Accuracy : {acc:.2f}%   (paper GPT-4o closed-ended: 45.40%)")
    if mode == "faithful":
        print(f"Refusals : {refusal:.1f}%    Random-fallback (unparsed->guess): {fallback:.1f}%")
    else:
        print(f"Refusals : {refusal:.1f}%    Unparseable (scored wrong): "
              f"{df['predicted'].isna().mean()*100:.1f}%")
    letters = {L: int((df['predicted'] == L).sum()) for L in "ABCD"}
    print(f"Pred A/B/C/D: {letters}")
    print(f"\nBy category:")
    cat_acc = df.groupby("category")["correct"].mean() * 100
    for cat, a in cat_acc.sort_values(ascending=False).items():
        n = len(df[df["category"] == cat])
        print(f"  {cat:<40} {a:.1f}%  (n={n})")


def run(model, k, results_path, data_path, mode="faithful", cot=False,
        limit=None, start=0, reasoning_effort="none", meta=None, detail="low"):
    # HARD-ENFORCE the benchmark-faithful config for the reproduction path. The faithful
    # prompt only reproduces the paper if the generation settings also match
    # config_mmoral_opg.json (temp=0, max_tokens=8192, img_detail=high). We force image detail
    # to the benchmark value and refuse a reasoning model (which cannot use temp=0/max_tokens
    # and so would silently deviate). temperature + max_tokens are pinned in gpt_client.call.
    if mode == "faithful":
        if gpt_client.is_reasoning_model(model):
            raise SystemExit(
                f"faithful reproduction requires a non-reasoning model (temp=0, "
                f"max_tokens={gpt_client.BENCHMARK_MAX_TOKENS}); got reasoning model {model!r}")
        if detail != gpt_client.BENCHMARK_IMG_DETAIL:
            print(f"[faithful] forcing image detail {detail!r} -> "
                  f"{gpt_client.BENCHMARK_IMG_DETAIL!r} (benchmark config)")
            detail = gpt_client.BENCHMARK_IMG_DETAIL
        print(f"[faithful] enforced benchmark config: temperature="
              f"{gpt_client.BENCHMARK_TEMPERATURE}, max_tokens={gpt_client.BENCHMARK_MAX_TOKENS}, "
              f"img_detail={gpt_client.BENCHMARK_IMG_DETAIL}  (source: config_mmoral_opg.json)")

    from dataio import eval_data
    full_df = eval_data.read_closed(data_path)

    if k > 0:
        pool_df = full_df.iloc[:POOL_SIZE].copy()
        test_df = full_df.iloc[POOL_SIZE:].copy()
    else:
        pool_df = full_df.iloc[:0].copy()
        test_df = full_df.copy()

    end = start + limit if limit else None
    test_df = test_df.iloc[start:end]

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    if os.path.exists(results_path):
        done = pd.read_csv(results_path)
        done_indices = set(done["index"].tolist())
        print(f"Resuming — {len(done_indices)}/{len(test_df)} already done")
    else:
        done = pd.DataFrame()
        done_indices = set()

    results = []
    for _, row in test_df.iterrows():
        if row["index"] in done_indices:
            continue
        examples = get_examples(pool_df, row, k=k, seed=int(row["index"]))
        system, content = build_prompt(row, examples=examples or None, cot=cot,
                                       mode=mode, detail=detail)
        try:
            raw = gpt_client.call(system, content, model=model, cot=cot,
                                  reasoning_effort=reasoning_effort)
        except APICallFailed as e:
            # skip: leave the item out of the CSV so it is retried on the next resume,
            # never write an error string as if it were a model answer (§1.0 rule 6).
            print(f"  [SKIP index {row['index']}] {e}")
            continue
        predicted, used_fallback, refused = extract(mode, raw, row, cot)
        correct = predicted == row["answer"]

        results.append({
            "index": row["index"], "file_name": row["file_name"],
            "category": row["category"], "question": row["question"],
            "answer": row["answer"], "predicted": predicted,
            "raw_response": raw, "correct": correct,
            "refused": refused, "used_fallback": used_fallback,
            "n_examples": len(examples), "prompt_mode": mode,
        })

        completed = len(done_indices) + len(results)
        acc = pd.concat([done, pd.DataFrame(results)], ignore_index=True)["correct"].mean() * 100
        tag = "OK" if correct else ("REFUSE" if refused else f"WRONG(got {predicted}, exp {row['answer']})")
        line = f"[{completed}/{len(test_df)}] {tag} | {row['category']} | acc {acc:.1f}%"
        print(line)
        with open("results/progress_gpt.txt", "w") as f:
            f.write(line + "\n")

        if len(results) % 10 == 0:
            pd.concat([done, pd.DataFrame(results)], ignore_index=True).to_csv(results_path, index=False)

    final = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
    # write a pristine CSV + its .meta.json sidecar (self-describing result)
    meta = dict(meta or {})
    meta.setdefault("model", model)
    meta.setdefault("config", f"{mode} {'cot' if cot else 'direct'} k={k}"
                    + (f" reasoning={reasoning_effort}" if gpt_client.is_reasoning_model(model) else ""))
    meta.setdefault("dataset", os.path.basename(data_path))
    meta.setdefault("image_detail", detail)
    meta.setdefault("gen_settings",
                    (f"temperature={gpt_client.BENCHMARK_TEMPERATURE} "
                     f"max_tokens={gpt_client.BENCHMARK_MAX_TOKENS} img_detail={detail} "
                     f"[benchmark-faithful, source config_mmoral_opg.json]")
                    if not gpt_client.is_reasoning_model(model)
                    else f"reasoning_effort={reasoning_effort} max_completion_tokens=2000")
    meta["n"] = int(len(final))
    meta["accuracy_pct"] = round(float(final["correct"].mean() * 100), 2)
    meta["refusal_pct"] = round(float(final["refused"].mean() * 100), 2)
    meta["random_fallback_pct"] = round(float(final["used_fallback"].mean() * 100), 2)
    meta.setdefault("command", "python " + " ".join(sys.argv))
    results_io.write_results(final, results_path, meta)
    print_summary(final, model, k, mode)
    print(f"\nWrote: {results_path}\n   +   {results_path}.meta.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-4o-2024-11-20")
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--data", default="data/closed_ended_clean.parquet")
    p.add_argument("--prompt", default="faithful", choices=["faithful", "coax"])
    p.add_argument("--cot", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--reasoning-effort", default="none")
    p.add_argument("--detail", default="low", choices=["low", "high", "auto"],
                   help="OpenAI image detail; VLMEvalKit faithful = low")
    p.add_argument("--exp", default="", help="experiment id, e.g. E0-repro / E1")
    p.add_argument("--paper-section", default="", help="e.g. §5.2.3")
    p.add_argument("--description", default="", help="one-line human description")
    args = p.parse_args()

    if args.out is None:
        tag = "clean-shuffled" if "clean_shuffled" in args.data else (
              "clean" if "clean" in args.data else (
              "shuffled" if "shuffled" in args.data else "whole"))
        cot_tag = "cot" if args.cot else "direct"
        args.out = f"results/closed_gpt_{args.model}_{args.prompt}_{cot_tag}_k{args.k}_{tag}.csv"

    meta = {"experiment": args.exp, "paper_section": args.paper_section,
            "description": args.description}

    run(model=args.model, k=args.k, results_path=args.out, data_path=args.data,
        mode=args.prompt, cot=args.cot, limit=args.limit, start=args.start,
        reasoning_effort=args.reasoning_effort, meta=meta, detail=args.detail)
