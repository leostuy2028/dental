import os
import random
import argparse
import sys
import pandas as pd
from dataio.data_loader import load_closed
from prompts.gemini import build_prompt
from clients import gemini_client
from utils import results_io

POOL_SIZE = 50


def get_examples(pool_df, test_row, k, seed=None):
    """Sample k same-category examples from pool (contamination-guarded)."""
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


def print_summary(df, model, k):
    accuracy = df["correct"].mean() * 100
    print(f"\n{'='*55}")
    print(f"Model    : {model}")
    print(f"Shot     : {k}-shot")
    print(f"Overall  : {accuracy:.2f}%")
    print(f"           paper closed-ended refs: Gemini-2.0-Flash 41.20 (RETIRED, not this model),")
    print(f"           GPT-4o 45.40, Claude-3.7-Sonnet 41.40  — gemini-2.5-flash is a NEW data point")
    print(f"\nBy category:")
    cat_acc = df.groupby("category")["correct"].mean() * 100
    for cat, acc in cat_acc.sort_values(ascending=False).items():
        n = len(df[df["category"] == cat])
        print(f"  {cat:<40} {acc:.1f}%  (n={n})")


def run(model, k, results_path, data_path="data/closed_ended.parquet",
        limit=None, start=0, thinking_budget=None, cot=False, mode="house", meta=None):
    # route the client at the requested model (default gemini-2.0-flash)
    gemini_client.MODEL = model

    full_df = pd.read_parquet(data_path)

    # 0-shot reproduces the paper's protocol on the full set; few-shot carves a pool.
    if k > 0:
        pool_df = full_df.iloc[:POOL_SIZE].copy()
        test_df = full_df.iloc[POOL_SIZE:].copy()
    else:
        pool_df = full_df.iloc[:0].copy()
        test_df = full_df.copy()

    # start/limit select a slice; start=50 limit=100 -> idx 50-149, the grid's canonical slice.
    end = start + limit if limit else None
    test_df = test_df.iloc[start:end]

    os.makedirs("results", exist_ok=True)

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
        parts = build_prompt(row, examples=examples if examples else None, cot=cot, mode=mode)
        predicted, raw = gemini_client.call(parts, thinking_budget=thinking_budget, cot=cot)
        correct = predicted == row["answer"]

        results.append({
            "index": row["index"],
            "file_name": row["file_name"],
            "category": row["category"],
            "question": row["question"],
            "answer": row["answer"],
            "predicted": predicted,
            "raw_response": raw,
            "correct": correct,
            "n_examples": len(examples),
        })

        completed = len(done_indices) + len(results)
        running_acc = pd.concat([done, pd.DataFrame(results)], ignore_index=True)["correct"].mean() * 100
        status = "OK" if correct else f"WRONG (got {predicted}, expected {row['answer']})"
        line = f"[{completed}/{len(test_df)}] {status} | {row['category']} | acc so far: {running_acc:.1f}%"
        print(line)

        with open("results/progress_gemini.txt", "w") as f:
            f.write(line + "\n")

        if len(results) % 10 == 0:
            pd.concat([done, pd.DataFrame(results)], ignore_index=True).to_csv(results_path, index=False)

    final = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
    meta = dict(meta or {})
    meta.setdefault("model", model)
    meta.setdefault("config", f"{mode} {'cot' if cot else 'direct'} think={thinking_budget}")
    meta.setdefault("dataset", os.path.basename(data_path))
    meta["n"] = int(len(final))
    meta["accuracy_pct"] = round(float(final["correct"].mean() * 100), 2)
    meta.setdefault("command", "python " + " ".join(sys.argv))
    results_io.write_results(final, results_path, meta)
    print_summary(final, model, k)
    print(f"Wrote: {results_path} + {results_path}.meta.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--data", default="data/closed_ended.parquet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="native thinking tokens; -1 dynamic, 0 off, omit to disable")
    parser.add_argument("--cot", action="store_true",
                        help="visible per-option chain-of-thought prompt (parse 'Answer: X')")
    parser.add_argument("--prompt", default="house", choices=["house", "coax"])
    parser.add_argument("--exp", default="")
    parser.add_argument("--paper-section", default="")
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    if args.out is None:
        tag = "shuffled" if "shuffled" in args.data else "original"
        think_tag = f"_think{args.thinking_budget}" if args.thinking_budget is not None else ""
        cot_tag = "_cot" if args.cot else ""
        args.out = f"results/closed_{args.model}_{args.k}shot_{tag}{cot_tag}{think_tag}.csv"

    run(model=args.model, k=args.k, results_path=args.out, data_path=args.data,
        limit=args.limit, start=args.start, thinking_budget=args.thinking_budget, cot=args.cot, mode=args.prompt,
        meta={"experiment": args.exp, "paper_section": args.paper_section, "description": args.description})
