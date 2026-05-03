import os
import random
import argparse
import pandas as pd
from data_loader import load_closed
from prompts.claude import build_prompt
from claude_client import call

POOL_SIZE = 50
TEST_START = 50
TEST_END = 150


def get_examples(pool_df, test_row, k, seed=None):
    """
    Sample k same-category examples from pool.
    Falls back to all available if fewer than k exist.
    Raises if any example shares an index with the test row (contamination guard).
    """
    if k == 0:
        return []

    same_cat = pool_df[pool_df["category"] == test_row["category"]]

    overlap = set(same_cat["index"]) & {test_row["index"]}
    assert len(overlap) == 0, f"DATA CONTAMINATION: index {overlap} appears in both pool and test set"

    if len(same_cat) == 0:
        return []

    rng = random.Random(seed)
    n = min(k, len(same_cat))
    return same_cat.sample(n=n, random_state=rng.randint(0, 2**31)).to_dict("records")


def prove_no_contamination(pool_df, test_df):
    pool_indices = set(pool_df["index"])
    test_indices = set(test_df["index"])
    overlap = pool_indices & test_indices

    print("=== CONTAMINATION PROOF ===")
    print(f"Pool indices  : {min(pool_indices)} – {max(pool_indices)}  ({len(pool_indices)} questions)")
    print(f"Test indices  : {min(test_indices)} – {max(test_indices)}  ({len(test_indices)} questions)")
    print(f"Overlap       : {len(overlap)} questions")
    assert len(overlap) == 0, f"CONTAMINATION DETECTED: {overlap}"
    print("Result        : CLEAN — zero overlap confirmed")
    print("=" * 28)
    print()


def print_summary(df, model, k):
    accuracy = df["correct"].mean() * 100
    print(f"\n{'='*55}")
    print(f"Model    : {model}")
    print(f"Shot     : {k}-shot")
    print(f"Overall  : {accuracy:.2f}%  (Haiku 3-shot: 34.00%,  Haiku 0-shot: 38.00%,  GPT-4o: 45.40%)")
    print(f"\nBy category:")
    cat_acc = df.groupby("category")["correct"].mean() * 100
    for cat, acc in cat_acc.sort_values(ascending=False).items():
        n = len(df[df["category"] == cat])
        print(f"  {cat:<40} {acc:.1f}%  (n={n})")


def run(model, k, results_path):
    full_df = load_closed()
    pool_df = full_df.iloc[:POOL_SIZE].copy()
    test_df = full_df.iloc[TEST_START:TEST_END].copy()

    prove_no_contamination(pool_df, test_df)

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
        system, messages = build_prompt(row, examples=examples if examples else None)
        predicted, raw = call(system, messages, model=model)
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
        status = "OK" if correct else f"WRONG (got {predicted}, expected {row['answer']})"
        print(f"[{completed}/{len(test_df)}] {status} | {row['category']} | {len(examples)}-shot")

        if len(results) % 10 == 0:
            pd.concat([done, pd.DataFrame(results)], ignore_index=True).to_csv(results_path, index=False)

    final = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
    final.to_csv(results_path, index=False)
    print_summary(final, model, k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = f"results/closed_{args.model.split('-')[1]}_{args.k}shot.csv"

    run(model=args.model, k=args.k, results_path=args.out)
