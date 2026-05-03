import os
import pandas as pd
from data_loader import load_closed
from prompts.claude import build_prompt
from claude_client import call

RESULTS_PATH = "results/closed_results.csv"


def print_summary(df):
    accuracy = df["correct"].mean() * 100
    print(f"\n{'='*55}")
    print(f"Overall accuracy : {accuracy:.2f}%  (GPT-4o baseline: 41.45%)")
    print(f"\nBy category:")
    cat_acc = df.groupby("category")["correct"].mean() * 100
    for cat, acc in cat_acc.sort_values(ascending=False).items():
        n = len(df[df["category"] == cat])
        print(f"  {cat:<40} {acc:.1f}%  (n={n})")


def run(limit=None):
    df = load_closed()
    if limit:
        df = df.head(limit)

    os.makedirs("results", exist_ok=True)

    if os.path.exists(RESULTS_PATH):
        done = pd.read_csv(RESULTS_PATH)
        done_indices = set(done["index"].tolist())
        print(f"Resuming — {len(done_indices)}/{len(df)} already done")
    else:
        done = pd.DataFrame()
        done_indices = set()

    results = []

    for _, row in df.iterrows():
        if row["index"] in done_indices:
            continue

        system, messages = build_prompt(row)
        predicted, raw = call(system, messages)
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
        })

        completed = len(done_indices) + len(results)
        total = len(df)
        status = "OK" if correct else f"WRONG (got {predicted}, expected {row['answer']})"
        print(f"[{completed}/{total}] {status} | {row['category']}")

        if len(results) % 10 == 0:
            pd.concat([done, pd.DataFrame(results)], ignore_index=True).to_csv(RESULTS_PATH, index=False)

    final = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
    final.to_csv(RESULTS_PATH, index=False)
    print_summary(final)


if __name__ == "__main__":
    run()
