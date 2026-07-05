"""
Re-derive the `predicted` / `correct` columns of a result CSV from its committed
`raw_response`, using the canonical extractor (clients.parsing.extract_letter).

Why this exists: an earlier eval harness stored a mis-parsed `predicted` for verbose
replies (it read "The correct answer is **D**" as 'A'). `raw_response` — the real API
output — was always correct, so the fix is a pure, no-API re-score. This script rewrites
only the two DERIVED columns; `raw_response` and every other column are untouched, and it
is idempotent (re-running changes nothing once a file is correct).

Usage:
  python dataio/rescore_predictions.py                 # re-score the known-affected files
  python dataio/rescore_predictions.py path/to/x.csv   # re-score specific file(s)
CoT vs direct parsing is inferred from a `_cot` marker in the filename.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clients.parsing import extract_letter  # noqa: E402

# the files the parser bug actually corrupted (verbose gemini-3.5-flash direct runs)
DEFAULT_FILES = [
    f"results/nshot/closed_gemini-3.5-flash_k{k}_cleanshuf_think0.csv" for k in (0, 1, 3, 5)
]


def rescore_file(path):
    df = pd.read_csv(path)
    if not {"raw_response", "predicted", "answer"} <= set(df.columns):
        print(f"  [skip] {path}: missing required columns")
        return 0
    cot = "_cot" in os.path.basename(path)
    new_pred = [extract_letter(str(r), cot=cot) for r in df["raw_response"].tolist()]
    new_pred = [p if p in ("A", "B", "C", "D") else None for p in new_pred]
    old_pred = df["predicted"].where(df["predicted"].isin(list("ABCD")))
    changed = sum(1 for o, n in zip(old_pred.tolist(), new_pred)
                  if (o if isinstance(o, str) else None) != n)
    if changed == 0:
        print(f"  [ok, no change] {path}")
        return 0
    old_acc = round(100 * (df["predicted"] == df["answer"]).mean(), 1)
    df["predicted"] = new_pred
    df["correct"] = [p == a for p, a in zip(new_pred, df["answer"].tolist())]
    new_acc = round(100 * df["correct"].mean(), 1)
    df.to_csv(path, index=False)
    print(f"  [rescored] {path}: {changed} rows changed, acc {old_acc} -> {new_acc}")
    return changed


def main():
    files = sys.argv[1:] or DEFAULT_FILES
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    total = 0
    for f in files:
        total += rescore_file(f if os.path.isabs(f) else os.path.join(repo, f))
    print(f"\ndone: {total} row(s) re-scored across {len(files)} file(s)")


if __name__ == "__main__":
    main()
