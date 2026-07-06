"""
GENERATOR for: PAPER_DRAFT.md §5.3.1 — what shuffling the key does to GPT-4o (2x2).

GPT-4o on the same 491 questions, under both prompts (the benchmark's own "original"/faithful
prompt and our "revised"/coax prompt) and both keys (the benchmark's original B-skewed key and
the position-balanced shuffled key). Each prompt is scored by its own pipeline exactly as in
§5.2 (original = benchmark parser with random fallback; revised = bare-letter). All four runs
share identical model-call settings (gpt-4o-2024-11-20, temperature 0, max_tokens 8192,
img_detail high); only the prompt and the option order differ.

Two things the 2x2 shows:
  * the skewed key pads BOTH prompts' raw scores (both fall when it is balanced), and
  * the revised prompt's advantage over the benchmark's own prompt LARGELY survives balancing
    the key (+11.0 original key -> +8.4 shuffled key), i.e. the prompt gain is mostly real, not
    an artifact of the skew.

The "guess-one-letter floor" (largest single-letter share of each key) is reported in the JSON
for the prose: 44.0% on the original key, 27.9% on the shuffled key.

Run:   python paper_analysis/shuffle_drop.py
Reads (all committed, no API):
  results/closed_ended/gpt-4o-...faithful...whole__n491.csv          (original prompt, original key)
  results/closed_ended/position_bias/gpt-4o-...faithful...shuffled__n491.csv  (original prompt, shuffled key)
  results/closed_ended/gpt-4o-...coax...whole__n491.csv              (revised prompt, original key)
  results/closed_ended/position_bias/gpt-4o-...coax...shuffled__n491.csv      (revised prompt, shuffled key)
Writes: paper_analysis/_generated/shuffle_drop_table.md + .values.json
"""
import os
import sys
import json
import math
import datetime
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "_generated")
sys.path.insert(0, REPO)
from paper_analysis.faithful_true_accuracy import high_conf   # noqa: E402
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# parser-scored runs (benchmark parser for faithful; bare-letter == parser for coax)
CSV = {
    ("original", "orig"): "results/closed_ended/gpt-4o-2024-11-20__faithful-direct-k0__whole__n491.csv",
    ("original", "shuf"): "results/closed_ended/position_bias/gpt-4o-2024-11-20__faithful-direct-k0__shuffled__n491.csv",
    ("revised", "orig"): "results/closed_ended/gpt-4o-2024-11-20__coax-direct-k0__whole__n491.csv",
    ("revised", "shuf"): "results/closed_ended/position_bias/gpt-4o-2024-11-20__coax-direct-k0__shuffled__n491.csv",
}
# TRUE (hand-read) accuracy applies only to the verbose original/faithful prompt: high_conf
# reads the letter the model actually committed to, backed by a committed hand-label for every
# ambiguous reply. (The revised prompt already replies in a bare letter, so its parser score IS
# its true score.) Two label files, one per key, because the shuffled replies differ.
TRUE = {
    "orig": ("results/closed_ended/gpt-4o-2024-11-20__faithful-direct-k0__whole__n491.csv",
             "paper_analysis/faithful_hand_labels.csv"),
    "shuf": ("results/closed_ended/position_bias/gpt-4o-2024-11-20__faithful-direct-k0__shuffled__n491.csv",
             "paper_analysis/faithful_shuffled_hand_labels.csv"),
}


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def true_accuracy(run_csv, labels_csv):
    """Accuracy of the letter the model actually stated (high_conf + committed hand-labels).
    Raises if an ambiguous reply has no committed label — never guesses."""
    df = pd.read_csv(os.path.join(REPO, *run_csv.split("/"))).set_index("index")
    lab = pd.read_csv(os.path.join(REPO, *labels_csv.split("/"))).set_index("index")
    correct, need = 0, []
    for i, r in df.iterrows():
        hc = high_conf(r["raw_response"])
        if hc is None:
            if i not in lab.index:
                need.append(int(i)); continue
            v = lab.loc[i, "true_letter"]
            hc = None if (isinstance(v, float) or str(v).strip() == "") else str(v).strip().upper()
        if hc is not None and hc == df.loc[i, "answer"]:
            correct += 1
    if need:
        raise SystemExit(f"unlabeled ambiguous replies in {labels_csv}: {need}")
    return correct, len(df)


def main():
    cell, ci, floor = {}, {}, {}
    n = None
    for (prompt, key), path in CSV.items():
        df = pd.read_csv(os.path.join(REPO, *path.split("/")))
        k, m = int(df["correct"].sum()), len(df)
        n = m
        cell[(prompt, key)] = round(100 * k / m, 1)
        ci[(prompt, key)] = wilson(k, m)
        floor[key] = round(float(df["answer"].value_counts(normalize=True).max() * 100), 1)

    true = {}
    for key, (run_csv, lab_csv) in TRUE.items():
        c, m = true_accuracy(run_csv, lab_csv)
        true[key] = round(100 * c / m, 1)
        ci[("true", key)] = wilson(c, m)

    def gap(a, b, key):  # a,b are prompt/row accuracies for the given key
        return round(a - b, 1)

    vals = {
        "n": n,
        "acc": {f"{p}_{k}": cell[(p, k)] for (p, k) in cell},
        "true_acc": {"orig": true["orig"], "shuf": true["shuf"]},
        "ci": {f"{p}_{k}": ci[(p, k)] for (p, k) in ci},
        "floor_original_key": floor["orig"],
        "floor_shuffled_key": floor["shuf"],
        # pipeline gap (revised parser − original parser) and real prompt gap (revised − original TRUE)
        "pipeline_gap_original_key": gap(cell[("revised", "orig")], cell[("original", "orig")], "orig"),
        "pipeline_gap_shuffled_key": gap(cell[("revised", "shuf")], cell[("original", "shuf")], "shuf"),
        "true_prompt_gap_original_key": gap(cell[("revised", "orig")], true["orig"], "orig"),
        "true_prompt_gap_shuffled_key": gap(cell[("revised", "shuf")], true["shuf"], "shuf"),
        "parser_cost_original_key": gap(true["orig"], cell[("original", "orig")], "orig"),
        "parser_cost_shuffled_key": gap(true["shuf"], cell[("original", "shuf")], "shuf"),
        "drop_original_prompt_parser": gap(cell[("original", "orig")], cell[("original", "shuf")], None),
        "drop_revised_prompt": gap(cell[("revised", "orig")], cell[("revised", "shuf")], None),
        "_generator": "paper_analysis/shuffle_drop.py",
        "_source_csv": [CSV[k] for k in CSV] + ["paper_analysis/faithful_hand_labels.csv",
                                                "paper_analysis/faithful_shuffled_hand_labels.csv"],
        "_generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    prov = (f"<!-- GENERATED by paper_analysis/shuffle_drop.py on {vals['_generated_utc']}. "
            f"Do not hand-edit; run `python paper_analysis/shuffle_drop.py`. -->")
    table = "\n".join([
        prov,
        f"| GPT-4o accuracy ({n} questions) | Original key | Shuffled key |",
        "|:--|--:|--:|",
        f"| Original prompt, benchmark parser | {cell[('original','orig')]}% | {cell[('original','shuf')]}% |",
        f"| Original prompt, true answer (hand-read) | {true['orig']}% | {true['shuf']}% |",
        f"| Revised prompt (bare letter) | {cell[('revised','orig')]}% | {cell[('revised','shuf')]}% |",
        "",
    ])
    with open(os.path.join(OUT_DIR, "shuffle_drop_table.md"), "w", encoding="utf-8") as f:
        f.write(table)
    with open(os.path.join(OUT_DIR, "shuffle_drop.values.json"), "w", encoding="utf-8") as f:
        json.dump(vals, f, indent=2)

    print(table)
    print(f"pipeline gap (revised − parser):  +{vals['pipeline_gap_original_key']} orig -> +{vals['pipeline_gap_shuffled_key']} shuffled")
    print(f"real prompt gap (revised − true): +{vals['true_prompt_gap_original_key']} orig -> +{vals['true_prompt_gap_shuffled_key']} shuffled")
    print(f"parser cost (true − parser): {vals['parser_cost_original_key']} orig, {vals['parser_cost_shuffled_key']} shuffled")
    print(f"floor: original key {floor['orig']}% -> shuffled key {floor['shuf']}%")


if __name__ == "__main__":
    main()
