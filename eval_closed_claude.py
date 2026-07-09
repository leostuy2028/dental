"""
Closed-ended (MCQ) eval harness for Claude, at parity with eval_closed_gemini.py:
supports --context (primer), --visual-exemplars (labeled example images), and
--thinking-budget (Anthropic extended thinking). Writes a pristine CSV + meta sidecar
(utils.results_io) so runs are reproducible and directly comparable to the Gemini runs.

Example (Opus 4.8 SOTA config, §5.4):
  python eval_closed_claude.py --model claude-opus-4-8 --prompt coax --thinking-budget 8192 \
    --data data/closed_ended_shuffled.parquet --context reference/opg_primer.txt \
    --visual-exemplars reference/exemplars_v2.json --out results/.../...csv
"""
import os
import sys
import argparse
import pandas as pd

from dataio import eval_data
from prompts.claude import build_prompt
from clients import claude_client
from clients.errors import APICallFailed
from utils import results_io


def print_summary(df, model, think_desc):
    accuracy = df["correct"].mean() * 100
    print(f"\n{'='*55}")
    print(f"Model    : {model}   {think_desc}")
    print(f"Overall  : {accuracy:.2f}%")
    letters = {L: int((df['predicted'] == L).sum()) for L in "ABCD"}
    print(f"Pred A/B/C/D: {letters}    "
          f"Unparseable (scored wrong): {df['predicted'].isna().mean()*100:.1f}%")
    if "refused" in df:
        print(f"Refusals : {df['refused'].mean()*100:.1f}%")
    print(f"\nBy category:")
    cat_acc = df.groupby("category")["correct"].mean() * 100
    for cat, acc in cat_acc.sort_values(ascending=False).items():
        n = len(df[df["category"] == cat])
        print(f"  {cat:<40} {acc:.1f}%  (n={n})")


def run(model, results_path, data_path, limit=None, start=0, thinking_budget=None,
        effort=None, cot=False, mode="coax", context=None, visual_exemplars=None, meta=None):
    test_df = eval_data.read_closed(data_path)
    end = start + limit if limit else None
    test_df = test_df.iloc[start:end]

    os.makedirs(os.path.dirname(results_path) or "results", exist_ok=True)

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

        system, messages = build_prompt(row, cot=cot, mode=mode,
                                        context=context, visual_exemplars=visual_exemplars)
        try:
            predicted, raw = claude_client.call(system, messages, model=model, cot=cot,
                                               thinking_budget=thinking_budget, effort=effort)
        except APICallFailed as e:
            # skip: leave the item out so it is retried on resume; never write an error
            # string as if it were a model answer (§1.0 rule 6).
            print(f"  [SKIP index {row['index']}] {e}")
            continue
        refused = claude_client.looks_like_refusal(raw)
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
            "refused": refused,
        })

        completed = len(done_indices) + len(results)
        running_acc = pd.concat([done, pd.DataFrame(results)], ignore_index=True)["correct"].mean() * 100
        status = "OK" if correct else f"WRONG (got {predicted}, expected {row['answer']})"
        line = f"[{completed}/{len(test_df)}] {status} | {row['category']} | acc so far: {running_acc:.1f}%"
        print(line)
        with open("results/progress_claude.txt", "w") as f:
            f.write(line + "\n")

        if len(results) % 10 == 0:
            pd.concat([done, pd.DataFrame(results)], ignore_index=True).to_csv(results_path, index=False)

    final = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
    if "correct" not in final.columns or len(final) == 0:
        print("No results written (all items skipped or empty slice) — check API errors above.")
        return
    think_tag = f"effort={effort}" if effort else f"think={thinking_budget}"
    meta = dict(meta or {})
    meta.setdefault("model", model)
    meta.setdefault("config", f"{mode} {'cot' if cot else 'direct'} {think_tag}")
    meta.setdefault("dataset", os.path.basename(data_path))
    meta["n"] = int(len(final))
    meta["accuracy_pct"] = round(float(final["correct"].mean() * 100), 2)
    meta.setdefault("command", "python " + " ".join(sys.argv))
    results_io.write_results(final, results_path, meta)
    print_summary(final, model, think_tag)
    print(f"Wrote: {results_path} + {results_path}.meta.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="data/closed_ended_shuffled.parquet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="legacy enabled-thinking budget in tokens (Haiku/Sonnet); omit to disable")
    parser.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                        help="Opus-4.8 adaptive-thinking effort level (preferred for Opus)")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--prompt", default="coax", choices=["house", "coax"])
    parser.add_argument("--context", default=None,
                        help="path to a reference text file prepended to each question (§5.4 primer)")
    parser.add_argument("--visual-exemplars", default=None,
                        help="path to a manifest.json of labeled example images to prepend (§5.4)")
    parser.add_argument("--exp", default="")
    parser.add_argument("--paper-section", default="")
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    context_text = open(args.context, encoding="utf-8").read() if args.context else None

    vis_ex = None
    if args.visual_exemplars:
        import json as _json
        man = _json.load(open(args.visual_exemplars, encoding="utf-8"))
        mdir = os.path.dirname(os.path.abspath(args.visual_exemplars))

        def _resolve(rel):
            for cand in (os.path.join(mdir, rel), os.path.join(mdir, os.path.basename(rel))):
                if os.path.exists(cand):
                    return cand
            raise FileNotFoundError(rel)

        vis_ex = [(open(_resolve(e["image"]), "rb").read(), e["caption"]) for e in man["exemplars"]]

    run(model=args.model, results_path=args.out, data_path=args.data, limit=args.limit,
        start=args.start, thinking_budget=args.thinking_budget, effort=args.effort,
        cot=args.cot, mode=args.prompt, context=context_text, visual_exemplars=vis_ex,
        meta={"experiment": args.exp, "paper_section": args.paper_section,
              "description": args.description, "context_file": args.context or "",
              "visual_exemplars": args.visual_exemplars or "", "effort": args.effort or ""})
