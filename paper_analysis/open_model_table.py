"""
§6.1 open-ended model-comparison table (NO API; RESEARCH_PLAN §1.0 rule 7).

Reads the committed open-ended score CSVs + ref_type and prints overall /
prose_ref / coord_ref accuracy for each model-arm. This is the single source for
the "preliminary note on other models" table in PAPER_DRAFT.md §6.1, so the table
is regenerable without re-running any model.

  python -m paper_analysis.open_model_table
"""
import os
import pandas as pd

RT = pd.read_parquet("eval_open/predictions.parquet")[["index", "ref_type"]]

# (label, prompt description, scores CSV) — all graded by gpt-4o judge, rubric='original' (unchanged)
ROWS = [
    ("GPT-4o", "primer + coordinate elicitation",
     "results/open/coordarms_gpt4o_cpc_all_scores.csv"),
    ("gpt-5-mini", "primer, no coordinates",
     "results/open/batched_gpt5mini_scores.csv"),
    ("gpt-5-mini + 12 visual exemplars", "primer + exemplars, no coordinates",
     "results/open/batched_gpt5mini_ex_scores.csv"),
]


def stats(csv):
    s = pd.read_csv(csv).merge(RT, on="index")
    return (len(s), s.score.mean() * 100,
            s[s.ref_type == "prose_ref"].score.mean() * 100,
            s[s.ref_type == "coord_ref"].score.mean() * 100)


def main():
    print("| Model | Prompt | Overall | prose-ref (n=477) | coord-ref (n=101) |")
    print("|:--|:--|--:|--:|--:|")
    for label, prompt, csv in ROWS:
        if not os.path.exists(csv):
            print(f"| {label} | {prompt} | `[PENDING]` | | |")
            continue
        n, o, p, c = stats(csv)
        print(f"| {label} | {prompt} | {o:.1f}% | {p:.1f}% | {c:.1f}% |")
    print(f"\n(all rows: GPT-4o judge, rubric='original' unchanged; n={len(RT)} items, "
          "477 prose-ref + 101 coord-ref)")


if __name__ == "__main__":
    main()
