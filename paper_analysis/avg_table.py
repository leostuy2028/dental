"""§6.3 balanced-average table (NO API). The paper's headline metric is
Avg = mean(closed-overall, open-overall). We compute it for gemini-3.5-flash
(the only model of ours run on BOTH halves) and set it beside the paper's numbers.

To keep the two halves on the SAME configuration, both use the coax prompt + the OPG
reading primer, no visual exemplars: closed = coax+primer on the balanced key (§5.4);
open = coax+primer full-578 free-text run (GPT-4o judge, §6.1). Exemplars are excluded
because they help multiple choice but hurt free text (§5.4 vs §6.1); only the thinking
budget differs (0 closed / 4000 open, each task-standard).
Paper numbers are quoted from arXiv:2509.09254 Tables 2-3 (not our computation).

  python -m paper_analysis.avg_table
"""
import pandas as pd

CLOSED_BAL = "results/closed_ended/knowledge_context/gemini-3.5-flash__coax-direct-ctx-opgprimer__shuffled__n491.csv"
CLOSED_ORIG = "results/closed_ended/knowledge_context/gemini-3.5-flash__coax-direct-ctx-opgprimer__whole__n491.csv"
closed = pd.read_csv(CLOSED_BAL).correct.mean() * 100
op = pd.read_csv("results/open/batched_gemini35_plain578_scores.csv").score.mean() * 100

# paper-reported (arXiv:2509.09254): (label, closed, open)
PAPER = [
    ("OralGPT (paper's best-avg model)", 39.60, 52.77),
    ("GPT-4o (paper)", 45.40, 37.50),
    ("Claude-3.7-Sonnet (paper)", 41.40, 40.67),
]

print("| Model | Closed (MCQ) | Open (free-text) | Avg. |")
print("|:--|--:|--:|--:|")
print(f"| **gemini-3.5-flash (coax + primer)** | {closed:.1f} | {op:.1f} | **{(closed+op)/2:.1f}** |")
for lbl, c, o in PAPER:
    print(f"| {lbl} | {c:.2f} | {o:.2f} | {(c+o)/2:.2f} |")
closed_orig = pd.read_csv(CLOSED_ORIG).correct.mean() * 100
print(f"\n(ours: closed = coax + primer on the balanced key (§5.4); open = coax + primer full-578 "
      f"free-text, GPT-4o judge (§6.1); no exemplars, matched across the two halves. Paper rows quoted "
      f"from arXiv:2509.09254 Tables 2-3. gemini closed on the ORIGINAL key = {closed_orig:.1f}% -> Avg "
      f"{(closed_orig+op)/2:.1f}.)")
