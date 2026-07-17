"""§6.3 balanced-average table (NO API). The paper's headline metric is
Avg = mean(closed-overall, open-overall). We can compute it for gemini-3.5-flash
(the only model of ours run on BOTH halves) and set it beside the paper's numbers.

Our closed = coax prompt on the position-balanced (shuffled) key (the corrected
number, §5); our open = the full-578 free-text run (coax+primer, GPT-4o judge, §6.1).
Paper numbers are quoted from arXiv:2509.09254 Tables 2-3 (not our computation).

  python -m paper_analysis.avg_table
"""
import pandas as pd

# ours (computed from committed CSVs)
closed = pd.read_csv("results/closed_ended/position_bias/gemini-3.5-flash__coax-direct-k0__shuffled__n491.csv").correct.mean() * 100
op = pd.read_csv("results/open/batched_gemini35_plain578_scores.csv").score.mean() * 100

# paper-reported (arXiv:2509.09254): (label, closed, open)
PAPER = [
    ("OralGPT (paper's best-avg model)", 39.60, 52.77),
    ("GPT-4o (paper)", 45.40, 37.50),
    ("Claude-3.7-Sonnet (paper)", 41.40, 40.67),
]

print("| Model | Closed (MCQ) | Open (free-text) | Avg. |")
print("|:--|--:|--:|--:|")
print(f"| **gemini-3.5-flash (ours)** | {closed:.1f} | {op:.1f} | **{(closed+op)/2:.1f}** |")
for lbl, c, o in PAPER:
    print(f"| {lbl} | {c:.2f} | {o:.2f} | {(c+o)/2:.2f} |")
print(f"\n(ours: closed = coax on the balanced key (§5); open = full-578 free-text, GPT-4o judge (§6.1). "
      f"Paper rows quoted from arXiv:2509.09254 Tables 2-3. gemini closed on the ORIGINAL key = "
      f"{pd.read_csv('results/closed_ended/position_bias/gemini-3.5-flash__coax-direct-k0__whole__n491.csv').correct.mean()*100:.1f}% -> Avg "
      f"{(pd.read_csv('results/closed_ended/position_bias/gemini-3.5-flash__coax-direct-k0__whole__n491.csv').correct.mean()*100+op)/2:.1f}.)")
