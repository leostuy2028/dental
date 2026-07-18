"""§6.3 balanced-average table (NO API). The paper's headline metric is
Avg = mean(closed-overall, open-overall). We compute it for gemini-3.5-flash
(the only model of ours run on BOTH halves) and set it beside the paper's numbers.

KEY CHOICE. This table is a head-to-head against the paper's *published* leaderboard,
and every paper number (OralGPT, GPT-4o, Claude) was measured on the benchmark's
ORIGINAL (position-skewed) key. To compare like-for-like we therefore put OUR closed
number on that same original key; scoring ourselves on the debiased balanced key while
the paper rows keep the bias would handicap us against inflated numbers. We still print
the debiased balanced-key figure in the footnote (it is the honest absolute number, and
it stays ahead of OralGPT).

Both of our halves use the SAME configuration — coax prompt + OPG reading primer, no
visual exemplars (exemplars help multiple choice but hurt free text, §5.4 vs §6.1) — so
the two numbers describe one model under one setup. Only the thinking budget differs
(0 closed / 4000 open, each task-standard).
Paper numbers are quoted from arXiv:2509.09254 Tables 2-3 (not our computation).

  python -m paper_analysis.avg_table
"""
import pandas as pd

CLOSED_ORIG = "results/closed_ended/knowledge_context/gemini-3.5-flash__coax-direct-ctx-opgprimer__whole__n491.csv"
CLOSED_BAL = "results/closed_ended/knowledge_context/gemini-3.5-flash__coax-direct-ctx-opgprimer__shuffled__n491.csv"
closed = pd.read_csv(CLOSED_ORIG).correct.mean() * 100      # original key = paper's key
closed_bal = pd.read_csv(CLOSED_BAL).correct.mean() * 100   # our debiased key (footnote)
op = pd.read_csv("results/open/batched_gemini35_plain578_scores.csv").score.mean() * 100

# paper-reported (arXiv:2509.09254): (label, closed, open) — all on the ORIGINAL key
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
print(f"\n(ours: closed = coax + primer on the ORIGINAL paper key (§5.4), matched to the key the paper's "
      f"leaderboard rows are measured on; open = coax + primer full-578 free-text, GPT-4o judge (§6.1); "
      f"no exemplars, matched across the two halves. Paper rows quoted from arXiv:2509.09254 Tables 2-3. "
      f"On our debiased balanced key the same config scores {closed_bal:.1f}% -> Avg "
      f"{(closed_bal+op)/2:.1f}, still above OralGPT's {(39.60+52.77)/2:.1f}.)")
