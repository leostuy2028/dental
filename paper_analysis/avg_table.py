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

SECOND TABLE (added 2026-07-25): the DECOMPOSITION. The headline "a generalist beats
the dental model" is weak on its own, because our best generalist is a newer model than
any the paper tested, so a reviewer can attribute the whole gap to model progress. The
decomposition answers that objection with the paper's OWN model: GPT-4o, same 491
questions, same X-rays, same benchmark parser, moves +11.0 on the closed half from a
prompt change alone. Averaged across both halves it reaches parity with OralGPT WITHOUT
any model progress; clearing OralGPT takes the prompt change AND a newer generation.
That separation is the defensible claim, so both tables are generated here together.

  python -m paper_analysis.avg_table
"""
import pandas as pd

CLOSED_ORIG = "results/closed_ended/knowledge_context/gemini-3.5-flash__coax-direct-ctx-opgprimer__whole__n491.csv"
CLOSED_BAL = "results/closed_ended/knowledge_context/gemini-3.5-flash__coax-direct-ctx-opgprimer__shuffled__n491.csv"
closed = pd.read_csv(CLOSED_ORIG).correct.mean() * 100      # original key = paper's key
closed_bal = pd.read_csv(CLOSED_BAL).correct.mean() * 100   # our debiased key (footnote)
op = pd.read_csv("results/open/batched_gemini35_plain578_scores.csv").score.mean() * 100

# GPT-4o, the paper's own model, under the two prompts — the decomposition's evidence.
G4_FAITHFUL = "results/closed_ended/gpt-4o-2024-11-20__faithful-direct-k0__whole__n491.csv"
G4_COAX = "results/closed_ended/gpt-4o-2024-11-20__coax-direct-k0__whole__n491.csv"
G4_OPEN = "results/open/batched_gpt4o_matched_scores.csv"        # prompt-matched: primer, NO coords
G4_OPEN_COORD = "results/open/coordarms_gpt4o_cpc_all_scores.csv"  # the older coordinate-elicited arm
g4_faithful = pd.read_csv(G4_FAITHFUL).correct.mean() * 100
g4_coax = pd.read_csv(G4_COAX).correct.mean() * 100
g4_open = pd.read_csv(G4_OPEN).score.mean() * 100
g4_open_coord = pd.read_csv(G4_OPEN_COORD).score.mean() * 100

# paper-reported (arXiv:2509.09254): (label, closed, open) — all on the ORIGINAL key
PAPER = [
    ("OralGPT (paper's best-avg model)", 39.60, 52.77),
    ("GPT-4o (paper)", 45.40, 37.50),
    ("Claude-3.7-Sonnet (paper)", 41.40, 40.67),
]
ORALGPT_AVG = (39.60 + 52.77) / 2

print("| Model | Closed (MCQ) | Open (free-text) | Avg. |")
print("|:--|--:|--:|--:|")
print(f"| **gemini-3.5-flash (coax + primer)** | {closed:.1f} | {op:.1f} | **{(closed+op)/2:.1f}** |")
for lbl, c, o in PAPER:
    print(f"| {lbl} | {c:.2f} | {o:.2f} | {(c+o)/2:.2f} |")
print(f"\n(ours: closed = coax + primer on the ORIGINAL paper key (§5.4), matched to the key the paper's "
      f"leaderboard rows are measured on; open = coax + primer full-578 free-text, GPT-4o judge (§6.1); "
      f"no exemplars, matched across the two halves. Paper rows quoted from arXiv:2509.09254 Tables 2-3. "
      f"On our debiased balanced key the same config scores {closed_bal:.1f}% -> Avg "
      f"{(closed_bal+op)/2:.1f}, still above OralGPT's {ORALGPT_AVG:.1f}.)")

# ---- decomposition table: how much of the specialist's lead is measurement? ----
g4_avg = (g4_coax + g4_open) / 2
print("\n\n| Step | Closed (MCQ) | Open (free-text) | Avg. | vs OralGPT |")
print("|:--|--:|--:|--:|:--|")
print(f"| GPT-4o as the paper measured it | {45.40:.1f} | {37.50:.1f} | {(45.40+37.50)/2:.1f} | "
      f"{(45.40+37.50)/2 - ORALGPT_AVG:+.1f} |")
print(f"| GPT-4o, our reproduction of that pipeline | {g4_faithful:.1f} | — | — | — |")
print(f"| **Same model, prompted properly** (no model change) | **{g4_coax:.1f}** | **{g4_open:.1f}** | "
      f"**{g4_avg:.1f}** | **{g4_avg - ORALGPT_AVG:+.1f}** |")
print(f"| OralGPT (the purpose-built dental model) | {39.60:.1f} | {52.77:.1f} | {ORALGPT_AVG:.1f} | — |")
print(f"| *(for reference)* a newer generation, same treatment | {closed:.1f} | {op:.1f} | "
      f"{(closed+op)/2:.1f} | {(closed+op)/2 - ORALGPT_AVG:+.1f} |")
print(f"\n(the prompt change on the paper's OWN model, same 491 items and same X-rays: "
      f"{g4_coax - g4_faithful:+.1f} points on the closed half, of which ~5.3 is the benchmark's parser "
      f"misreading answers the model got right (§5.2). Open half = the prompt-matched run "
      f"(primer, NO coordinate instruction, batched, gpt-4o-2024-11-20): {g4_open:.1f}% vs "
      f"{g4_open_coord:.1f}% for the coordinate-eliciting arm, a paired +{g4_open - g4_open_coord:.1f} "
      f"concentrated in the prose items. Both halves are the same pinned model snapshot, so the "
      f"{g4_avg:.1f} average involves NO model progress. Closed cell carries no primer while the open "
      f"cell does, which understates rather than flatters it.)")
