# paper_analysis — generators for every table and figure in the paper

**Rule (RESEARCH_PLAN §1.0 rule 7, MANDATORY).** No table or figure enters
`PAPER_DRAFT.md` without a committed generator here that produces it **from a committed
raw-output CSV** (the `raw_response` column of a result file in `results/`), with **zero
new API calls**. Run a generator to (re)produce its artifact into `_generated/`. This is
the analysis-side of the data-integrity guardrail: every quantitative element in the paper
is regenerable by running committed code against committed raw outputs.

## How it works

- Each generator is one script named for the paper element it produces.
- It reads committed CSV(s) under `results/`, computes the numbers, and writes into
  `_generated/`:
  - a **Markdown table fragment** (`<name>_table.md`) for a paper table, and/or
  - a **`<name>.values.json`** holding the scalars (accuracies, CIs, test statistics)
    plus provenance (`_generator`, `_source_csv`, `_generated_utc`), and/or
  - a **`figures/<name>.png`** for a chart.
- Every generated file carries provenance: table fragments begin with an HTML comment
  naming the generator + source CSV + timestamp; JSON carries the `_generator` /
  `_source_csv` / `_generated_utc` keys. Generated files are committed but never hand-edited.

## Manifest — paper element → generator → source → artifact

| Paper element | Generator | Source CSV(s) | Generated artifact(s) |
|---|---|---|---|
| **RESEARCH_PLAN §3.7** (NOT in paper) "none of the above" showcase: benchmark-pipeline vs true-answer vs coax on the 32 "None"-correct items. Retired from the paper in v0.10 (the paper's §5.2 makes the point on the full 491 and does not single out "None" items); kept as the most extreme illustration of the same effect. | `paper_analysis/none_showcase.py` | faithful + coax whole-491 CSVs (filtered to the 32 "None"-correct); canonical option text for the "None" index2ans | `_generated/none_showcase_table.md`, `_generated/none_showcase.values.json` |
| **§5.2** parsing-vs-prompting (the single §5.2 table, on the **full 491** released set): faithful parser-acc vs hand-verified TRUE acc vs coax (each with Wilson CI); 52 parser misreads; coax bare-reply + zero-fallback proof; benchmark-vs-coax pipeline McNemar (supersedes the retired `prompt_axis.py`) | `paper_analysis/faithful_true_accuracy.py` (+ committed hand-labels `paper_analysis/faithful_hand_labels.csv`) | faithful + coax whole-491 CSVs; hand-labels for the 9 ambiguous replies | `_generated/faithful_true_accuracy_table.md`, `_generated/faithful_true_accuracy.values.json` |
| **§5.3.1** answer-key distribution table (dataset-derived, not model outputs) | `paper_analysis/key_skew.py` | `data/closed_ended_clean.parquet` (+ complete, clean-shuffled) | `_generated/key_skew_table.md`, `_generated/key_skew.values.json` |
| **§5.3** content-stable / letter-stable / neither, per model | `paper_analysis/position_stability.py` (+ figure `dental_research/figures/make_position_stability.py`) | each model's coax run on clean (original key) + clean-shuffled, mapped through the two parquets | `_generated/position_stability_table.md`, `_generated/position_stability.values.json`, `dental_research/figures/position_stability.png` |
| **§5.3.2** few-shot × reasoning grid (F3/T1) — acc [CI], %A, χ² per cell | `paper_analysis/nshot_grid.py` | `results/nshot/closed_gemini-{2.5,3.5}-flash_k{0,1,3,5}_cleanshuf_think0[_cot].csv` — **predictions re-derived from `raw_response`** (never the stored `predicted`; the old harness mis-parsed verbose 3.5-flash replies) | `_generated/nshot_grid_table.md` (full 16-cell), `_generated/nshot_grid_condensed.md` (§5.3.2 table), `_generated/nshot_grid.values.json` |
| **§5.2** prompt-gain vs leaderboard figure (ranked bars) | `dental_research/figures/make_prompt_gain_leaderboard.py` *(figure generators that need the paper leaderboard live in the research repo; it reads our GPT-4o run CSVs from this code repo by sibling path)* | paper Table-2 leaderboard (`paper_text.txt`) + `results/closed_ended/{reproduction,prompt_axis}/…whole__n491.csv` | `dental_research/figures/prompt_gain_leaderboard.png` |
| **§6 (F8)** closed-vs-open landscape figure | *(to migrate here: currently `dental_research/figures/make_closed_vs_open.py`)* | paper leaderboard + our open-ended runs | `figures/closed_vs_open.png` |

*(Rows are added as each table/figure is built. Pending paper tables/figures — the §5.2
susceptibility panel (E1), the debiased mini-leaderboard, etc. — get a generator here the
same session their run lands.)*

## Figures convention

Every chart is a `make_<name>.py` (matplotlib) that reads a committed CSV and writes
`figures/<name>.png`. No chart is pasted or hand-drawn. The F8 generator
(`dental_research/figures/make_closed_vs_open.py`) is the existing pattern and should
migrate into `paper_analysis/figures/` so all generators live in one place in the code repo.

## Regenerate everything

Run each generator (they are independent):

```
python paper_analysis/none_showcase.py
# ...one line per generator as they are added
```

A single `build_paper.py` in the research repo (Tier 2, later) will run all of these,
inject the fresh table fragments into `PAPER_DRAFT.md` between `<!-- GEN -->` markers,
regenerate figures, and re-render the PDF.
