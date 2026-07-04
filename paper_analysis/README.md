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
| **§5.1** whole/clean/blank table + two-proportion z-test | `paper_analysis/blank_split.py` | `results/closed_ended/reproduction/gpt-4o-2024-11-20__faithful-direct-k0__whole__n491.csv` | `_generated/blank_split_table.md`, `_generated/blank_split.values.json` |
| **§5.2** faithful-vs-coax prompt table + McNemar (clean 453) | `paper_analysis/prompt_axis.py` | `results/closed_ended/prompt_axis/gpt-4o-2024-11-20__coax-direct-k0__whole__n491.csv` + the E0-repro faithful CSV | `_generated/prompt_axis_table.md`, `_generated/prompt_axis.values.json` |
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
python paper_analysis/blank_split.py
# ...one line per generator as they are added
```

A single `build_paper.py` in the research repo (Tier 2, later) will run all of these,
inject the fresh table fragments into `PAPER_DRAFT.md` between `<!-- GEN -->` markers,
regenerate figures, and re-render the PDF.
