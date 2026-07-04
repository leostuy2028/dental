# Results index — the single source of truth for "which file is which experiment"

All result CSVs are committed to this repo on purpose: they come from paid API runs and
must not be regenerated. This file maps **every** result file to its experiment, so results
are retrievable without guessing. **Rule: no result CSV is committed without its manifest row
below, added in the same session the file is produced.** (Convention: RESEARCH_PLAN §1.0.)

## Folder layout (one folder per paper section, mirrors PAPER_DRAFT.md §5–§6)

```
results/
  closed_ended/
    blank_answer/    # §5.1  blank-answer artifact + reasoning-tax re-scores
    position_bias/   # §5.2  key-skew, susceptibility grid, shuffle delta, corrected set
    cot_length/      # §5.3  reasoning-length sweep      (E-cot-length)
    nshot/           # §5.4  few-shot accuracy sweep     (E-nshot)
    no_image/        # §5.5  no-image / language-prior control (E3)
  open_ended/
    grader_audit/    # §6.1  grader-prompt analysis
    frontier/        # §6.2  answerer runs, judge comparison, reproduction (E9/E10)
```

## File naming

`<model>__<config>__<dataset>__<slice>.csv` — lowercase; fields joined by `__`; tokens
inside a field by `-`.
- **model**: `gemini-2.5-flash` | `gemini-3.5-flash` | `claude-haiku-4-5` | `claude-opus-4-8` | `gpt-4o`
- **config**: mode `direct`|`cot`; shots `k0`…`k5`; thinking `think-off`|`think-on`|`think-<budget>`; open-ended judge `judge-gpt4o`|`judge-gemini`|`judge-claude`
- **dataset**: `whole` | `clean` | `clean-shuffled` | `prose-ref` | `coord-ref`
- **slice**: `idx50-149` | `n100`, etc. Open-ended runs suffix `__answers` / `__scores`.

Example: `results/closed_ended/position_bias/gemini-3.5-flash__cot-k5__clean-shuffled__idx50-149.csv`

## Self-describing files (metadata inside every CSV)

Every result file also carries its own metadata as a leading `#`-comment block, so a file
can be reconstructed even without this manifest. Write with `results_io.write_results`,
read with `results_io.load_results` (repo root, `results_io.py`):

```
# experiment: E-nshot
# paper_section: §5.4
# model: gemini-3.5-flash
# config: cot k=5 think-off
# dataset: clean-shuffled
# slice: idx50-149
# n: 100
# command: python eval_closed_gemini.py --model gemini-3.5-flash --k 5 --cot ...
# code_commit: 4362b4f
# generated_utc: 2026-07-04T11:50:05+00:00
index,question,option1,...            <- data starts here
```

**Read these files with `results_io.load_results(path)`, not plain `pd.read_csv`.** The data
contains `#` (tooth numbers like `#44`); `load_results` strips only the leading comment block
by line-prefix, so `#` inside the data is preserved. (Plain `pd.read_csv(comment='#')` would
corrupt the data — do not use it here.)

## Manifest

| Path | Experiment (E#) | Paper § | Model | Config | Dataset | n | Date | Command |
|------|-----------------|---------|-------|--------|---------|---|------|---------|
| _(no results generated under this scheme yet — first regenerated run adds its row here)_ | | | | | | | | |

## Quarantine

Superseded-but-real results move to a `_superseded/` subfolder with a one-line reason and a
`SUPERSEDED` note in the manifest. Fabricated or coerced inputs are **never** committed
(kept local only — see the RESEARCH_PLAN Data-integrity guardrail).
