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
    prompt_axis/     # §5.2  faithful vs coax prompt comparison (E-prompt-axis)
    position_bias/   # §5.3  key-skew, susceptibility grid, shuffle delta, corrected set
    cot_length/      # §5.4  reasoning-length sweep      (E-cot-length)
    nshot/           # §5.5  few-shot accuracy sweep     (E-nshot)
    no_image/        # §5.6  no-image / language-prior control (E3)
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

## Self-describing files (metadata sidecar)

Every result `<name>.csv` is written together with a sidecar `<name>.csv.meta.json` that holds
its full identity, so a file reconstructs on its own even without this manifest. The **CSV
stays pristine** (no comment lines) — it opens correctly in plain `pd.read_csv`, Excel, or
grep, with no chance of confusing metadata with the data (which contains `#`, e.g. `#44`).

Write with `results_io.write_results(df, path, meta)`; read with
`results_io.load_results(path, return_meta=True)` (code repo, `utils/results_io.py`). Example sidecar:

```json
{
  "experiment": "E-nshot",
  "paper_section": "§5.4",
  "model": "gemini-3.5-flash",
  "config": "cot k=5 think-off",
  "dataset": "clean-shuffled",
  "slice": "idx50-149",
  "n": 100,
  "command": "python eval_closed_gemini.py --model gemini-3.5-flash --k 5 --cot ...",
  "code_commit": "eaa3e4a",
  "generated_utc": "2026-07-04T11:53:26+00:00"
}
```

Commit the `.csv` and its `.meta.json` together. The sidecar travels with the file; this
manifest is the searchable index across all files.

## Manifest

| Path | Experiment (E#) | Paper § | Model | Config | Dataset | n | Date | Command |
|------|-----------------|---------|-------|--------|---------|---|------|---------|
| `closed_ended/reproduction/gpt-4o-2024-11-20__faithful-direct-k0__whole__n491.csv` | E0-repro | §2/§5.1 | gpt-4o-2024-11-20 | faithful (VLMEvalKit verbatim prompt+parser), direct, k=0, img_detail=high, max_tokens=8192, temp=0, no resize/system; OpenAI-direct | whole (491) | 491 | 2026-07-04 | `python eval_closed_gpt.py --model gpt-4o-2024-11-20 --prompt faithful --detail high --data data/closed_ended.parquet --start 0` |
| `closed_ended/prompt_axis/gpt-4o-2024-11-20__coax-direct-k0__whole__n491.csv` | E-prompt-axis | §5.2 | gpt-4o-2024-11-20 | **coax** (persona + commit-to-one-letter, strict parse), direct, k=0, img_detail=high, temp=0; OpenAI-direct | whole (491) | 491 | 2026-07-04 | `python eval_closed_gpt.py --model gpt-4o-2024-11-20 --prompt coax --detail high --data data/closed_ended.parquet --start 0` |
| `closed_ended/position_bias/gpt-4o-2024-11-20__coax-direct-k0__clean-shuffled__n453.csv` | E1-stability | §5.3.3 | gpt-4o-2024-11-20 | coax, direct, k=0, img_detail=high, temp=0; OpenAI-direct | clean-shuffled (453) | 453 | 2026-07-04 | `python eval_closed_gpt.py --model gpt-4o-2024-11-20 --prompt coax --detail high --data data/closed_ended_clean_shuffled.parquet --start 0` |
| `closed_ended/position_bias/gemini-2.5-flash__coax-direct-k0__clean__n453.csv` | E1-stability | §5.3.3 | gemini-2.5-flash | coax, direct, k=0, temp=0, thinking off | clean (453) | 453 | 2026-07-05 | `python eval_closed_gemini.py --model gemini-2.5-flash --prompt coax --data data/closed_ended_clean.parquet` |
| `closed_ended/position_bias/gemini-2.5-flash__coax-direct-k0__clean-shuffled__n453.csv` | E1-stability | §5.3.3 | gemini-2.5-flash | coax, direct, k=0, temp=0, thinking off | clean-shuffled (453) | 453 | 2026-07-05 | `python eval_closed_gemini.py --model gemini-2.5-flash --prompt coax --data data/closed_ended_clean_shuffled.parquet` |
| `closed_ended/position_bias/gpt-4o-2024-11-20__faithful-direct-k0__clean-shuffled__n453.csv` | E1-stability (faithful robustness) | §5.3.3 | gpt-4o-2024-11-20 | faithful (VLMEvalKit verbatim), direct, k=0, img_detail=high, temp=0; OpenAI-direct | clean-shuffled (453) | 453 | 2026-07-05 | `python eval_closed_gpt.py --model gpt-4o-2024-11-20 --prompt faithful --detail high --data data/closed_ended_clean_shuffled.parquet --start 0` |

**E0-repro result:** 41.8% [95% CI 37.5–46.2] — reproduces the paper's GPT-4o closed-ended **45.40%** within CI. Real rows 44.2% (n=459), blank rows 6.2% (n=32). Only paper API model still callable; residual vs 45.40 = third-party proxy + model drift. Full write-up: `RESEARCH_PLAN §5.6` ledger + `PAPER_DRAFT §4`.

**Derived analysis E4/D1 (no API):** `paper_analysis/blank_split.py` re-scores the E0-repro CSV (its raw `raw_response` outputs) into the §5.1 whole-vs-clean table: whole (491) 41.8% [37.5–46.2], clean (453) 44.4% [39.9–49.0], the 32 unanswerable items 6.2% [1.7–20.1]; two-proportion z (clean vs blank) = 4.22, p = 2.4e-05. Run `python paper_analysis/blank_split.py` to regenerate `paper_analysis/_generated/blank_split_table.md` + `.values.json`. **Rule (§1.0 rule 7): every paper table/figure has a committed generator in `paper_analysis/`; every number regenerable from a raw-output CSV — no API calls, no hand-typed figures.**

**E-prompt-axis (§5.2):** faithful vs coax prompt for GPT-4o on the clean 453 (paired). faithful **44.4% [39.9–49.0]**, coax **53.6% [49.0–58.2]**, lift **+9.2 pts**, McNemar p **1.6e-04** (coax-right 80 / faithful-right 38); coax refusals 0%. Generator `paper_analysis/prompt_axis.py` → `_generated/prompt_axis_table.md`. Paper adopts coax going forward.

**E1-stability (§5.3.3):** each model run once on the clean key and once on clean-shuffled, paired at the item level: content-stable (same option text) / letter-stable (same letter, different option) / neither. Generator `paper_analysis/position_stability.py` (combined acc+stability table; `--compare-prompts` for the faithful check) + figure `figures/make_position_stability.py`.
- **GPT-4o coax:** content-stable **61.4%**, letter-stable 17.0%, neither 21.6%; acc 53.6→48.3 (drop 5.3).
- **Gemini-2.5-flash coax:** content-stable **61.8%**, letter-stable 12.1%, neither 26.0%; acc 43.5→43.3 (drop **0.2**).
- **Headline:** near-identical robustness (~61% content-stable) despite very different accuracy drops — both flip ~38% of items; gemini's cancel out at the aggregate, so **accuracy-under-shuffle is a misleading measure and content-stability is the honest one**.
- **GPT-4o faithful (robustness check) — `[PENDING re-elicitation]`:** the faithful-shuffled run (`...faithful-direct-k0__clean-shuffled__n453.csv`) is **partly corrupted**: an OpenAI **quota outage (429s)** mid-run left its last **102/453** answers empty (see the run's tail in `gpt_faithful_shuf.log`), which `faithful_predict` random-guessed. `position_stability.py` now **excludes** empty/failed rows. On the 351 valid items, faithful content-stable is **71.8%** (preliminary; not directly comparable to coax's 61.4% on n=453). The old committed **60.3%** was the corrupted figure and is retracted. Re-elicit the 102 rows (delete them, re-run the ledgered command; the fixed client now raises `APICallFailed` instead of writing empties) for a clean full-set number.

_Pilots under `results/_pilot/` are exploratory (prompt-mode + image-detail sweeps) and are intentionally NOT ledgered._

**Parser fix + re-score (2026-07-05).** The answer extractor (`clients/parsing.py::extract_letter`)
was rewritten to read a model's *last* answer declaration instead of the first A/B/C/D
character, which the old version mis-matched inside words ("**A**nswer" → A, "**B**ased on" → B)
on verbose replies. Only the four **`results/nshot/closed_gemini-3.5-flash_k{0,1,3,5}_cleanshuf_think0.csv`**
(direct) files were affected; their `predicted`/`correct` columns were re-derived in place from
the untouched `raw_response` via `python dataio/rescore_predictions.py` (no API). Effect: direct
accuracy k0 48→50, k3 57→60, k5 54→62 (k1 unchanged); %A stays low (no few-shot spike). 2.5-flash
and all CoT cells are byte-identical. The §5.3.2 grid regenerates from raw output via
`python paper_analysis/nshot_grid.py`.

## Quarantine

Superseded-but-real results move to a `_superseded/` subfolder with a one-line reason and a
`SUPERSEDED` note in the manifest. Fabricated or coerced inputs are **never** committed
(kept local only — see the RESEARCH_PLAN Data-integrity guardrail).
