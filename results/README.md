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
| `closed_ended/gpt-4o-2024-11-20__faithful-direct-k0__whole__n491.csv` | E-whole491 | §5.1/§5.2 | gpt-4o-2024-11-20 | faithful (VLMEvalKit verbatim prompt+parser), direct, k=0, img_detail=high, max_tokens=8192, temp=0 | whole 491 (canonical, blanks="None") | 491 | 2026-07-06 | MERGED, no API: `python dataio/build_whole491.py` (453 non-"None" rows + 38 "None" rows from `_superseded/`) |
| `closed_ended/gpt-4o-2024-11-20__coax-direct-k0__whole__n491.csv` | E-whole491 | §5.2 | gpt-4o-2024-11-20 | coax (persona + commit-to-one-letter), direct, k=0, img_detail=high, temp=0 | whole 491 (canonical) | 491 | 2026-07-06 | MERGED, no API: `python dataio/build_whole491.py` |
| `closed_ended/position_bias/…__clean__n453.csv`, `…__clean-shuffled__n453.csv` (gpt-4o + gemini-2.5) | E1-stability | §5.3.3 `[DEFERRED]` | gpt-4o / gemini-2.5-flash | coax/faithful, direct, k=0, temp=0 | clean / clean-shuffled (453) | 453 | 2026-07-04/05 | §5.3 content-stability runs; the `clean*` parquets they used were removed — §5.3 rework pending |
| `closed_ended/position_bias/gpt-4o-2024-11-20__coax-direct-k0__shuffled__n491.csv` | E-shuffle491 | §5.3.1 (Table 5.3) | gpt-4o-2024-11-20 | coax (revised prompt), direct, k=0, img_detail=high, max_tokens=8192, temp=0 | shuffled 491 (`data/closed_ended_shuffled.parquet`, frozen) | 491 | 2026-07-06 | `python eval_closed_gpt.py --model gpt-4o-2024-11-20 --prompt coax --detail high --data data/closed_ended_shuffled.parquet --start 0 --out …/…__shuffled__n491.csv`. Acc 48.9% (orig-key 54.2% → −5.3); 0 refusals, 0 fallbacks. Paired with the coax whole-491 (original key) at identical settings. Powers `paper_analysis/shuffle_drop.py`. |
| `closed_ended/position_bias/gpt-4o-2024-11-20__faithful-direct-k0__shuffled__n491.csv` | E-faithful-shuffle491 | §5.3.1 (Table 5.3, 2×2) | gpt-4o-2024-11-20 | faithful (VLMEvalKit prompt+parser, random fallback), direct, k=0, img_detail=high, max_tokens=8192, temp=0 | shuffled 491 (`data/closed_ended_shuffled.parquet`, frozen) | 491 | 2026-07-06 | `python eval_closed_gpt.py --model gpt-4o-2024-11-20 --prompt faithful --detail high --data data/closed_ended_shuffled.parquet --start 0 --out …/…faithful…__shuffled__n491.csv`. Acc 40.5% (orig-key 43.2% → −2.7); 1.2% random fallback, 0 refusals. Completes the prompt×key 2×2; feeds `paper_analysis/shuffle_drop.py`. |
| `closed_ended/position_bias/gemini-2.5-flash__coax-direct-k0__{whole,shuffled}__n491.csv` | E-gemini25-paired491 | §5.3.4 (Table 5.4) | gemini-2.5-flash | coax (revised) direct, k=0, thinking-budget 0, temp 0 | whole 491 + shuffled 491 (`closed_ended{,_shuffled}.parquet`, frozen) | 491 | 2026-07-06 | `python eval_closed_gemini.py --model gemini-2.5-flash --prompt coax --thinking-budget 0 --data data/closed_ended{,_shuffled}.parquet`. orig 44.6% → shuf 43.6%; position-robust (both-correct) 34.6%; letter-bias 8%. Feeds `paper_analysis/effective_score.py`. |
| `closed_ended/position_bias/gemini-3.5-flash__coax-direct-k0__{whole,shuffled}__n491.csv` | E-gemini35-paired | §5.3.4 (Table 5.4) | gemini-3.5-flash | coax (revised) direct, k=0, thinking-budget 0, temp 0 | whole 491 + shuffled 491 (frozen) | 491 | 2026-07-06 | `python eval_closed_gemini.py --model gemini-3.5-flash --prompt coax --thinking-budget 0 --data data/closed_ended{,_shuffled}.parquet`. orig 59.9% → shuf 58.0%; position-robust (both-correct) 50.1%; letter-bias 11%. Feeds `paper_analysis/effective_score.py`. |
| `_pilot/gemini25_determinism_probe.csv` | determinism-check | §5.3.3 / §7 | gemini-2.5-flash | coax direct k=0 think=0 temp=0 | original key, first 100 items | 100 | 2026-07-06 | determinism probe: re-run of the same original-key items ~1h after the whole-491 run. 99% agreement within session (vs 74% across the days to the old 453 run → model drift). Backs the §5.3.3 determinism claim + §7 drift caveat. Pilot, not a paper number. |
| `closed_ended/cot_length/gemini-3.5-flash__direct-think{0,512,2048,8192,-1}__hard50-shuffled__n50.csv` | E-thinking-hard | §5.4 (Table 5.5) | gemini-3.5-flash | coax (revised) direct, k=0, thinking-budget swept {0,512,2048,8192,dynamic}, temp 0 | 50 hard items (wrong for BOTH gpt-4o AND gemini-2.5 on the shuffled key), shuffled | 50 | 2026-07-07 | `python eval_closed_gemini.py --model gemini-3.5-flash --prompt coax --thinking-budget <B> --data data/closed_ended_hard50_shuffled.parquet`. Subset built by `python dataio/make_hard_subset.py` (regenerable; the 17MB parquet is NOT committed). Acc 46→48→50→50→54%; no paired step significant (0→8192 net +2 p=0.77; 0→dynamic net +4 p=0.39). Generator `paper_analysis/thinking_hard.py`. |
| `dentist_audit/survey_manifest.csv` | dentist-audit (probe) | §7 / DENTIST_AUDIT.md | — (dentist rater) | 60 closed (buckets T1/T2/T3/C on the shuffled key) + 15 open prose | selection from the 491, seed 20260707 | 75 | 2026-07-07 | `python dataio/make_dentist_survey.py`. Item manifest for the ground-truth probe; viewer bundle `python dataio/export_survey_bundle.py` → `survey/`. Analysis: `python paper_analysis/dentist_audit.py <submission.json>`. Not a model run. |
| `closed_ended/_superseded/*` | — | — | gpt-4o-2024-11-20 | the nan-rendered whole-491 runs + the blanks38-none runs | — | — | — | quarantined; the raw inputs merged into the two canonical 491 files above |

**E-whole491 (§5.1/§5.2):** GPT-4o on all **491** canonical questions (blanks rendered "None"), built by merging the 453 non-"None" rows (raw outputs identical under nan/None at temp=0) with the 38 "None" rows (`dataio/build_whole491.py`, **no API**). faithful (benchmark prompt + parser) **43.2%** — reproduces the paper's GPT-4o **45.40%** within CI (residual = third-party proxy + model drift); coax **54.2%**. The old nan-rendered whole-491 runs and the separate blanks38-none runs are quarantined in `closed_ended/_superseded/` as the merge inputs.

**§5.2 parsing-vs-prompting (all numbers regenerable, no API — §1.0 rule 7):**
- On the **453 questions with no "None" option**: benchmark prompt read by the **benchmark parser** = **44.4% [39.9–49.0]**; by the model's **true** (hand-verified) answer = **49.9% [45.3–54.5]** (parser misreads **49/453**, ~5.5 pts); **coax** = **53.6% [49.0–58.2]** (453/453 bare replies, 0 fallbacks). Combined **+9.2** = ~5.5 parser + ~3.7 prompt; McNemar p **1.6e-04**. Generator `paper_analysis/faithful_true_accuracy.py` (+ `faithful_hand_labels.csv`).
- On the **32 "None"-correct questions** (the §5.2 showcase): benchmark pipeline **25.0%** → model's true answer **28.1%** → **coax 53.1%** (parser +3, prompt +25). Generator `paper_analysis/none_showcase.py`.
- **Decision: coax prompt + keep the benchmark's own parser for all MCQ work** (no custom parser in any result).

**E1-stability (§5.3.3):** each model run once on the clean key and once on clean-shuffled, paired at the item level: content-stable (same option text) / letter-stable (same letter, different option) / neither. Generator `paper_analysis/position_stability.py` (combined acc+stability table; `--compare-prompts` for the faithful check) + figure `figures/make_position_stability.py`.
- **GPT-4o coax:** content-stable **61.4%**, letter-stable 17.0%, neither 21.6%; acc 53.6→48.3 (drop 5.3).
- **Gemini-2.5-flash coax:** content-stable **61.8%**, letter-stable 12.1%, neither 26.0%; acc 43.5→43.3 (drop **0.2**).
- **Headline:** near-identical robustness (~61% content-stable) despite very different accuracy drops — both flip ~38% of items; gemini's cancel out at the aggregate, so **accuracy-under-shuffle is a misleading measure and content-stability is the honest one**.
- **GPT-4o faithful (robustness check) — `[PENDING re-elicitation]`:** the faithful-shuffled run (`...faithful-direct-k0__clean-shuffled__n453.csv`) is **partly corrupted**: an OpenAI **quota outage (429s)** mid-run left its last **102/453** answers empty (see the run's tail in `gpt_faithful_shuf.log`), which `faithful_predict` random-guessed. `position_stability.py` now **excludes** empty/failed rows. On the 351 valid items, faithful content-stable is **71.8%** (preliminary; not directly comparable to coax's 61.4% on n=453). The old committed **60.3%** was the corrupted figure and is retracted. Re-elicit the 102 rows (delete them, re-run the ledgered command; the fixed client now raises `APICallFailed` instead of writing empties) for a clean full-set number.

_Pilots under `results/_pilot/` are exploratory (prompt-mode + image-detail sweeps) and are intentionally NOT ledgered._

**Data normalization to "None" (2026-07-05) — see `data/README.md`.** Blank options were being
rendered as `"nan"` (a reproduction bug); the benchmark renders them `"None"`
(`MMOral_OPG_CLOSED.post_build`). We now normalize once (`dataio/convert_to_none.py`) into the
canonical `data/closed_ended.parquet`; the raw NaN file is archived (`closed_ended_raw.parquet`)
and never used by a run (harnesses load via `dataio/eval_data.read_closed`, which refuses NaN).

_Blank-answer (`results/closed_ended/blank_answer/`) runs: GPT-4o on the 38 "None" items._
The `coax` run (bare letters, 53.1% on the 32 "None"-correct items) is clean. The `faithful`
run's committed `correct` column (6.2%) predates the parser `index2ans` fix and still reflects
the "nan" `index2ans` bug; its `raw_response` is real and re-scores to ~25% with "None"
`index2ans`. Both are kept as the record; a clean re-run on the None canonical is the next step.

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
