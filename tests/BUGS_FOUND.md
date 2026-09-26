# Suspected bugs found while writing the tests

None of these were fixed. The tests lock in what the code does **today**, and
each one is marked in the test file with `# CURRENT BEHAVIOR (looks like a bug)`.
When the rewrite fixes one, update that test on purpose so the change is a
decision, not an accident.

Checked by hand so far: the "a" → A problem and the hidden backspace character.
The rest were each reproduced by a test, so they are real behavior, but whether
each one *matters* still needs a human look.

## 1. Could change numbers in the paper

| Where | What happens |
|---|---|
| `clients/parsing.py` `extract_letter` | The English word "a" is read as answer **A**. `"Is it a cyst?"` → `A`. A reply with no answer at all can be scored as picking A, which is the exact bias the paper measures. |
| `utils/vlmeval_parse.py` (both functions) | With several letters in parentheses it takes the **earliest**, not the final answer: "Option (A) is wrong, the answer is (D)" → `A`. This copies the benchmark's own parser, so it may be intentional (reproducing their numbers). |
| `utils/vlmeval_parse.py` (both functions) | An answer written " B." or " B," is found, then its position can't be looked up, so the function falls back to a **random guess**. Also likely copied from the benchmark on purpose. |
| `paper_analysis/key_skew.py:35` | When two letters tie for most common, it silently picks the first in A-B-C-D order. |
| `paper_analysis/shuffle_drop.py:97` | The printed `n` and the "floor" numbers come from whichever results file is read **last**, not the original-key run. Correct today only because all four files have the same row count. |
| `paper_analysis/count_headtohead.py:111` | If the detector results file is missing, a row of left-over "Detector" numbers (e.g. 2200%) is still added to the table. |
| `paper_analysis/corrected_benchmark.py:108` | A question that is both a coordinate-answer item and "vague" is labeled only vague; its coordinate problem is dropped. |
| `paper_analysis/boneloss_footprint.py:39` | "Keyed no bone loss" only counts answers whose text starts with "No bone loss" (or similar). An option written "None" is missed, so the count may be low. |
| `paper_analysis/boneloss_audit.py:39` | An unexpected rating such as "9" is counted as "no bone loss" instead of flagged. |
| `paper_analysis/faithful_true_accuracy.py:72` | Reads a bolded `**B.**` but not a lowercase `**b.**`. |
| `paper_analysis/quality_audit.py:65` | Prints the **largest survey number** as the **number of surveys** (surveys 1 and 3 → "in 3 surveys"). |
| `curated/build_curated.py:84` | The "bone loss" warning is attached to any question mentioning bone loss, even ones marked KEEP. |
| `detector/infer_mmoral.py:20` (same in `tune_inference.py:53`) | Tooth-count reader needs the number right before "teeth"; "5 permanent teeth" gives no count. |
| `eval_open/judges.py:45` | A judge score of "1.5" is read as 1.0 instead of rejected as out of range. |
| `eval_open/run_frontier.py:41` | A blank answer (only spaces) counts as "complete". |
| `eval_open/run_coord_arms.py:49` | A missing answer (`None`) is not counted as a refusal. |
| `dataio/make_dentist_survey.py:49` | `N_T2` is never used; the real T2 count is whatever is left of 60. |

## 2. Safety checks that don't work

| Where | What happens |
|---|---|
| `dataio/export_quality_survey.py:260` | An invisible **backspace character** is inside the "nan leaked into the survey" check, so an option shown as "A) nan" is never caught. (Confirmed in the committed file.) |
| `dataio/check_na_roundtrip.py:55` | A `_superseded/` folder at the top of the repo is not skipped; nested ones are. |
| `eval_open/judges.py:132` | An "Invalid API Key" error is retried 15 times instead of stopping right away. |
| `eval_open/prompts_open.py:73` | An image width/height of 0 is accepted instead of rejected. |

## 3. Crashes

| Where | When it crashes |
|---|---|
| `eval_open/run_ensemble.py:122` | Resuming a partly finished run when there is new work to add. Resume doesn't really work. |
| `eval_open/run_inject_other.py:65` | A missing baseline score makes the progress print fail; it is then misreported as "STOPPED (Gemini cap?)" and the run ends early. |
| `eval_closed_claude.py:92`, `eval_closed_fewshot.py:136`, `eval_closed_gpt.py` | `--out` pointing anywhere other than `results/` (progress file path is hard-coded). The Gemini script is fine. |
| `eval_open/regrade.py:50` | `results/open/` doesn't exist yet. |
| `eval_open/finish_inject.py:46` | Its input file doesn't exist yet, although it is meant to run in stages. |
| `eval_open/run_frontier.py:35` | Fewer than 100 prose-reference questions (the similar scripts cope). |
| `eval_open/detect_teeth.py:64` | Not a crash: resume quietly mixes detections from two different models into one file. |
| `dataio/make_e11_subset.py:48` | Too few correct items to sample from. |
| `paper_analysis/dentist_audit.py:106` | A dentist answer with an unknown `item_id` (the main loop skips these, this check doesn't). |
| `paper_analysis/overlapping_options.py:60` | No question has overlapping options (should print "0 items overlap"). |
| `paper_analysis/position_stability.py:99` | Every item for a model failed on both runs (divide by zero). |
| `paper_analysis/boneloss_audit.py:120` | Some rows with a blank dentist answer, with pandas 3.x (depends on which other fields are blank). |
| `detector/model_card.py:57` | A checkpoint saved without training settings. |
| `detector/calibrate_arch.py:124` | A tooth index 1–8 never appears in the calibration data. |
| `eval_open/test_detector_inject.py` `anat()` | A malformed one-digit tooth code. |

## 4. Small things

- `paper_analysis/section54_table.py:62`: a row mismatch crashes with a message that doesn't say which rows.
- `paper_analysis/open_model_table.py:51`: the footnote "477 prose-ref + 101 coord-ref" is typed in, not computed.
- `eval_open/test_localize_detection.py:24`: `BASELINE` is never used; its progress line assumes 6 questions per image.
- `eval_open/detect_teeth.py`, `run_batched.py:68`: `overlay=True` silently wins over `detection_text` when both are given.
- `paper_analysis/resolution_check.py:15`: unused import.
- `wilson()` is written twice (`knowledge_context.py`, `nshot_grid.py`), one rounds and one doesn't.
