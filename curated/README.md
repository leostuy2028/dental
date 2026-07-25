# MMOral-Bench (curated) — v0.1.0

A corrected version of [MMOral-Bench](https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench),
the panoramic dental X-ray benchmark from *Towards Better Dental AI* (NeurIPS 2025,
arXiv:2509.09254).

We audited how the benchmark was built and scored, found faults that move scores without
any model reading an X-ray better, and fixed the ones that can be fixed. This directory is
the result. The reasoning behind every decision is in §7 of the accompanying paper.

## This is a patch, not a fork

**It contains no X-rays and no copied questions.** It contains the *changes*: for each of
the 1,069 released items, what we decided and why, plus the corrected answer key and the
rewritten reference answers.

Download MMOral-Bench from its own source, then join these files on `index`. The original
dataset stays canonical, our edits sit beside what they replace, and you can disagree with
any single one of them.

## Files

| File | Rows | What it holds |
|---|--:|---|
| `mmoral_curated_closed.csv` | 491 | multiple-choice: disposition, reason, original key, **balanced key**, flag reason |
| `mmoral_curated_open.csv` | 578 | free-text: disposition, reason, **rewritten reference**, scoring track |
| `VERSION.json` | — | counts, provenance, the code commit it was built from |

## What changed

**Multiple choice (491 questions)**

- **The answer key is rebalanced.** In the original, B is correct 44% of the time, so a
  model that always answers B scores 44% without looking at anything. After rebalancing,
  the best always-one-letter score is 27.9%. Option wording is untouched; only position
  moves. Use `balanced_answer`, not `original_answer`.
- **7 questions removed** (`DROP`). They ask what lies inside a bounding box at particular
  pixel coordinates, on images where no box is drawn and no coordinates appear. A dentist
  reviewing them said: "I don't see any coordinates."
- **71 flagged** (`FLAG`), kept and scored. Either their tooth numbers are valid in both
  the FDI and US Universal systems and mean different teeth in each (the benchmark never
  says which it uses), or they concern bone loss, the one topic a dentist audit found the
  key repeatedly wrong on. See `flag_reason`.
- **484 questions are in the reading score.**

**Free text (578 questions)**

- **82 reference answers rewritten** (`REPAIR`). These recorded a list of pixel boxes where
  the question asked something clinical. The boxes carry labels, so dropping the
  coordinates and keeping the labels recovers the intended sentence:

  ```
  question : Which historical treatments can be detected in the panoramic image?
  was      : [{"box_2d": [1531, 698, 1649, 769], "tooth_id": "36", "label": "Crown"}, ...]
  now      : Tooth #36: crown, root canal treatment. Tooth #38: filling.
  ```

  Nothing is invented. Every word comes from the released data; only coordinates were
  discarded. Both versions are in the file, so you can check each one.
- **107 moved to a writing track** (`SEPARATE`). "Please caption this X-ray with findings"
  has no determinate answer. Scoring it as reading accuracy lets fluency substitute for
  accuracy. Score these separately if at all, and never average them into a reading score.
- **6 need an overlap metric** (`SPATIAL`). These genuinely ask for a location and their
  coordinate answers are correct. A text-reading judge cannot score them; compare predicted
  and reference boxes instead.
- **13 flagged as malformed** (`MALFORMED`), unchanged. Their references are not valid JSON,
  because of a curly quotation mark where a straight one belongs. We report rather than
  guess at the intended text.
- **29 flagged** (`FLAG`) for asserting "no apparent bone loss".
- **452 questions are in the reading score.** The 13 malformed are excluded from it too: their reference is both a box list and unparseable, so the §7.2.1 repair could not run, and scoring plain English against coordinates is the mismatch this release exists to remove.

## How to use it

```python
import pandas as pd
bench  = pd.read_parquet("closed_ended.parquet")          # from the original release
patch  = pd.read_csv("curated/mmoral_curated_closed.csv")
df = bench.merge(patch, on="index")
df = df[df.in_reading_score]                              # drops the 7 unanswerable
# score against df.balanced_answer, NOT df.answer
```

```python
op    = pd.read_parquet("open_ended.parquet")
patch = pd.read_csv("curated/mmoral_curated_open.csv")
df = op.merge(patch, on="index")
df["reference"] = df.repaired_reference.where(df.reference_changed, df.answer)
reading = df[df.score_track == "reading"]                 # 452 questions
writing = df[df.score_track == "writing"]                 # 107, report separately
# also: score_track == "overlap" (6, needs an IoU metric), "excluded" (13, malformed)
```

**Report the two tracks separately.** A single number mixing reading accuracy with writing
quality is what we are trying to get away from.

## Also fix the grader

Curating the questions is only half of it. The free-text grader has faults of its own that
this patch cannot reach, because they live in the scoring prompt rather than in the data
(paper §7.3): it awards up to 0.2 for coordinate formatting on questions that never asked
for a location, it is given nine scored examples and no written rules, it returns a bare
number with no reasoning, and it is re-queried at rising temperature until a parseable score
appears. Changing judges moved one fixed set of answers by twelve points.

## Limitations

- **This is not a validated benchmark yet.** Everything here is a defect findable by reading
  the released files. A dentist audit found a kind no script can find: questions where the
  true answer is not among the four options. Six of sixty audited questions were
  unanswerable for one reason or another, so the real number to remove is higher than seven.
- **The expert review is partial.** 60 multiple-choice and 15 free-text items have been
  reviewed by one dentist, and about 3% of the key was found wrong. That is a probe, not a
  full audit, and one rater gives no inter-rater agreement.
- **The flags mark risk, not confirmed error.** A flagged question is not known to be wrong.
- **Single rater, single source.** See paper §5.8 and §8.

## Reproducing this

```bash
python -m paper_analysis.corrected_benchmark   # decide every item
python -m dataio.repair_coordinate_refs        # rewrite the 82 references
python dataio/prepare_datasets.py              # build the balanced key
python curated/build_curated.py                # assemble this directory
```

Deterministic, no API calls, no model outputs.

## Citing

Cite the original benchmark (arXiv:2509.09254) as the source of the data, and this work for
the corrections. `VERSION.json` records the code commit each build came from.

## Licence

MMOral-Bench is MIT-licensed. These derived annotations are released under the same terms.
