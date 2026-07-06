# Data — closed-ended MMOral-OPG sets (None-normalized)

**Provenance.** Source dataset: HuggingFace `OralGPT/MMOral-OPG-Bench` (MIT-licensed). In the
raw release, a "blank" answer option is stored as an **empty cell (NaN)**. The benchmark's own
eval code does **not** leave it NaN: `MMOral_OPG_CLOSED.post_build` runs
`self.data[col].fillna('None')` on the option columns, so both the prompt it shows the model
and its answer parser read the string **"None"** (a readable "none of the above" choice).

We do the same, **once**, at the data layer. Storing NaN and letting `str(NaN)` become `"nan"`
was a reproduction bug on our side (it made the model see `"D. nan"` and made the parser's
`index2ans` look for `"nan"`). The canonical set below removes NaN at the source.

## Files

| file | NaN? | role |
|---|---|---|
| `closed_ended_raw.parquet` | **has NaN** | **FROZEN** raw HF download. The *only* code that reads it is `dataio/convert_to_none.py`. Never used by a run. |
| `closed_ended.parquet` | 0 NaN | **canonical source of truth.** One-time `fillna('None')` of the raw. Every harness and derivative reads this. |
| `closed_ended_shuffled.parquet` | 0 NaN | derivative of the canonical: options permuted per-row (seeded by `index`), key flattened for the position-bias analysis. |
| `closed_ended_blanks38.parquet` | 0 NaN | derivative of the canonical: the 38 items that carry a "None" option (§3.7), for targeted re-runs. |
| `open_ended.parquet` | n/a | free-text half (no options). |

The old blanks-dropped "clean" sets (`closed_ended_clean*.parquet`) were **removed**: the
blanks are answerable "None" questions, not broken, so there is no reason to drop them. The
working set is the full 491 canonical.

## Rules (enforced)

1. **Never read `closed_ended_raw.parquet`** except in `dataio/convert_to_none.py`. All harnesses
   load through `dataio/eval_data.read_closed()`, which **raises** if a file still has NaN options.
2. **All derivatives build from `closed_ended.parquet`** (the None canonical), never from the raw.

## Regenerate

```
python dataio/convert_to_none.py      # raw -> canonical None (run once)
python dataio/prepare_datasets.py     # canonical -> shuffled
python dataio/make_blank_subset.py    # canonical -> blanks-38 subset + manifest
```
