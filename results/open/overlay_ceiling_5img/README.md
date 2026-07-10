# 5-image accurate-overlay CEILING test — artifacts

These are the hand-placed FDI tooth-number overlays used in the 2026-07-10 ceiling
test (see `RESEARCH_PLAN.md` "Current TODOs"). Each `<image>_overlay.jpg` is the
original panoramic from `data/open_ended.parquet` with green FDI codes drawn on a
black chip at each tooth-crown centre (PIL), following the arch curve.

| overlay file | source image | test result (gemini-3.5-flash, all Qs) |
|:--|:--|:--|
| 016690_overlay.jpg | 016690.jpg | 38% -> 80% (4 clear fillings; the clean win) |
| 016655_overlay.jpg | 016655.jpg | 5% -> 65% |
| 016713_overlay.jpg | 016713.jpg | 48% -> 53% |
| 016726_overlay.jpg | 016726.jpg | 28% -> 6% (REGRESSED; subtle finding, over-listing) |
| 016640_overlay.jpg | 016640.jpg | EXCLUDED from the score (disrupted dentition, missing teeth; could not be numbered accurately by hand) |

Per-question scores: `../overlay_needle.csv`.

## IMPORTANT — these overlays are CONTAMINATED (ceiling, not deployable)

They were placed by Opus **having already seen these images' ground-truth answers**
(the 5 were selected as the known-wrong-tooth LOCALIZE failures), so "accurate" here
means "agrees with the answer key". No answer text ever reached the model (the overlay
carries tooth NUMBERS only), and every score is real model output — but the numbering
was **not produced blind**. Treat the resulting +35 (teeth questions) as an UPPER BOUND.
The 100-image scale-up must produce numbering blind (trained FDI detector) to be a
defensible measurement. See the TODO in `dental_research/RESEARCH_PLAN.md`.
