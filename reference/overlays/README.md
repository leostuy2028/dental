# Blind FDI tooth-code overlays (LOCALIZE-39)

Hand-numbered FDI tooth-code overlays for the **39 LOCALIZE images** — the
`open_ended.parquet` panoramics whose Teeth-category question asks "which tooth
has X" (localization). Produced as the **BLIND hand-numbering fallback** for
RESEARCH_PLAN STEP 1 (the alternative to a trained detector).

## What's here
- `<image_name>.jpg` — the original panoramic with a green FDI tooth code drawn
  on a black chip at each tooth's crown center (39 files).
- `low_confidence.json` — per-image caveats for the disrupted / restored /
  edentulous cases (20 of the 39). Read this before trusting any posterior /
  restored-unit FDI.
- Source coordinates live in `../tooth_boxes.json`
  (`{image_name: [{"fdi", "cx", "cy"}, ...]}`, crown-center in ORIGINAL pixels),
  with a `_method` note as the first key.

Totals: 39 images, 1092 teeth labeled.

## Method (and why it's trustworthy for a benchmark-validity audit)
- **Blind.** Teeth were numbered by **anatomy only** — arch position → FDI
  quadrant (1st digit) and count-from-midline (2nd digit). The `answer`/`gt`
  column was **never loaded or read** during numbering. The numbering scripts
  (`worksheet.py`, `render.py`, `crop.py`) load only `image_name` + `image`.
- **Orientation.** Panoramic convention: image-LEFT = patient RIGHT
  (Q1 upper / Q4 lower); image-RIGHT = patient LEFT (Q2 upper / Q3 lower),
  confirmed against the R/L corner marker in each film.
- **Produced before grading.** These overlays and `tooth_boxes.json` are
  committed **before** any overlay-vs-plain grading run, so the labels cannot be
  contaminated by knowledge of the ground-truth answers.
- **Per-image QC.** Each image was numbered off a pixel-labeled coordinate grid,
  rendered, and visually checked; phantom labels on edentulous bone were removed
  and 2× crops were used to resolve gaps, drift, restorations and impacted molars.

## Limitations
- **Single rater.** All 39 were numbered by one annotator (Claude Opus). No
  second-rater agreement (e.g. Cohen's κ) was measured, so single-rater error is
  not quantified. Treat as a proxy reference, not a validated gold standard.
- **No exclusions.** Every image was numbered best-effort. Disrupted mouths were
  **not dropped**; instead they carry a caveat in `low_confidence.json`. On those
  (heavy restoration, implants, bridges, supra-eruption, large edentulous spans),
  the specific FDI of restored/implant units and gapped segments is a genuine
  best-effort guess — this is exactly the failure mode a trained detector is
  meant to fix.
- **±1-tooth molar imprecision** is accepted under the single-rater caveat.

## Low-confidence images (see low_confidence.json for details)
016640, 016713, 016726, 016823, 016839, 016955, 017113, 017118, 017137, 017144,
017168, 017204, 017220, 017239, 017656, 017690, 018160, 018328, 018384, 018466.

The other 19 images are clean/reliable natural or lightly-restored dentitions.
