# Dentist ground-truth surveys — record & reproduction

Expert audit of MMOral-Bench answer-key quality (the "is the benchmark's key actually
correct?" probe). Full design spec: `dental_research/DENTIST_AUDIT.md`.

## Reproduction (deterministic — same 75 items for every dentist)
- **Item selection:** `python dataio/make_dentist_survey.py` (seed 20260707)
  → `results/dentist_audit/survey_manifest.csv` (75 items: 60 closed blind + 15 open endorsement).
- **Viewer bundle:** `python dataio/export_survey_bundle.py` → `survey/` (static GitHub Pages viewer).
- **Analysis:** `python paper_analysis/dentist_audit.py results/dentist_audit/submission_<token>.json`
  → per-bucket certified key-error rate (Wilson CIs), T1+T2-vs-C predictiveness, re-weighted
  whole-key estimate, open endorsement.

## Survey log
| token | dentist | started | submitted | closed/open | certified key errors | re-weighted key-error est. | open endorsement | file |
|---|---|---|---|---|---|---|---|---|
| sandeep | Sandeep | 2026-07-07 | 2026-07-09 | 60 / 15 | 4 (T1: 3, T2: 1; C: 0) | ~3% | 13/15 AGREE (87%) | `submission_sandeep.json` |

## Convention for future surveys (so everything stays reproducible & poolable)
1. Send each dentist a viewer link with a **unique** `?token=<id>` — never reuse a token.
2. On return, save the JSON as `results/dentist_audit/submission_<token>.json` and add a
   top-level `"dentist_name"` field (the token may be an alias; the name is the record).
3. Run the analyzer; save the printout as `analysis_<token>.txt` next to it.
4. Append a row to the **Survey log** above.
5. Because the 75 items are the *same deterministic manifest* for every dentist, responses are
   directly comparable — pool them later for inter-rater agreement (Cohen's κ) and to confirm
   the handful of certified errors with a second reader.

## Survey 1 (Sandeep) — findings summary
- **Closed key is ~97% defensible.** 20 raw disagreements → only **4 certified key errors**
  after the 2nd-pass "is the key wrong?" adjudication; re-weighted whole-key estimate ≈ **3%**.
- **Model disagreement directionally predicts errors** (all 4 errors in the model-flagged
  T1+T2 buckets, 0 in the control C) — but underpowered (4 events, wide CIs).
- **Certified errors:** c220 & c134 (bone loss denied by the key), c147 (anatomical error —
  mandibular canals listed as a *maxillary* structure), c334 (filling vs root-canal at #36).
- **Question-quality signal (separate from key errors):** 3 "cannot determine" + 3 "none of
  these" responses — 2 are ill-posed coordinate questions (ask for a structure inside pixel
  coordinates that aren't shown; the dentist literally can't answer), the rest are
  option-coverage gaps (the true finding isn't among the 4 options).
- **Open-ended references strongly endorsed:** 13/15 AGREE, 1 partial, 1 disagree → the
  open-ended *reference content* is mostly sound (keep the open-metric critique on scoring/format).
- **Caveats:** single rater; expert-vs-key (not truth); only 4 error events → wide CIs; a probe
  to scope a full audit, not a definitive rate. One viewer-display flag on c72 to check.
