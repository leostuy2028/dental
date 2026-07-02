# n-shot × reasoning grid — Gemini 2.5 vs 3.5 (2026-07-02)

Full analysis + interpretation: **`dental-research/RESEARCH_PLAN.md` §3.8**.

**Setup:** `data/closed_ended_clean_shuffled.parquet` (453 rows, blank-answer rows
removed, options shuffled → balanced key). Test = idx 50–149 (100 items, identical
across cells). Exemplar pool = idx 0–49 (disjoint, same-category, from the shuffled
set). Internal thinking OFF (`--thinking-budget 0`). Realized k = 0 / 0.94 / 2.64 / 4.12.

Baselines on this slice (key A/B/C/D = 23/20/33/24): always-C = 33% is the bar; random 25%.

| Model | Mode | k | Acc % [95% CI] | A/B/C/D | χ² vs uniform |
|-------|------|---|----------------|---------|---------------|
| 2.5 | direct | 0 | 35.0 [26–45] | 23/26/31/20 | 2.6 |
| 2.5 | direct | 1 | 43.0 [34–53] | 44/20/19/17 | 19.4 * |
| 2.5 | direct | 3 | 49.0 [39–59] | 52/14/22/12 | 41.1 * |
| 2.5 | direct | 5 | 53.0 [43–62] | 49/14/25/12 | 34.6 * |
| 2.5 | CoT | 0 | 39.0 [30–49] | 28/25/25/20 | 1.3 |
| 2.5 | CoT | 1 | 48.0 [38–58] | 45/18/24/12 | 25.0 * |
| 2.5 | CoT | 3 | 47.0 [38–57] | 39/22/24/14 | 13.2 * |
| 2.5 | CoT | 5 | 52.0 [42–62] | 34/24/26/15 | 7.4 |
| 3.5 | direct | 0 | 48.0 [38–58] | 18/26/25/31 | 3.4 |
| 3.5 | direct | 1 | 58.0 [48–67] | 21/24/24/31 | 2.2 |
| 3.5 | direct | 3 | 57.0 [47–66] | 22/24/26/28 | 0.8 |
| 3.5 | direct | 5 | 54.0 [44–63] | 23/25/25/27 | 0.3 |
| 3.5 | CoT | 0 | 52.0 [42–62] | 21/18/29/32 | 5.2 |
| 3.5 | CoT | 1 | 54.0 [44–63] | 29/19/26/26 | 2.2 |
| 3.5 | CoT | 3 | 60.0 [50–69] | 25/22/27/26 | 0.6 |
| 3.5 | CoT | 5 | 63.0 [53–72] | 23/20/29/28 | 2.2 |

\* letter distribution significantly non-uniform (χ²>7.82, p<0.05, df=3).

**Takeaways:** (1) 2.5-flash has a severe few-shot-amplified A-bias (%A 23→52); 3.5-flash
does not (stays uniform) — bias is model-generational and shrank 2.5→3.5. (2) Few-shot
*helps* accuracy (2.5 35→53%, McNemar p<0.01), opposite the earlier Claude result.
(3) Visible CoT dampens 2.5's A-bias (k5: %A 49→34, χ² 35→7). (4) Best: 3.5 CoT k5 = 63%.
