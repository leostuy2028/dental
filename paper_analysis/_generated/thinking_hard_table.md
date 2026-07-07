<!-- GEN: thinking_hard (generator: paper_analysis/thinking_hard.py; source: results/closed_ended/cot_length/gemini-3.5-flash__direct-think*__hard50-shuffled__n50.csv; regenerate: python paper_analysis/thinking_hard.py) -->
| Thinking budget | Accuracy [95% CI] | A/B/C/D | n |
|:--|--:|:--|--:|
| off | 46.0% [33-60] | 5/14/14/17 | 50 |
| 512 | 48.0% [35-61] | 10/15/12/13 | 50 |
| 2048 | 50.0% [37-63] | 12/13/11/14 | 50 |
| 8192 | 50.0% [37-63] | 11/14/14/11 | 50 |
| dynamic | 54.0% [40-67] | 11/14/12/13 | 50 |
<!-- /GEN: thinking_hard -->

**Paired budget 0 -> 8192 (same 50 items):** 46.0% -> 50.0%; thinking rescued 7, broke 5 (net +2); McNemar exact p = 0.7744.
