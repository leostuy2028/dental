#!/usr/bin/env bash
# Section 5.4 reference arms on the ORIGINAL benchmark key (data/closed_ended.parquet,
# B-skewed), to sit beside the debiased shuffled-key numbers. Same config otherwise:
# gemini-3.5-flash, coax/direct, think=0, temp=0, full-res. No-primer already exists
# (position_bias/...__whole__n491.csv = 59.88%); this adds primer and primer+v2.
set -e
cd "$(dirname "$0")"
OUT=results/closed_ended/knowledge_context
COMMON="--model gemini-3.5-flash --k 0 --prompt coax --thinking-budget 0 --data data/closed_ended.parquet --start 0 --paper-section 5.4"

echo "=========== BENCHKEY 1/2: primer only ==========="
python eval_closed_gemini.py $COMMON \
  --context reference/opg_primer.txt \
  --out $OUT/gemini-3.5-flash__coax-direct-ctx-opgprimer__whole__n491.csv \
  --exp s54-primer-benchkey \
  --description "OPG text primer, ORIGINAL benchmark key (B-skewed) for reference"

echo "=========== BENCHKEY 2/2: primer + exemplars_v2 ==========="
python eval_closed_gemini.py $COMMON \
  --context reference/opg_primer.txt \
  --visual-exemplars reference/exemplars_v2.json \
  --out $OUT/gemini-3.5-flash__coax-direct-primerv1+visualex-v2__whole__n491.csv \
  --exp s54-primer-visualex-v2-benchkey \
  --description "STACKED primer + 12 visual exemplars, ORIGINAL benchmark key (B-skewed) for reference"

echo "=========== BENCHKEY BOTH DONE ==========="
