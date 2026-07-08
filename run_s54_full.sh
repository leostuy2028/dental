#!/usr/bin/env bash
# Section 5.4 full-benchmark runs (shuffled key, gemini-3.5-flash, coax/direct, think=0, FULL-RES).
# Baseline (no-primer) already exists: position_bias/...__coax-direct-k0__shuffled__n491.csv (58.04%).
# This produces the other two arms of the §5.4 table, both resumable (CSV checkpointing).
set -e
cd "$(dirname "$0")"
OUT=results/closed_ended/knowledge_context
COMMON="--model gemini-3.5-flash --k 0 --prompt coax --thinking-budget 0 --data data/closed_ended_shuffled.parquet --start 0 --paper-section 5.4"

echo "=========== ARM 1/2: primer only (full) ==========="
python eval_closed_gemini.py $COMMON \
  --context reference/opg_primer.txt \
  --out $OUT/gemini-3.5-flash__coax-direct-ctx-opgprimer__shuffled__n491.csv \
  --exp s54-primer-full \
  --description "OPG text primer (reference/opg_primer.txt), full benchmark shuffled key"

echo "=========== ARM 2/2: primer + exemplars_v2 (full) ==========="
python eval_closed_gemini.py $COMMON \
  --context reference/opg_primer.txt \
  --visual-exemplars reference/exemplars_v2.json \
  --out $OUT/gemini-3.5-flash__coax-direct-primerv1+visualex-v2__shuffled__n491.csv \
  --exp s54-primer-visualex-v2-full \
  --description "STACKED: OPG text primer + 12 combined visual exemplars (DENTEX pathology/numbering + Zenodo HisT/Jaw), full benchmark shuffled key"

echo "=========== BOTH ARMS DONE ==========="
