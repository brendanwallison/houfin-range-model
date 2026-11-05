#!/bin/bash
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_CLEAN_20261025
CODE=/mnt/c/Dev/houfin-range-model
SOURCE="$ROOT/code_compare"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/work/07980/bwa386/ls6/houfin/processed
export RESEARCH_SEALED_READ_LOG="$ROOT/reused_holdout_reads.csv" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=
for tick in $(seq 1 720); do
 if [ -f "$ROOT/evaluation/original2/EVALUATION_DONE" ] && [ -f "$ROOT/evaluation/clean2/EVALUATION_DONE" ]; then
  cd "$SOURCE"
  sha256sum -c SOURCE_MANIFEST.sha256 > "$ROOT/compare_source_verification.log"
  /home/breallis/houfin/venv/bin/python research/exp/final_clean_compare.py \
   --reference "$ROOT/evaluation/original2" --candidate "$ROOT/evaluation/clean2" \
   --cbc /home/breallis/houfin/work/houfin/research/FINAL_HOLDOUT_20261025/sealed/cbc_circles.npz \
   --out "$CODE/research/experiments/FINAL_CLEAN_RESULTS.json" --report "$CODE/research/experiments/FINAL_CLEAN_RESULTS.md"
  exit 0
 fi
 sleep 60
done
exit 2
