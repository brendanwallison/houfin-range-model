#!/bin/bash
# Fixed two additional thinning seeds, reusing the established fair-rival scorer.
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_HOLDOUT_20261025
cd "$ROOT/code_score"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/work/07980/bwa386/ls6/houfin/processed
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
export BLR_DEVICE=cuda BLR_MAX_ITER=100 BLR_STARTS=0.5 FINAL_FIT_CACHE="$ROOT/fit_cache"
export RESEARCH_SEALED_READ_LOG="$ROOT/sparse_followup_reads.csv"
for seed in 3 4; do
 for arm in v1_r64 v2_r64 v2_r128; do
  rank=64; split=0
  case "$arm" in v2_r64) split=32;; v2_r128) rank=128;split=64;; esac
  out="$ROOT/sparse_followup/${arm}_s${seed}"
  mkdir -p "$out"
  [ -f "$out/DONE" ] && continue
  /home/breallis/houfin/venv/bin/python research/exp/final_cached_fair.py \
   --cache "$ROOT/adapters/$arm" --out "$out" --target sealed \
   --confirmation-reason 'FINAL_SPARSE_FOLLOWUP: seeds3/4 sensitivity on previously opened heldout observations; no new confirmation or encoder tuning' \
   --rank "$rank" --desk-split "$split" --early 1966 1986 --modern 2005 2025 \
   --incremental --n-pos 96 --pos-seeds 1 2 --strong-base --lean --save-species \
   --tag _final --n-boot 1000 --z-keys-only --thin 0.03 --thin-unit row --thin-seed "$seed"
  date -Is > "$out/DONE"
 done
done
date -Is > "$ROOT/SPARSE_FOLLOWUP_DONE"
