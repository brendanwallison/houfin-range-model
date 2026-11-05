#!/bin/bash
# Existing observation-aware scorer, explicit seasonal halves, reused holdouts.
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_CLEAN_20261025
OUT="$ROOT/evaluation/clean2"
PY=/home/breallis/houfin/venv/bin/python
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/work/07980/bwa386/ls6/houfin/processed
export RESEARCH_SEALED_READ_LOG="$ROOT/reused_holdout_reads.csv"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export BLR_DEVICE=cuda BLR_MAX_ITER=100 BLR_STARTS=0.5 FINAL_FIT_CACHE="$ROOT/evaluation/fit_cache"
cd "$ROOT/analysis_code"
for spec in '1 1' '0.03 1' '0.03 2'; do
 read -r thin seed <<< "$spec"
 for arm in v2_w64 v2_s64; do
  dest="$OUT/summer/${arm}_f${thin}_s${seed}"; mkdir -p "$dest"
  [ -f "$dest/DONE" ] && continue
  "$PY" research/exp/final_cached_fair.py --cache "$OUT/adapters/$arm" --out "$dest" --target sealed \
   --confirmation-reason 'FINAL_SEASONAL: user-requested clean winter-only/summer-only sensitivity; reused heldouts, not fresh confirmation' \
   --rank 64 --desk-split 0 --early 1966 1986 --modern 2005 2025 \
   --incremental --n-pos 96 --pos-seeds 1 2 --strong-base --lean --save-species --tag _final --n-boot 1000 \
   --z-keys-only --thin "$thin" --thin-unit row --thin-seed "$seed"
  date -Is > "$dest/DONE"
 done
done
date -Is > "$OUT/SEASONAL_SUMMER_DONE"
