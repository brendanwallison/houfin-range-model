#!/bin/bash
set -euo pipefail
ROOT="${FINAL_HOLDOUT_ROOT:-/home/breallis/houfin/work/houfin/research/FINAL_HOLDOUT_20261025}"
cd "$ROOT/code_score"
export HOUFIN_DATA="${HOUFIN_DATA:-/home/breallis/houfin/scratch/houfin/data}" HOUFIN_PROCESSED=/work/07980/bwa386/ls6/houfin/processed
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export BLR_DEVICE=cuda BLR_MAX_ITER=100 BLR_STARTS=0.5 FINAL_FIT_CACHE="$ROOT/fit_cache"
unset CUDA_VISIBLE_DEVICES
ARM="$1"; LO="$2"; HI="$3"; THIN="${4:-1}"; SEED="${5:-1}"
RANK=64; SPLIT=0
case "$ARM" in v2_r64) SPLIT=32;; v2_r128) RANK=128; SPLIT=64;; esac
OUT="$ROOT/summer/${ARM}_e${LO}_${HI}_f${THIN}_s${SEED}"
mkdir -p "$OUT"
if [ -f "$OUT/DONE" ]; then exit 0; fi
"${FINAL_PYTHON:-/home/breallis/houfin/venv/bin/python}" research/exp/final_cached_fair.py \
  --cache "$ROOT/adapters/$ARM" --out "$OUT" --target sealed \
  --confirmation-reason 'FINAL_HELDOUT: predeclared final-candidate summer comparison, no encoder tuning' \
  --rank "$RANK" --desk-split "$SPLIT" --early "$LO" "$HI" --modern 2005 2025 \
  --incremental --n-pos 96 --pos-seeds 1 2 --strong-base --lean --save-species \
  --tag _final --n-boot 1000 --z-keys-only --thin "$THIN" --thin-unit row --thin-seed "$SEED"
date -Is > "$OUT/DONE"
