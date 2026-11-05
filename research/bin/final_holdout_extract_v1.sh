#!/bin/bash
set -euo pipefail
ROOT=/work/07980/bwa386/ls6/houfin/research/final_holdout_20261025
cd "$ROOT/code"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/work/07980/bwa386/ls6/houfin/venv/bin/python research/exp/final_holdout_extract.py \
  --encoder-root /work/07980/bwa386/ls6/houfin/processed/encoder \
  --cells "$ROOT/cells.npy" --family final --out "$ROOT/features/final"
