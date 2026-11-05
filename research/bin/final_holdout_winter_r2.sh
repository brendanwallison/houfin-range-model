#!/bin/bash
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_HOLDOUT_20261025
cd "$ROOT/code_winter_r2"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=
/home/breallis/houfin/venv/bin/python research/exp/final_winter_score.py \
  --root "$ROOT" --cbc "$ROOT/sealed/cbc_circles.npz" \
  --reason 'FINAL_HELDOUT: winter confirmation with every test circle excluded from readout calibration'
