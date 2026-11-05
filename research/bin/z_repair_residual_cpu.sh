#!/bin/bash
set -euo pipefail
ROOT=/mnt/c/Dev/houfin-range-model/.research_runtime/community_z_repair_20261026
CODE="$ROOT/residual_code"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/home/breallis/houfin/work/houfin/processed
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE:$CODE/src"
cd "$CODE"
mkdir -p "$ROOT/residuals/backcast40"
exec /home/breallis/houfin/venv/bin/python -u research/exp/z_repair_residual_fit.py --root "$ROOT" --rank 2 > "$ROOT/residuals/backcast40/rank2.log" 2>&1
