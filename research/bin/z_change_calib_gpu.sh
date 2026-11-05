#!/bin/bash
set -euo pipefail
ROOT=/mnt/c/Dev/houfin-range-model/.research_runtime/community_z_repair_20261026
CODE="$ROOT/change100_calib_code"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/home/breallis/houfin/work/houfin/processed
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE:$CODE/src"
cd "$CODE"
mkdir -p "$ROOT/models/backcast40"
for w in g0 g1; do
  /home/breallis/houfin/venv/bin/python -u research/exp/z_change_calib.py pass --root "$ROOT" --which $w >> "$ROOT/models/backcast40/change100_calib.log" 2>&1
done
cat "$ROOT/models/backcast40/change100/change_loss_calibration.json"
