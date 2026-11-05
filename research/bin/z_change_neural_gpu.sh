#!/bin/bash
set -euo pipefail
ARM="$1"
ROOT=/mnt/c/Dev/houfin-range-model/.research_runtime/community_z_repair_20261026
CODE="$ROOT/${ARM}_code"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/home/breallis/houfin/work/houfin/processed
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE:$CODE/src"
cd "$CODE"
mkdir -p "$ROOT/models/backcast40"
exec /home/breallis/houfin/venv/bin/python -u research/exp/z_change_neural.py run --root "$ROOT" --arm "$ARM" > "$ROOT/models/backcast40/$ARM.log" 2>&1
