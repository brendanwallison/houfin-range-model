#!/bin/bash
set -euo pipefail
ARM="$1"
ROOT=/mnt/c/Dev/houfin-range-model/.research_runtime/community_z_repair_20261026
CODE="$ROOT/evaluation_code"
BASE=/home/breallis/houfin/work/houfin/research_cache/desk_tempho_1995_s1
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/home/breallis/houfin/work/houfin/processed
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE:$CODE/src"
cd "$CODE"
mkdir -p "$ROOT/evaluation/backcast40/$ARM"
PY=/home/breallis/houfin/venv/bin/python
"$PY" -u research/exp/z_repair_evaluate.py --root "$ROOT" --base "$BASE" --arm "$ARM" > "$ROOT/evaluation/backcast40/$ARM/summer.log" 2>&1
"$PY" -u research/exp/z_repair_winter_evaluate.py --root "$ROOT" --base "$BASE" --cbc /mnt/e/Datasets/cbc_raw/prepared/cbc_circles_all.npz --arm "$ARM" > "$ROOT/evaluation/backcast40/$ARM/winter.log" 2>&1
