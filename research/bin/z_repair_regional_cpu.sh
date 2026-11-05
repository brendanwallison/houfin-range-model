#!/bin/bash
set -euo pipefail
ROOT=/mnt/c/Dev/houfin-range-model/.research_runtime/community_z_repair_20261026
CODE="$ROOT/regional_code"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE:$CODE/src"
cd "$CODE"
exec /home/breallis/houfin/venv/bin/python -u research/exp/z_repair_regional.py --root "$ROOT" > "$ROOT/evaluation/backcast40/regional.log" 2>&1
