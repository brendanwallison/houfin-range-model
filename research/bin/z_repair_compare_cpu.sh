#!/bin/bash
set -euo pipefail
ROOT=/mnt/c/Dev/houfin-range-model/.research_runtime/community_z_repair_20261026
CODE="$ROOT/evaluation_code"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export PYTHONPATH="$CODE:$CODE/src"
cd "$CODE"
exec /home/breallis/houfin/venv/bin/python research/exp/z_repair_compare.py --root "$ROOT"
