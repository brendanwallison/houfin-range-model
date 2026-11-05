#!/bin/bash
set -euo pipefail
ROOT=/mnt/c/Dev/houfin-range-model/.research_runtime/community_z_repair_20261026
CODE="$ROOT/change_cpu_code"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/home/breallis/houfin/work/houfin/processed
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE:$CODE/src"
cd "$CODE"
mkdir -p "$ROOT/change_cpu"
exec /home/breallis/houfin/venv/bin/python -u research/exp/z_change_cpu.py --root "$ROOT" --base /home/breallis/houfin/work/houfin/research_cache/desk_tempho_1995_s1 > "$ROOT/change_cpu/run.log" 2>&1
