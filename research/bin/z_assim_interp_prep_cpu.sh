#!/bin/bash
# In-era interpolation test data: project 1966-1995 training-place summer rows into the summer64 teacher.
set -euo pipefail
ROOT=/mnt/c/Dev/houfin-range-model/.research_runtime/community_z_repair_20261026
CODE=/mnt/c/Dev/houfin-range-model
BASE=/home/breallis/houfin/work/houfin/research_cache/desk_tempho_1995_s1
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/home/breallis/houfin/work/houfin/processed
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE:$CODE/src"
cd "$CODE"
exec /home/breallis/houfin/venv/bin/python -u research/exp/z_assim_interp_prep.py --root "$ROOT" --base "$BASE"
