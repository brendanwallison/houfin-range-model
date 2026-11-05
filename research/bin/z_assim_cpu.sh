#!/bin/bash
# Observation-anchored community field (research/exp/z_assim.py): inner grid or one arm fit + cube export.
set -euo pipefail
ROOT=/mnt/c/Dev/houfin-range-model/.research_runtime/community_z_repair_20261026
CODE=/mnt/c/Dev/houfin-range-model
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/home/breallis/houfin/work/houfin/processed
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE:$CODE/src"
cd "$CODE"
exec /home/breallis/houfin/venv/bin/python -u research/exp/z_assim.py "$@" --root "$ROOT"
