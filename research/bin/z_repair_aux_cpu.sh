#!/bin/bash
set -euo pipefail
TASK="$1"; WINTER="$2"
cd /mnt/c/Dev/houfin-range-model
ROOT="$PWD/.research_runtime/community_z_repair_20261026"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/home/breallis/houfin/work/houfin/processed
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
mkdir -p "$ROOT/aux/$TASK"
EXTRA=()
if [ -n "${3:-}" ]; then EXTRA=(--reuse-presence-from "$3"); fi
exec /home/breallis/houfin/venv/bin/python -u research/exp/z_repair_aux.py --root "$ROOT" --task "$TASK" --winter "$WINTER" "${EXTRA[@]}" > "$ROOT/aux/$TASK/$WINTER.log" 2>&1
