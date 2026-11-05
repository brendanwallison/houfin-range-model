#!/bin/bash
set -euo pipefail
TASK="$1"; SEASON="$2"
case "$TASK" in backcast40|backcast20) ;; *) exit 2;; esac
case "$SEASON" in summer|winter_full|winter_sparse10|winter_no_target) ;; *) exit 2;; esac
cd /mnt/c/Dev/houfin-range-model
ROOT="$PWD/.research_runtime/community_z_repair_20261026"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/home/breallis/houfin/work/houfin/processed
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
if [ "$SEASON" = summer ]; then
 POINTS="$ROOT/data/$TASK/summer_train_points"
else
 POINTS="$ROOT/data/$TASK/$SEASON/count_points"
fi
OUT="$ROOT/teachers/$TASK/$SEASON"
mkdir -p "$ROOT/teachers/$TASK"
exec /home/breallis/houfin/venv/bin/python -u research/exp/z_repair_teacher.py --package "$ROOT/data" --points "$POINTS" --out "$OUT" --threads 8 > "$ROOT/teachers/$TASK/$SEASON.log" 2>&1
