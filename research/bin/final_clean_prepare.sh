#!/bin/bash
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_CLEAN_20261025
cd "$ROOT/code"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data
export HOUFIN_PROCESSED=/home/breallis/houfin/work/houfin/processed
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=
PY=/home/breallis/houfin/venv/bin/python
ENC="$HOUFIN_PROCESSED/encoder"
CLEAN=/home/breallis/houfin/work/houfin/research/FINAL_HOLDOUT_20261025/audit/completeness_training_only.npz
COMMON=(--cbc /mnt/e/Datasets/cbc_raw/prepared/cbc_circles_all.npz --completeness "$CLEAN"
 --holdout "$ROOT/original_masks/holdout_cells.npy" --buffer "$ROOT/original_masks/buffer_cells.npy")
"$PY" research/exp/final_cbc_aux.py --z-dir "$ENC/esk_all369_breed/spacetime" --points "$ENC/bbs_points_all369" \
 "${COMMON[@]}" --out-dir "$ENC/cbc_aux_final_clean"
cp "$ENC/cbc_aux_final/community_all_nonsealed.csv" "$ENC/cbc_aux_final_clean/"
"$PY" research/exp/final_v2_aux.py --winter-z-dir "$ENC/esk_all369_winter/spacetime" --winter-points "$ENC/cbc_points_all369" \
 "${COMMON[@]}" --v1-aux "$ENC/cbc_aux_final_clean" --out-dir "$ENC/cbc_aux_final_v2_clean"
"$PY" "$ROOT/finalize.py"
