#!/bin/bash
# Score cube arms with the z_repair estimands, residency-stratified (research/exp/z_assim_evaluate.py).
set -euo pipefail
ROOT=/mnt/c/Dev/houfin-range-model/.research_runtime/community_z_repair_20261026
CODE=/mnt/c/Dev/houfin-range-model
BASE=/home/breallis/houfin/work/houfin/research_cache/desk_tempho_1995_s1
SIM=/home/breallis/houfin/work/houfin/research/E059/species_similarity.csv
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/home/breallis/houfin/work/houfin/processed
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE:$CODE/src"
cd "$CODE"
exec /home/breallis/houfin/venv/bin/python -u research/exp/z_assim_evaluate.py --root "$ROOT" --base "$BASE" --similarity "$SIM" --cbc /mnt/e/Datasets/cbc_raw/prepared/cbc_circles_all.npz "$@"
