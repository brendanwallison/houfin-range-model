#!/bin/bash
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_HOLDOUT_20261025
cd "$ROOT/code_damage_r2"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1
PY=/home/breallis/houfin/venv/bin/python
mkdir -p "$ROOT/sealed/pseudo_fixed_taxonomy"
"$PY" research/exp/cbc_prepare.py --raw-dir /mnt/e/Datasets/cbc_raw/pseudo_early3_w6695 \
  --products-csv '/mnt/e/Datasets/houfin/external/cbc_trends_2023/US and Canada Trends/cbc_indices_version_2023.0_na_web_download.csv' \
  --strata-shp /mnt/e/Datasets/houfin/external/bcr/bbsBayes/inst/maps/BBS_USGS_strata.shp \
  --strata-shp2 /mnt/e/Datasets/houfin/external/bcr/bbsBayes/inst/maps/BBS_CWS_strata.shp \
  --cache /home/breallis/houfin/work/houfin/research_cache/desk_tempho_1995_s1 --species-set sealed \
  --out "$ROOT/sealed/pseudo_fixed_taxonomy/cbc_circles.npz" \
  --taxonomy-reference "$ROOT/sealed/cbc_circles_meta.json" \
  --confirmation-reason 'FINAL_HELDOUT: fixed-readout check against existing E086v pseudo-early observation damage'
"$PY" research/exp/final_winter_damage_check.py --root "$ROOT" --damaged "$ROOT/sealed/pseudo_fixed_taxonomy/cbc_circles.npz"
