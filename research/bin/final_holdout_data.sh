#!/bin/bash
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_HOLDOUT_20261025
cd "$ROOT/code_data"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data
export HOUFIN_PROCESSED=/work/07980/bwa386/ls6/houfin/processed
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=
PY=/home/breallis/houfin/venv/bin/python
RAW=/mnt/e/Datasets/cbc_raw/20250205_FullSet_AllCountries_AllYears_CBC_Data_Request
PREP=/mnt/e/Datasets/cbc_raw/prepared
BASE=/home/breallis/houfin/work/houfin/research_cache/desk_tempho_1995_s1
TAX=$HOUFIN_DATA/avonet/eBird_taxonomy.csv
mkdir -p "$ROOT/sealed" "$ROOT/audit"
"$PY" -c 'from research.adopt import record_read; record_read("raw CBC completeness", "FINAL_HELDOUT: audit indirect sealed-species completeness influence before confirmation")'
"$PY" research/exp/cbc_completeness.py --raw-dir "$RAW" --prepared "$PREP/cbc_circles_all.npz" \
  --out "$ROOT/audit/completeness_all.npz" --bench-from 1966 --taxonomy "$TAX"
"$PY" research/exp/cbc_completeness.py --raw-dir "$RAW" --prepared "$PREP/cbc_circles_all.npz" \
  --out "$ROOT/audit/completeness_training_only.npz" --bench-from 1966 --taxonomy "$TAX" \
  --species-allowlist "$HOUFIN_PROCESSED/encoder/cbc_aux_final/community_all_nonsealed.csv"
"$PY" research/exp/cbc_prepare.py --raw-dir "$RAW" \
  --products-csv '/mnt/e/Datasets/houfin/external/cbc_trends_2023/US and Canada Trends/cbc_indices_version_2023.0_na_web_download.csv' \
  --strata-shp /mnt/e/Datasets/houfin/external/bcr/bbsBayes/inst/maps/BBS_USGS_strata.shp \
  --strata-shp2 /mnt/e/Datasets/houfin/external/bcr/bbsBayes/inst/maps/BBS_CWS_strata.shp \
  --cache "$BASE" --species-set sealed --out "$ROOT/sealed/cbc_circles.npz" \
  --confirmation-reason 'FINAL_HELDOUT: fixed v1/v2 summer-winter confirmation, no encoder tuning'
"$PY" research/exp/final_completeness_audit.py --original "$PREP/completeness_v3.npz" \
  --rebuilt "$ROOT/audit/completeness_all.npz" --clean "$ROOT/audit/completeness_training_only.npz" \
  --cbc "$PREP/cbc_circles_all.npz" --out "$ROOT/audit/result.json"
date -Is > "$ROOT/DATA_DONE"
