#!/bin/bash
#SBATCH -J f54_chkv2
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54_chkv2.o%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A DEB23008
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code/research/field54/env54.sh
export GDAL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date; python research/exp/field54_climate_v2_compare.py --s1 "$HOUFIN_PROCESSED54/states" \
  --s2 "$HOUFIN_PROCESSED54/states_v2" --missing "$HOUFIN_DATA54/climate_extra/missing_subpoints.csv" \
  --mask "$HOUFIN_DATA54/land_mask/ocean_mask_54km.tif" --out "$HOUFIN_PROCESSED54/field54_climate_v2_compare.json"
date; F54_CLIM_YEAR_STRIDE=5 /usr/bin/time -v python research/exp/field54_checks.py \
  --d27 "$HOUFIN_DATA27" --d54 "$HOUFIN_DATA54" --clim-dir climate_v2 --clim-grid climate_grid_monthly_v2 \
  --s27 /work/07980/bwa386/ls6/houfin/processed/encoder/states --s54 "$HOUFIN_PROCESSED54/states_v2" \
  --grid54-mask /work/07980/bwa386/ls6/houfin/processed/grid54/land_mask/ocean_mask_54km.tif \
  --out "$HOUFIN_PROCESSED54/field54_checks_v2.json" --sections mask,climate,states
date
