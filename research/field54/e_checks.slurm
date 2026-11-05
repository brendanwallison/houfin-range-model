#!/bin/bash
#SBATCH -J f54_check
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54_check.o%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A DEB23008
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code/research/field54/env54.sh
export GDAL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date; /usr/bin/time -v python research/exp/field54_checks.py \
  --d27 "$HOUFIN_DATA27" --d54 "$HOUFIN_DATA54" \
  --s27 /work/07980/bwa386/ls6/houfin/processed/encoder/states --s54 "$HOUFIN_PROCESSED54/states" \
  --grid54-mask /work/07980/bwa386/ls6/houfin/processed/grid54/land_mask/ocean_mask_54km.tif \
  --out "$HOUFIN_PROCESSED54/field54_checks.json" --sections "${F54_SECTIONS:-mask,luh,hyde,soil,elev,bui,climate,states}"
