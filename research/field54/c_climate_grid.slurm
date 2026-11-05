#!/bin/bash
#SBATCH -J f54_cgrid
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54_cgrid.o%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A DEB23008
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code/research/field54/env54.sh
export GDAL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date; /usr/bin/time -v python -m src.data.preprocess.climate_grid
date; ls $HOUFIN_DATA54/climate_grid_monthly | wc -l
