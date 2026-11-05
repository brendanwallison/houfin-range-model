#!/bin/bash
#SBATCH -J f54_luh
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54_luh.o%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A DEB23008
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code/research/field54/env54.sh
export GDAL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HOUFIN_PREPROCESS_WORKERS=${HOUFIN_PREPROCESS_WORKERS:-48}
date; /usr/bin/time -v python -m src.data.preprocess.hyde
date; /usr/bin/time -v python -m src.data.preprocess.luh3
date; ls $HOUFIN_DATA54/luh3_grid | wc -l; ls $HOUFIN_DATA54/hyde35_grid | wc -l; ls $HOUFIN_DATA54/luh3_grid_plainavg_diag | wc -l
