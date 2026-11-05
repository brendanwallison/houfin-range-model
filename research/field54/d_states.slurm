#!/bin/bash
#SBATCH -J f54_states
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54_states.o%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A DEB23008
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code/research/field54/env54.sh
export GDAL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -c "from src.config_utils import load_config as L; c=L(); print(c['paths']['hist_dir'], c['latent_cube']['water_mask_path']); [print(s['name'], s['grid_dir']) for s in c['states']['streams']]"
date; /usr/bin/time -v python -m src.data.combine.build_states --write-workers 16
date; ls $HOUFIN_PROCESSED54/states/yearly_states | wc -l
