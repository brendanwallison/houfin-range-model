#!/bin/bash
#SBATCH -J f54_statv2
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54_statv2.o%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A DEB23008
# States v2 into a SIBLING dir (processed/field54/states_v2); states v1 is never touched.
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code/research/field54/env54.sh
export ESK_DESK_CONFIG="$CODE/config/overlays/field54_states_v2.json"
export GDAL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -c "from src.config_utils import load_config as L; c=L(); h=c['paths']['hist_dir']; assert h.endswith('/field54/states_v2'), h; print(h); [print(s['name'], s['grid_dir']) for s in c['states']['streams']]"
date; /usr/bin/time -v python -m src.data.combine.build_states --write-workers 16
date; ls $HOUFIN_PROCESSED54/states_v2/yearly_states | wc -l
