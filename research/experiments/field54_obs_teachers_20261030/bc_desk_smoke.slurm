#!/bin/bash
#SBATCH -J f54d_dsmoke
#SBATCH -A DEB23008
#SBATCH -p development
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54d_dsmoke.o%j
# FIELD54 Stage D: prepare the twin's frozen 54 km DESK code/config, then a 2-epoch CPU smoke + cube + EMA cube.
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code_bc/research/field54/envbc.sh
env54
export OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64
TASK=${TASK:-backcast40}; ROOT=$([ "$TASK" = production ] && echo $PROD || echo $EVAL)
test -e $ROOT/coord_code || python -u research/exp/field54_desk.py prepare --root $ROOT --task $TASK \
  --data-config $CODE/config/overlays/field54_data.json
cd $ROOT/coord_code
export PYTHONPATH=$PWD:$PWD/src DATA_CONFIG=$ROOT/coord_code/config/field54_data.json
python -u research/exp/field54_desk.py smoke --root $ROOT --task $TASK --epochs 2
echo "[f54d_dsmoke] done $(date)"
