#!/bin/bash
#SBATCH -J f54bc_teach
#SBATCH -A DEB23008
#SBATCH -p development
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54bc_teach.o%j
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code_bc/research/field54/envbc.sh
env54
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
python -c "from src.config_utils import load_data_config as L; c=L(); assert c['grid']['target_res_m']==54000 and c['grid']['ref_raster'].startswith('/'), c['grid']; print('[env54] ok', c['grid']['ref_raster'])"
mkdir -p $EVAL $PROD
test -e $EVAL/data/DONE.json || python -u research/exp/z_repair_data.py --base $OBS/base_eval --points $OBS/bbs_points_eval54 --cbc $OBS/cbc_eval54.npz \
  --holdout $OBS/holdout_cells54.npy --buffer $OBS/buffer_cells54.npy --out $EVAL/data --tasks backcast40 --winters full
test -e $PROD/data/DONE.json || python -u research/exp/z_repair_data.py --production --base $OBS/base_prod --points $OBS/bbs_points_prod54 \
  --cbc $OBS/cbc_prod54.npz --out $PROD/data
echo "[f54bc_teach] packages done $(date)"
teach() { # root task season points
  mkdir -p $1/teachers/$2
  python -u research/exp/z_repair_teacher.py --package $1/data --points $4 --out $1/teachers/$2/$3 --threads 32 \
    > $1/teachers/$2/$3.log 2>&1
}
teach $EVAL backcast40 summer $EVAL/data/backcast40/summer_train_points &
teach $EVAL backcast40 winter_full $EVAL/data/backcast40/winter_full/count_points &
teach $PROD production summer $PROD/data/production/summer_train_points &
teach $PROD production winter_full $PROD/data/production/winter_full/count_points &
fail=0; for j in $(jobs -p); do wait $j || fail=1; done
tail -n 4 $EVAL/teachers/backcast40/*.log $PROD/teachers/production/*.log
[ $fail = 0 ] || { echo "teacher failure"; exit 1; }
echo "[f54bc_teach] teachers done $(date)"
mkdir -p $EVAL/aux/backcast40 $PROD/aux/production
(python -u research/exp/z_repair_aux.py --root $EVAL --task backcast40 --winter full --block-cells 3 > $EVAL/aux/backcast40/full.log 2>&1) &
(python -u research/exp/z_repair_aux.py --root $PROD --task production --winter full --block-cells 3 > $PROD/aux/production/full.log 2>&1) &
(python -u research/exp/z_assim_interp_prep.py --root $EVAL --base $OBS/base_eval > $EVAL/aux/backcast40/interp_prep.log 2>&1) &
fail=0; for j in $(jobs -p); do wait $j || fail=1; done
tail -n 6 $EVAL/aux/backcast40/full.log $PROD/aux/production/full.log $EVAL/aux/backcast40/interp_prep.log
[ $fail = 0 ] || { echo "aux failure"; exit 1; }
python -c "
from pathlib import Path; import sys
from research.exp.z_coord_run import write_points
print('interp points ->', write_points(Path('$EVAL'), 'backcast40'))"
python -u research/exp/field54_teacher_checks.py --eval $EVAL --prod $PROD --obs $OBS --out $F/runs/FIELD54_TEACHER_CHECKS.json
echo "[f54bc_teach] done $(date)"
