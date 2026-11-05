#!/bin/bash
#SBATCH -J f54e0_now
#SBATCH -A DEB23008
#SBATCH -p development
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54e0_now.o%j
# Stage E parts that need no 54 km DESK cube: observed teacher projection; rule-1 averaged arm scored (b3, b6).
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code_bc/research/field54/envbc.sh; env54
source $CODE/research/field54/e_common.sh
python -m pytest -q -p no:cacheprovider tests/test_field54_stage_e.py
test -e $EVAL/teacher_bottleneck/PROJECTION_DONE.json || python -u research/exp/z_repair_teacher_bottleneck.py --root $EVAL --base $OBS/base_eval --projection-only
need $EVAL/cubes/backcast40/fdinterp27_avg54_s1/cube_meta.json
for bc in 3 6; do
  python -u research/exp/z_assim_evaluate.py $EVALARGS --block-cells $bc --arms fdinterp27_avg54_s1 --no-teacher fdinterp27_avg54_s1 --score-only
done
echo "[f54e0] done $(date)"
