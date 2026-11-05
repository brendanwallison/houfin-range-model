#!/bin/bash
#SBATCH -J f54e3_score
#SBATCH -A DEB23008
#SBATCH -p development
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54e3_score.o%j
# Stage E twin scoring (withheld 54 km panel, k2; sedentary common first): block 3 (primary, 162 km) and 6 (sensitivity),
# then the pre-registered rules 1-3 (field54_stage_e.py rules).
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code_bc/research/field54/envbc.sh; env54
source $CODE/research/field54/e_common.sh
need $EVAL/cubes/backcast40/field_d_interp_s1/ASSIM_DONE.json $EVAL/cubes/backcast40/field_d_interp_ema_s1/ASSIM_DONE.json $ECUBE/EMA_DONE.json $EVAL/teacher_bottleneck/PROJECTION_DONE.json
ARMS="desk_interp_s1 desk_interp_s1_ema field_d_interp_s1 field_d_interp_ema_s1 fdinterp27_avg54_s1"
for bc in 3 6; do
  E="python -u research/exp/z_assim_evaluate.py $EVALARGS --block-cells $bc --no-teacher fdinterp27_avg54_s1"
  $E --arms $ARMS --out ASSIM_F54_MAIN_b$bc
  $E --arms desk_interp_s1_ema field_d_interp_s1 field_d_interp_ema_s1 --out ASSIM_F54_VSEMA_b$bc
  $E --arms field_d_interp_s1 field_d_interp_ema_s1 --out ASSIM_F54_EMAVSRAW_b$bc
  $E --arms fdinterp27_avg54_s1 field_d_interp_s1 field_d_interp_ema_s1 --out ASSIM_F54_NATIVEVSAVG_b$bc
  python research/exp/field54_stage_e.py rules --root $EVAL --block-cells $bc
done
echo "[f54e3] done $(date)"
