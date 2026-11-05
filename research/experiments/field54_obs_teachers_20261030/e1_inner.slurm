#!/bin/bash
#SBATCH -J f54e1_inner
#SBATCH -A DEB23008
#SBATCH -p development
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54e1_inner.o%j
# Stage E inner grid (pre-registered, FIELD54_FIELD_2026-10-30.md): field_d over lam_lap x lam_u on the twin's inner
# backcast (summer 1996-2005 hidden at training cells; training data only), D = desk_interp_s1 raw, 3-cell blocks.
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code_bc/research/field54/envbc.sh; env54
source $CODE/research/field54/e_common.sh
need $EVAL/models/backcast40/interp_s1/CUBE_DONE $DCUBE/Z_latent_2025.npy
GRID=$(python research/exp/field54_stage_e.py grid)
for i in 0 1 2 3 4; do
  python -u research/exp/z_assim.py inner --root $EVAL --task backcast40 --control-cube desk_interp_s1 --block-cells 3     --inner-grid "$GRID" --part $i/5 --threads 24 > $EVAL/assim_inner_$i.log 2>&1 &
done
fail=0; for j in $(jobs -p); do wait $j || fail=1; done
tail -n 3 $EVAL/assim_inner_*.log; [ $fail = 0 ] || exit 1
python research/exp/field54_stage_e.py select --root $EVAL
echo "[f54e1] done $(date)"
