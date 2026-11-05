#!/bin/bash
#SBATCH -J f54e4_prod
#SBATCH -A DEB23008
#SBATCH -p development
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54e4_prod.o%j
# Stage E PRODUCTION field fit (not scored): field_d on every summer row (1966-2025, --production) with the twin's
# winning D and the inner-grid setting. Submit with:  sbatch --export=ALL,DWIN=raw|ema e4_prod.slurm
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code_bc/research/field54/envbc.sh; env54
source $CODE/research/field54/e_common.sh
case "${DWIN:-}" in raw) D=desk_interp_s1;; ema) D=desk_interp_s1_ema;; *) echo "set DWIN=raw|ema (twin rule 2)"; exit 2;; esac
PROD=${ROOTP:-$PROD}   # v2 rebuild: ROOTP=$F/runs/field54_prod_v2
CH=$EVAL/assim/backcast40/FIELD54_INNER_CHOICE.json
need $CH $PROD/cubes/production/$D/Z_latent_2025.npy $PROD/models/production/interp_s1/CUBE_DONE
[ $D = desk_interp_s1 ] || need $PROD/cubes/production/$D/EMA_DONE.json
CFG=$(python -c "import json;print(json.dumps(json.load(open('$CH'))['cfg']))")
echo "[f54e4] D=$D cfg $CFG"
python -u research/exp/z_assim.py fit --root $PROD --task production --production --arm field_d --summer train \
  --tag _prod_$DWIN --control-cube $D --cfg "$CFG" --threads 60
echo "[f54e4] done $(date)"
