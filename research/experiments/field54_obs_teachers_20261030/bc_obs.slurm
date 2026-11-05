#!/bin/bash
#SBATCH -J f54bc_obs
#SBATCH -A DEB23008
#SBATCH -p development
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54bc_obs.o%j
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code_bc/research/field54/envbc.sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
for d in "$OBS" "$EVAL/data" "$PROD/data"; do test -e "$d/DONE.json" && { echo "refuse: $d done"; exit 1; }; done
env27
python -m pytest -q -p no:cacheprovider tests/test_field54_obs.py tests/test_z_repair_data.py
python -c "from src.config_utils import load_data_config as L; c=L(); assert c['grid']['target_res_m']==27000, c['grid']; print('[env27] ok', c['datasets_root'])"
python -u research/exp/field54_obs.py --mask54 /scratch/07980/bwa386/houfin/data54/land_mask/ocean_mask_54km.tif \
  --ref54 /scratch/07980/bwa386/houfin/data54/ref_grid_54km.tif --cache-meta $IN27/meta.json \
  --points27 /work/07980/bwa386/ls6/houfin/processed/encoder/bbs_points_all369 --route-years27 $IN27/route_years.csv \
  --cbc $IN27/cbc_circles_all.npz --cbc-sealed $IN27/sealed/cbc_circles.npz --holdout27 $IN27/holdout_cells.npy \
  --buffer27 $IN27/buffer_cells.npy --panel27 $IN27/summer_masks27_backcast40.npz --out $OBS \
  --sealed-read-logged "local research/sealed_reads.csv 2026-10-30T08:57:01-05:00 (FIELD54 production build)"
env54
python -c "from src.config_utils import load_data_config as L; c=L(); assert c['grid']['target_res_m']==54000, c['grid']; print('[env54] ok', c['datasets_root'])"
mkdir -p $EVAL $PROD
python -u research/exp/z_repair_data.py --base $OBS/base_eval --points $OBS/bbs_points_eval54 --cbc $OBS/cbc_eval54.npz \
  --holdout $OBS/holdout_cells54.npy --buffer $OBS/buffer_cells54.npy --out $EVAL/data --tasks backcast40 --winters full
python -u research/exp/z_repair_data.py --production --base $OBS/base_prod --points $OBS/bbs_points_prod54 \
  --cbc $OBS/cbc_prod54.npz --out $PROD/data
echo "[f54bc_obs] done $(date)"
