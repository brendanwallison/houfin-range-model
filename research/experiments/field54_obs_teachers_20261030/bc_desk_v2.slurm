#!/bin/bash
#SBATCH -J f54d_v2prep
#SBATCH -A DEB23008
#SBATCH -p development
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54d_v2prep.o%j
# Production rebuild on states_v2: new root field54_prod_v2 sharing field54_prod's data/teachers/aux (symlinks, read
# only), frozen 54 km DESK config with hist_dir = states_v2, 2-epoch CPU smoke; on success, queue the GPU task
# 0320_field54_desk_prod_v2.sh in the grid54 queue.
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code_bc/research/field54/envbc.sh
env54
export OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64
R=$F/runs/field54_prod_v2
mkdir -p $R
for d in data teachers aux; do test -e $R/$d || ln -s $PROD/$d $R/$d; done
test -e $R/coord_code || python -u research/exp/field54_desk.py prepare --root $R --task production \
  --data-config $CODE/config/overlays/field54_data.json --states states_v2 --climate-grid climate_grid_monthly_v2
cd $R/coord_code
export PYTHONPATH=$PWD:$PWD/src DATA_CONFIG=$R/coord_code/config/field54_data.json
grep -q states_v2/yearly_states config/z_coord_interp.json
python -u research/exp/field54_desk.py smoke --root $R --task production --epochs 2
Q=/work/07980/bwa386/ls6/houfin/research/grid54/queue
for d in pending running done failed; do test ! -e $Q/$d/0320_field54_desk_prod_v2.sh || { echo "task exists in $d"; exit 1; }; done
cp $F/logs/0320_field54_desk_prod_v2.sh.staged $Q/pending/.0320_field54_desk_prod_v2.sh.tmp
mv $Q/pending/.0320_field54_desk_prod_v2.sh.tmp $Q/pending/0320_field54_desk_prod_v2.sh
sha256sum $Q/pending/0320_field54_desk_prod_v2.sh
echo "[f54d_v2prep] done $(date)"
