#!/bin/bash
#SBATCH -J f54e2_fits
#SBATCH -A DEB23008
#SBATCH -p development
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54e2_fits.o%j
# Stage E twin field fits: field_d --summer interp (= 27 km field_d_interp) on D = desk_interp_s1 raw and on its EMA cube,
# with the pre-registered inner-grid choice (FIELD54_INNER_CHOICE.json).
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code_bc/research/field54/envbc.sh; env54
source $CODE/research/field54/e_common.sh
CH=$EVAL/assim/backcast40/FIELD54_INNER_CHOICE.json
need $CH $DCUBE/Z_latent_2025.npy $ECUBE/EMA_DONE.json
CFG=$(python -c "import json;print(json.dumps(json.load(open('$CH'))['cfg']))")
echo "[f54e2] cfg $CFG"
python -u research/exp/z_assim.py fit --root $EVAL --task backcast40 --arm field_d --summer interp --tag _interp   --control-cube desk_interp_s1 --cfg "$CFG" --threads 30 &
python -u research/exp/z_assim.py fit --root $EVAL --task backcast40 --arm field_d --summer interp --tag _interp_ema   --control-cube desk_interp_s1_ema --cfg "$CFG" --threads 30 &
fail=0; for j in $(jobs -p); do wait $j || fail=1; done
[ $fail = 0 ] || exit 1
echo "[f54e2] done $(date)"
