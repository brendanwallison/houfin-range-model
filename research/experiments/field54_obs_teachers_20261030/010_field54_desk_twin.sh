#!/bin/bash
# FIELD54 Stage D twin (research/experiments/FIELD54_DESK_2026-10-30.md): native 54 km DESK, the 27 km desk_interp
# recipe (frozen 27 km coord_code + 54 km config; diff in field54_eval/FIELD54_CONFIG_DIFF.json), 500 fixed epochs, no
# epoch selection on outcomes. Then the raw cube (z_coord_run) and the EMA cube at the trained half-life (z_ema_cube).
# RESUMABLE: DESK writes resume_checkpoint.pt every 10 epochs (configuration-fingerprinted); a cut task rerun from the
# top resumes training, skips it once TRAINING_DONE exists, re-exports the raw cube, then (re)writes the EMA cube.
# Expected ~35-45 min on an A100 (27 km: ~8.6 s/epoch; 54 km has ~1/4 of the cells).
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() > 0, 'no usable GPU on this node'"
nvidia-smi --query-gpu=name,memory.total --format=csv
R=/work/07980/bwa386/ls6/houfin/research/field54/runs/field54_eval
test -e $R/COORD_TRAINING_SMOKE_DONE
unset ESK_DESK_CONFIG
export HOUFIN_DATA54=/scratch/07980/bwa386/houfin/data54 HOUFIN_PROCESSED54=/work/07980/bwa386/ls6/houfin/processed/field54
export HOUFIN_DATA=$HOUFIN_DATA54 HOUFIN_PROCESSED=$HOUFIN_PROCESSED54
cd $R/coord_code
export PYTHONPATH=$PWD:$PWD/src DATA_CONFIG=$PWD/config/field54_data.json PYTHONUNBUFFERED=1
python -c "from src.config_utils import load_data_config as L; c=L(); assert c['grid']['target_res_m']==54000 and c['grid']['ref_raster'].startswith('/'), c['grid']"
python -u research/exp/z_coord_run.py run --root $R --arm interp
if [ ! -e $R/cubes/backcast40/desk_interp_s1_ema/EMA_DONE.json ]; then
  python -u research/exp/z_ema_cube.py --root $R --task backcast40 --model interp_s1 --cube desk_interp_s1
fi
echo "[010_field54_desk_twin] done $(date)"
