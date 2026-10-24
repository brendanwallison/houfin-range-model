#!/bin/bash
# E086ac fold 0: tempho T0 1996 seed 1 DESK on the fold's all-sedentary basis.
set -euo pipefail
export RESEARCH_OUT=$WORK/houfin/research/desk_temporal/runs/e086ac_f0_s
mkdir -p $RESEARCH_OUT/desk $RESEARCH_OUT/tmp; export TMPDIR=$RESEARCH_OUT/tmp MPLCONFIGDIR=$RESEARCH_OUT/tmp/mpl WANDB_MODE=offline WANDB_DIR=$RESEARCH_OUT
export ESK_DESK_CONFIG=research/overlays/desk_tempho_1995_seed1_ac_f0_s.json PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
git log --oneline -1
python scripts/run_encoder.py desk
