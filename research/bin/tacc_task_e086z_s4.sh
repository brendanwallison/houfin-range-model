#!/bin/bash
# E086z on TACC: tempho T0 1996 seed 4 + winter kernel half (sum kernel). Run dir is copied back for caching.
set -euo pipefail
export RESEARCH_OUT=$WORK/houfin/research/desk_temporal/runs/e086z_s4
mkdir -p $RESEARCH_OUT/desk $RESEARCH_OUT/tmp; export TMPDIR=$RESEARCH_OUT/tmp MPLCONFIGDIR=$RESEARCH_OUT/tmp/mpl WANDB_MODE=offline WANDB_DIR=$RESEARCH_OUT
export ESK_DESK_CONFIG=research/overlays/desk_tempho_1995_seed4_sumw.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
git log --oneline -1; nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/run_encoder.py desk
