#!/bin/bash
# E086w on TACC: DESK from 1902 (cbc) + cube. Runs from the desk-research worktree (worker sets cwd / env).
set -euo pipefail
export ESK_DESK_CONFIG=config/overlays/desk_cbc1902.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/tacc/gpu_monitor.sh; start_gpu_monitor "$WORK/houfin/research/desk_temporal/gpu_desk_cbc1902_${SLURM_JOB_ID}.csv" 30; trap stop_gpu_monitor EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv
git log --oneline -1
python scripts/run_encoder.py desk
python scripts/run_encoder.py cube
