#!/bin/bash
# Build the land-conditioned Z_disp feature cube, then ingest the complete
# 1902--2025 Z/Z_disp/BBS timeline into NumPyro-ready model inputs.
# Run after DESK + cube; run before 30_model_map.
#SBATCH -J houfin_modelprep
#SBATCH -o houfin_modelprep.o%j
#SBATCH -e houfin_modelprep.e%j
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=all
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
source "$SLURM_SUBMIT_DIR/scripts/tacc/gpu_monitor.sh"
module load python/3.12.11 2>/dev/null || true
module load remora 2>/dev/null || true
[ -n "${TACC_ALLOCATION:-}" ] && echo "allocation: $TACC_ALLOCATION"

gpu_preflight "Z_disp/model-ingest preparation"
start_gpu_monitor "$SLURM_SUBMIT_DIR/gpu_modelprep_${SLURM_JOB_ID}.csv" 30
trap stop_gpu_monitor EXIT

export STAGES="${STAGES:-path-features model-ingest}"
bash "$SLURM_SUBMIT_DIR/scripts/tacc/pipeline.sh"

echo "=== 25_model_prep complete ($STAGES) ==="
date
