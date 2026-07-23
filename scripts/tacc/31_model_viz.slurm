#!/bin/bash
# Post-MAP ecological diagnostics. Reconstructs fitted fields on GPU and writes
# modern fundamental-niche maps, the 1902--2025 niche trajectory, demographic
# rate maps, and BBS fit checks. Select MAP profile with HOUFIN_MAP_PROFILE.
#SBATCH -J houfin_mapviz
#SBATCH -o houfin_mapviz.o%j
#SBATCH -e houfin_mapviz.e%j
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=all
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
source "$SLURM_SUBMIT_DIR/scripts/tacc/gpu_monitor.sh"
module load python/3.12.11 2>/dev/null || true
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
gpu_preflight "post-MAP diagnostics"
start_gpu_monitor "$SLURM_SUBMIT_DIR/gpu_mapviz_${SLURM_JOB_ID}.csv" 30
trap stop_gpu_monitor EXIT
date

export HOUFIN_MAP_PROFILE="${HOUFIN_MAP_PROFILE:-standard}"
/usr/bin/time -v python scripts/viz/map_diagnostics.py --profile "$HOUFIN_MAP_PROFILE"

echo "=== 31_model_viz complete (profile=$HOUFIN_MAP_PROFILE) ==="
date
