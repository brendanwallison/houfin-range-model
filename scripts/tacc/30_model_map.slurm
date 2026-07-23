#!/bin/bash
#----------------------------------------------------------------------
# MAP fit of the age-structured model (NumPyro AutoDelta / SVI point estimate).
# GPU job. Reads the ingested inputs at age_model_config.input_dir
# (numpyro_input/: metadata.pkl + its versioned Z/Z_disp memmaps) and writes
# to results_dir/<run_names.map>/.
#
# CHECKPOINTED: age_run_map writes map_checkpoint.pkl every HOUFIN_MAP_CKPT_EVERY
# steps and resumes from it automatically. So on the 2 h dev queue, if the wall
# clock kills the job mid-fit, just resubmit -- it continues from the last
# checkpoint. Use submit_map.sh (which can auto-chain resubmissions).
#
# PREREQS (must exist first): the DESK cube -> path features (Z_disp) -> BBS npz ->
# scripts/ingest_model_data.py. This job does NOT build them.
#
#   bash scripts/tacc/submit_map.sh                       # gpu-a100-dev, 2 h
#   HOUFIN_MAP_STEPS=1800 RESUBMITS=3 bash scripts/tacc/submit_map.sh
#----------------------------------------------------------------------
#SBATCH -J houfin_map
#SBATCH -o houfin_map.o%j
#SBATCH -e houfin_map.e%j
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=all
##SBATCH -A REPLACE_WITH_PROJECT
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
source "$SLURM_SUBMIT_DIR/scripts/tacc/gpu_monitor.sh"
module load python/3.12.11 2>/dev/null || true
[ -n "${TACC_ALLOCATION:-}" ] && echo "allocation: $TACC_ALLOCATION"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
# Fail instead of silently falling back to CPU. Persist 30-second GPU/host-memory
# telemetry; /usr/bin/time adds peak process RSS to the SLURM log.
gpu_preflight "age-model MAP"
start_gpu_monitor "$SLURM_SUBMIT_DIR/gpu_map_${SLURM_JOB_ID}.csv" 30
trap stop_gpu_monitor EXIT
date

# Checkpoint cadence is set here.  The step target is deliberately left to
# age_run_map: standard defaults to 900, while an explicit MAP profile (such as
# quick90) supplies its own target.  An explicit HOUFIN_MAP_STEPS still wins.
export HOUFIN_MAP_CKPT_EVERY="${HOUFIN_MAP_CKPT_EVERY:-100}"
/usr/bin/time -v python -m src.model.age_run_map

echo "=== 30_model_map complete (steps=${HOUFIN_MAP_STEPS:-profile default}) ==="
date
