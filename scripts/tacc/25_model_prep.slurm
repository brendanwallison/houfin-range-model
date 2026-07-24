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

export STAGES="${STAGES:-path-features model-ingest}"

# path-features (Z_disp path integration / FFT kernel convolution) is the only
# stage here that actually needs GPU compute; model-ingest alone is numpy/
# memmap I/O plus a one-time kernel-array construction (fine on CPU/JAX-CPU).
# Only gate on a GPU when path-features is actually selected, so
# STAGES=model-ingest can run on a non-GPU queue (e.g. vm-small) to skip the
# GPU queue entirely for a data-prep-only rerun (e.g. after a spatiotemporal-
# basis config change -- no Z_disp regeneration needed).
case " $STAGES " in
    *" path-features "*)
        gpu_preflight "Z_disp path-feature generation"
        start_gpu_monitor "$SLURM_SUBMIT_DIR/gpu_modelprep_${SLURM_JOB_ID}.csv" 30
        trap stop_gpu_monitor EXIT
        ;;
    *)
        echo "[modelprep] STAGES='$STAGES' has no GPU-only stage; skipping gpu_preflight/telemetry."
        ;;
esac

bash "$SLURM_SUBMIT_DIR/scripts/tacc/pipeline.sh"

echo "=== 25_model_prep complete ($STAGES) ==="
date
