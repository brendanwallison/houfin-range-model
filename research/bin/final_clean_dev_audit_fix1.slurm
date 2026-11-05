#!/bin/bash
# Correct the audit-script CLI mismatch from job 3534241; keep its artifacts.
#SBATCH -J desk_clean_audit_fix1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu-a100-dev
#SBATCH -t 02:00:00
set -euo pipefail
ROOT=/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025
source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
CODE="$ROOT/code_clean2"
OUT="$ROOT/dev_diagnostics_clean2_fix1"
export PYTHONPATH="$CODE:$CODE/src" PYTHONUNBUFFERED=1
export AGE_MODEL_CONFIG="$CODE/config/overlays/map_run_fd_clean_v2_m2_r128.json"
export HOUFIN_MODEL_PRECISION=float64 HOUFIN_Z_FLOAT32=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HOUFIN_HMC_MAP_PRECISION=float64
export HOUFIN_HMC_MAP_DIR="$HOUFIN_PROCESSED/model_results/refined_float64__age_map_float32_fd_clean_v2_m2_r128_quick90"
unset HOUFIN_REFINE_SCALE_FILE HOUFIN_REFINE_TAG HOUFIN_REFINE_FRESH HOUFIN_REFINE_INIT HOUFIN_REFINE_FREEZE HOUFIN_HMC_ALLOW_MODEL_DRIFT
cd "$CODE"
mkdir -p "$OUT"
sha256sum -c SOURCE_MANIFEST.sha256 > "$OUT/source_verification.log"
python "$OUT/final_gradient_audit.py" --out "$OUT/stalled_gradient_audit.json" --steps 1e-7 1e-8 1e-9 > "$OUT/gradient_audit.log" 2>&1
date -Is > "$OUT/GRADIENT_AUDIT_DONE"
