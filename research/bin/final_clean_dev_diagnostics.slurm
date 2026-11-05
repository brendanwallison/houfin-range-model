#!/bin/bash
# Recover descriptive MAP diagnostics and audit the stalled point; no optimization.
#SBATCH -J desk_clean_audit
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu-a100-dev
#SBATCH -t 02:00:00
set -euo pipefail
ROOT=/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025
source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
export TWT="$ROOT/code_clean2" PYTHONPATH="$ROOT/code_clean2:$ROOT/code_clean2/src" PYTHONUNBUFFERED=1
cd "$TWT"
export AGE_MODEL_CONFIG="$TWT/config/overlays/map_run_fd_clean_v2_m2_r128.json"
export HOUFIN_MODEL_PRECISION=float64 HOUFIN_Z_FLOAT32=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HOUFIN_HMC_MAP_PRECISION=float64
unset HOUFIN_REFINE_SCALE_FILE HOUFIN_REFINE_TAG HOUFIN_REFINE_FRESH HOUFIN_REFINE_INIT HOUFIN_REFINE_FREEZE HOUFIN_HMC_ALLOW_MODEL_DRIFT
OUT="$ROOT/dev_diagnostics_clean2"
mkdir -p "$OUT"
sha256sum -c SOURCE_MANIFEST.sha256 > "$OUT/source_verification.log"
cp SOURCE_MANIFEST.sha256 "$OUT/code_manifest.sha256"
MAP="$HOUFIN_PROCESSED/model_results/age_map_float32_fd_clean_v2_m2_r128_quick90"
if [ ! -f "$OUT/MAP_DIAGNOSTICS_DONE" ]; then
 python scripts/viz/map_diagnostics.py --precision float64 --run-dir "$MAP" --out "$OUT/viz_map" > "$OUT/viz_map.log" 2>&1
 python research/age_model/exp/west_sink_audit.py --viz-dir "$OUT/viz_map" --out "$OUT/viz_map/west_sink_audit.json" > "$OUT/west_sink.log" 2>&1
 date -Is > "$OUT/MAP_DIAGNOSTICS_DONE"
fi
export HOUFIN_HMC_MAP_DIR="$HOUFIN_PROCESSED/model_results/refined_float64__age_map_float32_fd_clean_v2_m2_r128_quick90"
if [ ! -f "$OUT/GRADIENT_AUDIT_DONE" ]; then
 python "$ROOT/diagnostics/final_gradient_audit.py" --out "$OUT/stalled_gradient_audit.json" --steps 1e-7 1e-8 1e-9 > "$OUT/gradient_audit.log" 2>&1
 date -Is > "$OUT/GRADIENT_AUDIT_DONE"
fi
date -Is > "$OUT/DIAGNOSTICS_DONE"
