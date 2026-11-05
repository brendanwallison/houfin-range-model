#!/bin/bash
set -euo pipefail
STEP=$1
case "$STEP" in step31_nestonly|step33_nestcbc_s41|step33_nestcbc_natdep_s41|step33_nestcbc_abs_s41) ;; *) exit 2 ;; esac
ROOT=/work/07980/bwa386/ls6/houfin
R=$ROOT/research/grid54/runs/$STEP; CODE=$R/code
source "$ROOT/age-research/scripts/tacc/env.sh"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HOUFIN_PROCESSED=$ROOT/processed HOUFIN_Z_FLOAT32=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.85
unset HOUFIN_SYNTHETIC_OBS
cd "$CODE"
export PYTHONPATH=$CODE:$CODE/src AGE_MODEL_CONFIG=$CODE/config/overlays/grid54_$STEP.json DATA_CONFIG=$CODE/config/overlays/grid54_data.json
REF=$ROOT/processed/model_results/refined_float64__age_map_float32_grid54_${STEP}_quick90
for precision in float64 float32; do
 export HOUFIN_MODEL_PRECISION=$precision
 "$ROOT/venv/bin/python" "$ROOT/research/grid54/monitor/range_pathology_readout.py" --precision "$precision" --save-state --run "$REF" --out "$R/out/precision_$precision.json" > "$R/out/precision_$precision.log" 2>&1
done
"$ROOT/venv/bin/python" "$ROOT/research/grid54/monitor/compare_point_precision.py" "$R/out/precision_float64.state.npz" "$R/out/precision_float32.state.npz" --out "$ROOT/research/grid54/overnight/precision_$STEP.json"
