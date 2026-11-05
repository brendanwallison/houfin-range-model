#!/bin/bash
set -uo pipefail
ROOT=/work/07980/bwa386/ls6/houfin
source "$ROOT/age-research/scripts/tacc/env.sh"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HOUFIN_PROCESSED=$ROOT/processed HOUFIN_Z_FLOAT32=1 HOUFIN_MODEL_PRECISION=float64 XLA_PYTHON_CLIENT_MEM_FRACTION=.85
unset HOUFIN_SYNTHETIC_OBS
rc=0
for STEP in step3_native step3_native_v2 step8a_native step8a_native_v2 step25_natdep step26_mdd110 step26_mdd200 step29_yb_native_s42; do
 R=$ROOT/research/grid54/runs/$STEP; CODE=$R/code
 cd "$CODE" || exit 1
 export PYTHONPATH=$CODE:$CODE/src AGE_MODEL_CONFIG=$R/out/resolved_config.json DATA_CONFIG=$CODE/config/overlays/grid54_data.json
 REF=$ROOT/processed/model_results/refined_float64__age_map_float32_grid54_${STEP}_quick90
 if timeout 600 "$ROOT/venv/bin/python" "$ROOT/research/grid54/monitor/range_pathology_readout.py" --run "$REF" --out "$R/out/pathology.json" > "$R/out/pathology.log" 2>&1; then
  cp "$R/out/pathology.json" "$ROOT/research/grid54/overnight/.retrospective_$STEP.tmp"
  mv "$ROOT/research/grid54/overnight/.retrospective_$STEP.tmp" "$ROOT/research/grid54/overnight/retrospective_$STEP.json"
 else rc=1; fi
 echo "[retrospective] checked $STEP $(date -Is)"
done
exit "$rc"
