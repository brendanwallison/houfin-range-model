#!/bin/bash
set -uo pipefail
ROOT=/work/07980/bwa386/ls6/houfin
source "$ROOT/age-research/scripts/tacc/env.sh"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HOUFIN_MODEL_PRECISION=float64 HOUFIN_Z_FLOAT32=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.85
unset HOUFIN_SYNTHETIC_OBS
PY=$ROOT/venv/bin/python
overall=0
for STEP in step29_yb_native step31_nestonly step31_nest step31_nest_bloose step32_cbc_yb; do
 R=$ROOT/research/grid54/runs/$STEP; CODE=$R/code
 cd "$CODE" || exit 1
 export AGE_MODEL_CONFIG=$CODE/config/overlays/grid54_$STEP.json DATA_CONFIG=$CODE/config/overlays/grid54_data.json
 export PYTHONPATH=$CODE:$CODE/src HOUFIN_PROCESSED=$ROOT/processed
 REF=$ROOT/processed/model_results/refined_float64__age_map_float32_grid54_${STEP}_quick90
 if [ ! -f "$R/out/pathology.json" ]; then
  timeout 1200 "$PY" "$ROOT/research/grid54/monitor/range_pathology_readout.py" --run "$REF" --out "$R/out/pathology.json" > "$R/out/pathology.log" 2>&1 || overall=1
 fi
 if [ "$STEP" = step31_nestonly ] && [ ! -f "$R/out/front_speed.json" ]; then
  timeout 1200 "$PY" "$ROOT/research/grid54/monitor/grid54_front_speed.py" "$REF" "$R/out/front_speed.json" > "$R/out/front_speed.log" 2>&1 || overall=1
 fi
 echo "[readout] checked $STEP $(date -Is)"
done
exit "$overall"
