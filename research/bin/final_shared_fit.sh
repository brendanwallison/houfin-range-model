#!/bin/bash
# One matched cold fit; source and inputs are frozen by the round1 staging script.
set -euo pipefail
ARM="$1"
case "$ARM" in s64|w64|r128) ;; *) exit 2;; esac
if [ "${FINAL_SHARED_LOCAL:-0}" = 1 ]; then
 ROOT=/home/breallis/houfin/work/houfin/research/FINAL_SHARED_ROUND1_20261025
 PY=/home/breallis/houfin/venv/bin/python
 export PYTHONPATH="/home/breallis/houfin/work/houfin/research/FINAL_LOCAL_JAX_20261025/runtime:$ROOT/code:$ROOT/code/src"
 export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/work/07980/bwa386/ls6/houfin/processed
 export XLA_PYTHON_CLIENT_MEM_FRACTION=0.72 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
else
 ROOT=/work/07980/bwa386/ls6/houfin/research/final_shared_round1_20261025
 source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
 PY=python
 export PYTHONPATH="$ROOT/code:$ROOT/code/src" XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
fi
export AGE_MODEL_CONFIG="$ROOT/code/config/overlays/map_run_shared_clean_${ARM}.json"
export HOUFIN_Z_FLOAT32=1 HOUFIN_MAP_PROFILE=quick90 HOUFIN_REFINE_MAX_ITER=16000 PYTHONUNBUFFERED=1
unset HOUFIN_REFINE_SCALE_FILE HOUFIN_REFINE_TAG HOUFIN_REFINE_FRESH HOUFIN_REFINE_INIT HOUFIN_REFINE_FREEZE HOUFIN_MAP_REUSE HOUFIN_HMC_ALLOW_MODEL_DRIFT
cd "$ROOT/code"
OUT="$ROOT/$ARM"; mkdir -p "$OUT"
sha256sum -c ROUND1_MANIFEST.sha256 > "$OUT/source_verification.log"
"$PY" -c 'import json; from src.config_utils import load_age_model_config; print(json.dumps(load_age_model_config(),indent=2))' > "$OUT/resolved_config.json"
MAP="$HOUFIN_PROCESSED/model_results/age_map_float32_shared_clean_${ARM}_quick90"
if [ ! -f "$OUT/MAP_DONE" ]; then
 export HOUFIN_MODEL_PRECISION=float32
 "$PY" -m src.model.age_run_map > "$OUT/map.log" 2>&1
 date -Is > "$OUT/MAP_DONE"
fi
export HOUFIN_MODEL_PRECISION=float64
if [ ! -f "$OUT/MAP_DIAGNOSTICS_DONE" ]; then
 "$PY" scripts/viz/map_diagnostics.py --precision float64 --run-dir "$MAP" --out "$OUT/viz_map" > "$OUT/viz_map.log" 2>&1
 "$PY" research/age_model/exp/west_sink_audit.py --viz-dir "$OUT/viz_map" --out "$OUT/viz_map/west_sink_audit.json" > "$OUT/west_sink_map.log" 2>&1
 date -Is > "$OUT/MAP_DIAGNOSTICS_DONE"
fi
export HOUFIN_HMC_MAP_PRECISION=float32 HOUFIN_HMC_MAP_DIR="$MAP"
"$PY" -m src.model.age_refine_map > "$OUT/refine.log" 2>&1
REF="$HOUFIN_PROCESSED/model_results/refined_float64__age_map_float32_shared_clean_${ARM}_quick90"
export REPORT="$REF/refine_report.json"
"$PY" -c 'import json,os; d=json.load(open(os.environ["REPORT"])); assert d["valid"],d'
"$PY" scripts/viz/map_diagnostics.py --precision float64 --run-dir "$REF" --out "$OUT/viz_ref" > "$OUT/viz_ref.log" 2>&1
"$PY" research/age_model/exp/west_sink_audit.py --viz-dir "$OUT/viz_ref" --out "$OUT/viz_ref/west_sink_audit.json" > "$OUT/west_sink_ref.log" 2>&1
date -Is > "$OUT/FIT_DONE"
