#!/bin/bash
# Resume from the completed MAP with data-slice barriers; preserve failed outputs.
set -euo pipefail
ARM="$1"
case "$ARM" in s64|w64|r128) ;; *) exit 2;; esac
if [ "${FINAL_SHARED_LOCAL:-0}" = 1 ]; then
 ROOT=/home/breallis/houfin/work/houfin/research/FINAL_SHARED_ROUND1_20261025
 CODE="$ROOT/code_projectionfix1"
 PY=/home/breallis/houfin/venv/bin/python
 export PYTHONPATH="/home/breallis/houfin/work/houfin/research/FINAL_LOCAL_JAX_20261025/runtime:$CODE:$CODE/src"
 export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/work/07980/bwa386/ls6/houfin/processed
 export XLA_PYTHON_CLIENT_MEM_FRACTION=0.72 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
else
 ROOT=/work/07980/bwa386/ls6/houfin/research/final_shared_round1_20261025
 CODE="$ROOT/code_projectionfix1"
 source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
 PY=python
 export PYTHONPATH="$CODE:$CODE/src" XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
fi
cd "$CODE"
OUT="$ROOT/projectionfix1/$ARM"; mkdir -p "$OUT"
"$PY" - <<'PY' > "$OUT/source_verification.log"
from pathlib import Path
import hashlib,json
manifest=json.loads(Path('PROJECTIONFIX_MANIFEST.json').read_text())
for name,sha in manifest.items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
print('Verified',len(manifest),'source/config files')
PY
export AGE_MODEL_CONFIG="$CODE/config/overlays/map_run_shared_clean_${ARM}.json"
export HOUFIN_MODEL_PRECISION=float64 HOUFIN_Z_FLOAT32=1 HOUFIN_MAP_PROFILE=quick90 HOUFIN_REFINE_MAX_ITER=16000 PYTHONUNBUFFERED=1
export HOUFIN_HMC_MAP_PRECISION=float32
export HOUFIN_HMC_MAP_DIR="$HOUFIN_PROCESSED/model_results/age_map_float32_shared_clean_${ARM}_quick90"
# This snapshot differs only by two data-only optimization barriers in age_fields.
export HOUFIN_HMC_ALLOW_MODEL_DRIFT=1 HOUFIN_REFINE_TAG=projectionfix1
unset HOUFIN_REFINE_SCALE_FILE HOUFIN_REFINE_FRESH HOUFIN_REFINE_INIT HOUFIN_REFINE_FREEZE
"$PY" -m src.model.age_refine_map > "$OUT/refine.log" 2>&1
REF="$HOUFIN_PROCESSED/model_results/refined_float64__age_map_float32_shared_clean_${ARM}_quick90__projectionfix1"
export REPORT="$REF/refine_report.json"
"$PY" -c 'import json,os; d=json.load(open(os.environ["REPORT"])); assert d["valid"],d'
date -Is > "$OUT/REFINEMENT_DONE"
"$PY" scripts/viz/map_diagnostics.py --precision float64 --run-dir "$REF" --out "$OUT/viz_ref" > "$OUT/viz_ref.log" 2>&1
"$PY" research/age_model/exp/west_sink_audit.py --viz-dir "$OUT/viz_ref" --out "$OUT/viz_ref/west_sink_audit.json" > "$OUT/west_sink.log" 2>&1
date -Is > "$OUT/DIAGNOSTICS_DONE"
