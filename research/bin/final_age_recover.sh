#!/bin/bash
set -euo pipefail
ARM="$1"
ROOT=/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025
source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
export TWT="$ROOT/code_memoryfix1" PYTHONPATH="$ROOT/code_memoryfix1:$ROOT/code_memoryfix1/src" PYTHONUNBUFFERED=1
cd "$TWT"
export AGE_MODEL_CONFIG="$TWT/config/overlays/map_run_${ARM}.json"
export HOUFIN_MODEL_PRECISION=float64 HOUFIN_Z_FLOAT32=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# The sole source delta casts the same float32 inputs on host before transfer;
# no values, shapes, priors or objective change. Permit this recorded hash delta.
export HOUFIN_HMC_ALLOW_MODEL_DRIFT=1
export HOUFIN_REFINE_MAX_ITER=16000 HOUFIN_REFINE_TAG=finite_restart1
unset HOUFIN_REFINE_FRESH HOUFIN_REFINE_INIT HOUFIN_REFINE_FREEZE
MAP=/work/07980/bwa386/ls6/houfin/processed/model_results/age_map_float32_${ARM}_quick90
if [ "$ARM" = fd_v2_r128 ]; then
 export HOUFIN_HMC_MAP_DIR="$MAP" HOUFIN_HMC_MAP_PRECISION=float32
else
 export HOUFIN_HMC_MAP_DIR="/work/07980/bwa386/ls6/houfin/processed/model_results/refined_float64__age_map_float32_${ARM}_quick90" HOUFIN_HMC_MAP_PRECISION=float64
fi
OUT=/work/07980/bwa386/ls6/houfin/research/age_model/FINAL_DESK_MAP/$ARM/finite_restart1
mkdir -p "$OUT"
cp SOURCE_MANIFEST.sha256 "$OUT/code_manifest.sha256"
python -m src.model.age_refine_map > "$OUT/refine.log" 2>&1
REF=$(grep -Eo 'use HOUFIN_HMC_MAP_DIR=[^ ]+' "$OUT/refine.log" | tail -1 | cut -d= -f2)
test -n "$REF"
export REPORT="$REF/refine_report.json"
python -c 'import json,os; d=json.load(open(os.environ["REPORT"])); assert d["valid"],d; print(d["reason"])'
echo "$REF" > "$OUT/refine_dir.txt"
python scripts/viz/map_diagnostics.py --precision float64 --run-dir "$REF" --out "$OUT/viz_ref" > "$OUT/viz_ref.log" 2>&1
python research/age_model/exp/west_sink_audit.py --viz-dir "$OUT/viz_ref" --out "$OUT/viz_ref/west_sink_audit.json" > "$OUT/west_sink.log" 2>&1
if [ "$ARM" = fd_v2_r128 ]; then
 python scripts/viz/map_diagnostics.py --precision float64 --run-dir "$MAP" --out "$OUT/viz_map" > "$OUT/viz_map.log" 2>&1
fi
date -Is > "$ROOT/${ARM}_RECOVERED"
