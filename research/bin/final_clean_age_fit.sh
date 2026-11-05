#!/bin/bash
# Diagnostic two-seed clean candidate; no direct clean-effect comparison to old mean4.
set -euo pipefail
ROOT=/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025
[ -f "$ROOT/clean_v2_m2_HOUSEFINCH_DONE" ] && exit 0
source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
export TWT="$ROOT/code_clean2" PYTHONPATH="$ROOT/code_clean2:$ROOT/code_clean2/src" PYTHONUNBUFFERED=1
cd "$TWT"
export AGE_MODEL_CONFIG="$TWT/config/overlays/map_run_fd_clean_v2_m2_r128.json"
export HOUFIN_Z_FLOAT32=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 HOUFIN_MAP_PROFILE=quick90 HOUFIN_REFINE_MAX_ITER=16000
unset HOUFIN_REFINE_SCALE_FILE HOUFIN_REFINE_TAG HOUFIN_REFINE_FRESH HOUFIN_REFINE_INIT HOUFIN_REFINE_FREEZE HOUFIN_MAP_REUSE HOUFIN_HMC_ALLOW_MODEL_DRIFT
OUT=/work/07980/bwa386/ls6/houfin/research/age_model/FINAL_DESK_MAP/fd_clean_v2_m2_r128
MAP="$HOUFIN_PROCESSED/model_results/age_map_float32_fd_clean_v2_m2_r128_quick90"
mkdir -p "$OUT"
cp SOURCE_MANIFEST.sha256 "$OUT/code_manifest.sha256"
python -c 'import json; from src.config_utils import load_age_model_config; print(json.dumps(load_age_model_config(),indent=2))' > "$OUT/resolved_config.json"
export HOUFIN_MODEL_PRECISION=float32
python -m src.model.age_run_map > "$OUT/map.log" 2>&1
export HOUFIN_MODEL_PRECISION=float64 HOUFIN_HMC_MAP_PRECISION=float32 HOUFIN_HMC_MAP_DIR="$MAP"
python -m src.model.age_refine_map > "$OUT/refine.log" 2>&1
REF=$(grep -Eo 'use HOUFIN_HMC_MAP_DIR=[^ ]+' "$OUT/refine.log" | tail -1 | cut -d= -f2)
test -n "$REF"
export REPORT="$REF/refine_report.json"
python -c 'import json,os; d=json.load(open(os.environ["REPORT"])); assert d["valid"],d'
echo "$REF" > "$OUT/refine_dir.txt"
for stage in map ref; do
 RUN="$MAP"; [ "$stage" = ref ] && RUN="$REF"
 python scripts/viz/map_diagnostics.py --precision float64 --run-dir "$RUN" --out "$OUT/viz_$stage" > "$OUT/viz_${stage}.log" 2>&1
 python research/age_model/exp/west_sink_audit.py --viz-dir "$OUT/viz_$stage" --out "$OUT/viz_$stage/west_sink_audit.json" > "$OUT/west_sink_${stage}.log" 2>&1
done
date -Is > "$ROOT/clean_v2_m2_HOUSEFINCH_DONE"
