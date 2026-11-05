#!/bin/bash
# One isolated conditioning pilot, unchanged density and raw-gradient convergence gates.
set -euo pipefail
ROOT=/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025
source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
export TWT="$ROOT/code_precondition1" PYTHONPATH="$ROOT/code_precondition1:$ROOT/code_precondition1/src" PYTHONUNBUFFERED=1
cd "$TWT"
export AGE_MODEL_CONFIG="$TWT/config/overlays/map_run_fd_control64.json"
export HOUFIN_MODEL_PRECISION=float64 HOUFIN_HMC_MAP_PRECISION=float64 HOUFIN_Z_FLOAT32=1 HOUFIN_HMC_ALLOW_MODEL_DRIFT=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HOUFIN_HMC_MAP_DIR=/work/07980/bwa386/ls6/houfin/processed/model_results/refined_float64__age_map_float32_fd_control64_quick90__finite_restart1
export HOUFIN_REFINE_SCALE_FILE="$ROOT/gradient_audit/fd_control64_scale.json"
export HOUFIN_REFINE_TAG=blockscale1 HOUFIN_REFINE_MAX_ITER=2000
unset HOUFIN_REFINE_FRESH HOUFIN_REFINE_INIT HOUFIN_REFINE_FREEZE
test -f "$HOUFIN_REFINE_SCALE_FILE"
OUT=/work/07980/bwa386/ls6/houfin/research/age_model/FINAL_DESK_MAP/fd_control64/blockscale1
mkdir -p "$OUT"
cp SOURCE_MANIFEST.sha256 "$OUT/code_manifest.sha256"
python -m src.model.age_refine_map > "$OUT/refine.log" 2>&1
REF=$(grep -Eo 'use HOUFIN_HMC_MAP_DIR=[^ ]+' "$OUT/refine.log" | tail -1 | cut -d= -f2)
test -n "$REF"
export REPORT="$REF/refine_report.json"
python -c 'import json,os; d=json.load(open(os.environ["REPORT"])); assert d["valid"], d; print(d["reason"])'
echo "$REF" > "$OUT/refine_dir.txt"
python scripts/viz/map_diagnostics.py --precision float64 --run-dir "$REF" --out "$OUT/viz_ref" > "$OUT/viz_ref.log" 2>&1
python research/age_model/exp/west_sink_audit.py --viz-dir "$OUT/viz_ref" --out "$OUT/viz_ref/west_sink_audit.json" > "$OUT/west_sink.log" 2>&1
