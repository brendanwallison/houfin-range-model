#!/bin/bash
# Resume the transferred control task, preserving its optimizer fingerprint/gates.
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_LOCAL_JAX_20261025
P=/work/07980/bwa386/ls6/houfin/processed
test -f "$ROOT/BASELINE_AUDIT_DONE"
test -f "$ROOT/CONTROL_CONTINUATION_STAGED"
export PYTHONPATH="$ROOT/runtime:$ROOT/code_precondition1:$ROOT/code_precondition1/src"
export AGE_MODEL_CONFIG="$ROOT/code_precondition1/config/overlays/map_run_fd_control64.json"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED="$P"
export HOUFIN_MODEL_PRECISION=float64 HOUFIN_HMC_MAP_PRECISION=float64 HOUFIN_Z_FLOAT32=1
export HOUFIN_HMC_ALLOW_MODEL_DRIFT=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.72
export HOUFIN_HMC_MAP_DIR="$P/model_results/refined_float64__age_map_float32_fd_control64_quick90__finite_restart1"
export HOUFIN_REFINE_SCALE_FILE="$ROOT/tacc_control_baseline.json"
export HOUFIN_REFINE_TAG=blockscale1 HOUFIN_REFINE_MAX_ITER=8000
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
unset HOUFIN_REFINE_FRESH HOUFIN_REFINE_INIT HOUFIN_REFINE_FREEZE
cd "$ROOT/code_precondition1"
sha256sum -c SOURCE_MANIFEST.sha256 > "$ROOT/continuation_source_verification.log"
/home/breallis/houfin/venv/bin/python -m src.model.age_refine_map > "$ROOT/control_continue8000.log" 2>&1
REF="$P/model_results/refined_float64__age_map_float32_fd_control64_quick90__finite_restart1__blockscale1"
export REPORT="$REF/refine_report.json"
/home/breallis/houfin/venv/bin/python -c 'import json,os; d=json.load(open(os.environ["REPORT"])); print(d); assert d["valid"],d'
/home/breallis/houfin/venv/bin/python scripts/viz/map_diagnostics.py --precision float64 --run-dir "$REF" --out "$ROOT/viz_control_ref" > "$ROOT/viz_control_ref.log" 2>&1
/home/breallis/houfin/venv/bin/python research/age_model/exp/west_sink_audit.py --viz-dir "$ROOT/viz_control_ref" --out "$ROOT/viz_control_ref/west_sink_audit.json" > "$ROOT/west_sink_control.log" 2>&1
date -Is > "$ROOT/CONTROL_CONTINUATION_DONE"
