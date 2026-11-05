#!/bin/bash
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
SNAP=/scratch/07980/bwa386/age_wt/0329ze_81e_E081
cd "$SNAP"
export PYTHONPATH="$SNAP:$SNAP/src"
export AGE_MODEL_CONFIG="$SNAP/config/overlays/map_run81e_desk_ctl1902.json"
export HOUFIN_MODEL_PRECISION=float64 HOUFIN_HMC_MAP_PRECISION=float32 HOUFIN_Z_FLOAT32=1
export HOUFIN_HMC_MAP_DIR=/work/07980/bwa386/ls6/houfin/processed/model_results/age_map_float32_run_81e_quick90
export HOUFIN_REFINE_MAX_ITER=16000 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
unset HOUFIN_REFINE_FRESH HOUFIN_REFINE_INIT HOUFIN_REFINE_FREEZE
OUT=/work/07980/bwa386/ls6/houfin/research/age_model/E081/81e
REF=/work/07980/bwa386/ls6/houfin/processed/model_results/refined_float64__age_map_float32_run_81e_quick90
export REF_REPORT="$REF/refine_report.json"
python -m src.model.age_refine_map > "$OUT/refine_continue_16000.log" 2>&1
python -c 'import json,os; d=json.load(open(os.environ["REF_REPORT"])); print(d["status"], d["iterations"], d["reason"]); assert d["valid"], "refinement remains invalid"' \
  2>&1
python scripts/viz/map_diagnostics.py --precision float64 --run-dir "$REF" --out "$OUT/viz_ref_continue16000" > "$OUT/viz_ref_continue16000.log" 2>&1
python research/age_model/exp/west_sink_audit.py --viz-dir "$OUT/viz_ref_continue16000" --out "$OUT/viz_ref_continue16000/west_sink_audit.json" > "$OUT/west_sink_continue16000.log" 2>&1
