#!/bin/bash
# Numerical recovery only: same finite point/model, fresh L-BFGS history, isolated output.
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
SNAP=/scratch/07980/bwa386/age_wt/0329za_81a_E081
cd "$SNAP"
export PYTHONPATH="$SNAP:$SNAP/src" PYTHONUNBUFFERED=1
export AGE_MODEL_CONFIG="$SNAP/config/overlays/map_run81a_desk_cbc1902.json"
export HOUFIN_MODEL_PRECISION=float64 HOUFIN_HMC_MAP_PRECISION=float64 HOUFIN_Z_FLOAT32=1
export HOUFIN_HMC_MAP_DIR=/work/07980/bwa386/ls6/houfin/processed/model_results/refined_float64__age_map_float32_run_81a_quick90
export HOUFIN_REFINE_TAG=finite_restart1 HOUFIN_REFINE_MAX_ITER=8000 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
unset HOUFIN_REFINE_FRESH HOUFIN_REFINE_INIT HOUFIN_REFINE_FREEZE
OUT=/work/07980/bwa386/ls6/houfin/research/age_model/E081/81a/finite_restart1
mkdir -p "$OUT"
python -m src.model.age_refine_map > "$OUT/refine.log" 2>&1
REF=$(grep -Eo 'use HOUFIN_HMC_MAP_DIR=[^ ]+' "$OUT/refine.log" | tail -1 | cut -d= -f2)
test -n "$REF"
export REF_REPORT="$REF/refine_report.json"
python -c 'import json,os; d=json.load(open(os.environ["REF_REPORT"])); assert abs(d["U_start"]-144094.7171325632)<1, d; assert d["valid"], d; print(d["reason"])'
python scripts/viz/map_diagnostics.py --precision float64 --run-dir "$REF" --out "$OUT/viz_ref" > "$OUT/viz_ref.log" 2>&1
python research/age_model/exp/west_sink_audit.py --viz-dir "$OUT/viz_ref" --out "$OUT/viz_ref/west_sink_audit.json" > "$OUT/west_sink.log" 2>&1
echo "$REF" > "$OUT/refine_dir.txt"
