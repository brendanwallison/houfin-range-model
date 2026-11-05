#!/bin/bash
# One 54 km ladder fit: cold MAP (quick90, float32) -> diagnostics -> float64 refine -> diagnostics.
# Code and overlay are frozen per step under $R/code (CODE_MANIFEST.sha256). Usage: grid54_fit.sh STEP OVERLAY_NAME
set -euo pipefail
STEP="$1"; OVL="$2"
if [ -d /mnt/e/Datasets/houfin/grid54 ]; then   # local 3090 (WSL)
 R=/mnt/e/Datasets/houfin/grid54/runs/$STEP
 CODE="$R/code"
 PY=/home/breallis/houfin/venv/bin/python
 export PYTHONPATH="/home/breallis/houfin/work/houfin/research/FINAL_LOCAL_JAX_20261025/runtime:$CODE:$CODE/src"
 export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/work/07980/bwa386/ls6/houfin/processed
 export XLA_PYTHON_CLIENT_MEM_FRACTION=0.72 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
 # ptxas must match jaxlib's CUDA 12.8; the system 12.3 ptxas (first on PATH via .bashrc) segfaults intermittently
 export PATH=/home/breallis/houfin/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin:$PATH
 # XLA command buffers (CUDA graphs) segfault libcuda under WSL in the float64 refine of process-noise models
 # (deterministic, after L-BFGS iteration 1; off = runs clean, 2026-10-30)
 export XLA_FLAGS="--xla_gpu_enable_command_buffer= ${XLA_FLAGS:-}"
else                                             # TACC (Lonestar6 A100); inputs under $HOUFIN_PROCESSED/grid54
 R=/work/07980/bwa386/ls6/houfin/research/grid54/runs/$STEP
 CODE="$R/code"
 source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
 PY=python
 export PYTHONPATH="$CODE:$CODE/src" XLA_PYTHON_CLIENT_MEM_FRACTION=${GRID54_MEM_FRACTION:-0.95}
fi
export AGE_MODEL_CONFIG="$CODE/config/overlays/$OVL"
export DATA_CONFIG="$CODE/config/overlays/grid54_data.json"
export HOUFIN_Z_FLOAT32=1 HOUFIN_MAP_PROFILE=quick90 HOUFIN_REFINE_MAX_ITER=16000 PYTHONUNBUFFERED=1
unset HOUFIN_REFINE_SCALE_FILE HOUFIN_REFINE_TAG HOUFIN_REFINE_FRESH HOUFIN_REFINE_INIT HOUFIN_REFINE_FREEZE HOUFIN_MAP_REUSE HOUFIN_HMC_ALLOW_MODEL_DRIFT
cd "$CODE"
OUT="$R/out"; mkdir -p "$OUT"
(cd "$CODE" && sha256sum --quiet -c ../CODE_MANIFEST.sha256) > "$OUT/source_verification.log" 2>&1
"$PY" -c 'import json; from src.config_utils import load_age_model_config; print(json.dumps(load_age_model_config(),indent=2))' > "$OUT/resolved_config.json"
RUN=$("$PY" -c 'from src.config_utils import load_age_model_config as l; print(l()["run_names"]["map"].format(precision="float32"))')_$HOUFIN_MAP_PROFILE
MAP="$HOUFIN_PROCESSED/model_results/$RUN"
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
REF="$HOUFIN_PROCESSED/model_results/refined_float64__$RUN"
export REPORT="$REF/refine_report.json"
"$PY" -c 'import json,os; d=json.load(open(os.environ["REPORT"])); assert d["valid"],d'
date -Is > "$OUT/REFINEMENT_DONE"
"$PY" scripts/viz/map_diagnostics.py --precision float64 --run-dir "$REF" --out "$OUT/viz_ref" > "$OUT/viz_ref.log" 2>&1
"$PY" research/age_model/exp/west_sink_audit.py --viz-dir "$OUT/viz_ref" --out "$OUT/viz_ref/west_sink_audit.json" > "$OUT/west_sink_ref.log" 2>&1
# PNG-only copies: Windows Photos pre-loads a folder's neighbours and crashes on the float GeoTIFF in viz_*/
for v in viz_map viz_ref; do mkdir -p "$OUT/${v}_png"; cp "$OUT/$v"/*.png "$OUT/${v}_png/" 2>/dev/null || true; done
date -Is > "$OUT/FIT_DONE"
