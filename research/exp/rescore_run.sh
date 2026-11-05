#!/bin/bash
# Rescore one GRID54 run's refined point with the SDM-baseline scorer, using the run's OWN frozen code and overlay.
# Usage: rescore_run.sh STEP [REF_STEP]     e.g. rescore_run.sh step10_pn05 step3_obsroute
# CPU only, 2 threads, cores 0-3, nice 19 (a GPU fit may be running). Output: $ROOT/<STEP>/{design.npz,pred.npz,rescore.json}
set -euo pipefail
STEP="$1"; REFSTEP="${2:-step3_obsroute}"
HERE=/mnt/e/Datasets/houfin/grid54/baselines/b0          # staged copies of research/exp/sdm_*.py
ROOT=/mnt/e/Datasets/houfin/grid54/baselines/rescore
R=/mnt/e/Datasets/houfin/grid54/runs/$STEP
CODE=$R/code
if [ -f "$CODE/config/overlays/grid54_${STEP}.json" ]; then   # seed copies carry their source's overlay too
  OVL="grid54_${STEP}.json"
else
  OVL=$(ls "$CODE/config/overlays/" | grep '^grid54_' | grep -v '^grid54_data' | grep -v '_data\.json$')
  [ "$(echo "$OVL" | wc -l)" = 1 ] || { echo "ambiguous overlay for $STEP: $OVL"; exit 2; }
fi
export PY=/home/breallis/houfin/venv/bin/python
export PYTHONPATH="/home/breallis/houfin/work/houfin/research/FINAL_LOCAL_JAX_20261025/runtime:$CODE:$CODE/src"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/work/07980/bwa386/ls6/houfin/processed
export AGE_MODEL_CONFIG="$CODE/config/overlays/$OVL" DATA_CONFIG="$CODE/config/overlays/grid54_data.json"
export HOUFIN_Z_FLOAT32=1 HOUFIN_MODEL_PRECISION=float64 PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
REF=$HOUFIN_PROCESSED/model_results/refined_float64__age_map_float32_grid54_${STEP}_quick90
OUT=$ROOT/$STEP; mkdir -p "$OUT"
echo "[$STEP] overlay $OVL; refined point $REF"
cd "$CODE"
if [ ! -f "$OUT/pred.npz" ]; then
  nice -n 19 taskset -c 0-3 "$PY" "$HERE/sdm_b0_design.py" --run-dir "$REF" --out "$OUT" --no-z --pred-name pred.npz \
    > "$OUT/export.log" 2>&1 || { echo "[$STEP] export FAILED"; tail -20 "$OUT/export.log"; exit 1; }
fi
nice -n 19 taskset -c 0-3 "$PY" "$HERE/sdm_rescore.py" --root "$ROOT" --step "$STEP" --ref "$REFSTEP"
