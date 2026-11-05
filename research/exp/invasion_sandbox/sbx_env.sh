#!/bin/bash
# Run a sandbox command inside one GRID54 run's OWN frozen code and overlay (recipe from
# .research_runtime/range_monitor/rescore_step43_factor_p2_veit_s41.sh). CPU float64, capped threads, nice.
# Usage: sbx_env.sh STEP CMD [ARGS...]      e.g. sbx_env.sh step3_obsroute python harness.py --spec s.json
# SBX_THREADS (default 2) and SBX_CORES (default 4-7) control the CPU footprint.
set -euo pipefail
STEP="$1"; shift
R=/mnt/e/Datasets/houfin/grid54/runs/$STEP
CODE=$R/code
[ -d "$CODE" ] || { echo "no frozen code for $STEP"; exit 2; }
if [ -f "$CODE/config/overlays/grid54_${STEP}.json" ]; then
  OVL="grid54_${STEP}.json"
else
  OVL=$(ls "$CODE/config/overlays/" | grep '^grid54_' | grep -v '^grid54_data' | grep -v '_data\.json$')
  [ "$(echo "$OVL" | wc -l)" = 1 ] || { echo "ambiguous overlay for $STEP: $OVL"; exit 2; }
fi
T=${SBX_THREADS:-2}
SBX=/mnt/c/Dev/houfin-range-model/research/exp/invasion_sandbox
export PYTHONPATH="/home/breallis/houfin/work/houfin/research/FINAL_LOCAL_JAX_20261025/runtime:$CODE:$CODE/src:$CODE/scripts/viz:$SBX"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/work/07980/bwa386/ls6/houfin/processed
export AGE_MODEL_CONFIG="${SBX_OVERLAY:-$CODE/config/overlays/$OVL}" DATA_CONFIG="$CODE/config/overlays/grid54_data.json"
export HOUFIN_Z_FLOAT32=1 HOUFIN_MODEL_PRECISION=${SBX_PRECISION:-float64} PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
# SBX_DEVICE=gpu: the local 3090 (command buffers off -- the known deterministic XLA crash); SBX_MEM_FRACTION lets
# several sandbox jobs share the card. SBX_PRECISION=float32|float64 (harness reads it; default float64).
if [ "${SBX_DEVICE:-cpu}" = gpu ]; then
  export JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_MEM_FRACTION=${SBX_MEM_FRACTION:-.3}
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
else
  export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
  export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$T"
fi
export SBX_PRECISION=${SBX_PRECISION:-float64}
export SBX_STEP="$STEP" SBX_REFINED="$HOUFIN_PROCESSED/model_results/refined_float64__age_map_float32_grid54_${STEP}_quick90"
cd "$CODE"
PYBIN=/home/breallis/houfin/venv/bin/python
if [ "$1" = python ]; then shift; set -- "$PYBIN" "$@"; fi
exec nice -n 15 taskset -c "${SBX_CORES:-4-7}" "$@"
