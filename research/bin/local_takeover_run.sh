#!/usr/bin/env bash
# Execute against the reconciled worktree only, never fall back to a frozen tree.
set -euo pipefail
ROOT=${TAKEOVER_CODE_ROOT:?Set TAKEOVER_CODE_ROOT to the isolated source worktree}
ROOT=$(realpath "$ROOT")
test -f "$ROOT/src/model/density_regulation.py"
RUNTIME=${TAKEOVER_JAX_RUNTIME:-/home/breallis/houfin/work/houfin/research/FINAL_LOCAL_JAX_20261025/runtime}
export PYTHONPATH="$RUNTIME:$ROOT:$ROOT/src:$ROOT/scripts/viz:$ROOT/research/exp/invasion_sandbox"
export HOUFIN_DATA=${HOUFIN_DATA:-/home/breallis/houfin/scratch/houfin/data}
export HOUFIN_PROCESSED=${HOUFIN_PROCESSED:-/work/07980/bwa386/ls6/houfin/processed}
export AGE_MODEL_CONFIG=${AGE_MODEL_CONFIG:-$ROOT/config/overlays/grid54_step43_factor_p2_veit_s41.json}
export DATA_CONFIG=${DATA_CONFIG:-$ROOT/config/overlays/grid54_data.json}
export HOUFIN_Z_FLOAT32=${HOUFIN_Z_FLOAT32:-1} HOUFIN_MODEL_PRECISION=${SBX_PRECISION:-float64}
export SBX_PRECISION=${SBX_PRECISION:-float64} PYTHONUNBUFFERED=1
export SBX_STEP=step43_factor_p2_veit_s41
export SBX_REFINED=${SBX_REFINED:-$HOUFIN_PROCESSED/model_results/refined_float64__age_map_float32_grid54_step43_factor_p2_veit_s41_quick90}
export OMP_NUM_THREADS=${SBX_THREADS:-8} OPENBLAS_NUM_THREADS=${SBX_THREADS:-8} MKL_NUM_THREADS=${SBX_THREADS:-8}
# Small transient files use RAM: drvfs breaks pytest's unlink/truncate capture,
# and the WSL root filesystem has little free space. Persistent caches stay on C:.
export TMPDIR=${TAKEOVER_TMPDIR:-/dev/shm/houfin-takeover-$UID}
export XDG_CACHE_HOME=${TAKEOVER_CACHE:-/mnt/c/Dev/houfin-range-model/.research_runtime/takeover/cache}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ "${SBX_DEVICE:-cpu}" = gpu ]; then
    export JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_command_buffer=
    export XLA_PYTHON_CLIENT_MEM_FRACTION=${SBX_MEM_FRACTION:-.7}
else
    export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$OMP_NUM_THREADS"
fi
cd "$ROOT"
exec nice -n 10 taskset -c "${SBX_CORES:-0-15}" /home/breallis/houfin/venv/bin/python "$@"
