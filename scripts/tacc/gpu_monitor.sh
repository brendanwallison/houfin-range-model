#!/bin/bash
# Source from GPU SLURM jobs. Records accelerator utilization plus host-memory
# pressure so VRAM exhaustion/unified-memory spill is visible after the run.

gpu_preflight () {
    local label="${1:-GPU job}"
    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "ERROR: $label has no nvidia-smi; was it submitted to a GPU node?"
        return 1
    }
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader
    python -c "import jax; g=jax.devices('gpu'); assert g, 'no JAX GPU'; print('[gpu] jax', jax.__version__, 'devices', g)" || {
        echo "ERROR: $label would fall back to CPU. Install a CUDA JAX build."
        return 1
    }
    if [ "${XLA_PYTHON_CLIENT_MEM_FRACTION:-0}" != "0" ]; then
        echo "[gpu] XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"
    fi
    echo "[gpu] XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-<default>}"
}

start_gpu_monitor () {
    GPU_MONITOR_LOG="${1:?monitor log path required}"
    local interval="${2:-30}"
    # Fail LOUDLY rather than writing a header-only CSV for the whole job. The one-shot
    # nvidia-smi check lives in gpu_preflight (called by 20_encoder, 25_model_prep,
    # 30_model_map and 31_model_viz), but that one is JAX-based, so it is the wrong
    # preflight for the torch encoder -- hence this separate check.
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[gpu] WARNING: no nvidia-smi on $(hostname) -- telemetry DISABLED (not a GPU node?)" >&2
        GPU_MONITOR_PID=""
        export GPU_MONITOR_LOG GPU_MONITOR_PID
        return 0
    fi
    # `user_all_rss_kib` is the sum over EVERY process this user has on the node, not the
    # trainer's -- named honestly so nobody reads it as the job's footprint. The trainer logs its
    # own RSS and torch's peak VRAM per epoch (desk_training), which is the figure to trust.
    {
        echo "timestamp,gpu_index,util_gpu_pct,util_mem_pct,vram_used_mib,vram_total_mib,temp_c,power_w,user_all_rss_kib,host_mem_available_kib,host_swap_free_kib"
        while true; do
            local stamp user_rss host_avail swap_free
            stamp="$(date --iso-8601=seconds)"
            user_rss="$(ps -u "$USER" -o rss= | awk '{s+=$1} END {print s+0}')"
            host_avail="$(awk '/MemAvailable:/{print $2}' /proc/meminfo)"
            swap_free="$(awk '/SwapFree:/{print $2}' /proc/meminfo)"
            nvidia-smi \
                --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
                --format=csv,noheader,nounits |
                while IFS= read -r row; do
                    echo "$stamp,$row,$user_rss,$host_avail,$swap_free"
                done
            sleep "$interval"
        done
    # stdout -> the CSV, stderr -> the JOB LOG. Previously `2>&1` folded stderr into the CSV, so
    # a mid-run nvidia-smi failure was buried among the data rows instead of being visible.
    } > "$GPU_MONITOR_LOG" &
    GPU_MONITOR_PID=$!
    export GPU_MONITOR_LOG GPU_MONITOR_PID
    echo "[gpu] telemetry every ${interval}s -> $GPU_MONITOR_LOG (pid $GPU_MONITOR_PID)"
}

stop_gpu_monitor () {
    if [ -n "${GPU_MONITOR_PID:-}" ]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
}
