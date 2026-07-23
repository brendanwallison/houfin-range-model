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
    {
        echo "timestamp,gpu_index,util_gpu_pct,util_mem_pct,vram_used_mib,vram_total_mib,temp_c,power_w,user_process_rss_kib,host_mem_available_kib,host_swap_free_kib"
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
    } > "$GPU_MONITOR_LOG" 2>&1 &
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
