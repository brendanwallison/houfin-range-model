"""Runtime diagnostics for GPU-backed JAX model jobs."""
from __future__ import annotations

import os
import resource

import jax


def require_gpu(context: str = "model") -> object:
    """Return the active GPU or fail before an expensive job falls back to CPU."""
    gpus = jax.devices("gpu")
    if not gpus:
        raise RuntimeError(
            f"{context} requires a JAX GPU, but jax.devices('gpu') is empty. "
            "Install/use a CUDA JAX build and submit to a GPU partition."
        )
    device = gpus[0]
    print(
        f"[gpu] {context}: backend={jax.default_backend()} device={device}; "
        f"XLA_PYTHON_CLIENT_PREALLOCATE={os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', '<default>')} "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', '<default>')}"
    )
    return device


def memory_snapshot(label: str, device=None) -> dict:
    """Print JAX allocator counters and process RSS; return the numeric snapshot."""
    device = device or jax.devices()[0]
    stats = device.memory_stats() or {}
    out = {
        "bytes_in_use": int(stats.get("bytes_in_use", -1)),
        "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", -1)),
        "bytes_limit": int(stats.get("bytes_limit", -1)),
        "largest_free_block_bytes": int(stats.get("largest_free_block_bytes", -1)),
        # Linux reports KiB; macOS reports bytes. GPU production jobs are Linux.
        "process_maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    def gb(n):
        return "n/a" if n < 0 else f"{n / 2**30:.2f} GiB"
    print(
        f"[memory:{label}] device_in_use={gb(out['bytes_in_use'])} "
        f"device_peak={gb(out['peak_bytes_in_use'])} "
        f"device_limit={gb(out['bytes_limit'])} "
        f"largest_free={gb(out['largest_free_block_bytes'])} "
        f"process_maxrss={out['process_maxrss_kib'] / 2**20:.2f} GiB"
    )
    return out
