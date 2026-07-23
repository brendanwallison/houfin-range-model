"""Shared loader for the age-structured model's numpyro inputs.

Consolidates the ``load_data`` / ``load_data_to_gpu`` copies that previously
lived in ``age_run_map``, ``age_run_svi``, ``age_run_hmc`` and
``age_resume_hmc``. Heavy inputs default to explicit device residency. A
JIT-compiled differentiable ``lax.scan`` cannot truly stream NumPy slices from
host memory; calling these arrays "streaming" previously obscured XLA's copies.
Set ``HOUFIN_MODEL_INPUT_RESIDENCY=host`` only as an experimental mode.
"""
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

LARGE_INPUT_KEYS = {"Z_gathered", "Z_disp_gathered", "st_basis"}


def load_data(input_dir, target_device=None, precision="float32", verbose=True):
    """Load model inputs, casting to ``precision`` and placing device arrays.

    If ``target_device`` is given, arrays are materialized on it unless explicit
    experimental host residency is requested.
    """
    meta_path = os.path.join(input_dir, "metadata.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    f_type_cpu = np.float32 if precision == "float32" else np.float64
    f_type_target = jnp.float32 if precision == "float32" else jnp.float64
    i_type_target = jnp.int32 if precision == "float32" else jnp.int64

    z_shape = (meta["time"], meta["N_land"], meta["M"])
    z_mem = np.memmap(
        os.path.join(input_dir, meta["z_gathered_path"]),
        dtype="float32", mode="r", shape=z_shape,
    )

    z_disp_shape = (meta["time"], meta["N_land"], meta["K"], meta["M"])
    z_disp_mem = np.memmap(
        os.path.join(input_dir, meta["z_disp_gathered_path"]),
        dtype="float32", mode="r", shape=z_disp_shape,
    )

    meta["Z_gathered"] = np.asarray(z_mem, dtype=f_type_cpu)
    meta["Z_disp_gathered"] = np.asarray(z_disp_mem, dtype=f_type_cpu)
    residency = os.environ.get("HOUFIN_MODEL_INPUT_RESIDENCY", "device").lower()
    if residency not in {"device", "host"}:
        raise ValueError("HOUFIN_MODEL_INPUT_RESIDENCY must be 'device' or 'host'")

    if verbose:
        print(f"Iterating through metadata and casting to {precision} on device: {target_device}...")

    def _to_device(value):
        if np.issubdtype(value.dtype, np.floating):
            return jnp.array(value).astype(f_type_target)
        elif np.issubdtype(value.dtype, np.integer):
            return jnp.array(value).astype(i_type_target)
        return jnp.array(value)

    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            if key in LARGE_INPUT_KEYS and residency == "host":
                meta[key] = value.astype(f_type_cpu)
                if verbose:
                    print(f"  [HOST, experimental] {key}: {meta[key].nbytes / 1e9:.2f} GB")
            else:
                if target_device is not None:
                    meta[key] = jax.device_put(_to_device(value), target_device)
                else:
                    meta[key] = _to_device(value)
                if verbose:
                    print(f"  [DEVICE] {key}: {meta[key].nbytes / 1e6:.1f} MB")

    if verbose:
        device_bytes = sum(
            x.nbytes for x in meta.values()
            if hasattr(x, "device") and not isinstance(x, np.ndarray)
        )
        host_bytes = sum(x.nbytes for x in meta.values() if isinstance(x, np.ndarray))
        print(f"--- Explicit input residency: device={device_bytes / 1e9:.2f} GB, "
              f"host={host_bytes / 1e9:.2f} GB (mode={residency}) ---")

    return meta


def load_data_to_gpu(input_dir, precision="float32"):
    """Backward-compatible alias used by the visualization/analysis scripts."""
    return load_data(input_dir, target_device=None, precision=precision, verbose=True)
