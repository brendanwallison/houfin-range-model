"""Shared loader for the age-structured model's numpyro inputs.

Consolidates the ``load_data`` / ``load_data_to_gpu`` copies that previously
lived in ``age_run_map``, ``age_run_svi``, ``age_run_hmc`` and
``age_resume_hmc``. The heavy ``Z_gathered`` / ``Z_disp_gathered`` arrays are
kept in host RAM (streamed from memmap); everything else is cast and placed on
the compute device.
"""
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

# Arrays intentionally kept in host RAM rather than resident on the device.
STREAMING_KEYS = {"Z_gathered", "Z_disp_gathered", "st_basis"}


def load_data(input_dir, target_device=None, precision="float32", verbose=True):
    """Load model inputs, casting to ``precision`` and placing device arrays.

    If ``target_device`` is given, non-streaming arrays are materialized on that
    device; otherwise they land on JAX's default device.
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

    meta["Z_gathered"] = np.array(z_mem).astype(f_type_cpu)
    meta["Z_disp_gathered"] = np.array(z_disp_mem).astype(f_type_cpu)

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
            if key in STREAMING_KEYS:
                meta[key] = value.astype(f_type_cpu)
                if verbose:
                    print(f"  [CPU] {key}: {meta[key].nbytes / 1e9:.2f} GB")
            else:
                if target_device is not None:
                    with jax.default_device(target_device):
                        meta[key] = _to_device(value)
                else:
                    meta[key] = _to_device(value)
                if verbose:
                    print(f"  [GPU] {key}: {meta[key].nbytes / 1e6:.1f} MB")

    if precision == "float32" and meta.get("pseudo_zero", 0) < 1e-7:
        meta["pseudo_zero"] = 1e-7

    if verbose:
        vram_gb = sum(x.nbytes for x in meta.values() if isinstance(x, jnp.ndarray)) / 1e9
        print(f"--- Total Resident VRAM: {vram_gb:.2f} GB ---")

    return meta


def load_data_to_gpu(input_dir, precision="float32"):
    """Backward-compatible alias used by the visualization/analysis scripts."""
    return load_data(input_dir, target_device=None, precision=precision, verbose=True)
