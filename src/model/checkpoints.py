"""Small, strict helpers for atomic inference artifacts."""
from __future__ import annotations

import os
import pickle


def save_pickle_atomic(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_map_params(map_dir):
    """Load params bundled with a complete v2 MAP checkpoint."""
    path = os.path.join(map_dir, "map_checkpoint.pkl")
    with open(path, "rb") as fh:
        checkpoint = pickle.load(fh)
    if checkpoint.get("format_version") != 2 or not checkpoint.get("fingerprint"):
        raise RuntimeError(
            f"{path} is a legacy/unverifiable checkpoint; rerun MAP fresh under "
            "the current code before using it to initialize posterior inference."
        )
    if int(checkpoint.get("step", -1)) != len(checkpoint.get("losses", ())):
        raise RuntimeError(f"{path} has inconsistent step/loss history")
    if "params" not in checkpoint:
        raise RuntimeError(f"{path} does not contain step-matched MAP parameters")
    return checkpoint["params"], checkpoint


def auto_delta_params_to_latents(params):
    """Convert NumPyro AutoDelta parameter names to model sample-site names."""
    suffix = "_auto_loc"
    converted = {}
    for name, value in params.items():
        if not name.endswith(suffix):
            raise ValueError(f"unexpected AutoDelta MAP parameter name: {name}")
        converted[name[:-len(suffix)]] = value
    return converted
