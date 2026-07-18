"""Shared N-stream covariate IO for the DESK encoder (trainer, validate, cube).

Replaces the hardcoded 2-stream PRISM/BUI reads. Loads per-year
``state_{year}.npz`` (one array per stream) using the ``state_schema.json`` sidecar
written by ``streams.run_states`` for the channel layout, applies optional
per-stream transforms (from the ``states`` config), and provides the split into
per-stream tensors that ``MultiStreamAutoencoder.forward(*streams)`` expects.

Per-channel normalization stats are computed once (on the labeled snapshot) and
reused verbatim for the history bag and every per-year state at cube time, so the
whole pipeline shares one normalization.
"""
import json
import os

import numpy as np


def load_schema(states_dir):
    """Load ``state_schema.json`` from ``states_dir`` or its parent."""
    for cand in (states_dir, os.path.dirname(os.path.normpath(states_dir))):
        p = os.path.join(cand, "state_schema.json")
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh)
    raise FileNotFoundError(f"state_schema.json not found in/above {states_dir}")


def _transform(arr, spec):
    """Apply a stream's optional transform (``{'type':'pow','p':..}`` | ``log1p``)."""
    t = spec.get("transform")
    if not t:
        return arr
    kind = t.get("type")
    if kind == "pow":
        return np.power(np.clip(arr, 0.0, None), float(t["p"]))
    if kind == "log1p":
        return np.log1p(np.clip(arr, 0.0, None))
    raise ValueError(f"unknown stream transform {t!r}")


def stream_dims(schema):
    """Per-stream channel widths, in schema order (the ``dims`` for the model)."""
    return [int(s["dim"]) for s in schema["streams"]]


def load_state_stack(year, states_dir, schema):
    """Load one year's state as ``(H, W, C)`` (streams concatenated, transforms applied)."""
    z = np.load(os.path.join(states_dir, f"state_{year}.npz"))
    bands = [_transform(z[s["name"]].astype("float32"), s) for s in schema["streams"]]
    return np.concatenate(bands, axis=-1)


def transform_flat(bag, schema):
    """Apply per-stream transforms to a flat ``(N, C)`` bag (e.g. history_vectors)."""
    bag = np.asarray(bag, dtype="float32").copy()
    for s in schema["streams"]:
        sl = slice(int(s["start"]), int(s["end"]))
        bag[:, sl] = _transform(bag[:, sl], s)
    return bag


def fit_norm(cov_flat):
    """Per-channel mean/std over ``(N, C)`` (post-transform). Returns ``(mu, sd)``."""
    mu = cov_flat.mean(0)
    sd = cov_flat.std(0)
    return mu.astype("float32"), sd.astype("float32")


def apply_norm(cov, mu, sd):
    """Standardize ``(..., C)`` with stored stats (broadcast over leading dims)."""
    return (cov - mu) / (sd + 1e-6)


def split_streams(x, schema):
    """Split a ``(..., C)`` array/tensor into per-stream ``(..., dim)`` pieces."""
    return [x[..., int(s["start"]):int(s["end"])] for s in schema["streams"]]
