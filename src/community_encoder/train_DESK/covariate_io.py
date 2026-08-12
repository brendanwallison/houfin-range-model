"""Shared N-stream covariate IO for the DESK encoder (trainer, validate, cube).

Loads per-year ``state_{year}.npz`` (one array per stream) using the ``state_schema.json``
sidecar written by ``streams.run_states`` for the channel layout, applies optional
per-stream transforms (from the ``states`` config), and provides the split into per-stream
tensors that ``MultiStreamAutoencoder.forward(*streams)`` expects.

Per-channel normalization stats are fit ONCE, on the supervised training pixels only
(holdout and buffer cells excluded, so the evaluation distribution cannot leak into the
inputs), then reused verbatim for every per-year state at cube time -- one normalization
for the whole pipeline. That reuse is why ``assert_schema_compatible`` matters: mu/sd are
POSITIONAL, so a channel reorder that preserves the total width would misnormalize every
channel with no other symptom.
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


def assert_schema_compatible(saved, live, context=""):
    """Verify an on-disk state layout matches the one a model was fitted against.

    ``mu``/``sd`` and the encoder's input widths are positional, so a states dir
    rebuilt with different channels is not interchangeable with a checkpoint even
    when the total width happens to match. Compares stream names, widths, and —
    when both sides recorded them — the variable name lists, so a *reordering*
    that preserves ``dim`` is caught too. That case is otherwise completely
    silent: every array shape agrees while each channel gets another channel's
    normalization.
    """
    where = f" ({context})" if context else ""
    s_names = [s["name"] for s in saved["streams"]]
    l_names = [s["name"] for s in live["streams"]]
    if s_names != l_names:
        raise SystemExit(f"state schema mismatch{where}: streams {s_names} (model) "
                         f"vs {l_names} (on disk)")
    for ss, ls in zip(saved["streams"], live["streams"]):
        if int(ss["dim"]) != int(ls["dim"]):
            raise SystemExit(
                f"state schema mismatch{where}: stream {ss['name']!r} is "
                f"{ss['dim']} ch in the model but {ls['dim']} ch on disk. The "
                f"saved mu/sd and encoder input width are positional — rebuild "
                f"states and retrain rather than mixing them.")
        sv, lv = list(ss.get("variables") or []), list(ls.get("variables") or [])
        if sv and lv and sv != lv:
            diff = next((i for i, (a, b) in enumerate(zip(sv, lv)) if a != b), None)
            raise SystemExit(
                f"state schema mismatch{where}: stream {ss['name']!r} has the same "
                f"width but a different channel ORDER (first difference at index "
                f"{diff}: model {sv[diff]!r} vs disk {lv[diff]!r}). Normalization "
                f"is positional, so this would silently apply the wrong stats.")


def load_state_stack(year, states_dir, schema):
    """Load one year's state as ``(H, W, C)`` (streams concatenated, transforms applied)."""
    z = np.load(os.path.join(states_dir, f"state_{year}.npz"))
    bands = []
    for s in schema["streams"]:
        arr = z[s["name"]]
        # Cheap explicit guard: without it a width mismatch only surfaces later as
        # a broadcast error inside apply_norm, which reads as a normalization bug
        # rather than a stale-states one.
        if arr.shape[-1] != int(s["dim"]):
            raise SystemExit(
                f"state_{year}.npz stream {s['name']!r} has {arr.shape[-1]} channels "
                f"but the schema says {s['dim']}. The states dir and the schema are "
                f"out of sync — rebuild states (src.data.combine.build_states).")
        bands.append(_transform(arr.astype("float32"), s))
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


def norm_grid(cov_stack, mu, sd):
    """Standardize a ``(H, W, C)`` covariate grid on its finite cells.

    Grid-native forward (spatial conv) needs the *whole* grid, not a gather of
    valid pixels, so invalid (any-NaN) cells are zero-filled -- 0 == the post-norm
    channel mean, a neutral value the masked conv also excludes. Returns
    ``(covn (H,W,C) float32, mask (H,W) bool)`` where ``mask`` marks finite cells.
    """
    mask = ~np.isnan(cov_stack).any(axis=-1)
    covn = np.zeros(cov_stack.shape, dtype="float32")
    covn[mask] = apply_norm(cov_stack[mask].astype("float32"), mu, sd)
    return covn, mask
