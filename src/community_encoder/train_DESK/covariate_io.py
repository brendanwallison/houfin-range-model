"""Shared N-stream covariate IO for the DESK encoder (trainer, validate, cube).

Loads per-year ``state_{year}.npz`` (one array per stream) using the ``state_schema.json``
sidecar written by ``streams.run_states`` for the channel layout, applies each stream's
optional transform, and provides the split into per-stream tensors that
``MultiStreamAutoencoder.forward(*streams)`` expects.

The schema sidecar is the whole interface: nothing here reads ``states.streams`` from the
config, so a stream property only reaches this module if ``streams.schema_entry`` copies it
into the schema. States are written in RAW units and transformed on load (both here and in
``transform_flat`` for ``history_vectors.npy``), so the transform is applied exactly once,
before mu/sd are fit.

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


def _indicator_index(spec):
    """Position of a stream's availability channel within that stream, or None."""
    ind = spec.get("indicator_variable")
    if not ind:
        return None
    variables = [str(v) for v in (spec.get("variables") or [])]
    if str(ind) not in variables:
        raise SystemExit(
            f"stream {spec.get('name')!r} declares indicator_variable {ind!r}, which is "
            f"not among its {len(variables)} variables. Rebuild states.")
    return variables.index(str(ind))


def indicator_channels(schema):
    """Channel positions, in the concatenated ``(..., C)`` array, of availability channels.

    An availability channel says whether its stream has real data in this cell. It is a
    flag, not a measurement, and the two things it must NOT go through are the stream
    transform and the standardization -- both of which move the value that means "absent"
    away from 0. That matters because the augmentation masks by multiplying by 0, so if 0
    does not mean "absent", masking writes a value that never occurs in the real data.
    Left alone, the channel reaches the encoder as the raw coverage fraction: 0.0 absent,
    1.0 fully covered, 0.5 half covered.
    """
    out = []
    for s in schema["streams"]:
        j = _indicator_index(s)
        if j is not None:
            out.append(int(s["start"]) + j)
    return out


def _transform(arr, spec):
    """Apply a stream's optional transform (``{'type':'pow','p':..}`` | ``log1p``).

    The stream's availability channel is exempt: the transform exists for the stream's
    measurements (raw BUI is square feet running to ~1e8), and a coverage fraction is not
    one of them.
    """
    t = spec.get("transform")
    if not t:
        return arr
    kind = t.get("type")
    if kind == "pow":
        out = np.power(np.clip(arr, 0.0, None), float(t["p"]))
    elif kind == "log1p":
        out = np.log1p(np.clip(arr, 0.0, None))
    else:
        raise ValueError(f"unknown stream transform {t!r}")
    j = _indicator_index(spec)
    if j is not None:
        out[..., j] = arr[..., j]
    return out


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
        # The transform runs BEFORE mu/sd are fit, so changing it moves every
        # channel of the stream onto a different scale while leaving name, dim
        # and variables identical -- the same silent-misnormalization failure as
        # a reorder, and equally invisible without this check.
        if (ss.get("transform") or None) != (ls.get("transform") or None):
            raise SystemExit(
                f"state schema mismatch{where}: stream {ss['name']!r} was built with "
                f"transform {ss.get('transform')!r} in the model but "
                f"{ls.get('transform')!r} on disk. The saved mu/sd were fit on the "
                f"post-transform scale, so rebuild states and retrain.")
        # Same failure class as the transform, one channel narrower. indicator_variable
        # exempts a channel from BOTH the transform and the standardization, so adding,
        # removing or moving it changes what mu/sd mean for that channel while name, dim,
        # variables and transform all stay identical.
        # ema_tau is baked into the written arrays and changes nothing about their shape,
        # names, order, transform or indicator -- only how smooth they are in time. So two
        # states dirs built at different tau are interchangeable by every other check here and
        # by every count, while the model was fitted on one smoothness and would be run on
        # another. Compared only when BOTH sides recorded it: state dirs built before ema_tau
        # became provenance carry no such key, and a run against one of those must keep
        # working rather than fail on an absence.
        s_tau, l_tau = ss.get("ema_tau"), ls.get("ema_tau")
        if s_tau is not None and l_tau is not None and float(s_tau) != float(l_tau):
            raise SystemExit(
                f"state schema mismatch{where}: stream {ss['name']!r} was built with "
                f"ema_tau={s_tau} in the model but {l_tau} on disk. The input smoothing is "
                f"baked into the arrays and is invisible in every shape and name check, so "
                f"these two states dirs are not interchangeable -- point the run at the "
                f"states dir it was trained against, or rebuild and retrain.")
        if (ss.get("indicator_variable") or None) != (ls.get("indicator_variable") or None):
            raise SystemExit(
                f"state schema mismatch{where}: stream {ss['name']!r} declares "
                f"indicator_variable {ss.get('indicator_variable')!r} in the model but "
                f"{ls.get('indicator_variable')!r} on disk. That channel is exempt from "
                f"the transform and the standardization, so the saved mu/sd do not "
                f"transfer -- rebuild states and retrain.")


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


def fit_norm(cov_flat, schema=None):
    """Per-channel mean/std over ``(N, C)`` (post-transform). Returns ``(mu, sd)``.

    Availability channels are left un-standardized. ``apply_norm`` computes
    ``(x - mu) / sd`` over all channels at once, so switching it off for one channel means
    giving that channel ``mu = 0`` and ``sd = 1``: the subtraction and division then leave
    the value unchanged. Standardizing a 0/1 flag would put "absent" at roughly -1.2 and
    "present" at +0.8, so 0 -- the value the augmentation mask writes -- would mean neither.

    ``schema`` is optional only so callers that predate availability channels still work;
    pass it whenever the stack has one.
    """
    mu = cov_flat.mean(0).astype("float32")
    sd = cov_flat.std(0).astype("float32")
    for ch in (indicator_channels(schema) if schema else []):
        mu[ch] = 0.0
        sd[ch] = 1.0
    return mu, sd


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
