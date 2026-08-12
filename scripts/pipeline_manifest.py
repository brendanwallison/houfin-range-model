#!/usr/bin/env python
"""Record and compare a fingerprint of every pipeline artifact, to prove a refactor changed nothing.

    # BEFORE re-running anything, on the machine that holds the artifacts:
    python scripts/pipeline_manifest.py record -o /work/houfin/manifest_pre.json

    # ... re-run stages with the new code ...
    python scripts/pipeline_manifest.py compare /work/houfin/manifest_pre.json

Config-driven, so it follows ``$ESK_DESK_CONFIG`` / ``$AGE_MODEL_CONFIG`` / ``$DATA_CONFIG``
and works against an overlay. Needs only numpy — no torch, no JAX — so it runs on a login node.

WHY A HASH IS NOT ENOUGH, AND WHAT THIS DOES INSTEAD
----------------------------------------------------
"Byte-identical" is the right invariant for the deterministic stages, and a content hash
settles those outright. It is the WRONG test for anything downstream of DESK training:
torch on CUDA with AMP is not bit-reproducible run-to-run, so ``env_model_semisup.pth``
and everything derived from it can differ in the last bits after a re-run with *zero* code
change. A pure hash diff there produces false alarms and teaches you to ignore it.

So every array artifact gets BOTH a content hash and a numeric summary (shape, dtype,
finite/NaN counts, sum, mean, min, max, and the sum of |x| to catch sign-flip
cancellation). The verdict is then three-valued:

    SAME        hash equal. Definitely unchanged.
    EQUIVALENT  hash differs, every numeric summary within tolerance. Consistent with
                floating-point nondeterminism in the producing stage, not a code change.
                Expected for DESK weights and anything downstream; NOT expected for the
                preprocess/trend/ESK stages, where it means look closer.
    DIFFERS     a numeric summary moved. This is the signal.

THREE FILES DIFFER BY DESIGN ON THIS BRANCH
-------------------------------------------
``desk_meta.npz`` dropped ``bbs_mode`` and ``ebird_frac`` and now always writes
``output_ema=True``; ``cube_meta.json`` gained the DESK-checkpoint provenance keys; and
``metadata.pkl``'s ``z_kernel_contract`` gained two. Comparing those as bytes would report
a failure that is the intended change. They are compared FIELD-WISE against an explicit
expected-added / expected-removed list, so an unlisted key change still fails.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config_utils import load_age_model_config, load_config, load_data_config  # noqa: E402

# Relative tolerance for the EQUIVALENT verdict. 1e-9 is far tighter than float32
# nondeterminism (~1e-6 relative after a long reduction) yet far looser than a real change,
# which moves a mean by orders of magnitude more.
RTOL = 1e-9
ATOL = 1e-12

# Keys this branch intentionally added/removed. Anything else is a failure.
EXPECTED_META_CHANGES = {
    "desk_meta.npz": {"added": {"output_ema"}, "removed": {"bbs_mode", "ebird_frac"}},
    "cube_meta.json": {"added": {"desk_checkpoint", "desk_meta", "esk_basis_dir"}, "removed": set()},
    "metadata.pkl": {"added": {"desk_checkpoint", "esk_basis_dir"}, "removed": set()},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def array_summary(a):
    """Numeric fingerprint of one array, NaN-safe and cheap on a memmap."""
    a = np.asarray(a)
    if a.dtype.kind in "OUS":                       # object/str arrays: hash their repr
        return {"dtype": str(a.dtype), "shape": list(a.shape),
                "repr_sha256": hashlib.sha256(repr(a.tolist()).encode()).hexdigest()}
    f = np.asarray(a, dtype="float64").ravel()
    finite = np.isfinite(f)
    g = f[finite]
    return {
        "dtype": str(a.dtype), "shape": list(a.shape),
        "n_finite": int(finite.sum()), "n_nonfinite": int((~finite).sum()),
        # sum AND abs-sum: a sign flip that cancels in the sum shows up in abs.
        "sum": float(g.sum()) if g.size else 0.0,
        "abs_sum": float(np.abs(g).sum()) if g.size else 0.0,
        "mean": float(g.mean()) if g.size else 0.0,
        "min": float(g.min()) if g.size else 0.0,
        "max": float(g.max()) if g.size else 0.0,
    }


def describe(path, kind="array"):
    """Record for one artifact: identity +, for arrays, per-key numeric summaries."""
    p = Path(path)
    rec = {"path": str(p), "exists": p.exists(), "kind": kind}
    if not p.exists():
        return rec
    rec["size"] = p.stat().st_size
    rec["sha256"] = sha256(p)
    try:
        if p.suffix == ".npy":
            rec["arrays"] = {"": array_summary(np.load(p, mmap_mode="r", allow_pickle=False))}
        elif p.suffix == ".npz":
            with np.load(p, allow_pickle=True) as z:
                rec["arrays"] = {k: array_summary(z[k]) for k in sorted(z.files)}
        elif p.suffix == ".json":
            rec["json_keys"] = sorted(json.loads(p.read_text()).keys())
    except Exception as exc:                        # a corrupt artifact is itself a finding
        rec["read_error"] = f"{type(exc).__name__}: {exc}"
    return rec


def describe_metadata_pkl(path):
    """metadata.pkl: record the scalar/shape surface, not the whole pickle."""
    p = Path(path)
    rec = {"path": str(p), "exists": p.exists(), "kind": "metadata"}
    if not p.exists():
        return rec
    import pickle
    with p.open("rb") as fh:
        md = pickle.load(fh)
    rec["keys"] = sorted(md.keys())
    fields = {}
    for k, v in sorted(md.items()):
        if isinstance(v, np.ndarray):
            fields[k] = array_summary(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            fields[k] = v
        elif isinstance(v, dict):
            fields[k] = {kk: (vv if isinstance(vv, (int, float, str, bool)) or vv is None
                              else f"<{type(vv).__name__}>") for kk, vv in sorted(v.items())}
        else:
            fields[k] = f"<{type(v).__name__}>"
    rec["fields"] = fields
    return rec


def collect():
    """Every artifact worth pinning, resolved from the configs."""
    cfg, dcfg, acfg = load_config(), load_data_config(), load_age_model_config()
    trends = dcfg.get("trends", {})
    points = Path(cfg["trend"]["points_dir"])
    basis = Path(cfg["desk"]["z_dir"])
    desk_out = Path(cfg["paths"]["desk_output_dir"])
    cube = Path(cfg["latent_cube"]["output_dir"])
    zdisp = Path(acfg["raw_z_dir"])
    inputs = Path(acfg["input_dir"])

    items = {}

    def add(stage, name, path, kind="array"):
        items[f"{stage}/{name}"] = describe(path, kind)

    # --- deterministic: these MUST be byte-identical -------------------------------
    for key in ("bbs_trend_grid", "bbs_abund_grid", "ebird_trend_grid"):
        if key in trends:
            add("preprocess", key, trends[key])
    for name in ("X_points.npy", "point_index.npy", "points_meta.json"):
        add("trend_points", name, points / name)
    for name in ("Z.npy", "valid_mask.npy", "esk_landmarks.npy", "esk_projmat.npy", "meta.json"):
        add("esk", name, basis / name)

    # --- DESK: weights are not bit-reproducible; the metadata is compared field-wise -
    for name in ("env_model_semisup.pth", "output_ema.pth", "holdout_cells.npy",
                 "buffer_cells.npy", "training_mask.npy"):
        add("desk", name, desk_out / name)
    items["desk/desk_meta.npz"] = describe(desk_out / "desk_meta.npz", kind="meta")

    # --- cube / path features: sampled, not exhaustive (hundreds of per-year files) --
    cube_years = sorted(cube.glob("Z_latent_*.npy")) if cube.is_dir() else []
    items["cube/_inventory"] = _inventory(cube_years)
    for p in _sample(cube_years):
        add("cube", p.name, p)
    items["cube/cube_meta.json"] = describe(cube / "cube_meta.json", kind="meta")

    disp = sorted(zdisp.glob("Z_disp_*.npz")) if zdisp.is_dir() else []
    items["path_features/_inventory"] = _inventory(disp)
    for p in _sample(disp):
        add("path_features", p.name, p)
    add("path_features", "path_feature_meta.json", zdisp / "path_feature_meta.json")

    # --- model inputs ---------------------------------------------------------------
    items["model_inputs/metadata.pkl"] = describe_metadata_pkl(inputs / "metadata.pkl")
    if inputs.is_dir():
        for p in sorted(inputs.glob("*.dat")):
            add("model_inputs", p.name, p)
    return items


def _sample(paths, k=6):
    """First, last and evenly spaced middles, for the DETAILED numeric summaries.

    Only a sample gets a full summary -- there are ~124 cube years and as many Z_disp
    files. The sample is not the safety net: ``_inventory`` digests every file, so a
    change in an unsampled year is still caught. The sample exists to tell you what KIND
    of change it is once the digest says there was one.
    """
    if len(paths) <= k:
        return list(paths)
    idx = sorted({0, len(paths) - 1, *[round(i * (len(paths) - 1) / (k - 1)) for i in range(k)]})
    return [paths[i] for i in idx]


def _inventory(paths):
    """Names, count, and a digest over EVERY file's content hash.

    Sampling detailed summaries is a deliberate cost trade; sampling the CHANGE DETECTION
    would not be -- a one-year perturbation in an unsampled file would read as SAME. This
    reads every byte once so it cannot.
    """
    per_file = {p.name: sha256(p) for p in paths}
    digest = hashlib.sha256(
        json.dumps(per_file, sort_keys=True).encode()).hexdigest() if per_file else ""
    return {"kind": "inventory", "n_files": len(paths),
            "names": sorted(per_file), "digest": digest, "per_file": per_file}


def _close(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return bool(np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True))
    return a == b


def compare_arrays(old, new):
    """(all_close, [differing field descriptions])"""
    diffs = []
    for key in sorted(set(old) | set(new)):
        if key not in old:
            diffs.append(f"array {key!r} is new"); continue
        if key not in new:
            diffs.append(f"array {key!r} disappeared"); continue
        for f in sorted(set(old[key]) | set(new[key])):
            a, b = old[key].get(f), new[key].get(f)
            if not _close(a, b):
                label = f"{key}.{f}" if key else f
                diffs.append(f"{label}: {a!r} -> {b!r}")
    return (not diffs), diffs


def compare(pre, now):
    verdicts = []
    for name in sorted(set(pre) | set(now)):
        o, n = pre.get(name), now.get(name)
        if o is None:
            verdicts.append((name, "NEW", ["absent from the baseline"])); continue
        if n is None:
            verdicts.append((name, "GONE", ["present in the baseline, absent now"])); continue

        if o.get("kind") == "inventory":
            miss = sorted(set(o["names"]) - set(n["names"]))
            extra = sorted(set(n["names"]) - set(o["names"]))
            op, np_ = o.get("per_file", {}), n.get("per_file", {})
            changed = sorted(k for k in set(op) & set(np_) if op[k] != np_[k])
            notes = []
            if miss:
                notes.append(f"{len(miss)} missing: {miss[:4]}")
            if extra:
                notes.append(f"{len(extra)} new: {extra[:4]}")
            if changed:
                notes.append(f"{len(changed)} of {len(op)} files changed content: {changed[:6]}")
            verdicts.append((name, "DIFFERS" if notes else "SAME",
                             notes or [f"{o['n_files']} files, digest matches"]))
            continue

        if not o.get("exists") and not n.get("exists"):
            verdicts.append((name, "ABSENT", ["not produced in either run"])); continue
        if o.get("exists") != n.get("exists"):
            verdicts.append((name, "DIFFERS", [f"exists {o.get('exists')} -> {n.get('exists')}"]))
            continue

        if o.get("kind") in ("meta", "metadata"):
            verdicts.append(compare_meta(name, o, n)); continue

        if o.get("sha256") == n.get("sha256"):
            verdicts.append((name, "SAME", [])); continue
        ok, diffs = compare_arrays(o.get("arrays", {}), n.get("arrays", {}))
        if o.get("json_keys") is not None and o["json_keys"] != n.get("json_keys"):
            ok, diffs = False, diffs + [f"json keys {o['json_keys']} -> {n.get('json_keys')}"]
        verdicts.append((name, "EQUIVALENT" if ok else "DIFFERS",
                         diffs or ["hash differs; every numeric summary matches"]))
    return verdicts


def compare_meta(name, o, n):
    """Field-wise, tolerating only the additions/removals this branch declares."""
    base = Path(name).name
    exp = EXPECTED_META_CHANGES.get(base, {"added": set(), "removed": set()})
    ok, diffs = compare_arrays(o.get("arrays", {}), n.get("arrays", {}))
    of, nf = o.get("fields", {}), n.get("fields", {})
    added, removed = set(nf) - set(of), set(of) - set(nf)
    unexpected_add, unexpected_rm = added - exp["added"], removed - exp["removed"]
    if unexpected_add:
        diffs.append(f"UNEXPECTED new fields: {sorted(unexpected_add)}")
    if unexpected_rm:
        diffs.append(f"UNEXPECTED removed fields: {sorted(unexpected_rm)}")
    for f in sorted(set(of) & set(nf)):
        if not _close(of[f], nf[f]):
            diffs.append(f"{f}: {of[f]!r} -> {nf[f]!r}")
    note = []
    if added & exp["added"]:
        note.append(f"declared additions {sorted(added & exp['added'])}")
    if removed & exp["removed"]:
        note.append(f"declared removals {sorted(removed & exp['removed'])}")
    real = [d for d in diffs if d]
    return (name, "DIFFERS" if real else "SAME", real or note or ["identical"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="write a manifest of the artifacts on disk now")
    r.add_argument("-o", "--out", required=True)
    c = sub.add_parser("compare", help="compare the artifacts on disk now against a manifest")
    c.add_argument("baseline")
    c.add_argument("-v", "--verbose", action="store_true", help="list SAME entries too")
    args = ap.parse_args()

    if args.cmd == "record":
        items = collect()
        Path(args.out).write_text(json.dumps(items, indent=2, sort_keys=True))
        present = sum(1 for v in items.values()
                      if v.get("exists") or v.get("kind") == "inventory")
        print(f"recorded {present}/{len(items)} artifacts -> {args.out}")
        for name, v in sorted(items.items()):
            if v.get("kind") != "inventory" and not v.get("exists"):
                print(f"  (absent) {name}")
        return 0

    pre = json.loads(Path(args.baseline).read_text())
    verdicts = compare(pre, collect())
    order = {"DIFFERS": 0, "NEW": 1, "GONE": 1, "EQUIVALENT": 2, "ABSENT": 3, "SAME": 4}
    counts = {}
    for name, verdict, notes in sorted(verdicts, key=lambda v: (order[v[1]], v[0])):
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict == "SAME" and not args.verbose:
            continue
        print(f"{verdict:11s} {name}")
        for d in notes:
            if d:
                print(f"              {d}")
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    bad = counts.get("DIFFERS", 0) + counts.get("GONE", 0)
    if bad:
        print(f"\nFAIL: {bad} artifact(s) changed. EQUIVALENT is fine for DESK and anything "
              f"downstream of it (CUDA/AMP is not bit-reproducible); it is NOT fine for "
              f"preprocess, trend_points or esk, which are deterministic.")
    else:
        print("\nOK: nothing changed beyond declared metadata fields.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
