"""Ensemble-mean z cube for the age model: average per-seed Z_latent_<year>.npy (raw instantaneous z, the
build_final_z_cube layout) after checking the seeds share coordinates (z is supervised to the same ESK targets).

    python research/exp/final_mean_cube.py --cubes <cube dir> ... --out <dir> [--min-corr 0.8]

Per seed pair and latent dim, the correlation over valid cells of a reference year is reported; the build refuses if
the median per-dim correlation of any pair is below --min-corr (unaligned seeds must not be averaged). NaN pattern is
kept where any seed is NaN.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.community_encoder.cube_contract import compatible_contract, validate_cube_contract


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cubes", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-corr", type=float, default=0.8); ap.add_argument("--ref-year", type=int, default=2020)
    ap.add_argument("--shared-coordinate-targets", action="store_true",
                    help="require shared hashed coordinate targets; retain low seed correlation as uncertainty, not an arbitrary rotation")
    a = ap.parse_args()
    if len(a.cubes) < 2 or len(set(map(os.path.realpath, a.cubes))) != len(a.cubes):
        raise ValueError("supply at least two distinct seed cubes")
    if os.path.realpath(a.out) in set(map(os.path.realpath, a.cubes)):
        raise ValueError("ensemble output must not overwrite a seed cube")
    metas = []
    for c in a.cubes:
        with open(os.path.join(c, "cube_meta.json")) as fh:
            m = json.load(fh)
        validate_cube_contract(m)
        if m.get("temporal_output") != "raw_instantaneous":
            raise ValueError("ensemble requires raw_instantaneous cubes")
        if not m.get("desk_checkpoint", {}).get("content_sha256"):
            raise ValueError("seed cube lacks checkpoint identity")
        metas.append(m)
    if any(compatible_contract(m) != compatible_contract(metas[0]) for m in metas[1:]):
        raise ValueError("seed cube feature layouts or basis identities differ")
    if a.shared_coordinate_targets:
        for block in metas[0].get("feature_blocks", []):
            identities = block.get("basis_identity", {})
            if not block.get("coordinate_target") or not all(
                    identities.get(k, {}).get("content_sha256") for k in
                    ("meta.json", "esk_projmat.npy", "esk_landmarks.npy")):
                raise ValueError("shared-coordinate mode requires hashed basis and coordinate target provenance")
            if block["start"] and not block.get("coordinate_target_identity", {}).get("content_sha256"):
                raise ValueError("private seasonal coordinates need a hashed coordinate target")
        if not metas[0].get("feature_blocks"):
            raise ValueError("shared-coordinate mode needs explicit target blocks")
    files = [sorted(glob.glob(os.path.join(c, "Z_latent_*.npy"))) for c in a.cubes]
    years = [os.path.basename(f)[9:-4] for f in files[0]]
    if not years or any([os.path.basename(f)[9:-4] for f in fs] != years for fs in files):
        raise ValueError("seed cubes cover different or empty years")
    if any([int(y) for y in years] != m.get("years") for m in metas):
        raise ValueError("seed cube files disagree with metadata years")
    shape = None
    for fs in files:
        for f in fs:
            z = np.load(f, mmap_mode="r")
            validate_cube_contract(metas[0], z.shape[-1])
            shape = z.shape if shape is None else shape
            if z.shape != shape or z.ndim != 3:
                raise ValueError("seed cube array shapes differ")
    ref = [np.load(os.path.join(c, f"Z_latent_{a.ref_year}.npy")) for c in a.cubes]
    ok = np.all([np.isfinite(r).all(-1) for r in ref], 0)
    rep = {}
    for i in range(len(ref)):
        for j in range(i + 1, len(ref)):
            cc = [np.corrcoef(ref[i][ok][:, d], ref[j][ok][:, d])[0, 1] for d in range(ref[i].shape[-1])]
            rep[f"{i}-{j}"] = {"median": float(np.median(cc)), "min": float(np.min(cc)), "first24_median": float(np.median(cc[:24]))}
            rep[f"{i}-{j}"]["blocks"] = {
                b["name"]: float(np.median(cc[b["start"]:b["stop"]]))
                for b in metas[0].get("feature_blocks", [
                    {"name": "summer", "start": 0, "stop": len(cc)}])}
    print("[mean-cube] per-dim seed correlation (ref year):", json.dumps(rep), flush=True)
    scores = [score for v in rep.values() for score in [v["median"], *v["blocks"].values()]]
    if not np.isfinite(scores).all() or (min(scores) < a.min_corr and not a.shared_coordinate_targets):
        raise SystemExit("seeds not aligned enough to average -- refusing")
    os.makedirs(a.out, exist_ok=True)
    for y in years:
        st = np.stack([np.load(os.path.join(c, f"Z_latent_{y}.npy")) for c in a.cubes])
        np.save(os.path.join(a.out, f"Z_latent_{y}.npy"), st.mean(0).astype(np.float32))
    json.dump({"seeds": a.cubes, "alignment": rep, "years": [years[0], years[-1]]},
              open(os.path.join(a.out, "ensemble_meta.json"), "w"), indent=1)
    cube_meta = {**compatible_contract(metas[0]), "years": [int(y) for y in years],
                 "kernel_contract": metas[0]["kernel_contract"],
                 "ensemble_method": "arithmetic_mean_of_aligned_features",
                 "coordinate_alignment": "shared_hashed_targets" if a.shared_coordinate_targets else "seed_correlation_gate",
                 "seed_correlation_diagnostics": rep,
                 "ensemble_members": [{"cube_dir": c, "desk_checkpoint": m["desk_checkpoint"],
                                       "desk_meta": m.get("desk_meta")}
                                      for c, m in zip(a.cubes, metas)]}
    with open(os.path.join(a.out, "cube_meta.json"), "w") as fh:
        json.dump(cube_meta, fh, indent=2, allow_nan=False)
    print(f"[mean-cube] {len(a.cubes)} seeds x {len(years)} years -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
