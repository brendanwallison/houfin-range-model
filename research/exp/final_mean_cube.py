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

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cubes", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-corr", type=float, default=0.8); ap.add_argument("--ref-year", type=int, default=2020)
    a = ap.parse_args()
    files = [sorted(glob.glob(os.path.join(c, "Z_latent_*.npy"))) for c in a.cubes]
    years = [os.path.basename(f)[9:-4] for f in files[0]]
    assert all([os.path.basename(f)[9:-4] for f in fs] == years for fs in files), "seed cubes cover different years"
    ref = [np.load(os.path.join(c, f"Z_latent_{a.ref_year}.npy")) for c in a.cubes]
    ok = np.all([np.isfinite(r).all(-1) for r in ref], 0)
    rep = {}
    for i in range(len(ref)):
        for j in range(i + 1, len(ref)):
            cc = [np.corrcoef(ref[i][ok][:, d], ref[j][ok][:, d])[0, 1] for d in range(ref[i].shape[-1])]
            rep[f"{i}-{j}"] = {"median": float(np.median(cc)), "min": float(np.min(cc)), "first24_median": float(np.median(cc[:24]))}
    print("[mean-cube] per-dim seed correlation (ref year):", json.dumps(rep), flush=True)
    if min(v["median"] for v in rep.values()) < a.min_corr:
        raise SystemExit("seeds not aligned enough to average -- refusing")
    os.makedirs(a.out, exist_ok=True)
    for y in years:
        st = np.stack([np.load(os.path.join(c, f"Z_latent_{y}.npy")) for c in a.cubes])
        np.save(os.path.join(a.out, f"Z_latent_{y}.npy"), st.mean(0).astype(np.float32))
    json.dump({"seeds": a.cubes, "alignment": rep, "years": [years[0], years[-1]]},
              open(os.path.join(a.out, "ensemble_meta.json"), "w"), indent=1)
    print(f"[mean-cube] {len(a.cubes)} seeds x {len(years)} years -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
