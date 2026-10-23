"""E086aa: CBC WINTER community points in the bbs_community_points artifact format, for a winter ESK basis.

    python research/exp/e086aa_winter_points.py --cbc cbc_circles_all.npz --species community_all_nonsealed.csv
        --out-dir <points dir> [--years 1966 2024]

Rows: grid cell x winter (year_end) with >= 1 CBC list with party-hours in --years. Counts standardized to the median
party-hours with exponent 0.8 (E064 / E081), averaged over a cell-year's lists, then log1p -- the same transform the
breeding points use. Columns: the --species list in its order, restricted to the species the CBC file carries (the
rest are absent from winter counts or dropped by name harmonization); points_meta.json records which.
Writes X_points.npy (N, S) float32, point_index.npy (N, 3) int32 (row, col, year), point_weights.npy (N,) ones,
points_meta.json. Sealed species and the House Finch never appear (the CBC file is built from the cache's non-sealed
lists; asserted).
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cbc", required=True); ap.add_argument("--species", required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--years", type=int, nargs=2, default=[1966, 2024])
    a = ap.parse_args()
    import pandas as pd
    codes = [str(c).lower() for c in pd.read_csv(a.species)["species_code"]]
    assert "houfin" not in codes
    D = np.load(a.cbc, allow_pickle=False); cs = list(D["species"])
    assert "houfin" not in cs
    keep = [c for c in codes if c in cs]
    ph, yr = D["party_hours"].astype(float), D["year_end"]
    ok = (D["row"] >= 0) & (yr >= a.years[0]) & (yr <= a.years[1]) & np.isfinite(ph) & (ph > 0)
    Y = D["Y"][ok][:, [cs.index(c) for c in keep]].astype(float) * ((np.median(ph[ok]) / ph[ok])[:, None] ** 0.8)
    df = pd.DataFrame(Y); df["r"], df["c"], df["y"] = D["row"][ok], D["col"][ok], yr[ok]
    g = df.groupby(["r", "c", "y"]).mean()
    idx = np.array([list(k) for k in g.index], np.int32)
    X = np.log1p(g.to_numpy()).astype(np.float32)
    nz = X.sum(1) > 0
    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, "X_points.npy"), X[nz]); np.save(os.path.join(a.out_dir, "point_index.npy"), idx[nz])
    np.save(os.path.join(a.out_dir, "point_weights.npy"), np.ones(int(nz.sum()), np.float32))
    meta = {"source": "cbc_winter", "species": keep, "n_species": len(keep), "n_rows": int(nz.sum()),
            "absent_from_cbc": [c for c in codes if c not in cs], "years": a.years, "ruzicka_log1p": True,
            "effort": "party-hours^0.8 to the median", "cbc_file": a.cbc}
    json.dump(meta, open(os.path.join(a.out_dir, "points_meta.json"), "w"), indent=1)
    print(f"[E086aa] winter points: {int(nz.sum()):,} cell-years x {len(keep)} species ({len(codes) - len(keep)} of the list "
          f"absent from CBC) -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
