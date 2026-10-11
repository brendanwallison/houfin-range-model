"""E051: reservoir covariates from the National Inventory of Dams on DESK's 27 km grid.

    python -I research/exp/nid_covariates.py --states <dir of nid_<ST>.json> --grid <ref_grid_27km.tif>
        --out <nid_27km.npz> [--y0 1940 --y1 2025]

Per dam (NID /query, addProps=all): lat/lon, yearCompleted, surfaceArea (acres), normalStorage (acre-feet). Where
surfaceArea is missing it is imputed from normalStorage by the log-log fit over dams having both. Per cell and year y,
over dams completed by y:
    nid_surface   cumulative reservoir surface area / cell area
    nid_storage   log1p(cumulative normal storage, acre-feet per km^2 of cell)
    nid_count     cumulative number of dams
Dams without a completion year are left out (their timing is unknown), and so are dams outside the grid. Reads
downloaded files only; run with python -I.
"""
import argparse
import glob
import json
import os

import numpy as np

NAMES = ["nid_surface", "nid_storage", "nid_count"]


def fnum(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--states")
    ap.add_argument("--grid")
    ap.add_argument("--out")
    ap.add_argument("--y0", type=int, default=1940)
    ap.add_argument("--y1", type=int, default=2025)
    a = ap.parse_args()
    import rasterio
    from pyproj import Transformer
    grid = rasterio.open(a.grid)
    H, W = grid.height, grid.width
    recs = []
    for f in sorted(glob.glob(os.path.join(a.states, "nid_*.json"))):
        for d in json.load(open(f, encoding="utf-8")):
            recs.append((fnum(d.get("latitude")), fnum(d.get("longitude")), fnum(d.get("yearCompleted")),
                         fnum(d.get("surfaceArea")), fnum(d.get("normalStorage"))))
    r = np.array(recs, "float64")
    lat, lon, yc, sa, st = r.T
    both = np.isfinite(sa) & np.isfinite(st) & (sa > 0) & (st > 0)
    b1, b0 = np.polyfit(np.log(st[both]), np.log(sa[both]), 1)
    sa_imp = np.where(np.isfinite(sa) & (sa > 0), sa,
                      np.where(np.isfinite(st) & (st > 0), np.exp(b0 + b1 * np.log(np.maximum(st, 1e-9))), 0.0))
    st0 = np.where(np.isfinite(st) & (st > 0), st, 0.0)
    tr = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    ok_ll = np.isfinite(lat) & np.isfinite(lon)
    x, y = tr.transform(lon[ok_ll], lat[ok_ll])
    col, row = ~grid.transform * (np.asarray(x), np.asarray(y))
    col, row = np.floor(col).astype(int), np.floor(row).astype(int)
    idx_ll = np.where(ok_ll)[0]
    in_g = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    keep = idx_ll[in_g]
    cell = row[in_g] * W + col[in_g]
    yk = yc[keep]
    has_year = np.isfinite(yk) & (yk > 1600) & (yk <= a.y1)
    cell, yk, sak, stk = cell[has_year], yk[has_year], sa_imp[keep][has_year], st0[keep][has_year]
    cell_m2 = abs(grid.transform.a * grid.transform.e)
    years = np.arange(a.y0, a.y1 + 1)
    out = np.zeros((len(years), len(NAMES), H, W), "float32")
    for i, yy in enumerate(years):
        m = yk <= yy
        s_ = np.bincount(cell[m], weights=sak[m] * 4046.8564224, minlength=H * W) / cell_m2
        v_ = np.bincount(cell[m], weights=stk[m], minlength=H * W) / (cell_m2 / 1e6)
        n_ = np.bincount(cell[m], minlength=H * W).astype("float64")
        out[i] = np.stack([s_, np.log1p(v_), n_]).reshape(3, H, W)
    info = {"dams_total": len(r), "with_latlon": int(ok_ll.sum()), "in_grid": int(in_g.sum()),
            "with_year": int(has_year.sum()), "surface_reported": int((np.isfinite(sa) & (sa > 0)).sum()),
            "area_storage_fit": [float(b0), float(b1)],
            "completed_1966_2005": int(((yk > 1966) & (yk <= 2005)).sum())}
    np.savez(a.out, years=years, nid=out, nid_names=np.array(NAMES), info=np.array(repr(info)))
    print(f"[nid] -> {a.out}: {info}", flush=True)


if __name__ == "__main__":
    main()
