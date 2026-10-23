"""E050: auxiliary habitat covariates on DESK's 27 km grid -- forest age/succession (Pan et al. stand age) and CRP.

    python -I research/exp/aux_covariates.py --grid <ref_grid_27km.tif> --out <aux_27km.npz>
        --forest-age <stand age raster> --forest-ref-year 2006
        --crp-csv <CRP cumulative county csv> --counties <county shapefile> [--y0 1940 --y1 2025]

FOREST (back-cast from one stand-age snapshot): per 27 km cell a 1-year histogram of the reference-year stand age of
its forest pixels, as fractions of the cell's area (all source pixels count in the denominator, so water and
non-forest dilute). A pixel of age A in the reference year is age A - (ref - y) in year y, and not yet established if
that is negative, so any year's age structure is a shift of the histogram:
    forest_est    share of the cell in forest established by year y
    forest_young  share aged 0-20 in year y          forest_mid   20-60          forest_old   60+
    forest_age    mean age of the established forest (0 where none)
Only the latest stand-replacing establishment is known (earlier states of a pixel, e.g. older forest later cut, are
not), so this is succession since the last disturbance, not full history.
CRP: county acres under contract (FSA, 1986-2008) / county land area (ALAND), area-weighted onto cells through a 1 km
rasterization of the counties; zero before the first year, the last year held after it; counties reported as XX999
(fewer than 4 contracts) are dropped.
Reads downloaded files only; run with python -I.
"""
import argparse
import csv
import os

import numpy as np

FOREST_NAMES = ["forest_est", "forest_young", "forest_mid", "forest_old", "forest_age"]
MAX_AGE = 400


def forest_hist(path, grid, block_rows=512):
    """Per-cell 1-year histogram of reference-year stand age, as fractions of the cell's area. Non-forest is nodata in
    the source, so only forest pixels are projected; the denominator is the cell's area in source pixels (cells
    partly outside the source extent are under-counted -- none of them are CONUS)."""
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window
    H, W = grid.height, grid.width
    inv = ~grid.transform
    hist = np.zeros((H * W, MAX_AGE + 1), "float64")
    with rasterio.open(path) as s:
        tr = Transformer.from_crs(s.crs, grid.crs, always_xy=True)
        nod = s.nodata
        px_per_cell = abs(grid.transform.a * grid.transform.e) / abs(s.transform.a * s.transform.e)
        for r0 in range(0, s.height, block_rows):
            nr = min(block_rows, s.height - r0)
            a = s.read(1, window=Window(0, r0, s.width, nr))
            forest = (a >= 0) & (a <= MAX_AGE)
            if nod is not None:
                forest &= a != nod
            if not forest.any():
                continue
            rr, cc = np.nonzero(forest)
            xs, ys = s.transform * (cc + 0.5, rr + r0 + 0.5)
            gx, gy = tr.transform(xs, ys)
            col, row = inv * (np.asarray(gx), np.asarray(gy))
            col, row = np.floor(col).astype(int), np.floor(row).astype(int)
            ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
            np.add.at(hist, (row[ok] * W + col[ok], a[rr[ok], cc[ok]].astype(int)), 1.0)
    hist /= px_per_cell
    total = np.full((H, W), px_per_cell)
    return hist.reshape(H, W, MAX_AGE + 1), total


def forest_features(hist, ref, years):
    """(Y, 5, H, W) from the reference-year age histogram (fractions of cell area)."""
    H, W, _ = hist.shape
    ages = np.arange(MAX_AGE + 1)
    cum = np.cumsum(hist[..., ::-1], axis=-1)[..., ::-1]          # cum[..., k] = share with ref age >= k
    out = np.zeros((len(years), len(FOREST_NAMES), H, W), "float32")
    for i, y in enumerate(years):
        lag = ref - y                                              # age in year y = ref age - lag
        if lag < 0:
            raise ValueError("years after the reference year are not back-cast")
        share_ge = lambda a: cum[..., min(lag + a, MAX_AGE)] if lag + a <= MAX_AGE else np.zeros((H, W))
        est = share_ge(0)
        young = est - share_ge(20)
        mid = share_ge(20) - share_ge(60)
        old = share_ge(60)
        w = np.where(ages >= lag, ages - lag, 0.0)
        mass = (hist * (ages >= lag)).sum(-1)
        mean_age = np.where(mass > 0, (hist * w).sum(-1) / np.maximum(mass, 1e-12), 0.0)
        out[i] = np.stack([est, young, mid, old, mean_age]).astype("float32")
    return out


def crp_features(crp_csv, counties, grid, years, fine=27):
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    H, W = grid.height, grid.width
    rows = list(csv.reader(open(crp_csv, encoding="utf-8")))
    hdr_i = next(i for i, r in enumerate(rows) if r and r[0].strip().upper() == "STATE")
    yrs = [int(float(x)) for x in rows[hdr_i][4:] if x.strip()]
    acres = {}
    for r in rows[hdr_i + 1:]:
        if len(r) < 5 or not r[1].strip() or not r[3].strip():
            continue
        try:
            st, co = int(float(r[1])), int(float(r[3]))
        except ValueError:
            continue
        if co == 999:
            continue
        vals = [float(x) if x.strip() else 0.0 for x in r[4:4 + len(yrs)]]
        acres[f"{st:02d}{co:03d}"] = np.array(vals)
    g = gpd.read_file(counties)
    g["FIPS"] = g["STATEFP"].astype(str).str.zfill(2) + g["COUNTYFP"].astype(str).str.zfill(3)
    g = g.to_crs(grid.crs)
    tr_f = grid.transform * rasterio.Affine.scale(1.0 / fine)
    ids = rasterize(((geom, i + 1) for i, geom in enumerate(g.geometry)), out_shape=(H * fine, W * fine),
                    transform=tr_f, fill=0, dtype="int32")
    frac = np.zeros((len(yrs), len(g)), "float64")
    matched = 0
    for i, f in enumerate(g["FIPS"]):
        if f in acres:
            frac[:, i] = acres[f] * 4046.8564224 / max(float(g["ALAND"].iloc[i]), 1.0)
            matched += 1
    frac = np.clip(frac, 0.0, 1.0)
    out = np.zeros((len(years), H, W), "float32")
    cid = ids.reshape(H, fine, W, fine).transpose(0, 2, 1, 3).reshape(H, W, fine * fine)
    for i, y in enumerate(years):
        if y < yrs[0]:
            continue
        j = yrs.index(min(y, yrs[-1]))
        val = np.concatenate([[0.0], frac[j]])                     # id 0 = no county
        out[i] = val[cid].mean(-1)
    return out, {"counties_in_shapefile": len(g), "counties_with_crp": len(acres), "matched": matched,
                 "crp_years": [yrs[0], yrs[-1]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grid")
    ap.add_argument("--out")
    ap.add_argument("--forest-age", default=None)
    ap.add_argument("--forest-ref-year", type=int, default=2006)
    ap.add_argument("--crp-csv", default=None)
    ap.add_argument("--counties", default=None)
    ap.add_argument("--y0", type=int, default=1940)
    ap.add_argument("--y1", type=int, default=2025)
    a = ap.parse_args()
    import rasterio
    grid = rasterio.open(a.grid)
    years = np.arange(a.y0, a.y1 + 1)
    out = {"years": years}
    info = {}
    if a.forest_age:
        hist, total = forest_hist(a.forest_age, grid)
        fy = np.minimum(years, a.forest_ref_year)                 # after the snapshot: hold it
        out["forest"] = forest_features(hist, a.forest_ref_year, fy)
        out["forest_names"] = np.array(FOREST_NAMES)
        info["forest"] = {"ref_year": a.forest_ref_year, "cells_with_data": int((total > 0).sum()),
                          "mean_forest_share_ref": float(hist.sum(-1)[total > 0].mean())}
    if a.crp_csv:
        out["crp"], info["crp"] = crp_features(a.crp_csv, a.counties, grid, years)
    np.savez(a.out, info=np.array(repr(info)), **out)
    print(f"[aux] -> {a.out}: {info}", flush=True)


if __name__ == "__main__":
    main()
