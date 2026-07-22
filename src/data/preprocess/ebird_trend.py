"""Rasterize eBird Status & Trends 'trends' parquets onto the model grid.

Each species' trends parquet has one row per native 27 km cell with WGS84
``longitude``/``latitude`` centroids and per-cell trend estimates. We project the
centroids to the ref-grid CRS, bin each row to a model cell, and average within a
cell (eBird's 27 km sinusoidal grid maps ~1:1 onto the 27 km Albers grid). Two
fields are gridded per species:

  ``abd_ppy``  -- percent-per-year trend in relative abundance (the recent-domain
                  rate blended with the BBS long-term rate in trend_community.py)
  ``abd``      -- relative abundance at the middle of the trend window (a
                  diagnostic / optional reference; the modern anchor is the 2023
                  eBird abundance raster, not this).

Output ``trends.ebird_trend_grid`` (.npz): ``abd_ppy`` (n_species, H, W) float32,
``abd`` (n_species, H, W) float32, ``species_code`` (n_species,), ``valid`` (H, W)
bool, ``start_year``/``end_year`` (n_species,) int.

    python -m src.data.preprocess.ebird_trend
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from src.config_utils import load_data_config


def _find_parquet(trends_dir, code):
    """The main (non-folds) trends-estimates parquet for one species code."""
    hits = [p for p in glob.glob(os.path.join(trends_dir, f"{code}_*_ebird-trends_*.parquet"))
            if "_folds_" not in os.path.basename(p)]
    return sorted(hits)[0] if hits else None


def rasterize_parquet(path, ref_crs, transform, H, W, value_cols=("abd_ppy", "abd")):
    """Bin a trends parquet's per-cell rows onto the model grid (mean within cell)."""
    from rasterio.warp import transform as warp_transform

    df = pd.read_parquet(path, columns=["longitude", "latitude", "start_year", "end_year",
                                        *value_cols])
    xs, ys = warp_transform("EPSG:4326", ref_crs, df["longitude"].tolist(), df["latitude"].tolist())
    cols, rows = (~transform) * (np.asarray(xs), np.asarray(ys))  # world -> fractional pixel
    col = np.floor(cols).astype(int)
    row = np.floor(rows).astype(int)
    inb = (row >= 0) & (row < H) & (col >= 0) & (col < W)

    grids = {}
    for vc in value_cols:
        v = df[vc].to_numpy(dtype="float64")
        m = inb & np.isfinite(v)
        acc = np.zeros((H, W), "float64")
        cnt = np.zeros((H, W), "float64")
        np.add.at(acc, (row[m], col[m]), v[m])
        np.add.at(cnt, (row[m], col[m]), 1.0)
        g = np.full((H, W), np.nan, "float32")
        nz = cnt > 0
        g[nz] = (acc[nz] / cnt[nz]).astype("float32")
        grids[vc] = g
    yr = (int(df["start_year"].iloc[0]) if len(df) else 0,
          int(df["end_year"].iloc[0]) if len(df) else 0)
    return grids, yr


def build(community_csv, trends_dir, out_path):
    import rasterio

    comm = pd.read_csv(community_csv)
    dcfg = load_data_config()
    with rasterio.open(dcfg["grid"]["ref_raster"]) as ref:
        ref_crs, transform = ref.crs, ref.transform
        H, W = ref.height, ref.width

    ppy, abd, codes, sy, ey, missing = [], [], [], [], [], []
    for _, r in comm.iterrows():
        code = str(r["species_code"])
        path = _find_parquet(trends_dir, code)
        if path is None:
            missing.append(code)
            continue
        grids, (s, e) = rasterize_parquet(path, ref_crs, transform, H, W)
        ppy.append(grids["abd_ppy"])
        abd.append(grids["abd"])
        codes.append(code)
        sy.append(s)
        ey.append(e)
    if not codes:
        raise SystemExit(f"no eBird trends parquets found in {trends_dir} for {len(comm)} species")

    abd_ppy = np.stack(ppy).astype("float32")
    abd_arr = np.stack(abd).astype("float32")
    valid = np.isfinite(abd_ppy).any(axis=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, abd_ppy=abd_ppy, abd=abd_arr, species_code=np.array(codes),
             valid=valid, start_year=np.array(sy, dtype=int), end_year=np.array(ey, dtype=int))
    print(f"[ebird_trend] {len(codes)} species gridded to {H}x{W} "
          f"({int(valid.sum())} covered cells); {len(missing)} missing parquets"
          + (f": {missing}" if missing else "."))
    print(f"[ebird_trend] wrote -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default=None, help="community_trend.csv (default: config).")
    ap.add_argument("--trends-dir", default=None, help="Dir of eBird trends parquets (default: config).")
    ap.add_argument("--out", default=None, help="Output .npz (default: trends.ebird_trend_grid).")
    args = ap.parse_args()

    dcfg = load_data_config()
    dr = dcfg["datasets_root"]
    community = args.community or dcfg["community_trend_list"]
    trends_dir = args.trends_dir or os.path.join(dr, dcfg.get("ebird_trends_subdir", "ebird_trends_2022"))
    out = args.out or dcfg["trends"]["ebird_trend_grid"]
    build(community, trends_dir, out)


if __name__ == "__main__":
    main()
