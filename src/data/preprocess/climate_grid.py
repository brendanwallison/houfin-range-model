"""Rasterize the per-centroid climate CSVs into per-year model-grid GeoTIFFs.

climr writes long-format ``climate_{q10,q50,q90}.csv`` (``id, PERIOD, <monthly
cols>``) — a dead-end format that only the viz script read. This turns them into
the same per-year raster layout every other covariate uses, so the covariate
assembler (``streams.run_states``) can ingest climate like LUH-3/HYDE. For each
level and each model **bio-year** ``T`` (Aug(T-1)→Jul(T)), the 12 monthly columns
are collapsed per base variable (PPT-like summed, temperatures averaged; see
``climate_io.bioyear_aggregate``) and scattered onto the model grid, writing
``climate_grid/{base}_{lvl}_{T}_grid.tif`` (aligned to ``grid.ref_raster``).

The level is folded into the variable token so the file matches the streamer's
``{var}_{year}_grid.tif`` pattern with ``var = "{base}_{lvl}"`` (e.g.
``Tmax_q50_1980_grid.tif``).

    python -m src.data.preprocess.climate_grid \
        --climate-dir $HOUFIN_DATA/climate --centroids $HOUFIN_DATA/elevation/cell_centroids.csv \
        --out $HOUFIN_DATA/climate_grid
"""
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config_utils import load_data_config
from src.data.combine.climate_io import bioyear_aggregate, grid_from_centroids, parse_month_columns
from src.processing import regrid
from src.temporal import load_timeline, model_years

LEVELS = ("q10", "q50", "q90")


def _ref_template():
    """Single-band model-grid template DataArray (float32) + its (ny, nx)."""
    ref = regrid.load_ref()
    band0 = ref.isel(band=0) if "band" in ref.dims else ref
    tmpl = band0.astype("float32")
    ny, nx = int(tmpl.sizes["y"]), int(tmpl.sizes["x"])
    return tmpl, ny, nx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--climate-dir", default=None, help="dir with climate_{lvl}.csv")
    ap.add_argument("--centroids", default=None, help="cell_centroids.csv (id,row,col,...)")
    ap.add_argument("--out", default=None, help="output climate_grid dir")
    ap.add_argument("--levels", default=",".join(LEVELS))
    args = ap.parse_args()

    cfg = load_data_config()
    dr = cfg["datasets_root"]
    climate_dir = args.climate_dir or os.path.join(dr, "climate")
    centroids = args.centroids or os.path.join(dr, "elevation", "cell_centroids.csv")
    out = args.out or os.path.join(dr, "climate_grid")
    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    os.makedirs(out, exist_ok=True)

    tl = load_timeline()
    start_month = tl["bio_year_start_month"]
    years = model_years(tl)
    cen = pd.read_csv(centroids, usecols=["id", "row", "col"])
    tmpl, ny, nx = _ref_template()

    for lvl in levels:
        csv = os.path.join(climate_dir, f"climate_{lvl}.csv")
        if not os.path.exists(csv):
            print(f"[skip {lvl}] missing {csv}", flush=True)
            continue
        df = pd.read_csv(csv)
        groups = parse_month_columns(df.columns)
        bases = list(groups)
        print(f"[{lvl}] {len(bases)} base vars {bases} x {len(years)} bio-years", flush=True)
        for yr in tqdm(years, desc=f"climate {lvl}", mininterval=2):
            # Resume: skip a year whose rasters all already exist.
            paths = {b: os.path.join(out, f"{b}_{lvl}_{yr}_grid.tif") for b in bases}
            if all(os.path.exists(p) for p in paths.values()):
                continue
            agg = bioyear_aggregate(df, yr, start_month, month_groups=groups)
            if agg.empty:
                continue  # bio-year straddles a data gap (e.g. before obs start)
            agg = agg.reset_index()
            for base in bases:
                grid = grid_from_centroids(agg, cen, ny, nx, value_col=base)
                da = tmpl.copy(data=grid)
                da.rio.write_nodata(np.nan, inplace=True)
                da.rio.to_raster(paths[base])
    print(f"Done -> {out}", flush=True)


if __name__ == "__main__":
    main()
