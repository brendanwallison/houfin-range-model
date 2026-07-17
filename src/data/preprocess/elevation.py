"""Static elevation summaries for climate downscaling.

Aggregates a fine DEM to the model grid as three per-cell elevation quantiles
(p10 / p50 / p90) and emits the model-grid cell centroids reprojected to WGS84.
Those two products drive the climate acquire step: `climr` is queried at each
centroid for each of the three representative elevations, giving climate at
low/median/high sub-cell elevation without ever materializing 1 km climate (see
docs/TEMPORAL.md and the climate acquire module). Elevation is time-invariant,
so this runs once.
"""
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.warp import transform as warp_transform

from src.processing import regrid

ELEV_QUANTILES = (0.10, 0.50, 0.90)
DEFAULT_FINE_FACTOR = 14  # ref cells subdivided this many times before quantiling


def dem_to_fine_grid(dem_path, ref_transform, ref_crs, H, W, fine_factor):
    """Reproject any DEM onto a grid aligned to the ref, ``fine_factor`` x finer.

    Returns a (H*fine_factor, W*fine_factor) array in the ref CRS whose blocks
    nest exactly into the model cells, so ``block_quantiles(..., fine_factor)``
    gives per-model-cell elevation quantiles. Handles a geographic or projected
    DEM at any resolution (unlike integer ``block_factor``, which requires the
    source to be an integer sub-multiple of the target already).
    """
    ff = int(fine_factor)
    fine_transform = ref_transform * rasterio.Affine.scale(1.0 / ff, 1.0 / ff)
    fine = np.full((H * ff, W * ff), np.nan, dtype="float64")
    with rasterio.open(dem_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=fine,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=fine_transform, dst_crs=ref_crs,
            src_nodata=src.nodata, dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    return fine, ff


def elevation_quantiles(dem, block, quantiles=ELEV_QUANTILES):
    """Per model cell, the (p10, p50, p90) elevation of its fine DEM subpixels.

    ``dem`` is the fine DEM aligned to the model grid (nodata = NaN); ``block``
    is fine cells per model cell. Returns (n_quantiles, ny_t, nx_t).
    """
    return regrid.block_quantiles(np.asarray(dem, dtype="float64"), block, quantiles)


def cell_centroids_wgs84(transform, nrows, ncols, src_crs):
    """Model-grid cell-center lon/lat (WGS84) for every cell.

    Returns a dict of flat arrays: id, row, col, long, lat (row-major order).
    `climr` needs geographic lon/lat, so centers are computed in the grid CRS
    and reprojected — the grid stays in its projected CRS, climr works in lon/lat.
    """
    rows, cols = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")
    rows = rows.ravel(); cols = cols.ravel()
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    lon, lat = warp_transform(src_crs, "EPSG:4326", xs, ys)
    return {
        "id": np.arange(rows.size),
        "row": rows, "col": cols,
        "long": np.asarray(lon), "lat": np.asarray(lat),
    }


def main():
    """Write the 3 elevation-quantile rasters + the centroid table (config-driven).

    Consumes ANY DEM (geographic or projected, any resolution): it is reprojected
    onto a fine sub-grid of the model ref grid, then quantiled per model cell. The
    DEM defaults to the ``dem`` block downloaded by acquire/dem.py; ``fine_factor``
    from the ``elevation`` config. This is a data-box step; the pure helpers above
    are unit-tested without data.
    """
    import argparse
    import csv
    import glob
    import os

    from src.config_utils import load_data_config

    cfg = load_data_config()
    ecfg = cfg.get("elevation", {})
    dcfg = cfg.get("dem", {})

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dem", help="DEM GeoTIFF (default: the acquire/dem.py download).")
    ap.add_argument("--out-dir", help="Default: {datasets_root}/elevation.")
    ap.add_argument("--fine-factor", type=int)
    args = ap.parse_args()

    dem_path = args.dem
    if not dem_path:  # default to the downloaded DEM under datasets_root/<dem.out_subdir>
        dem_dir = os.path.join(cfg["datasets_root"], dcfg.get("out_subdir", "dem"))
        found = sorted(glob.glob(os.path.join(dem_dir, "*.tif")))
        if not found:
            raise SystemExit(f"no DEM in {dem_dir}; run scripts/download_dem.py or pass --dem")
        dem_path = found[0]
    out_dir = args.out_dir or os.path.join(cfg["datasets_root"], "elevation")
    fine_factor = args.fine_factor or ecfg.get("fine_factor", DEFAULT_FINE_FACTOR)

    with rasterio.open(cfg["grid"]["ref_raster"]) as ref:
        ref_transform, crs, H, W = ref.transform, ref.crs, ref.height, ref.width

    fine, block = dem_to_fine_grid(dem_path, ref_transform, crs, H, W, fine_factor)
    q = elevation_quantiles(fine, block)             # (3, H, W)
    target_transform = ref_transform

    os.makedirs(out_dir, exist_ok=True)
    prof = dict(driver="GTiff", height=q.shape[1], width=q.shape[2], count=1,
                dtype="float32", crs=crs, transform=target_transform, nodata=np.nan)
    for name, band in zip(("q10", "q50", "q90"), q):
        with rasterio.open(os.path.join(out_dir, f"elev_{name}.tif"), "w", **prof) as dst:
            dst.write(band.astype("float32"), 1)

    cen = cell_centroids_wgs84(target_transform, q.shape[1], q.shape[2], crs)
    with open(os.path.join(out_dir, "cell_centroids.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "row", "col", "long", "lat", "elev_q10", "elev_q50", "elev_q90"])
        flat = [b.ravel() for b in q]
        for i in range(cen["id"].size):
            w.writerow([cen["id"][i], cen["row"][i], cen["col"][i],
                        cen["long"][i], cen["lat"][i],
                        flat[0][i], flat[1][i], flat[2][i]])
    print(f"Wrote 3 elevation quantile rasters + cell_centroids.csv to {out_dir}")


if __name__ == "__main__":
    main()
