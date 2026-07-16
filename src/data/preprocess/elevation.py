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
from rasterio.warp import transform as warp_transform

from src.processing import regrid

ELEV_QUANTILES = (0.10, 0.50, 0.90)


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


def _read_dem(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.transform, src.crs, abs(src.transform.a)


def main():
    """Write the 3 elevation-quantile rasters + the centroid table (config-driven).

    Requires a fine DEM already reprojected/aligned to the model Albers grid
    (integer block factor to grid.target_res_m). This is a data-box step; the
    pure helpers above are unit-tested without data.
    """
    import argparse
    import csv
    import os

    from src.config_utils import load_data_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dem", required=True, help="Fine DEM aligned to the model grid (GeoTIFF).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = load_data_config()
    target = cfg["grid"]["target_res_m"]
    dem, transform, crs, native = _read_dem(args.dem)
    block = regrid.block_factor(native, target)
    q = elevation_quantiles(dem, block)              # (3, ny_t, nx_t)
    target_transform = transform * rasterio.Affine.scale(block, block)

    os.makedirs(args.out_dir, exist_ok=True)
    prof = dict(driver="GTiff", height=q.shape[1], width=q.shape[2], count=1,
                dtype="float32", crs=crs, transform=target_transform, nodata=np.nan)
    for name, band in zip(("q10", "q50", "q90"), q):
        with rasterio.open(os.path.join(args.out_dir, f"elev_{name}.tif"), "w", **prof) as dst:
            dst.write(band.astype("float32"), 1)

    cen = cell_centroids_wgs84(target_transform, q.shape[1], q.shape[2], crs)
    with open(os.path.join(args.out_dir, "cell_centroids.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "row", "col", "long", "lat", "elev_q10", "elev_q50", "elev_q90"])
        flat = [b.ravel() for b in q]
        for i in range(cen["id"].size):
            w.writerow([cen["id"][i], cen["row"][i], cen["col"][i],
                        cen["long"][i], cen["lat"][i],
                        flat[0][i], flat[1][i], flat[2][i]])
    print(f"Wrote 3 elevation quantile rasters + cell_centroids.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
