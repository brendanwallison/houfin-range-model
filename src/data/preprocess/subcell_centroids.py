"""Sub-cell centroids for a more principled climate quantile (optional path).

The default climate path (elevation.py + climate_climr.py) samples each 25 km cell
once at its centroid, at three elevation quantiles — capturing *elevation-only*
sub-grid variability at a fixed lon/lat. This module instead lays a ``grid``x``grid``
mesh of points inside each model cell, each at its TRUE fine-DEM lon/lat + mean
elevation, so climr is evaluated across the cell's actual spatial *and* elevation
distribution. The q10/q50/q90 are then real **spatial** quantiles of the downscaled
climate within each cell (computed in the climate step), capturing horizontal
gradients (coast, rain shadow, latitude) the centroid-at-3-elevations method misses.

Emits ``subcell_centroids.csv``: one row per sub-point
``id, parent_id, row, col, long, lat, elev`` — ``parent_id`` is the 25 km cell id
(``row*W + col``, matching ``cell_centroids.csv``), ``id`` a unique sub-point id.
Gated by ``climate.subgrid`` in data_config; the climate step auto-generates it.
"""
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

from src.data.preprocess.elevation import dem_to_fine_grid

DEFAULT_GRID = 5   # grid x grid sub-points per model cell (5x5 = 25)


def build_subcell_centroids(dem_path, ref_transform, crs, H, W, grid=DEFAULT_GRID,
                            land_mask=None):
    """Sub-point table for a ``grid``x``grid`` mesh per model cell (NaN-elev dropped).

    Returns a dict of flat arrays ``id, parent_id, row, col, long, lat, elev``.
    ``parent_id = parent_row*W + parent_col`` matches ``cell_centroids.csv`` ids, so
    quantile aggregation in the climate step keys straight onto the model grid.

    ``land_mask`` (optional ``(H, W)`` boolean, True = land) drops every sub-point
    whose parent cell is not land. The finite-elevation filter alone is NOT enough:
    the DEM assigns ocean a finite value (0 / bathymetry), so ocean sub-points
    survive it, inflating the point count and producing tiles that climr's refmap
    can't cover (offshore -> "Empty tile - not enough data"). Masking by the model's
    ocean mask removes them at the source.
    """
    fine, g = dem_to_fine_grid(dem_path, ref_transform, crs, H, W, grid)  # (H*g, W*g)
    fine_transform = ref_transform * rasterio.Affine.scale(1.0 / g, 1.0 / g)
    fr, fc = np.meshgrid(np.arange(H * g), np.arange(W * g), indexing="ij")
    fr = fr.ravel(); fc = fc.ravel()
    elev = fine.ravel()
    xs = fine_transform.c + (fc + 0.5) * fine_transform.a + (fr + 0.5) * fine_transform.b
    ys = fine_transform.f + (fc + 0.5) * fine_transform.d + (fr + 0.5) * fine_transform.e
    lon, lat = warp_transform(crs, "EPSG:4326", xs, ys)
    p_row, p_col = fr // g, fc // g
    parent = p_row * W + p_col
    keep = np.isfinite(elev)                       # drop DEM nodata sub-points
    if land_mask is not None:                      # drop ocean sub-points (parent cell not land)
        if land_mask.shape != (H, W):
            raise ValueError(f"land_mask shape {land_mask.shape} != grid ({H}, {W})")
        keep &= land_mask[p_row, p_col]
    return {
        "id": np.arange(fr.size)[keep],
        "parent_id": parent[keep],
        "row": p_row[keep], "col": p_col[keep],
        "long": np.asarray(lon)[keep], "lat": np.asarray(lat)[keep],
        "elev": elev[keep],
    }


def write_csv(path, cols):
    """Write the sub-cell table to CSV (columns in climr-friendly order)."""
    import csv
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    order = ["id", "parent_id", "row", "col", "long", "lat", "elev"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(order)
        n = cols["id"].size
        for i in range(n):
            w.writerow([cols[k][i] for k in order])


def main():
    import argparse
    import glob
    import os

    from src.config_utils import load_data_config
    from src.data.preprocess.bbs import load_grid_reference

    cfg = load_data_config()
    ccfg = cfg.get("climate", {}).get("subgrid", {})
    dcfg = cfg.get("dem", {})

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dem", help="DEM GeoTIFF (default: the acquire/dem.py download).")
    ap.add_argument("--out", help="Default: {datasets_root}/elevation/subcell_centroids.csv")
    ap.add_argument("--mask", help="Ocean mask TIF (1=ocean,0=land). Default: config "
                                   "latent_cube.water_mask_path or land_mask/ocean_mask_25km.tif")
    ap.add_argument("--grid", type=int, default=int(ccfg.get("grid", DEFAULT_GRID)))
    args = ap.parse_args()

    dem_path = args.dem
    if not dem_path:
        dem_dir = os.path.join(cfg["datasets_root"], dcfg.get("out_subdir", "dem"))
        found = sorted(glob.glob(os.path.join(dem_dir, "*.tif")))
        if not found:
            raise SystemExit(f"no DEM in {dem_dir}; run scripts/download_dem.py or pass --dem")
        dem_path = found[0]
    out = args.out or os.path.join(cfg["datasets_root"], "elevation", "subcell_centroids.csv")

    # Model ocean mask (same grid): drop ocean sub-points at the source. Falls back
    # to elevation-only if the mask is absent (with a loud warning).
    mask_path = args.mask or cfg.get("latent_cube", {}).get("water_mask_path") \
        or os.path.join(cfg["datasets_root"], "land_mask", "ocean_mask_25km.tif")
    land_mask = None
    if os.path.exists(mask_path):
        land_mask, _, _, _, _, _ = load_grid_reference(mask_path)
    else:
        print(f"[subcell] WARNING: ocean mask not found at {mask_path}; keeping all "
              f"finite-elevation sub-points (ocean points will NaN out in climate).")

    with rasterio.open(cfg["grid"]["ref_raster"]) as ref:
        cols = build_subcell_centroids(dem_path, ref.transform, ref.crs, ref.height,
                                       ref.width, args.grid, land_mask=land_mask)
    write_csv(out, cols)
    n_parent = np.unique(cols["parent_id"]).size
    masked = " (land-masked)" if land_mask is not None else ""
    print(f"Wrote {cols['id'].size} sub-points ({args.grid}x{args.grid}/cell) over "
          f"{n_parent} cells{masked} -> {out}")


if __name__ == "__main__":
    main()
