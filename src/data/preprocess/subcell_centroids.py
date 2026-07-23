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


def rasterize_land_fine(land_source, crs, ref_transform, H, W, grid=DEFAULT_GRID,
                        lake_source=None, exclusion_source=None, exclude_iso_a3=()):
    """Binary (1=land, 0=water) grid at the SUB-POINT resolution ``(H*grid, W*grid)``,
    rasterized from the same land polygon the 25 km ocean mask uses (Natural Earth).

    This is the *high-resolution* land test the 25 km parent mask can't do: it drops
    water sub-points INSIDE a coastal cell that the 25 km mask calls land -- exactly
    the within-cell coastal structure the subgrid method exists to resolve.
    """
    import geopandas as gpd
    import rasterio.features
    fine_transform = ref_transform * rasterio.Affine.scale(1.0 / grid, 1.0 / grid)
    gdf = gpd.read_file(land_source).to_crs(crs)
    fine_land = rasterio.features.rasterize(
        ((geom, 1) for geom in gdf.geometry),
        out_shape=(H * grid, W * grid), transform=fine_transform, fill=0, dtype="uint8")
    if lake_source:
        lakes = gpd.read_file(lake_source).to_crs(crs)
        fine_lakes = rasterio.features.rasterize(
            ((geom, 1) for geom in lakes.geometry),
            out_shape=(H * grid, W * grid), transform=fine_transform, fill=0, dtype="uint8")
        fine_land[fine_lakes > 0] = 0
    if exclusion_source and exclude_iso_a3:
        from src.data.preprocess.land_mask import rasterize_country_exclusions
        fine_excluded = rasterize_country_exclusions(
            exclusion_source, exclude_iso_a3, crs, fine_land.shape, fine_transform)
        fine_land[fine_excluded > 0] = 0
    return fine_land


def build_subcell_centroids(dem_path, ref_transform, crs, H, W, grid=DEFAULT_GRID,
                            land_mask=None, fine_land=None):
    """Sub-point table for a ``grid``x``grid`` mesh per model cell (NaN-elev dropped).

    Returns a dict of flat arrays ``id, parent_id, row, col, long, lat, elev``.
    ``parent_id = parent_row*W + parent_col`` matches ``cell_centroids.csv`` ids, so
    quantile aggregation in the climate step keys straight onto the model grid.

    Two optional ocean filters (the finite-elevation filter alone is NOT enough --
    the DEM assigns ocean a finite value (0 / bathymetry), so ocean sub-points
    survive it, inflating the point count and producing tiles climr's refmap can't
    cover -> offshore "Empty tile - not enough data"):
    - ``land_mask`` (``(H, W)`` bool, True = land): drops sub-points whose PARENT
      25 km cell is not land -> aligns the point set to the modeled grid (removes
      fully-offshore cells).
    - ``fine_land`` (``(H*grid, W*grid)`` 0/1, from :func:`rasterize_land_fine`):
      drops each sub-point whose OWN fine location is water -> removes coastal
      water sub-points a 'land' 25 km cell would otherwise keep.
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
    if land_mask is not None:                      # drop sub-points in non-land 25 km cells
        if land_mask.shape != (H, W):
            raise ValueError(f"land_mask shape {land_mask.shape} != grid ({H}, {W})")
        keep &= land_mask[p_row, p_col]
    if fine_land is not None:                      # drop water sub-points at fine resolution
        if fine_land.shape != (H * g, W * g):
            raise ValueError(f"fine_land shape {fine_land.shape} != fine grid ({H * g}, {W * g})")
        keep &= (fine_land[fr, fc] > 0)
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
    ap.add_argument("--mask", help="Parent ocean mask TIF (1=ocean,0=land). Default: config "
                                   "latent_cube.water_mask_path or land_mask/ocean_mask_{res}km.tif")
    ap.add_argument("--land-source", dest="land_source",
                    help="Land polygon for the fine sub-point mask (default: coastline.land_source)")
    ap.add_argument("--lake-source", dest="lake_source",
                    help="Polygonal lakes removed from fine land (default: coastline.lake_source)")
    ap.add_argument("--exclusion-source", help="Admin-0 polygons for configured study exclusions")
    ap.add_argument("--exclude-iso-a3", help="Comma-separated ISO-A3 study exclusions")
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

    # Ocean filters: (1) 25 km parent mask aligns to the modeled grid; (2) fine land
    # mask (same Natural Earth polygon as the 25 km mask, rasterized at the sub-point
    # grid) drops coastal water sub-points. Both fall back gracefully if absent.
    res_km = cfg["grid"]["target_res_m"] // 1000
    mask_path = args.mask or cfg.get("latent_cube", {}).get("water_mask_path") \
        or os.path.join(cfg["datasets_root"], "land_mask", f"ocean_mask_{res_km}km.tif")
    land_mask = None
    if os.path.exists(mask_path):
        land_mask, _, _, _, _, _ = load_grid_reference(mask_path)
    else:
        print(f"[subcell] WARNING: parent ocean mask not found at {mask_path}.")

    land_source = args.land_source or cfg.get("coastline", {}).get("land_source")
    if land_source and not os.path.isabs(land_source):
        land_source = os.path.join(cfg["datasets_root"], land_source)
    lake_source = args.lake_source if args.lake_source is not None else cfg.get("coastline", {}).get("lake_source")
    if lake_source and not os.path.isabs(lake_source):
        lake_source = os.path.join(cfg["datasets_root"], lake_source)
    exclusion_source = args.exclusion_source if args.exclusion_source is not None else cfg.get("coastline", {}).get("study_exclusion_source")
    if exclusion_source and not os.path.isabs(exclusion_source):
        exclusion_source = os.path.join(cfg["datasets_root"], exclusion_source)
    exclude_iso_a3 = ([x.strip().upper() for x in args.exclude_iso_a3.split(",") if x.strip()]
                      if args.exclude_iso_a3 is not None
                      else list(cfg.get("coastline", {}).get("study_exclude_iso_a3", [])))

    with rasterio.open(cfg["grid"]["ref_raster"]) as ref:
        fine_land = None
        if land_source and os.path.exists(land_source):
            fine_land = rasterize_land_fine(land_source, ref.crs, ref.transform,
                                            ref.height, ref.width, args.grid,
                                            lake_source=lake_source,
                                            exclusion_source=exclusion_source,
                                            exclude_iso_a3=exclude_iso_a3)
        else:
            print(f"[subcell] WARNING: land polygon not found ({land_source}); "
                  f"no fine sub-point ocean mask (coastal water points will NaN out).")
        cols = build_subcell_centroids(dem_path, ref.transform, ref.crs, ref.height,
                                       ref.width, args.grid, land_mask=land_mask,
                                       fine_land=fine_land)
    write_csv(out, cols)
    n_parent = np.unique(cols["parent_id"]).size
    tags = ",".join(t for t, on in [("25km", land_mask is not None),
                                     ("fine", fine_land is not None)] if on)
    masked = f" (land-masked: {tags})" if tags else ""
    print(f"Wrote {cols['id'].size} sub-points ({args.grid}x{args.grid}/cell) over "
          f"{n_parent} cells{masked} -> {out}")


if __name__ == "__main__":
    main()
