"""Build the model-grid reference raster (grid geometry only, not a data product).

The reference raster defines the CRS / transform / extent that every product is
reprojected onto (``grid.ref_raster``). It is pure geometry: a single band of
zeros at ``grid.target_res_m`` over the study-area bounding box (the fixed
North-American extent the project keeps). The extent is taken from a base
raster's bounds (any raster covering the box — the study area is unchanged) or
given explicitly; the reference is independent of whatever product supplied the
bounds.
"""
import argparse

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.config_utils import load_data_config


def build_ref_grid(bounds, crs, res_m, out_path):
    """Write a single-band zero raster at ``res_m`` covering ``bounds`` in ``crs``.

    ``bounds`` = (left, bottom, right, top). Cell count is ceil'd to cover the
    box; the transform originates at (left, top).
    """
    left, bottom, right, top = bounds
    ncols = int(np.ceil((right - left) / res_m))
    nrows = int(np.ceil((top - bottom) / res_m))
    transform = from_origin(left, top, res_m, res_m)
    profile = dict(driver="GTiff", height=nrows, width=ncols, count=1,
                   dtype="uint8", crs=crs, transform=transform, nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.zeros((nrows, ncols), dtype="uint8"), 1)
    print(f"Ref grid: {nrows}x{ncols} @ {res_m} m, {crs} -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-raster", help="Raster whose bounds/CRS define the study-area box.")
    ap.add_argument("--bounds", nargs=4, type=float, metavar=("L", "B", "R", "T"),
                    help="Explicit bounds (with --crs) instead of --base-raster.")
    ap.add_argument("--crs", help="CRS for --bounds (e.g. ESRI:102039).")
    ap.add_argument("--out", help="Output ref raster (default: grid.ref_raster).")
    args = ap.parse_args()

    cfg = load_data_config()
    res_m = cfg["grid"]["target_res_m"]
    out = args.out or cfg["grid"]["ref_raster"]

    if args.base_raster:
        with rasterio.open(args.base_raster) as src:
            bounds, crs = tuple(src.bounds), src.crs
    elif args.bounds and args.crs:
        bounds, crs = tuple(args.bounds), rasterio.crs.CRS.from_string(args.crs)
    elif cfg["grid"].get("box_bounds") and cfg["grid"].get("box_crs"):
        # Default: the fixed study-area box from config -> builds standalone.
        bounds = tuple(cfg["grid"]["box_bounds"])
        crs = rasterio.crs.CRS.from_string(cfg["grid"]["box_crs"])
    else:
        raise SystemExit("provide --base-raster, or --bounds with --crs, or set grid.box_bounds/box_crs")
    build_ref_grid(bounds, crs, res_m, out)


if __name__ == "__main__":
    main()
