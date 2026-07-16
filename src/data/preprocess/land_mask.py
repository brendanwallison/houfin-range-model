"""De-dilated coastline: land-fraction land mask + gated observation snapping.

Replaces the old rule (ocean iff *all* BUI quantile bands were NaN), which made
any model-grid cell containing even one land subpixel "land" — dilating the
coast. Here a cell is land iff at least ``tau`` of its finest-resolution
subpixels are land, computed from a **continental** land/water source so
Canadian/Mexican land inside the bounding box is real land (not nodata).

De-dilating can strand coastal observations on newly-ocean cells, so
observations are **snapped** to the nearest land cell within a small radius;
anything farther (genuinely offshore, e.g. a pelagic eBird smear) is dropped
rather than pulled in. Snapping reuses ``scipy.ndimage.distance_transform_edt``,
the same primitive the cube gap-fill uses.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.processing import regrid

DEFAULT_TAU = 0.5
DEFAULT_FINE_FACTOR = 25  # ref cells subdivided this many times to estimate fraction


def compute_land_fraction(fine_land, block):
    """Per target cell, the fraction (0..1) of finest-res subpixels that are land.

    ``fine_land`` is a binary array (1 = land, 0 = water) at the fine source
    resolution; ``block`` = fine cells per target cell. A block-mean of the
    binary field is exactly the land fraction.
    """
    return regrid.block_reduce(np.asarray(fine_land, dtype=float), block, how="mean")


def land_mask_from_fraction(frac, tau=0.5):
    """Boolean model-grid land mask: a cell is land iff land_fraction >= tau."""
    return np.asarray(frac) >= tau


def snap_to_nearest_land(rows, cols, land_mask, max_cells=1):
    """Snap observation cells to the nearest land cell, within ``max_cells``.

    Returns ``(snapped_rows, snapped_cols, keep)``. Points already on land are
    unchanged. Ocean points within ``max_cells`` of land snap to that nearest
    land cell; ocean points farther than ``max_cells`` are marked ``keep=False``
    (genuinely offshore — drop, don't invent land for them).
    """
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    land_mask = np.asarray(land_mask, dtype=bool)
    # Distance (in cells) from every cell to the nearest land cell, plus the
    # index of that nearest land cell.
    dist, (iy, ix) = distance_transform_edt(~land_mask, return_indices=True)
    on_land = land_mask[rows, cols]
    d = dist[rows, cols]
    keep = on_land | (d <= max_cells)
    out_rows = np.where(on_land, rows, iy[rows, cols])
    out_cols = np.where(on_land, cols, ix[rows, cols])
    return out_rows, out_cols, keep


def build_land_mask(land_source, ref_path, out_path, tau=DEFAULT_TAU,
                    fine_factor=DEFAULT_FINE_FACTOR):
    """Rasterize a continental land polygon → per-cell land fraction → land mask.

    Rasterizes ``land_source`` (a land/water polygon, e.g. Natural Earth) onto a
    grid ``fine_factor``× finer than the model ref grid, block-averages to the
    land fraction per model cell, and thresholds at ``tau``. Writes the mask in
    the project convention (1 = ocean, 0 = land) aligned to the ref grid. Only
    the fine binary grid is held in RAM (uint8), so this is light.
    """
    import geopandas as gpd
    import rasterio
    import rasterio.features
    from rasterio.transform import Affine

    with rasterio.open(ref_path) as ref:
        crs, transform = ref.crs, ref.transform
        H, W = ref.height, ref.width
        profile = ref.profile

    ff = int(fine_factor)
    fine_transform = transform * Affine.scale(1.0 / ff, 1.0 / ff)
    gdf = gpd.read_file(land_source).to_crs(crs)
    fine_land = rasterio.features.rasterize(
        ((geom, 1) for geom in gdf.geometry),
        out_shape=(H * ff, W * ff), transform=fine_transform, fill=0, dtype="uint8",
    )
    frac = compute_land_fraction(fine_land, ff)
    ocean = (~land_mask_from_fraction(frac, tau)).astype("uint8")  # 1=ocean, 0=land

    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(ocean, 1)
    n_land = int((ocean == 0).sum())
    print(f"Land mask: {n_land}/{ocean.size} land cells (tau={tau}, fine={ff}x) -> {out_path}")
    return out_path


def main():
    import argparse
    import os

    from src.config_utils import load_data_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--land-source", help="Land polygon (default: coastline.land_source).")
    ap.add_argument("--ref", help="Ref grid raster (default: grid.ref_raster).")
    ap.add_argument("--out", help="Output mask (default: {datasets_root}/land_mask/ocean_mask_{res}km.tif).")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--fine-factor", type=int)
    args = ap.parse_args()

    cfg = load_data_config()
    ccfg = cfg.get("coastline", {})
    res_km = cfg["grid"]["target_res_m"] // 1000
    land_source = args.land_source or ccfg.get("land_source")
    if not land_source:
        raise SystemExit("no land source (set coastline.land_source or --land-source)")
    if not os.path.isabs(land_source):  # relative paths resolve under datasets_root
        land_source = os.path.join(cfg["datasets_root"], land_source)
    ref = args.ref or cfg["grid"]["ref_raster"]
    out = args.out or os.path.join(cfg["datasets_root"], "land_mask", f"ocean_mask_{res_km}km.tif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    build_land_mask(land_source, ref, out,
                    tau=args.tau or ccfg.get("land_fraction_tau", DEFAULT_TAU),
                    fine_factor=args.fine_factor or ccfg.get("fine_factor", DEFAULT_FINE_FACTOR))


if __name__ == "__main__":
    main()
