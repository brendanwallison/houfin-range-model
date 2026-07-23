"""Terrestrial mask: land minus polygonal inland water, plus observation snapping.

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


def subtract_lakes(fine_land, fine_lakes):
    """Remove polygonal lake water from a fine terrestrial mask.

    The lake input is deliberately polygonal, not river-line hydrography. Its
    effect is area-weighted at model resolution, so narrow features cannot turn
    into artificially wide water corridors.
    """
    land = np.asarray(fine_land, dtype="uint8")
    lakes = np.asarray(fine_lakes, dtype=bool)
    if land.shape != lakes.shape:
        raise ValueError(f"fine land shape {land.shape} != fine lakes shape {lakes.shape}")
    return np.where(lakes, 0, land).astype("uint8", copy=False)


def rasterize_country_exclusions(source, iso_a3, crs, shape, transform):
    """Rasterize selected Natural Earth admin-0 polygons as model exclusions."""
    import geopandas as gpd
    import rasterio.features

    codes = {str(x).upper() for x in iso_a3}
    gdf = gpd.read_file(source).to_crs(crs)
    code_col = next((c for c in ("ADM0_A3", "ISO_A3", "SOV_A3") if c in gdf.columns), None)
    if code_col is None:
        raise ValueError(f"{source} has no recognized ISO-A3 column")
    selected = gdf[gdf[code_col].astype(str).str.upper().isin(codes)]
    missing = codes - set(selected[code_col].astype(str).str.upper())
    if missing:
        raise ValueError(f"{source} has no polygons for requested ISO-A3 codes: {sorted(missing)}")
    return rasterio.features.rasterize(
        ((geom, 1) for geom in selected.geometry), out_shape=shape,
        transform=transform, fill=0, dtype="uint8")


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
                    fine_factor=DEFAULT_FINE_FACTOR, lake_source=None,
                    exclusion_source=None, exclude_iso_a3=()):
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
    if lake_source:
        lakes = gpd.read_file(lake_source).to_crs(crs)
        fine_lakes = rasterio.features.rasterize(
            ((geom, 1) for geom in lakes.geometry),
            out_shape=(H * ff, W * ff), transform=fine_transform, fill=0, dtype="uint8",
        )
        fine_land = subtract_lakes(fine_land, fine_lakes)
    if exclusion_source and exclude_iso_a3:
        fine_excluded = rasterize_country_exclusions(
            exclusion_source, exclude_iso_a3, crs, fine_land.shape, fine_transform)
        fine_land[fine_excluded > 0] = 0
    frac = compute_land_fraction(fine_land, ff)
    ocean = (~land_mask_from_fraction(frac, tau)).astype("uint8")  # 1=ocean, 0=land

    # 0 is a real semantic value (land), so it must never also be nodata.
    profile.update(count=1, dtype="uint8", nodata=255)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(ocean, 1)
    n_land = int((ocean == 0).sum())
    lake_note = f", lakes={lake_source}" if lake_source else ""
    exclude_note = f", excludes={','.join(exclude_iso_a3)}" if exclude_iso_a3 else ""
    print(f"Land mask: {n_land}/{ocean.size} terrestrial cells (tau={tau}, fine={ff}x{lake_note}{exclude_note}) -> {out_path}")
    return out_path


def main():
    import argparse
    import os

    from src.config_utils import load_data_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--land-source", help="Land polygon (default: coastline.land_source).")
    ap.add_argument("--lake-source", help="Polygonal inland-water layer (default: coastline.lake_source).")
    ap.add_argument("--exclusion-source", help="Admin-0 polygons used for study-domain exclusions.")
    ap.add_argument("--exclude-iso-a3", help="Comma-separated ISO-A3 territories excluded from this model mask.")
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
    lake_source = args.lake_source if args.lake_source is not None else ccfg.get("lake_source")
    if lake_source and not os.path.isabs(lake_source):
        lake_source = os.path.join(cfg["datasets_root"], lake_source)
    if lake_source and not os.path.exists(lake_source):
        raise SystemExit(
            f"lake polygon not found: {lake_source}. Run scripts/tacc/download_all.sh "
            "or pass --lake-source; refusing to build a coastline-only terrestrial mask.")
    exclusion_source = args.exclusion_source if args.exclusion_source is not None else ccfg.get("study_exclusion_source")
    if exclusion_source and not os.path.isabs(exclusion_source):
        exclusion_source = os.path.join(cfg["datasets_root"], exclusion_source)
    exclude_iso_a3 = ([x.strip().upper() for x in args.exclude_iso_a3.split(",") if x.strip()]
                      if args.exclude_iso_a3 is not None
                      else list(ccfg.get("study_exclude_iso_a3", [])))
    if exclude_iso_a3 and (not exclusion_source or not os.path.exists(exclusion_source)):
        raise SystemExit(f"study exclusion polygons not found: {exclusion_source}")
    ref = args.ref or cfg["grid"]["ref_raster"]
    out = args.out or os.path.join(cfg["datasets_root"], "land_mask", f"ocean_mask_{res_km}km.tif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    build_land_mask(land_source, ref, out,
                    tau=args.tau or ccfg.get("land_fraction_tau", DEFAULT_TAU),
                    fine_factor=args.fine_factor or ccfg.get("fine_factor", DEFAULT_FINE_FACTOR),
                    lake_source=lake_source, exclusion_source=exclusion_source,
                    exclude_iso_a3=exclude_iso_a3)


if __name__ == "__main__":
    main()
