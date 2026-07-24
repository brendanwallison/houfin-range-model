"""Reproject the aggregated SoilGrids 5000 m tiles onto the model grid (static).

SoilGrids COGs are in interrupted Goode Homolosine (ESRI:54052), so they must be
reprojected — not block-aggregated — to the model Albers grid; `reproject_match`
with ``average`` is the linear areal aggregate (5 km → ~25 km). Soil is
time-invariant, so this runs once, one output raster per property×depth.
"""
import glob
import os

import numpy as np
import rioxarray  # noqa: F401  (registers .rio)
from scipy.ndimage import distance_transform_edt

from src.config_utils import load_data_config
from src.data.masks import read_land_mask
from src.processing import regrid


def fill_terrestrial_nodata(arr, land_mask):
    """Fill static-soil nodata from the nearest valid *terrestrial* neighbour.

    Reprojection can leave a few mixed coastal cells without a SoilGrids source
    footprint.  These are covariate-source holes, not habitat absence: fill them
    once here, before state construction, and never let them reach DESK's generic
    Z fallback. Ocean/off-domain cells remain NaN and never donate values.
    """
    x = np.asarray(arr, dtype="float32").copy()
    land = np.asarray(land_mask, dtype=bool)
    if x.shape != land.shape:
        raise ValueError(f"soil array {x.shape} != terrestrial mask {land.shape}")
    target = land & ~np.isfinite(x)
    if not target.any():
        return x, 0
    valid = land & np.isfinite(x)
    if not valid.any():
        raise ValueError("cannot fill terrestrial SoilGrids nodata: no valid terrestrial source cells")
    _, nearest = distance_transform_edt(~valid, return_indices=True)
    x[target] = x[tuple(nearest[:, target])]
    return x, int(target.sum())


def preprocess(in_dir, out_dir, ref, land_mask=None):
    """Reproject every ``*_mean_5000.tif`` in ``in_dir`` onto ``ref``. Returns count."""
    os.makedirs(out_dir, exist_ok=True)
    tiles = sorted(glob.glob(os.path.join(in_dir, "*_mean_5000.tif")))
    for tif in tiles:
        da = rioxarray.open_rasterio(tif, masked=True)
        # SoilGrids COGs are interrupted Goode Homolosine (ESRI:54052). Assert it
        # so a source in a different projection fails loudly rather than being
        # silently misaligned by reproject_match (which trusts the source CRS).
        crs = da.rio.crs
        wkt = crs.to_wkt() if crs else ""
        if "Homolosine" not in wkt and (crs is None or crs.to_epsg() != 54052):
            raise ValueError(
                f"{os.path.basename(tif)} CRS {crs} is not the expected SoilGrids "
                f"Goode Homolosine (ESRI:54052); verify the product before reprojecting.")
        out = regrid.reproject_to_ref(da, ref, resampling="average")
        if land_mask is not None:
            # SoilGrids input is single-band; retain the rioxarray metadata while
            # replacing only the terrestrial nodata values in its pixel array.
            values = np.asarray(out.values)
            if values.ndim == 3 and values.shape[0] == 1:
                filled, n = fill_terrestrial_nodata(values[0], land_mask)
                out.values[0] = filled
            elif values.ndim == 2:
                filled, n = fill_terrestrial_nodata(values, land_mask)
                out.values[...] = filled
            else:
                raise ValueError(f"expected one SoilGrids band, got {values.shape} for {tif}")
            if n:
                print(f"[soilgrids] filled {n} terrestrial coastal nodata cells: {os.path.basename(tif)}")
        out.rio.to_raster(os.path.join(out_dir, os.path.basename(tif).replace("_5000.tif", "_grid.tif")))
    print(f"SoilGrids: reprojected {len(tiles)} tiles -> {out_dir}")
    return len(tiles)


def main():
    cfg = load_data_config()
    dr = cfg["datasets_root"]
    scfg = cfg.get("soilgrids", {})
    in_dir = os.path.join(dr, scfg.get("out_subdir", "soilgrids_5000m"))
    out_dir = os.path.join(dr, "soilgrids_grid")
    res_km = cfg["grid"]["target_res_m"] // 1000
    mask_path = os.path.join(dr, "land_mask", f"ocean_mask_{res_km}km.tif")
    preprocess(in_dir, out_dir, regrid.load_ref(cfg), land_mask=read_land_mask(mask_path))


if __name__ == "__main__":
    main()
