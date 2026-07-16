#!/usr/bin/env python3

"""
Reproject weekly eBird relative abundance GeoTIFFs onto the model reference grid.

For each input .tif file:
    - Read the single-band abundance raster
    - Enforce CRS = EPSG:8857 (from file metadata you provided)
    - Reproject onto the model grid (``grid.ref_raster``) via reproject_match
      with ``average`` -- the linear areal aggregate straight from eBird's
      ~2.96 km native cells to the model grid (any ratio, CRS resolved too)
    - Apply the model-grid ocean mask (1=ocean, 0=land)
    - Save a single-band GeoTIFF aligned with the model grid
    - Save a PNG quick-look
"""

import os
import glob
import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio
import matplotlib.pyplot as plt

from src.config_utils import load_data_config
from src.processing import regrid
_CFG = load_data_config()
_DR = _CFG["datasets_root"]

# Paths (config-driven)
# Input is the raw eBird dir the downloader writes to; output is the same name
# with a "_grid" suffix (the reprojected model grid the encoder's ebird_folder
# points at). Single source of truth via data_config's ebird_raw_subdir + grid.
_EBIRD_SUBDIR = _CFG.get("ebird_raw_subdir", "ebird_weekly_2023")
_RES_KM = _CFG["grid"]["target_res_m"] // 1000
EBIRD_DIR = f"{_DR}/{_EBIRD_SUBDIR}"
OUT_DIR = f"{_DR}/{_EBIRD_SUBDIR}_grid"
OCEAN_MASK = f"{_DR}/land_mask/ocean_mask_{_RES_KM}km.tif"

PNG_POWER = 0.25

# CRS for eBird rasters (from file metadata)
EBIRD_CRS = "EPSG:8857"


def save_png(array, out_path, cmap="viridis"):
    """Save a PNG using a global monotone power transform."""
    arr = array.astype(float)
    arr[arr < 0] = 0
    vis = np.power(arr, PNG_POWER)
    vis[np.isnan(vis)] = 0
    plt.imsave(out_path, vis, cmap=cmap)


def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load the model reference grid (CRS/transform/extent at grid.target_res_m)
    ref = regrid.load_ref(_CFG)

    # Load the model-grid ocean mask (1=ocean, 0=land)
    with rasterio.open(OCEAN_MASK) as src:
        ocean_mask = src.read(1)

    # Process all eBird .tif files
    tif_files = sorted(glob.glob(os.path.join(EBIRD_DIR, "*.tif")))

    for tif_path in tif_files:
        fname = os.path.basename(tif_path)
        out_tif = os.path.join(OUT_DIR, fname.replace(".tif", "_grid.tif"))
        out_png = os.path.join(OUT_DIR, fname.replace(".tif", "_grid.png"))

        if os.path.exists(out_tif):
            print(f"Skipping existing: {out_tif}")
            continue

        print(f"Processing {fname}")

        # Load eBird raster
        da = rxr.open_rasterio(tif_path, masked=True)

        # Validate, then enforce, the assumed CRS. eBird S&T weekly rasters are
        # EPSG:8857; if a raster declares a *different* CRS, that's a data-format
        # surprise we want to fail on rather than silently overwrite.
        existing = da.rio.crs
        if existing is not None and existing.to_epsg() != 8857:
            raise ValueError(
                f"{os.path.basename(tif_path)} declares CRS {existing} != assumed "
                f"{EBIRD_CRS}; verify the eBird product before overwriting.")
        da = da.rio.write_crs(EBIRD_CRS, inplace=False)

        # Ensure nodata is nan
        da = da.rio.write_nodata(float("nan"), inplace=False)

        # Reproject onto the model grid (any native:target ratio; average is the
        # linear areal aggregate -- correct for relative abundance).
        da_reproj = regrid.reproject_to_ref(da, ref, resampling="average")

        # Apply ocean mask
        if da_reproj.shape[-2:] != ocean_mask.shape:
            raise ValueError(
                f"Shape mismatch after reprojection for {tif_path}: "
                f"da_reproj.shape={da_reproj.shape}, "
                f"ocean_mask.shape={ocean_mask.shape}"
            )

        data = da_reproj.values.astype("float32")
        data[0][ocean_mask == 1] = np.nan
        da_reproj = da_reproj.copy(data=data)

        # Save GeoTIFF
        da_reproj.rio.to_raster(out_tif)

        # Save PNG quick-look
        save_png(data[0], out_png)

        print(f"Saved → {out_tif}")
        print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()