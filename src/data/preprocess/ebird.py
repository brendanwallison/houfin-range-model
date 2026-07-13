#!/usr/bin/env python3

"""
Reproject weekly eBird relative abundance GeoTIFFs onto the canonical 4 km BUI grid.

For each input .tif file:
    - Read the single-band abundance raster
    - Enforce CRS = EPSG:8857 (from file metadata you provided)
    - Reproject to the BUI 4 km grid using rioxarray.reproject_match
    - Apply the 4 km ocean mask (1=ocean, 0=land)
    - Save a single-band GeoTIFF aligned with BUI
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
_CFG = load_data_config()
_DR = _CFG["datasets_root"]

# -----------------------------
# Paths (config-driven)
# -----------------------------
# Input is the raw eBird dir the downloader writes to; output is the same name
# with an "_albers" suffix (the reprojected grid the encoder's ebird_folder
# points at). Single source of truth via data_config's ebird_raw_subdir.
_EBIRD_SUBDIR = _CFG.get("ebird_raw_subdir", "ebird_weekly_2023")
EBIRD_DIR = f"{_DR}/{_EBIRD_SUBDIR}"
OUT_DIR = f"{_DR}/{_EBIRD_SUBDIR}_albers"
BUI_REF = f"{_DR}/HBUI/BUI/2020_BUI_4km.tif"
OCEAN_MASK = f"{_DR}/land_mask/ocean_mask_4km.tif"

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

    # -----------------------------
    # Load reference BUI grid
    # -----------------------------
    bui_ref = rxr.open_rasterio(BUI_REF)

    # -----------------------------
    # Load 4 km ocean mask
    # -----------------------------
    with rasterio.open(OCEAN_MASK) as src:
        ocean_mask = src.read(1)

    # -----------------------------
    # Process all eBird .tif files
    # -----------------------------
    tif_files = sorted(glob.glob(os.path.join(EBIRD_DIR, "*.tif")))

    for tif_path in tif_files:
        fname = os.path.basename(tif_path)
        out_tif = os.path.join(OUT_DIR, fname.replace(".tif", "_bui4km.tif"))
        out_png = os.path.join(OUT_DIR, fname.replace(".tif", "_bui4km.png"))

        if os.path.exists(out_tif):
            print(f"Skipping existing: {out_tif}")
            continue

        print(f"Processing {fname}")

        # -----------------------------
        # Load eBird raster
        # -----------------------------
        da = rxr.open_rasterio(tif_path, masked=True)

        # Enforce CRS = EPSG:8857
        da = da.rio.write_crs(EBIRD_CRS, inplace=False)

        # Ensure nodata is nan
        da = da.rio.write_nodata(float("nan"), inplace=False)

        # -----------------------------
        # Reproject to BUI 4 km grid
        # -----------------------------
        da_reproj = da.rio.reproject_match(
            bui_ref,
            resampling="bilinear",
            nodata=float("nan"),
        )

        # -----------------------------
        # Apply ocean mask
        # -----------------------------
        if da_reproj.shape[-2:] != ocean_mask.shape:
            raise ValueError(
                f"Shape mismatch after reprojection for {tif_path}: "
                f"da_reproj.shape={da_reproj.shape}, "
                f"ocean_mask.shape={ocean_mask.shape}"
            )

        data = da_reproj.values.astype("float32")
        data[0][ocean_mask == 1] = np.nan
        da_reproj = da_reproj.copy(data=data)

        # -----------------------------
        # Save GeoTIFF
        # -----------------------------
        da_reproj.rio.to_raster(out_tif)

        # -----------------------------
        # Save PNG quick-look
        # -----------------------------
        save_png(data[0], out_png)

        print(f"Saved → {out_tif}")
        print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()