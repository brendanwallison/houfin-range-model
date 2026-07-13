#!/usr/bin/env python3

"""
Create the model-grid ocean mask from the aggregated BUI quantile raster.

Convention:
    1 = ocean
    0 = land

Logic:
    A model-grid cell is ocean if all BUI quantile bands are NaN -- i.e. it
    contained no land (250 m) subpixels. This matches the "any-land -> land"
    convention used elsewhere. Resolution follows grid.target_res_m, so at
    16 km it reads 2020_BUI_16km.tif and writes ocean_mask_16km.tif.

Also generates a PNG quick-look using a categorical colormap.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt

from src.config_utils import load_data_config
_CFG = load_data_config()
_DR = _CFG["datasets_root"]
_RES_KM = _CFG["grid"]["target_res_m"] // 1000

# Aggregated BUI quantile raster at the model resolution (from preprocess/bui.py)
BUI_PATH = f"{_DR}/HBUI/2020_BUI_{_RES_KM}km.tif"

# Output mask + PNG quick-look
OUT_MASK = f"{_DR}/land_mask/ocean_mask_{_RES_KM}km.tif"
OUT_PNG = f"{_DR}/land_mask/ocean_mask_{_RES_KM}km.png"


def save_mask_png(mask, out_path):
    """
    Save a PNG for a binary mask using a categorical colormap.

    - 0 = land (light gray)
    - 1 = ocean (blue)
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap="Blues")  # ocean = blue, land = white
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    # Load the aggregated BUI quantile raster at model resolution
    with rasterio.open(BUI_PATH) as src:
        arr = src.read()  # shape: (7, H, W)
        profile = src.profile

    # Ocean = all bands NaN
    ocean = np.all(np.isnan(arr), axis=0).astype("uint8")

    # Update profile for single-band mask
    profile.update(count=1, dtype="uint8")

    # Save mask
    with rasterio.open(OUT_MASK, "w", **profile) as dst:
        dst.write(ocean, 1)

    print(f"Saved 4 km ocean mask → {OUT_MASK}")

    # Save PNG quick-look
    save_mask_png(ocean, OUT_PNG)
    print(f"Saved PNG quick-look → {OUT_PNG}")


if __name__ == "__main__":
    main()