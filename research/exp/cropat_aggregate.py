"""E047: aggregate CropAT-US annual 1 km cropland density to DESK's 27 km grid (area mean, nodata excluded).

    python research/exp/cropat_aggregate.py --src <dir of extracted CropAT-US GeoTIFFs> --grid <ref_grid_27km.tif>
        --out <cropat_27km.npz> [--pattern density] [--y0 1850 --y1 2021]

Finds one cropland-density GeoTIFF per year (file name containing --pattern and a 4-digit year), reprojects it onto
the reference grid with average resampling, and saves {years (Y,), crop (Y, H, W) float32 fraction in [0, 1], valid
share (Y, H, W) of the 27 km cell covered by valid 1 km pixels, files}. Values reported in percent are rescaled.
Reads the downloaded files only; run with python -I.
"""
import argparse
import os
import re
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src")
    ap.add_argument("--grid")
    ap.add_argument("--out")
    ap.add_argument("--pattern", default="density")
    ap.add_argument("--y0", type=int, default=1850)
    ap.add_argument("--y1", type=int, default=2021)
    a = ap.parse_args()
    import rasterio
    from rasterio.warp import Resampling, reproject
    g = rasterio.open(a.grid)
    H, W = g.height, g.width
    files = {}
    for root, _, fs in os.walk(a.src):
        for f in fs:
            if not f.lower().endswith((".tif", ".tiff")) or a.pattern.lower() not in f.lower():
                continue
            m = re.findall(r"(1[89]\d\d|20[0-2]\d)", f)
            if m:
                y = int(m[-1])
                if a.y0 <= y <= a.y1:
                    files[y] = os.path.join(root, f)
    if not files:
        sys.exit(f"no files matching {a.pattern!r} with a year under {a.src}")
    years = np.array(sorted(files))
    crop = np.full((len(years), H, W), np.nan, "float32")
    vshare = np.zeros((len(years), H, W), "float32")
    scale = None
    for i, y in enumerate(years):
        with rasterio.open(files[y]) as s:
            arr = s.read(1).astype("float32")
            nod = s.nodata
            valid = np.isfinite(arr) if nod is None else (arr != nod) & np.isfinite(arr)
            if scale is None:
                mx = float(np.nanmax(np.where(valid, arr, np.nan)))
                scale = 100.0 if mx > 1.5 else 1.0
                print(f"[cropat] {os.path.basename(files[y])}: crs {s.crs}, res {s.res}, max {mx:.3g} -> scale /{scale:g}",
                      flush=True)
            num = np.where(valid, arr / scale, 0.0).astype("float32")
            den = valid.astype("float32")
            dst_n = np.zeros((H, W), "float32")
            dst_d = np.zeros((H, W), "float32")
            for src_a, dst in ((num, dst_n), (den, dst_d)):
                reproject(src_a, dst, src_transform=s.transform, src_crs=s.crs, dst_transform=g.transform,
                          dst_crs=g.crs, resampling=Resampling.average)
        crop[i] = np.where(dst_d > 0.05, dst_n / np.maximum(dst_d, 1e-9), np.nan)
        vshare[i] = dst_d
    np.savez(a.out, years=years, crop=crop, valid_share=vshare, files=np.array([files[y] for y in years]))
    print(f"[cropat] {len(years)} years {years.min()}-{years.max()} -> {a.out}; CONUS mean crop fraction "
          f"{np.nanmean(crop[0]):.3f} ({years[0]}) -> {np.nanmean(crop[-1]):.3f} ({years[-1]})", flush=True)


if __name__ == "__main__":
    main()
