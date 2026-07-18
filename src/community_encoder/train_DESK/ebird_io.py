"""Torch-free loader for the eBird weekly community stack.

Kept separate from ``esk_kernel`` (which imports torch) so the CPU data-prep
stages — ``ebird_cache`` and the amplitude-modulation builder — can reproject the
eBird rasters without pulling in torch. ``esk_kernel`` re-exports
``load_tifs_structured`` for backward compatibility.
"""
import glob
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio

from src.processing import regrid


def load_tifs_structured(folder, pattern="*_abundance_median_*.tif", target_res_m=None):
    """
    Parses filenames to enforce strict (Species, Time) ordering.
    Returns: stack (H, W, S*T), meta dict.

    If ``target_res_m`` is given, each raster is reprojected onto the model
    reference grid (``grid.ref_raster``) as it loads, via ``reproject_match``
    with ``average`` -- the linear areal aggregate. Relative abundance
    aggregates linearly, so this is the correct order: aggregate abundance
    first, then build the (nonlinear) Ruzicka kernel + PCA at the target
    resolution -- giving Z directly at the model grid rather than mean-pooling
    the finished embedding downstream (meaningless for a kernel-PCA latent).

    Reproject (not integer ``block_reduce``) is used so this works for any
    native:target ratio -- eBird's ~2.96 km cells do not divide the 25 km grid
    evenly -- and so the eBird CRS (EPSG:8857) is resolved onto the Albers grid
    in the same step, going straight from finest native resolution to the model
    grid without an intermediate fixed-resolution reprojection.
    """
    import rioxarray  # noqa: F401  (registers .rio)
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise ValueError(f"No files found in {folder} matching {pattern}")

    regex = re.compile(r"([a-z0-9]+)_abundance_median_(\d{4}-\d{2}-\d{2})")
    records = []
    for fpath in files:
        fname = os.path.basename(fpath)
        match = regex.match(fname)
        if match:
            records.append({
                "species": match.group(1),
                "date": datetime.strptime(match.group(2), "%Y-%m-%d"),
                "path": fpath,
            })
        else:
            print(f"Skipping non-matching file: {fname}")

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No filenames matched regex.")

    n_species = df["species"].nunique()
    n_weeks = df["date"].nunique()
    if len(df) != n_species * n_weeks:
        raise ValueError(f"Grid incomplete. Expected {n_species * n_weeks} files, found {len(df)}.")

    df_sorted = df.sort_values(by=["species", "date"])
    ordered_paths = df_sorted["path"].tolist()

    with rasterio.open(ordered_paths[0]) as src:
        H, W = src.shape
        native_res = abs(src.transform.a)

    ref = None
    if target_res_m is not None:
        ref = regrid.load_ref()
        H, W = int(ref.rio.height), int(ref.rio.width)
        print(f"Reprojecting {len(ordered_paths)} rasters {native_res:.0f}m -> "
              f"model grid {H}x{W} @ {target_res_m}m as they load "
              f"({n_species} sp x {n_weeks} wks)...")
    else:
        print(f"Loading {len(ordered_paths)} rasters at native {native_res:.0f}m "
              f"({n_species} sp x {n_weeks} wks)...")

    full_stack = np.zeros((H, W, len(ordered_paths)), dtype=np.float32)

    for i, p in enumerate(ordered_paths):
        if ref is not None:
            da = rioxarray.open_rasterio(p, masked=True).squeeze()
            band = regrid.reproject_to_ref(da, ref, resampling="average").values
        else:
            with rasterio.open(p) as src:
                band = src.read(1).astype(np.float64)
                if src.nodata is not None:
                    band[band == src.nodata] = np.nan
        full_stack[:, :, i] = band.astype(np.float32)

    # Species block order = sorted unique species (matches df_sorted column blocks),
    # so downstream code (e.g. the BBS amplitude modulation) can align per-species
    # scalars to the right 52-week block.
    species = sorted(df["species"].unique().tolist())
    return full_stack, {"n_species": n_species, "n_weeks": n_weeks,
                        "native_res_m": native_res, "target_res_m": target_res_m,
                        "species": species}
