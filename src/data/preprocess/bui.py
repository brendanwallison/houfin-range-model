"""Aggregate the 250 m HISDAC-US BUI series to the model grid — resolution-deferral aware.

Rewrite of the old ``aggregate_and_interpolate_bui.py``. What changed and why:

Old (broken) order                         New (this module)
------------------------------------------  ------------------------------------------
per-cell quantiles from 250 m  [OK]          same, but vectorized via regrid._blocks
Python double-loop over cells (slow)         (exact np.nanquantile, no binning error)
linear-interpolate quantile bands in time    interpolate the raw within-cell 250 m value
  + re-sort bands to force monotonicity        stacks in time, THEN quantile (correct order;
                                              monotonic by construction, no re-sort)
write x^0.25 viridis PNGs used AS DATA        write linear-space GeoTIFFs; defer any power
                                              transform / standardization to model-input time
BLOCK_SIZE = 16 hardcoded                    block factor derived from config grid + native res

The quantile is the model's BUI feature, and it is applied at the target
resolution from the finest data — so re-running at a different
``grid.target_res_m`` simply re-derives everything correctly. Temporal
interpolation is done on the raw 250 m values (a linear operation) with the
nonlinear quantile applied last, so interpolated years yield true quantiles of
the time-interpolated surface — not interpolated quantiles.

Run:
    python scripts/aggregate_and_interpolate_bui.py            # snapshots only
    python scripts/aggregate_and_interpolate_bui.py --interpolate  # + yearly
"""
import argparse
import glob
import os
import re
import warnings

import numpy as np
import rasterio
from rasterio.transform import Affine

from src.config_utils import load_data_config
from src.processing import regrid

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.90, 0.97, 0.99]
_YEAR_RE = re.compile(r"(\d{4})_BUI\.tif$")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_bui_rasters(bui_dir: str) -> list:
    """Return [(year, path)] for raw ``NNNN_BUI.tif`` rasters, recursively.

    Handles the depositor's deeply-nested archive layout and skips macOS
    ``._`` AppleDouble junk files that ship in the tarball.
    """
    hits = []
    for path in glob.glob(os.path.join(bui_dir, "**", "*_BUI.tif"), recursive=True):
        base = os.path.basename(path)
        if base.startswith("._"):
            continue
        m = _YEAR_RE.search(base)
        if m:
            hits.append((int(m.group(1)), path))
    return sorted(hits)


# ---------------------------------------------------------------------------
# Aggregation (finest-resolution -> target grid)
# ---------------------------------------------------------------------------

def _read_land(path: str, watermask: np.ndarray = None):
    """Read band 1 as float, nodata/water -> NaN. Returns (arr, profile)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        profile = src.profile
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    arr[arr < 0] = np.nan
    if watermask is not None:
        arr[watermask[: arr.shape[0], : arr.shape[1]] == 1] = np.nan
    return arr, profile


def cell_values(path, block, watermask=None):
    """Within-cell 250 m values grouped by target cell: (H4, W4, block*block).

    NaN for nodata/water. This is the exact per-cell sample the quantiles and
    the temporal interpolation are both computed from.
    """
    arr, profile = _read_land(path, watermask)
    vals = regrid._blocks(arr, block)
    target_transform = profile["transform"] * Affine.scale(block, block)
    return vals, target_transform, profile


def quantiles_from_values(vals, quantiles=QUANTILES) -> np.ndarray:
    """(Q, H4, W4) exact per-cell quantiles. Monotonic in q by construction."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cell -> NaN
        q = np.nanquantile(vals, quantiles, axis=2)
    return q.astype("float32")


def write_quantile_geotiff(bands, transform, profile, out_path):
    """Write the (Q, H4, W4) quantile stack as a linear-space multiband GeoTIFF."""
    prof = profile.copy()
    prof.update(
        height=bands.shape[1], width=bands.shape[2], count=bands.shape[0],
        transform=transform, dtype="float32", nodata=np.nan,
    )
    with rasterio.open(out_path, "w", **prof) as dst:
        for k in range(bands.shape[0]):
            dst.write(bands[k], k + 1)


# ---------------------------------------------------------------------------
# Temporal interpolation (on additive histograms, NOT on quantiles)
# ---------------------------------------------------------------------------

def interpolate_values(vals0, vals1, y0, y1, year):
    """Linear-in-time interpolation of the raw within-cell 250 m values.

    Interpolating the raw values (a linear op) and taking the quantile
    afterwards gives the true quantile of the time-interpolated surface. The
    old code interpolated the quantiles themselves (biased) and re-sorted to
    hide the resulting non-monotonicity.
    """
    if year == y0:
        return vals0
    if year == y1:
        return vals1
    w = (year - y0) / (y1 - y0)
    return (1.0 - w) * vals0 + w * vals1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_watermask(cfg):
    path = cfg.get("bui_watermask")
    if path and os.path.exists(path):
        with rasterio.open(path) as src:
            return src.read(1)
    print("[info] no BUI water mask configured/found; masking nodata/negatives only.")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bui-dir", help="Raw BUI dir (default: {datasets_root}/HBUI).")
    ap.add_argument("--out-dir", help="Output dir (default: same as --bui-dir).")
    ap.add_argument("--interpolate", action="store_true",
                    help="Also write per-year interpolated quantile GeoTIFFs.")
    args = ap.parse_args()

    cfg = load_data_config()
    dr = cfg["datasets_root"]
    bui_dir = args.bui_dir or os.path.join(dr, "HBUI")
    out_dir = args.out_dir or bui_dir
    os.makedirs(out_dir, exist_ok=True)

    target_res = regrid.load_grid_spec(cfg)["target_res_m"]
    watermask = _load_watermask(cfg)

    rasters = discover_bui_rasters(bui_dir)
    if not rasters:
        raise SystemExit(f"No *_BUI.tif rasters under {bui_dir}")
    years = [y for y, _ in rasters]
    print(f"Found {len(rasters)} BUI snapshots: {years[0]}..{years[-1]}")

    native = regrid.native_res_m(rasters[0][1])
    block = regrid.block_factor(native, target_res)
    print(f"native={native:.0f}m target={target_res}m -> block factor {block}")
    res_tag = f"{target_res // 1000}km"

    def out_path(year):
        return os.path.join(out_dir, f"{year}_BUI_{res_tag}.tif")

    # Snapshot aggregation: exact per-cell quantiles -> linear-space GeoTIFF.
    # In --interpolate mode we keep the previous snapshot's value stack so each
    # gap is filled from just two adjacent snapshots (memory-bounded).
    prev = None  # (year, vals, transform, profile)
    for year, path in rasters:
        vals, transform, profile = cell_values(path, block, watermask)
        write_quantile_geotiff(quantiles_from_values(vals), transform, profile, out_path(year))
        print(f"  {year}: wrote {os.path.basename(out_path(year))}")

        if args.interpolate and prev is not None:
            py, pvals, ptransform, pprofile = prev
            for yr in range(py + 1, year):  # snapshot endpoints already written
                vi = interpolate_values(pvals, vals, py, year, yr)
                write_quantile_geotiff(quantiles_from_values(vi), ptransform, pprofile, out_path(yr))
            if year - py > 1:
                print(f"    interpolated {py + 1}..{year - 1}")
        prev = (year, vals, transform, profile) if args.interpolate else None


if __name__ == "__main__":
    main()
