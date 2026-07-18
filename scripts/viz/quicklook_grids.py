#!/usr/bin/env python3
"""Thumbnail-PNG quicklooks of the 25 km gridded products, for visual validation.

Walks each product dir for GeoTIFFs, renders band 1 as a small percentile-stretched
PNG (NaN / ocean transparent) into ``<out>/<dataset>/<name>.png``. Dense time-series
dirs are evenly sub-sampled to ``--sample`` rasters per dataset (use ``--all`` for
every raster -- can be thousands). Then tar the folder and scp it to a workstation.

    python scripts/viz/quicklook_grids.py               # default dirs, sampled
    python scripts/viz/quicklook_grids.py --all         # every raster (large)
    python scripts/viz/quicklook_grids.py --include-ebird --sample 100

eBird already writes ``_grid.png`` next to each grid tif during preprocessing, so
it's skipped by default (--include-ebird to render its tifs too).
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config_utils import load_data_config


def _even_sample(items, k):
    """Evenly-spaced sub-sample of <= k items (keeps first/last), order preserved."""
    if k <= 0 or len(items) <= k:
        return items
    idx = sorted(set(np.linspace(0, len(items) - 1, k).round().astype(int)))
    return [items[i] for i in idx]


def render_one(task):
    """Render one GeoTIFF band-1 to a percentile-stretched thumbnail PNG."""
    tif, png, max_dim, cmap = task
    try:
        with rasterio.open(tif) as src:
            a = src.read(1, masked=True)
        arr = np.ma.filled(a.astype("float32"), np.nan)
        h, w = arr.shape
        step = max(1, round(max(h, w) / max_dim))   # cheap decimate to ~max_dim
        arr = arr[::step, ::step]
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return "empty"
        lo, hi = np.nanpercentile(finite, [2, 98])
        if hi <= lo:
            hi = lo + 1e-9
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad(alpha=0.0)                        # NaN/ocean transparent
        os.makedirs(os.path.dirname(png), exist_ok=True)
        plt.imsave(png, np.ma.masked_invalid(arr), cmap=cm, vmin=lo, vmax=hi)
        return "ok"
    except Exception as e:  # noqa: BLE001  (one bad raster shouldn't kill the batch)
        return f"error:{type(e).__name__}"


def default_groups(dr):
    """Standard 25 km raster products (eBird + CSV-only climate handled separately)."""
    return {
        "ref_grid":  [p for p in [os.path.join(dr, "ref_grid_25km.tif")] if os.path.exists(p)],
        "land_mask": sorted(glob.glob(os.path.join(dr, "land_mask", "*.tif"))),
        "luh3":      sorted(glob.glob(os.path.join(dr, "luh3_grid", "*.tif"))),
        "hyde":      sorted(glob.glob(os.path.join(dr, "hyde35_grid", "*.tif"))),
        "soilgrids": sorted(glob.glob(os.path.join(dr, "soilgrids_grid", "**", "*.tif"), recursive=True)),
        "elevation": sorted(glob.glob(os.path.join(dr, "elevation", "**", "*.tif"), recursive=True)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output dir (default: $HOUFIN_PROCESSED/quicklooks)")
    ap.add_argument("--sample", type=int, default=48,
                    help="max rasters per dataset (evenly sub-sampled); default 48")
    ap.add_argument("--all", action="store_true", help="render every raster (can be thousands)")
    ap.add_argument("--max-dim", type=int, default=300, help="thumbnail max dimension in px")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16))
    ap.add_argument("--include-ebird", action="store_true",
                    help="also render the eBird grid tifs (5000+; it already writes _grid.png)")
    args = ap.parse_args()

    cfg = load_data_config()
    dr = cfg["datasets_root"]
    out = args.out or os.path.join(os.environ.get("HOUFIN_PROCESSED", dr), "quicklooks")

    groups = default_groups(dr)
    if args.include_ebird:
        sub = cfg.get("ebird_raw_subdir", "ebird_weekly_2023") + "_grid"
        groups["ebird"] = sorted(glob.glob(os.path.join(dr, sub, "*.tif")))

    tasks = []
    for ds, files in groups.items():
        if not files:
            continue
        chosen = files if args.all else _even_sample(files, args.sample)
        for f in chosen:
            name = os.path.splitext(os.path.basename(f))[0]
            tasks.append((f, os.path.join(out, ds, name + ".png"), args.max_dim, args.cmap))
        print(f"{ds}: {len(chosen)}/{len(files)} rasters", flush=True)

    if not tasks:
        raise SystemExit(f"no rasters found under {dr} (has preprocess run?)")

    print(f"rendering {len(tasks)} thumbnails, {args.workers} workers -> {out}", flush=True)
    counts = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for status in tqdm(ex.map(render_one, tasks), total=len(tasks), mininterval=2):
            counts[status] = counts.get(status, 0) + 1
    print("done:", counts, flush=True)

    parent, base = os.path.dirname(out), os.path.basename(out)
    print(f"\nBundle + scp:\n"
          f"  tar czf {base}.tgz -C {parent} {base}\n"
          f"  # from your PC:  scp <user>@ls6.tacc.utexas.edu:{parent}/{base}.tgz .")


if __name__ == "__main__":
    main()
