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

The ~5k weekly rasters are independent, so they reproject in parallel across a
process pool (each worker opens its own files -- fork-safe, no shared GDAL/HDF5
handles). Worker count comes from ``HOUFIN_PREPROCESS_WORKERS`` (else
``SLURM_CPUS_ON_NODE``, capped) so a whole Lonestar6 node is actually used.
"""

import os
import re
import csv
import glob
import collections
import multiprocessing as mp

import numpy as np
import xarray as xr  # noqa: F401  (kept: rioxarray extends xarray)
import rioxarray as rxr
import rasterio
import matplotlib
matplotlib.use("Agg")  # headless: compute nodes have no display, and safe in workers
import matplotlib.pyplot as plt
from tqdm import tqdm

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

# eBird S&T weekly abundance = 52 weekly surfaces/year. Species with fewer weeks
# present are incomplete and can't enter the encoder's rectangular species x week
# grid, so the reference community is the top-N *complete* species by rank.
EXPECTED_WEEKS = int(_CFG.get("ebird_weeks", 52))
# Reference-community size. The raw dir may hold leftovers from earlier (uncapped)
# downloads; selection here is authoritative regardless of what's on disk.
EBIRD_TOPN = int(os.environ.get("HOUFIN_EBIRD_TOPN", _CFG.get("ebird_topn", 100)))

_SP_RE = re.compile(r"([a-z0-9]+)_abundance_median")


def _species_of(path):
    """eBird species code parsed from a raster filename, or None."""
    m = _SP_RE.match(os.path.basename(path))
    return m.group(1) if m else None


def _read_ranked():
    """Ranked species codes (best-first) from the config species_list CSV."""
    path = _CFG.get("species_list")
    if not path or not os.path.exists(path):
        return []
    focal = str(_CFG.get("focal_species_code") or "").strip().lower()
    with open(path, newline="") as fh:
        codes = [row["species_code"].strip()
                 for row in csv.DictReader(fh) if row.get("species_code")]
    # Defense-in-depth: never let the focal (transfer target, e.g. House Finch) into
    # the community even if a stale ranked list still contains it (avonet drops it too).
    return [c for c in codes if c.lower() != focal] if focal else codes


def select_reference_community(tif_files, n_target):
    """Choose the reprojection set: top-``n_target`` *complete* species by rank.

    Groups the present rasters by species, keeps only species with the full
    EXPECTED_WEEKS coverage, and takes the first ``n_target`` of those in ranked
    order. Ignores incomplete species and any leftover extras from earlier
    downloads. Returns (selected_files, kept_species).
    """
    per = collections.defaultdict(list)
    for f in tif_files:
        sp = _species_of(f)
        if sp:
            per[sp].append(f)
    complete = {sp for sp, fs in per.items() if len(fs) >= EXPECTED_WEEKS}
    ranked = _read_ranked()
    kept = [sp for sp in ranked if sp in complete][:n_target]
    if not ranked:
        # No ranked list: fall back to all complete species (order-stable).
        kept = sorted(complete)[:n_target] if n_target else sorted(complete)
    files = sorted(f for sp in kept for f in per[sp])
    return files, kept, per, complete

# Per-worker read-only state (loaded once per process by _init_worker).
_WREF = None
_WOCEAN = None


def save_png(array, out_path, cmap="viridis"):
    """Save a PNG using a global monotone power transform."""
    arr = array.astype(float)
    arr[arr < 0] = 0
    vis = np.power(arr, PNG_POWER)
    vis[np.isnan(vis)] = 0
    plt.imsave(out_path, vis, cmap=cmap)


def _init_worker():
    """Load the shared, read-only grid + ocean mask once per worker process.

    Runs after fork, so each worker gets its own GDAL handles (no sharing across
    processes). Pin GDAL to one thread per worker so N workers don't each spawn
    warp threads and oversubscribe the node.
    """
    global _WREF, _WOCEAN
    os.environ.setdefault("GDAL_NUM_THREADS", "1")
    # Bound the GDAL block cache PER WORKER. Default is 5% of node RAM per process
    # (~12 GB on a 238 GB node); N fork workers each claim that independently, so at
    # high worker counts the cache ceiling alone exceeds node RAM and reprojection
    # OOMs (a tiny allocation fails because the cache already consumed everything).
    # 512 MB/worker is ample for these single-band warps; 48 workers => ~24 GB cache.
    os.environ.setdefault("GDAL_CACHEMAX", "512MB")
    _WREF = regrid.load_ref(_CFG)
    with rasterio.open(OCEAN_MASK) as src:
        _WOCEAN = src.read(1)


def _process_one(tif_path):
    """Reproject one eBird raster onto the model grid. Returns (name, status)."""
    fname = os.path.basename(tif_path)
    out_tif = os.path.join(OUT_DIR, fname.replace(".tif", "_grid.tif"))
    out_png = os.path.join(OUT_DIR, fname.replace(".tif", "_grid.png"))

    if os.path.exists(out_tif):
        return (fname, "exists")

    da = rxr.open_rasterio(tif_path, masked=True)

    # Validate, then enforce, the assumed CRS. eBird S&T weekly rasters are
    # EPSG:8857; if a raster declares a *different* CRS, that's a data-format
    # surprise we want to fail on rather than silently overwrite.
    existing = da.rio.crs
    if existing is not None and existing.to_epsg() != 8857:
        raise ValueError(
            f"{fname} declares CRS {existing} != assumed {EBIRD_CRS}; "
            f"verify the eBird product before overwriting.")
    da = da.rio.write_crs(EBIRD_CRS, inplace=False)
    da = da.rio.write_nodata(float("nan"), inplace=False)

    # Reproject onto the model grid (any native:target ratio; average is the
    # linear areal aggregate -- correct for relative abundance).
    da_reproj = regrid.reproject_to_ref(da, _WREF, resampling="average")

    if da_reproj.shape[-2:] != _WOCEAN.shape:
        raise ValueError(
            f"Shape mismatch after reprojection for {tif_path}: "
            f"da_reproj.shape={da_reproj.shape}, ocean_mask.shape={_WOCEAN.shape}")

    data = da_reproj.values.astype("float32")
    data[0][_WOCEAN == 1] = np.nan
    da_reproj = da_reproj.copy(data=data)

    da_reproj.rio.to_raster(out_tif)
    save_png(data[0], out_png)
    return (fname, "ok")


def _worker_count(n_items):
    """Parallel workers: HOUFIN_PREPROCESS_WORKERS, else SLURM/cpu count, capped.

    Capped at 64 by default to bound peak RAM (each worker holds a native raster
    + reprojection buffers); raise HOUFIN_PREPROCESS_WORKERS once remora shows
    memory headroom. Never exceed the number of rasters.
    """
    env = os.environ.get("HOUFIN_PREPROCESS_WORKERS")
    if env:
        n = int(env)
    else:
        slurm = os.environ.get("SLURM_CPUS_ON_NODE")
        n = int(slurm) if slurm else (os.cpu_count() or 1)
        n = min(n, 64)
    return max(1, min(n, n_items or 1))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_tifs = sorted(glob.glob(os.path.join(EBIRD_DIR, "*.tif")))
    if not all_tifs:
        raise SystemExit(f"No eBird .tif files found in {EBIRD_DIR}")

    # Authoritative selection: top-N complete species by rank. Ignores incomplete
    # species and leftover extras from earlier (uncapped) downloads, so the grid
    # output the encoder reads is a clean rectangular species x week set.
    tif_files, kept, per, complete = select_reference_community(all_tifs, EBIRD_TOPN)
    print(f"eBird: {len(per)} species present ({len(complete)} complete @ "
          f"{EXPECTED_WEEKS} wk); selected top-{EBIRD_TOPN} by rank -> {len(kept)} "
          f"species, {len(tif_files)} rasters -> {OUT_DIR}", flush=True)
    if not tif_files:
        raise SystemExit("No complete eBird species to reproject "
                         "(check species_list and per-species week counts).")
    if len(kept) < EBIRD_TOPN:
        print(f"[WARN] only {len(kept)} complete species available (< target {EBIRD_TOPN}).",
              flush=True)

    workers = _worker_count(len(tif_files))
    print(f"eBird reproject: {len(tif_files)} rasters, {workers} workers", flush=True)

    counts = {"ok": 0, "exists": 0}
    if workers == 1:
        _init_worker()
        it = (_process_one(t) for t in tif_files)
        for _, status in tqdm(it, total=len(tif_files), desc="ebird", mininterval=5):
            counts[status] = counts.get(status, 0) + 1
    else:
        # fork-based pool: workers inherit imported modules; _init_worker loads
        # each worker's own grid/ocean handles. imap_unordered yields as each
        # raster finishes, so tqdm shows real completion progress.
        with mp.Pool(processes=workers, initializer=_init_worker) as pool:
            for _, status in tqdm(
                pool.imap_unordered(_process_one, tif_files, chunksize=4),
                total=len(tif_files), desc="ebird", mininterval=5,
            ):
                counts[status] = counts.get(status, 0) + 1

    print(f"eBird reproject done: reprojected={counts.get('ok', 0)} "
          f"already-present={counts.get('exists', 0)} -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
