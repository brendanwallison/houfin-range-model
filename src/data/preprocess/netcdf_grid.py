"""Shared helpers for reprojecting global netCDF stacks onto the model grid.

Both HYDE (population) and LUH-3 (land use) ship global lat/lon netCDF stacks
with a time axis; both need the same steps: find the data variable(s) and the
spatial/time dims, normalize longitudes to -180..180, and reproject each in-range
time slice onto the model grid one at a time (so peak RAM is a single global
slice, not the whole stack). Product-specific choices — which variables, average
vs sum, output naming — stay in the callers.
"""
import os
import multiprocessing as mp

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor used in workers)
from tqdm import tqdm

Y_NAMES = {"lat", "latitude", "y"}
X_NAMES = {"lon", "longitude", "x"}


def detect_3d_vars(ds):
    """All (time, y, x) data variables, in file order (raises if none)."""
    cands = [v for v in ds.data_vars if ds[v].ndim == 3]
    if not cands:
        raise ValueError(f"no 3-D (time,y,x) data variables in {list(ds.data_vars)}")
    return cands


def spatial_dims(da):
    """Return (y_dim, x_dim), tolerant of lat/lon vs y/x naming."""
    ydim = next((d for d in da.dims if d.lower() in Y_NAMES), None)
    xdim = next((d for d in da.dims if d.lower() in X_NAMES), None)
    if ydim is None or xdim is None:
        raise ValueError(f"could not find lat/lon dims in {da.dims}")
    return ydim, xdim


def time_dim(da, ydim, xdim):
    """The remaining (non-spatial) dim; raises if not exactly one."""
    others = [d for d in da.dims if d not in (ydim, xdim)]
    if len(others) != 1:
        raise ValueError(f"expected one time dim besides {ydim},{xdim}; got {others}")
    return others[0]


def years_of(coord):
    """Integer year per time step (datetime64, cftime, or numeric-year coords)."""
    if np.issubdtype(coord.dtype, np.datetime64):
        return coord.dt.year.values
    try:  # cftime coords expose .dt.year too, but not the datetime64 dtype
        return coord.dt.year.values
    except (TypeError, AttributeError):
        return coord.values.astype(int)


def normalize_lon(da, xdim):
    """Shift 0..360 longitudes to -180..180 and sort ascending (no-op if already so)."""
    lon = da[xdim].values
    if lon.max() > 180:
        da = da.assign_coords({xdim: ((lon + 180) % 360) - 180}).sortby(xdim)
    return da


def reproject_time_slices(da, ref, resampling, year_lo, year_hi, out_dir, name_fn):
    """Reproject each in-[year_lo,year_hi] slice of ``da`` onto ``ref``, one at a time.

    ``name_fn(year) -> filename`` names each output GeoTIFF (written under
    ``out_dir``). Returns the list of years written. Sets EPSG:4326 on each slice
    (global lat/lon source) before ``reproject_match``.
    """
    import os

    from src.processing import regrid

    ydim, xdim = spatial_dims(da)
    tdim = time_dim(da, ydim, xdim)
    years = years_of(da[tdim])
    written = []
    for i, yr in enumerate(years):
        yr = int(yr)
        if yr < year_lo or yr > year_hi:
            continue
        sl = normalize_lon(da.isel({tdim: i}), xdim)
        sl = sl.rio.set_spatial_dims(x_dim=xdim, y_dim=ydim).rio.write_crs("EPSG:4326")
        out = regrid.reproject_to_ref(sl, ref, resampling=resampling)
        out.rio.to_raster(os.path.join(out_dir, name_fn(yr)))
        written.append(yr)
    return written


# --- Parallel reprojection -----------------------------------------------------
# Each worker opens its OWN netCDF handle (HDF5 handles are not fork-safe, so a
# shared open dataset must never cross a fork). Work is one (file, variable, year)
# slice per task; the pool fans them across the node with a live progress bar.

_WORKER_REF = None


def worker_count(n_items, cap=48):
    """Parallel workers: HOUFIN_PREPROCESS_WORKERS, else SLURM/cpu count, capped.

    Capped (default 48) to bound peak RAM (each worker holds one global slice +
    reprojection buffers); never exceed the number of work items.
    """
    env = os.environ.get("HOUFIN_PREPROCESS_WORKERS")
    if env:
        n = int(env)
    else:
        slurm = os.environ.get("SLURM_CPUS_ON_NODE")
        n = int(slurm) if slurm else (os.cpu_count() or 1)
        n = min(n, cap)
    return max(1, min(n, n_items or 1))


def _worker_init(cfg):
    global _WORKER_REF
    os.environ.setdefault("GDAL_NUM_THREADS", "1")  # no per-worker warp threads
    from src.processing import regrid
    _WORKER_REF = regrid.load_ref(cfg)


def _reproject_one(item):
    """Reproject one (file, var, time-index) slice; opens its own netCDF handle.

    Skips (and reports) if the output already exists, so an interrupted run
    resumes instead of redoing completed slices.
    """
    import xarray as xr
    from src.processing import regrid
    if os.path.exists(item["out_path"]):
        return "exists"
    with xr.open_dataset(item["nc_path"], decode_times=True) as ds:
        sl = ds[item["var"]].isel({item["tdim"]: item["tindex"]})
        sl = normalize_lon(sl, item["xdim"])
        sl = (sl.rio.set_spatial_dims(x_dim=item["xdim"], y_dim=item["ydim"])
                .rio.write_crs("EPSG:4326"))
        out = regrid.reproject_to_ref(sl, _WORKER_REF, resampling=item["resampling"])
        out.rio.to_raster(item["out_path"])
    return "ok"


def enumerate_slices(nc_path, variables, resampling, year_lo, year_hi, out_dir, name_fn):
    """Metadata-only scan of one netCDF -> list of reproject work items (no data read).

    ``name_fn(var, year) -> filename``. ``resampling`` applies to every variable in
    this file, so callers pass the product-correct method (``average`` for intensive
    fields like fractions/densities, ``sum`` for extensive counts).
    """
    import xarray as xr
    items = []
    with xr.open_dataset(nc_path, decode_times=True) as ds:
        varlist = variables or detect_3d_vars(ds)
        for var in varlist:
            if var not in ds.data_vars:
                raise KeyError(f"{var} not in {nc_path} (have {list(ds.data_vars)})")
            da = ds[var]
            ydim, xdim = spatial_dims(da)
            tdim = time_dim(da, ydim, xdim)
            for i, yr in enumerate(years_of(da[tdim])):
                yr = int(yr)
                if year_lo <= yr <= year_hi:
                    items.append(dict(
                        nc_path=nc_path, var=var, tdim=tdim, tindex=i,
                        xdim=xdim, ydim=ydim, year=yr, resampling=resampling,
                        out_path=os.path.join(out_dir, name_fn(var, yr))))
    return items


def reproject_parallel(items, cfg, workers=None, desc="reproject"):
    """Reproject all work items across a process pool, with a tqdm progress bar.

    Returns (n_reprojected, n_already_present). Fork-safe: each worker opens its
    own netCDF handle in ``_reproject_one``.
    """
    if not items:
        print(f"{desc}: nothing to do (all present, or no in-range years).", flush=True)
        return 0, 0
    workers = workers or worker_count(len(items))
    counts = {"ok": 0, "exists": 0}
    print(f"{desc}: {len(items)} slices, {workers} workers", flush=True)
    if workers == 1:
        _worker_init(cfg)
        for it in tqdm(items, desc=desc, mininterval=5):
            counts[_reproject_one(it)] += 1
    else:
        with mp.Pool(processes=workers, initializer=_worker_init, initargs=(cfg,)) as pool:
            for status in tqdm(pool.imap_unordered(_reproject_one, items, chunksize=8),
                               total=len(items), desc=desc, mininterval=5):
                counts[status] += 1
    print(f"{desc}: reprojected={counts['ok']} already-present={counts['exists']}", flush=True)
    return counts["ok"], counts["exists"]
