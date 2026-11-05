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
    if item.get("parents"):  # opt-in (luh3 parent_weighted); absent -> unchanged path below
        return _reproject_parent_weighted(item)
    with xr.open_dataset(item["nc_path"], decode_times=True) as ds:
        sl = ds[item["var"]].isel({item["tdim"]: item["tindex"]})
        sl = normalize_lon(sl, item["xdim"])
        sl = (sl.rio.set_spatial_dims(x_dim=item["xdim"], y_dim=item["ydim"])
                .rio.write_crs("EPSG:4326"))
        out = regrid.reproject_to_ref(sl, _WORKER_REF, resampling=item["resampling"])
        out.rio.to_raster(item["out_path"])
    return "ok"


# --- Parent-weighted aggregation (opt-in; used by luh3 ``parent_weighted``) ------
# Some LUH-3 layers are intensive PER UNIT OF ANOTHER LAYER, not per unit of cell
# area: a fertilization rate is kg/ha of the crop's own area, an irrigated fraction
# is a fraction of that crop, secondary mean age is per unit of secondary land. The
# correct coarse value of such a rate r over its parent area a is
#     sum(r * a * A) / sum(a * A)          (A = source cell area)
# i.e. the parent-area-weighted mean, NOT the plain area mean sum(r * A) / sum(A),
# which dilutes the rate by pixels where the parent land use is absent. Both terms
# are linear area aggregates, so each is one GDAL ``average`` reprojection with the
# SAME valid mask (hence the same overlap weights), and the ratio is exact up to the
# within-cell variation of A, which GDAL's source-pixel weighting ignores exactly as
# the plain ``average`` path does. Where the parent area aggregates to 0 the rate is
# undefined; the plain area-mean of the rate (the default rule) is the fallback.


def _open_geo_slice(nc_path, var, tdim, tindex, xdim, ydim):
    """One (lat, lon) slice as an EPSG:4326 DataArray, values loaded, handle closed."""
    import xarray as xr
    with xr.open_dataset(nc_path, decode_times=True) as ds:
        sl = ds[var].isel({tdim: tindex}).load()
    sl = normalize_lon(sl, xdim)
    return sl.rio.set_spatial_dims(x_dim=xdim, y_dim=ydim).rio.write_crs("EPSG:4326")


def parent_weighted_reproject(rate, parent, ref, reproject_fn):
    """Parent-area-weighted mean of ``rate`` onto ``ref``; also returns the plain mean.

    ``rate`` and ``parent`` are aligned source DataArrays (same grid; NaN = nodata).
    ``reproject_fn(da) -> DataArray`` performs the ``average`` reprojection (injected
    so tests can run without the config glue). Returns ``(weighted, plain, n_fallback)``
    where ``plain`` is exactly the default-rule output of ``rate``, ``weighted`` equals
    avg(rate * parent) / avg(parent) over the common valid mask, and ``n_fallback``
    counts target cells where avg(parent) <= 0 (or is NaN) but ``plain`` is finite, so
    ``plain`` was used.
    """
    plain = reproject_fn(rate)
    r = np.asarray(rate.values, dtype="float64")
    a = np.asarray(parent.values, dtype="float64")
    if r.shape != a.shape:
        raise ValueError(f"rate {r.shape} and parent {a.shape} are not on one grid")
    valid = np.isfinite(r) & np.isfinite(a)
    xd, yd, crs = rate.rio.x_dim, rate.rio.y_dim, rate.rio.crs
    num_src, den_src = (
        rate.copy(data=v).rio.set_spatial_dims(x_dim=xd, y_dim=yd).rio.write_crs(crs)
        .rio.write_nodata(np.nan, encoded=False)
        for v in (np.where(valid, r * a, np.nan), np.where(valid, a, np.nan)))
    num = np.asarray(reproject_fn(num_src).values, dtype="float64")
    den = np.asarray(reproject_fn(den_src).values, dtype="float64")
    pv = np.asarray(plain.values, dtype="float64")
    use = np.isfinite(den) & (den > 0) & np.isfinite(num)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(use, num / np.where(use, den, 1.0), pv)
    n_fallback = int((~use & np.isfinite(pv)).sum())
    weighted = plain.copy(data=w.astype(plain.dtype))
    return weighted, plain, n_fallback


def _reproject_parent_weighted(item):
    """Worker body for an item carrying ``parents`` (list of slice specs whose SUM is
    the parent area fraction). Writes the weighted raster to ``out_path`` and, when
    ``plain_out_path`` is set, the default-rule (plain average) raster there too."""
    from src.processing import regrid
    keys = ("nc_path", "var", "tdim", "tindex", "xdim", "ydim")
    rate = _open_geo_slice(*(item[k] for k in keys))
    parent = None
    for p in item["parents"]:
        sl = _open_geo_slice(*(p[k] for k in keys))
        if sl.shape != rate.shape:
            raise ValueError(f"parent {p['var']} grid {sl.shape} != {item['var']} {rate.shape}")
        parent = sl.values.astype("float64") if parent is None else parent + sl.values
    parent = rate.copy(data=parent)

    def _rp(da):
        return regrid.reproject_to_ref(da, _WORKER_REF, resampling=item["resampling"])

    weighted, plain, _ = parent_weighted_reproject(rate, parent, _WORKER_REF, _rp)
    weighted.rio.to_raster(item["out_path"])
    if item.get("plain_out_path") and not os.path.exists(item["plain_out_path"]):
        plain.rio.to_raster(item["plain_out_path"])
    return "ok"


def attach_parents(items, parent_map, var_index, plain_dir=None):
    """Mark items whose variable is parent-weighted (opt-in; returns a NEW list).

    ``parent_map``: ``{var: [parent_var, ...]}`` (the parent area is the SUM of the
    listed fractions). ``var_index``: ``{var: (nc_path, tdim, xdim, ydim, {year: tindex})}``
    for every variable that may serve as a parent (excluded layers included -- a parent
    need not be an output). ``plain_dir``: if set, the default-rule raster of each
    weighted variable is also written there under the same name (diagnostic).
    """
    out = []
    for it in items:
        parents = parent_map.get(it["var"])
        if not parents:
            out.append(it)
            continue
        it = dict(it)
        specs = []
        for pv in parents:
            if pv not in var_index:
                raise KeyError(f"parent {pv!r} of {it['var']!r} not found in any LUH file")
            nc_path, tdim, xdim, ydim, tmap = var_index[pv]
            if it["year"] not in tmap:
                raise KeyError(f"parent {pv!r} has no year {it['year']}")
            specs.append(dict(nc_path=nc_path, var=pv, tdim=tdim, tindex=tmap[it["year"]],
                              xdim=xdim, ydim=ydim))
        it["parents"] = specs
        if plain_dir:
            it["plain_out_path"] = os.path.join(plain_dir, os.path.basename(it["out_path"]))
        out.append(it)
    return out


def index_variables(nc_paths):
    """``{var: (nc_path, tdim, xdim, ydim, {year: tindex})}`` for every 3-D variable."""
    import xarray as xr
    idx = {}
    for nc_path in nc_paths:
        with xr.open_dataset(nc_path, decode_times=True) as ds:
            for var in detect_3d_vars(ds):
                da = ds[var]
                ydim, xdim = spatial_dims(da)
                tdim = time_dim(da, ydim, xdim)
                tmap = {int(y): i for i, y in enumerate(years_of(da[tdim]))}
                idx[var] = (nc_path, tdim, xdim, ydim, tmap)
    return idx


def enumerate_slices(nc_path, variables, resampling, year_lo, year_hi, out_dir, name_fn,
                     exclude=None):
    """Metadata-only scan of one netCDF -> list of reproject work items (no data read).

    ``name_fn(var, year) -> filename``. ``resampling`` applies to every variable in
    this file, so callers pass the product-correct method (``average`` for intensive
    fields like fractions/densities, ``sum`` for extensive counts). ``variables`` is
    an optional allow-list (None -> every 3-D var); ``exclude`` then drops names from
    that set (e.g. LUH-3 management layers that are identically zero over the region).
    """
    import xarray as xr
    items = []
    excl = set(exclude or ())
    with xr.open_dataset(nc_path, decode_times=True) as ds:
        varlist = [v for v in (variables or detect_3d_vars(ds)) if v not in excl]
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
