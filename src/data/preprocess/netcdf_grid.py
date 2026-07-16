"""Shared helpers for reprojecting global netCDF stacks onto the model grid.

Both HYDE (population) and LUH-3 (land use) ship global lat/lon netCDF stacks
with a time axis; both need the same steps: find the data variable(s) and the
spatial/time dims, normalize longitudes to -180..180, and reproject each in-range
time slice onto the model grid one at a time (so peak RAM is a single global
slice, not the whole stack). Product-specific choices — which variables, average
vs sum, output naming — stay in the callers.
"""
import numpy as np

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
