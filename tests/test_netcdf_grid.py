"""Tests for the netCDF->grid machinery shared by hyde.py and luh3.py.

Exercises year filtering, multi-variable handling, output naming, longitude
normalization, and average-of-constant on synthetic data. The geographic->Albers
reprojection itself is validated separately on real HYDE/SoilGrids rasters.
Runs standalone or under pytest; needs xarray + rioxarray + rasterio.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401  (registers .rio)
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.preprocess import luh3
from src.data.preprocess import netcdf_grid as ncg
from src.data.preprocess.build_ref_grid import build_ref_grid


def _synthetic_ds(values_by_var, years, lon0_360=False):
    """A tiny global-ish lat/lon dataset over North America, one time axis."""
    lon = np.arange(-130, -60, 1.0)  # 1 deg, covers CONUS
    if lon0_360:
        lon = lon % 360  # 230..300, unsorted wrt -180..180
    lat = np.arange(55, 20, -1.0)    # descending, like real products
    data_vars = {}
    for name, val in values_by_var.items():
        arr = np.full((len(years), len(lat), len(lon)), val, dtype="float32")
        data_vars[name] = (("time", "lat", "lon"), arr)
    return xr.Dataset(data_vars, coords={"time": years, "lat": lat, "lon": lon})


def _small_ref(path):
    # ~1 deg WGS84 grid over the same window; geographic->geographic keeps it fast.
    build_ref_grid((-125.0, 25.0, -65.0, 50.0), "EPSG:4326", 1.0, path)
    return rioxarray.open_rasterio(path)


def test_years_of_numeric_and_datetime():
    assert list(ncg.years_of(xr.DataArray([1901, 1902, 2025]))) == [1901, 1902, 2025]
    dt = xr.DataArray(pd.to_datetime(["1901-06-01", "2025-06-01"]))
    assert list(ncg.years_of(dt)) == [1901, 2025]


def test_normalize_lon_wraps_and_sorts():
    ds = _synthetic_ds({"v": 1.0}, [2000], lon0_360=True)
    out = ncg.normalize_lon(ds["v"], "lon")
    lon = out["lon"].values
    assert lon.max() <= 180 and lon.min() >= -180
    assert np.all(np.diff(lon) > 0)  # sorted ascending


def test_year_filter_and_average_of_constant():
    with tempfile.TemporaryDirectory() as d:
        ref = _small_ref(os.path.join(d, "ref.tif"))
        ds = _synthetic_ds({"v": 7.0}, [1899, 1902, 1950, 2030])  # 1899/2030 out of range
        out = os.path.join(d, "out"); os.makedirs(out)
        yrs = ncg.reproject_time_slices(ds["v"], ref, "average", 1902, 2025, out,
                                        name_fn=lambda y: f"v_{y}_grid.tif")
        assert yrs == [1902, 1950]  # out-of-range years dropped
        with rasterio.open(os.path.join(out, "v_1902_grid.tif")) as s:
            a = s.read(1, masked=True)
            assert (s.width, s.height) == (ref.rio.width, ref.rio.height)
            assert abs(float(a.mean()) - 7.0) < 1e-3  # average of a constant field


def test_luh3_multi_variable_writes_per_var_year():
    with tempfile.TemporaryDirectory() as d:
        ref_path = os.path.join(d, "ref.tif"); _small_ref(ref_path)
        ref = rioxarray.open_rasterio(ref_path)
        ds = _synthetic_ds({"primf": 0.5, "pastr": 0.2}, [1901, 1902, 1903])
        nc = os.path.join(d, "states.nc"); ds.to_netcdf(nc)
        out = os.path.join(d, "grid")
        written = luh3.preprocess_file(nc, out, ref, 1902, 1903)
        assert set(written) == {"primf", "pastr"}
        assert written["primf"] == [1902, 1903]
        for v in ("primf", "pastr"):
            for y in (1902, 1903):
                assert os.path.exists(os.path.join(out, f"{v}_{y}_grid.tif"))
        assert not os.path.exists(os.path.join(out, "primf_1901_grid.tif"))  # filtered


if __name__ == "__main__":
    test_years_of_numeric_and_datetime()
    print("[years_of] numeric + datetime64 -> integer years OK")
    test_normalize_lon_wraps_and_sorts()
    print("[normalize_lon] 0..360 -> -180..180 sorted OK")
    test_year_filter_and_average_of_constant()
    print("[reproject] year filter + average-of-constant + ref alignment OK")
    test_luh3_multi_variable_writes_per_var_year()
    print("[luh3] per-variable per-year rasters, year-filtered OK")
    print("\nALL NETCDF-GRID CHECKS PASSED")
