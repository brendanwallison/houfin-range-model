"""Reproject LUH-3 land-use netCDF onto the model grid, one raster per var-year.

LUH-3 states/management files are global 0.25 deg annual netCDF stacks with many
data variables (states: 12 land-use fractions primf/primn/secdf/.../pastr/range;
management: crop/fertilizer/irrigation/wood-harvest layers). Every variable is
an intensive per-cell fraction or rate, so all reproject with **average** (the
areal aggregate 0.25 deg -> 25 km, ~1:1). "Use every LUH-3 covariate": all 3-D
variables in each file are written, one GeoTIFF per variable per in-range year
(``{var}_{year}_grid.tif``).

Only years within the model timeline (minus a short warm-up) are written; the
combine streamer EMA-carries any covariate that lags end_year (LUH-3 ends 2024).
Slices are read/reprojected one at a time (see ``netcdf_grid``), so peak RAM is a
single global 0.25 deg slice, not the whole stack.
"""
import os

import xarray as xr

from src.config_utils import load_data_config
from src.data.preprocess import netcdf_grid as ncg
from src.processing import regrid
from src.temporal import load_timeline

WARMUP_YEARS = 20


def preprocess_file(nc_path, out_dir, ref, year_lo, year_hi, variables=None):
    """Reproject every (or ``variables``) 3-D var to 25 km per year. Returns {var: [years]}."""
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    with xr.open_dataset(nc_path, decode_times=True) as ds:
        varlist = variables or ncg.detect_3d_vars(ds)
        for var in varlist:
            if var not in ds.data_vars:
                raise KeyError(f"{var} not in {nc_path} (have {list(ds.data_vars)})")
            yrs = ncg.reproject_time_slices(
                ds[var], ref, "average", year_lo, year_hi, out_dir,
                name_fn=lambda yr, v=var: f"{v}_{yr}_grid.tif")
            written[var] = yrs
            print(f"LUH-3 {var}: {len(yrs)} rasters -> {out_dir}")
    return written


def main():
    cfg = load_data_config()
    dr = cfg["datasets_root"]
    zcfg = cfg.get("zenodo", {})
    ldef = zcfg.get("datasets", {}).get("luh3", {})
    tl = load_timeline(cfg)
    year_lo, year_hi = tl["first_year"] - WARMUP_YEARS, tl["end_year"]

    in_dir = os.path.join(dr, zcfg.get("out_subdirs", {}).get("luh3", "LUH3"))
    out_dir = os.path.join(dr, "luh3_grid")
    ref = regrid.load_ref(cfg)
    # Optional per-file variable allow-list (else: every 3-D variable).
    var_map = ldef.get("variables", {})
    for fname in ldef.get("files", []):
        nc_path = os.path.join(in_dir, fname)
        if not os.path.exists(nc_path):
            print(f"[skip] {nc_path} not present")
            continue
        preprocess_file(nc_path, out_dir, ref, year_lo, year_hi, var_map.get(fname))


if __name__ == "__main__":
    main()
