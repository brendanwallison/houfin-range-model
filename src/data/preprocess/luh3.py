"""Reproject LUH-3 land-use netCDF onto the model grid, one raster per var-year.

LUH-3 states/management files are global 0.25 deg annual netCDF stacks with many
data variables (states: 12 land-use fractions primf/primn/secdf/.../pastr/range;
management: crop/fertilizer/irrigation/wood-harvest layers). Every variable is an
intensive per-cell fraction or rate (no categorical/class layers), so all reproject
with **average** (the areal aggregate 0.25 deg -> 27 km, ~1:1, where average ~= sum
per cell anyway). "Use every LUH-3 covariate": all 3-D variables in each file are
written, one GeoTIFF per variable per in-range year (``{var}_{year}_grid.tif``).

Only years within the model timeline (minus a short warm-up) are written; the
combine streamer EMA-carries any covariate that lags end_year (LUH-3 ends 2024).
Slices reproject in parallel, one (file, var, year) per task, each worker opening
its own netCDF handle (see ``netcdf_grid.reproject_parallel``).
"""
import os

from src.config_utils import load_data_config
from src.data.preprocess import netcdf_grid as ncg
from src.temporal import load_timeline

WARMUP_YEARS = 20


def preprocess_file(nc_path, out_dir, ref, year_lo, year_hi, variables=None):
    """Serial single-file entry point used by tests and small local runs."""
    import xarray as xr

    os.makedirs(out_dir, exist_ok=True)
    written = {}
    with xr.open_dataset(nc_path, decode_times=True) as ds:
        for var in variables or ncg.detect_3d_vars(ds):
            written[var] = ncg.reproject_time_slices(
                ds[var], ref, "average", year_lo, year_hi, out_dir,
                name_fn=lambda year, var=var: f"{var}_{year}_grid.tif",
            )
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
    os.makedirs(out_dir, exist_ok=True)
    var_map = ldef.get("variables", {})  # optional per-file variable allow-list
    exclude = ldef.get("exclude", [])    # variables to DROP (e.g. layers identically 0 over NA)
    if exclude:
        print(f"[luh3] excluding {len(exclude)} variables: {exclude}")

    # Enumerate every (file, variable, in-range year) slice; all intensive -> average.
    items = []
    for fname in ldef.get("files", []):
        nc_path = os.path.join(in_dir, fname)
        if not os.path.exists(nc_path):
            print(f"[skip] {nc_path} not present")
            continue
        items += ncg.enumerate_slices(
            nc_path, var_map.get(fname), "average", year_lo, year_hi, out_dir,
            name_fn=lambda v, yr: f"{v}_{yr}_grid.tif", exclude=exclude)

    # OPT-IN (default off; absent -> every layer is a plain area average as before):
    # ``parent_weighted: {var: [parent, ...]}`` aggregates a per-parent-land-use rate as
    # the parent-area-weighted mean (see netcdf_grid.parent_weighted_reproject).
    parent_map = ldef.get("parent_weighted") or {}
    if parent_map:
        nc_paths = [os.path.join(in_dir, f) for f in ldef.get("files", [])
                    if os.path.exists(os.path.join(in_dir, f))]
        plain_dir = ldef.get("parent_weighted_plain_dir")
        if plain_dir:
            os.makedirs(plain_dir, exist_ok=True)
        items = ncg.attach_parents(items, parent_map, ncg.index_variables(nc_paths),
                                   plain_dir=plain_dir)
        n_pw = sum(1 for it in items if it.get("parents"))
        print(f"[luh3] parent-weighted aggregation for {len(parent_map)} variables "
              f"({n_pw} slices): {parent_map}", flush=True)
        import json
        with open(os.path.join(out_dir, "parent_weighting.json"), "w") as fh:
            json.dump({"rule": "sum(rate*parent)/sum(parent) via two GDAL average "
                               "reprojections on a common valid mask; plain average "
                               "where the parent aggregates to 0",
                       "parent_weighted": parent_map}, fh, indent=2, sort_keys=True)

    ncg.reproject_parallel(items, cfg, desc="luh3")


if __name__ == "__main__":
    main()
