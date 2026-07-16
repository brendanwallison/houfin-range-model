"""Reproject HYDE 3.5 population netCDF onto the model grid, one year per raster.

HYDE ships one global 5-arc-min netCDF per variable (population_density.nc,
urban_population.nc, rural_population.nc), each spanning all HYDE time points.
Density is intensive (people/km^2 -> reproject **average**); the urban/rural
*population* layers are extensive counts (people/cell -> reproject **sum**, so
totals are conserved 5' -> 25 km). Aggregation per file is inferred from the
name ("density" -> average, else sum) and overridable via ``hyde.aggregation``.

Only time points within the model timeline (minus a short warm-up) are written;
the combine streamer maps model years to the nearest available HYDE year and
EMA-carries the rest. Slices are read and reprojected one at a time (see
``netcdf_grid``), so peak RAM is a single global 5' grid, not the whole century.
"""
import os

import xarray as xr

from src.config_utils import load_data_config
from src.data.preprocess import netcdf_grid as ncg
from src.processing import regrid
from src.temporal import load_timeline

WARMUP_YEARS = 20  # write time points this far before first_year for EMA warm-up


def _resampling_for(fname, overrides):
    """average for density (intensive), sum for population counts (extensive)."""
    if fname in overrides:
        return overrides[fname]
    return "average" if "density" in fname.lower() else "sum"


def preprocess_file(nc_path, out_dir, ref, resampling, year_lo, year_hi):
    """Write one 25 km raster per in-range HYDE time point. Returns years written."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(nc_path))[0]
    with xr.open_dataset(nc_path, decode_times=True) as ds:
        var = ncg.detect_3d_vars(ds)[0]  # one variable per HYDE file
        written = ncg.reproject_time_slices(
            ds[var], ref, resampling, year_lo, year_hi, out_dir,
            name_fn=lambda yr: f"{stem}_{yr}_grid.tif")
    print(f"HYDE {stem}: {len(written)} rasters ({resampling}) -> {out_dir}")
    return written


def main():
    cfg = load_data_config()
    dr = cfg["datasets_root"]
    hcfg = cfg.get("hyde", {})
    tl = load_timeline(cfg)
    year_lo, year_hi = tl["first_year"] - WARMUP_YEARS, tl["end_year"]
    overrides = hcfg.get("aggregation", {})

    in_dir = os.path.join(dr, hcfg.get("out_subdir", "hyde35"))
    out_dir = os.path.join(dr, "hyde35_grid")
    ref = regrid.load_ref(cfg)
    for fname in hcfg.get("files", []):
        nc_path = os.path.join(in_dir, fname)
        if not os.path.exists(nc_path):
            print(f"[skip] {nc_path} not present")
            continue
        preprocess_file(nc_path, out_dir, ref, _resampling_for(fname, overrides), year_lo, year_hi)


if __name__ == "__main__":
    main()
