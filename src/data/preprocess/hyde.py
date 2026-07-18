"""Reproject HYDE 3.5 population netCDF onto the model grid, one year per raster.

HYDE ships one global 5-arc-min netCDF per variable (population_density.nc,
urban_population.nc, rural_population.nc), each spanning all HYDE time points.
Density is intensive (people/km^2 -> reproject **average**); the urban/rural
*population* layers are extensive counts (people/cell -> reproject **sum**, so
totals are conserved 5' -> 25 km). Aggregation per file is inferred from the
name ("density" -> average, else sum) and overridable via ``hyde.aggregation``.

Only time points within the model timeline (minus a short warm-up) are written;
the combine streamer maps model years to the nearest available HYDE year and
EMA-carries the rest. Slices reproject in parallel, each worker opening its own
netCDF handle (see ``netcdf_grid.reproject_parallel``).
"""
import os

from src.config_utils import load_data_config
from src.data.preprocess import netcdf_grid as ncg
from src.temporal import load_timeline

WARMUP_YEARS = 20  # write time points this far before first_year for EMA warm-up


def _resampling_for(fname, overrides):
    """average for density (intensive), sum for population counts (extensive)."""
    if fname in overrides:
        return overrides[fname]
    return "average" if "density" in fname.lower() else "sum"


def main():
    cfg = load_data_config()
    dr = cfg["datasets_root"]
    hcfg = cfg.get("hyde", {})
    tl = load_timeline(cfg)
    year_lo, year_hi = tl["first_year"] - WARMUP_YEARS, tl["end_year"]
    overrides = hcfg.get("aggregation", {})

    in_dir = os.path.join(dr, hcfg.get("out_subdir", "hyde35"))
    out_dir = os.path.join(dr, "hyde35_grid")
    os.makedirs(out_dir, exist_ok=True)

    # One variable per HYDE file; resampling chosen per file (density vs counts).
    items = []
    for fname in hcfg.get("files", []):
        nc_path = os.path.join(in_dir, fname)
        if not os.path.exists(nc_path):
            print(f"[skip] {nc_path} not present")
            continue
        resampling = _resampling_for(fname, overrides)
        stem = os.path.splitext(fname)[0]
        items += ncg.enumerate_slices(
            nc_path, None, resampling, year_lo, year_hi, out_dir,
            name_fn=lambda v, yr, s=stem: f"{s}_{yr}_grid.tif")

    ncg.reproject_parallel(items, cfg, desc="hyde")


if __name__ == "__main__":
    main()
