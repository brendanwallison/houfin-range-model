#!/usr/bin/env python3
"""Labeled sample maps of a few raw covariates (HYDE, LUH-3, climr, Z) at
three eras, on the project's model grid, for a first visual look at what
these inputs actually contain.

Periods (mean over available native time-steps in range):
  1900-1915, 1950-1965, 2010-2025

Variables (a handful, not the full covariate set):
  HYDE   urban_population        (5 arc-min annual/decadal population count)
  LUH-3  primf, urban, c3ann     (0.25 deg annual land-use fraction)
  climr  CMD_q50, CMI_q50        (already 27 km gridded annual climate-moisture
                                  normals -- deficit and index -- median
                                  elevation tercile; from CLIMATE_DIR, fetched
                                  from TACC since no R/climr is installed here)
  Z      components 0-3          (DESK encoder's latent spacetime cube --
                                  Z_latent_{year}.npy, (H,W,64) float32, ALREADY
                                  on the 27km model grid and ALREADY NaN-masked
                                  over water by the TACC pipeline that built it;
                                  no reprojection or land-mask rasterization
                                  needed. Not named covariates -- these are
                                  individual kernel-PCA-style latent dimensions.
                                  Uses the BASE config/data_config.json grid
                                  (27km), not this machine's possibly-overridden
                                  local_data_config.json, since Z's shape is
                                  fixed at whatever resolution it was built at.)

Each PNG: percentile-stretched raster on the model grid, ocean masked out
(or, for Z, already-NaN water left as-is), coastline drawn, titled with
dataset/variable/era, with a colorbar.

Output: docs/img/covariate_samples/{dataset}_{var}_{era}.png
"""
import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.features import rasterize
from shapely.ops import unary_union

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config_utils import load_data_config
from src.processing import regrid
from src.data.preprocess import netcdf_grid as ncg

OUT_DIR = os.path.join(REPO_ROOT, "docs", "img", "covariate_samples")

PERIODS = [(1900, 1915), (1950, 1965), (2010, 2025)]

LUH3_NC = os.path.join(
    REPO_ROOT, "data", "LUH3",
    "multiple-states_input4MIPs_landState_CMIP_UofMD-landState-3-1-2_gn_0850-2024.nc",
)
# Not under the repo's data/ root -- scp'd down from TACC ($SCRATCH/houfin/data/*)
# into these local paths by hand, per the calling conversation.
CLIMATE_DIR = os.path.expanduser("~/Data/climate_grid")
HYDE_GRID_DIR = os.path.expanduser("~/Data/hyde35_grid")
# Still being copied down from TACC as of this writing -- variables land in it
# gradually. period_mean() below auto-prefers a variable here over the raw
# netCDF once enough of its {var}_{year}_grid.tif files exist for the requested
# period, so LUH-3 rows upgrade from "netcdf" to pre-gridded on their own as the
# copy completes; no script edit needed once it's done.
LUH3_GRID_DIR = os.path.expanduser("~/Data/luh3_grid")
# DESK Z cube: Z_latent_{year}.npy, (H, W, 64), scp'd from TACC
# ($WORK/houfin/processed/encoder/cube) into this local path.
Z_DIR = os.path.expanduser("~/Data/cube")
Z_COMPONENTS = (0, 1, 2, 3)

# (dataset, kind, source, variable, resampling, display label, value transform, cmap)
# kind "netcdf": source is a global lat/lon .nc stack (LUH-3 states, not yet gridded).
# kind "grid_tif": source is a dir of already-27km-gridded {var}_{year}_grid.tif
# (HYDE, climr -- built by the TACC preprocessing pipeline, just reprojected here
# onto whatever grid this machine's data_config points at).
# kind "z_component": source is Z_DIR; ``var`` is "z{k}" for latent component k.
# Already on the model's 27km grid and already NaN-masked over water -- no
# reprojection or land-mask rasterization applied (see period_mean()/main()).
VARIABLES = [
    ("HYDE", "grid_tif", HYDE_GRID_DIR, "urban_population", "sum",
     "HYDE: Urban Population (log1p count / grid cell)", np.log1p, "magma"),
    ("HYDE", "grid_tif", HYDE_GRID_DIR, "population_density", "average",
     "HYDE: Population Density (log1p people/km2)", np.log1p, "viridis"),
    ("LUH-3", "netcdf", LUH3_NC, "primf", "average",
     "LUH-3: Primary Forest Fraction", None, "YlGn"),
    ("LUH-3", "netcdf", LUH3_NC, "urban", "average",
     "LUH-3: Urban Land Fraction", None, "magma"),
    ("LUH-3", "netcdf", LUH3_NC, "c3ann", "average",
     "LUH-3: C3 Annual Crop Fraction", None, "YlOrBr"),
    ("climr", "grid_tif", CLIMATE_DIR, "CMD_q50", "average",
     "climr: Climate Moisture Deficit (q50 elevation)", None, "YlOrRd"),
    ("climr", "grid_tif", CLIMATE_DIR, "CMI_q50", "average",
     "climr: Climate Moisture Index (q50 elevation)", None, "BrBG"),
] + [
    ("Z", "z_component", Z_DIR, f"z{k}", None,
     f"Z: Latent Component {k}", None, "RdBu_r")
    for k in Z_COMPONENTS
]


def _base_grid_spec():
    """The COMMITTED config/data_config.json grid section, bypassing any
    local_data_config.json override. Z is built on TACC at whatever resolution
    the base config specifies (27km production), independent of this machine's
    local dev convenience override (25km) -- so its extent must come from here,
    not from load_data_config()/regrid.load_ref()."""
    with open(os.path.join(REPO_ROOT, "config", "data_config.json")) as f:
        base_cfg = json.load(f)
    box_minx, box_miny, box_maxx, box_maxy = base_cfg["grid"]["box_bounds"]
    return {
        "box_bounds": (box_minx, box_miny, box_maxx, box_maxy),
        "extent": (box_minx, box_maxx, box_miny, box_maxy),
        "target_res_m": base_cfg["grid"]["target_res_m"],
    }


_Z_YEAR_RE = re.compile(r"Z_latent_(\d{4})\.npy$")


def _z_component_period_mean(z_dir, component_idx, year_lo, year_hi):
    """Mean of latent component ``component_idx`` over every Z_latent_{year}.npy
    in [year_lo, year_hi]. Uses mmap so only the one requested channel of each
    64-channel cube is actually read. Returns a plain (H, W) array, already
    NaN-masked over water by the TACC pipeline that built Z -- no additional
    land masking needed."""
    paths = []
    for path in glob.glob(os.path.join(z_dir, "Z_latent_*.npy")):
        m = _Z_YEAR_RE.search(os.path.basename(path))
        if m and year_lo <= int(m.group(1)) <= year_hi:
            paths.append(path)
    if not paths:
        raise ValueError(f"no Z_latent_*.npy in [{year_lo}, {year_hi}] under {z_dir}")

    slices = [np.load(p, mmap_mode="r")[..., component_idx] for p in paths]
    return np.nanmean(np.stack([np.asarray(s, dtype="float64") for s in slices]), axis=0)


def _land_mask(cfg, ref, ny, nx, transform):
    import geopandas as gpd
    from shapely.geometry import box as shapely_box

    project_crs = cfg["grid"]["box_crs"]
    box_bounds = tuple(cfg["grid"]["box_bounds"])
    box_geom = shapely_box(*box_bounds)
    land_path = os.path.join(cfg["datasets_root"], cfg["coastline"]["land_source"])
    land_gdf = gpd.read_file(land_path).to_crs(project_crs)
    land_geom = unary_union(land_gdf.geometry.buffer(0)).intersection(box_geom)
    mask = rasterize(
        [(land_geom, 1)], out_shape=(ny, nx), transform=transform,
        fill=0, all_touched=True, dtype="uint8",
    ).astype(bool)
    return mask, land_geom


def _period_mean_on_grid(nc_path, var, resampling, year_lo, year_hi, ref):
    """Mean of every in-[year_lo, year_hi] native time-step of ``var``, reprojected
    onto ``ref``. Loads only the selected slices (not the whole time series)."""
    import xarray as xr

    with xr.open_dataset(nc_path, decode_times=True) as ds:
        da = ds[var]
        ydim, xdim = ncg.spatial_dims(da)
        tdim = ncg.time_dim(da, ydim, xdim)
        years = ncg.years_of(da[tdim])
        idx = [i for i, yr in enumerate(years) if year_lo <= int(yr) <= year_hi]
        if not idx:
            raise ValueError(f"no {var} time-steps in [{year_lo}, {year_hi}]")
        sl = da.isel({tdim: idx}).mean(dim=tdim, skipna=True).load()

    sl = ncg.normalize_lon(sl, xdim)
    sl = sl.rio.set_spatial_dims(x_dim=xdim, y_dim=ydim).rio.write_crs("EPSG:4326")
    sl = sl.rio.write_nodata(float("nan"), inplace=False)
    return regrid.reproject_to_ref(sl, ref, resampling=resampling)


def _grid_tif_period_mean(grid_dir, var, resampling, year_lo, year_hi, ref):
    """Mean of every in-[year_lo, year_hi] {var}_{year}_grid.tif (already gridded,
    one year per file), reprojected onto ``ref``."""
    paths = [
        os.path.join(grid_dir, f"{var}_{yr}_grid.tif")
        for yr in range(year_lo, year_hi + 1)
    ]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise ValueError(f"no {var} grid files in [{year_lo}, {year_hi}] under {grid_dir}")

    das = [rioxarray.open_rasterio(p, masked=True).squeeze("band", drop=True) for p in paths]
    stacked = das[0].copy(data=np.nanmean(np.stack([d.values for d in das]), axis=0))
    stacked = stacked.rio.write_crs(das[0].rio.crs).rio.write_nodata(float("nan"), inplace=False)
    return regrid.reproject_to_ref(stacked, ref, resampling=resampling)


def _grid_tif_coverage(grid_dir, var, year_lo, year_hi):
    """Fraction of years in [year_lo, year_hi] that have a {var}_{year}_grid.tif."""
    n = year_hi - year_lo + 1
    have = sum(
        os.path.exists(os.path.join(grid_dir, f"{var}_{yr}_grid.tif"))
        for yr in range(year_lo, year_hi + 1)
    )
    return have / n


def period_mean(row, year_lo, year_hi, ref):
    """Dispatch a VARIABLES row to the right loader by ``kind``.

    "netcdf" rows auto-upgrade to LUH3_GRID_DIR's pre-gridded tifs once that
    variable has full coverage for the requested period (it's still mid-copy
    from TACC as of this writing), so no script edit is needed once it lands.
    """
    dataset, kind, source, var, resampling, _label, _tf, _cmap = row
    if kind == "z_component":
        component_idx = int(var[1:])  # "z3" -> 3
        return _z_component_period_mean(source, component_idx, year_lo, year_hi)
    if kind == "netcdf" and dataset == "LUH-3":
        if _grid_tif_coverage(LUH3_GRID_DIR, var, year_lo, year_hi) >= 1.0:
            return _grid_tif_period_mean(LUH3_GRID_DIR, var, resampling, year_lo, year_hi, ref)
    if kind == "netcdf":
        return _period_mean_on_grid(source, var, resampling, year_lo, year_hi, ref)
    return _grid_tif_period_mean(source, var, resampling, year_lo, year_hi, ref)


def _plot(arr, land, extent, box_bounds, title, cmap, out_png, vlim=None):
    """``vlim``, if given, is a shared (lo, hi) so multiple eras of the same
    variable share a color scale -- otherwise each panel's independent 2/98
    percentile stretch can hide a real but small temporal change under a much
    larger spatial (e.g. orographic) range."""
    masked = np.where(land, arr, np.nan)
    finite = masked[np.isfinite(masked)]
    if finite.size == 0:
        print(f"[skip] {out_png}: no finite land values")
        return
    if vlim is not None:
        lo, hi = vlim
    else:
        lo, hi = np.nanpercentile(finite, [2, 98])
    if hi <= lo:
        hi = lo + 1e-9

    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.set_facecolor("white")
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("white", alpha=0)
    im = ax.imshow(masked, extent=extent, origin="upper", cmap=cm, vmin=lo, vmax=hi)
    ax.set_xlim(box_bounds[0], box_bounds[2])
    ax.set_ylim(box_bounds[1], box_bounds[3])
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def _plot_diff(diff, land, extent, box_bounds, title, out_png):
    """Diverging change map (era_hi minus era_lo), its own scale centered on 0 --
    a variable's absolute-value color range is often dominated by a much larger
    spatial (e.g. orographic) contrast than the temporal change, which can make
    genuinely real change look invisible in the absolute-value panels."""
    from matplotlib.colors import TwoSlopeNorm

    masked = np.where(land, diff, np.nan)
    finite = masked[np.isfinite(masked)]
    if finite.size == 0:
        print(f"[skip] {out_png}: no finite land values")
        return
    lo, hi = np.nanpercentile(finite, [2, 98])
    bound = max(abs(lo), abs(hi), 1e-9)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)

    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.set_facecolor("white")
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("white", alpha=0)
    im = ax.imshow(masked, extent=extent, origin="upper", cmap=cm, norm=norm)
    ax.set_xlim(box_bounds[0], box_bounds[2])
    ax.set_ylim(box_bounds[1], box_bounds[3])
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def _to_array(grid):
    """period_mean() returns an xarray DataArray for reprojected sources, or a
    plain (H, W) ndarray for "z_component" (nothing to reproject)."""
    return grid.values if hasattr(grid, "values") else grid


def main():
    cfg = load_data_config()
    ref = regrid.load_ref(cfg)
    box_bounds = tuple(cfg["grid"]["box_bounds"])

    ny, nx = ref.shape[-2], ref.shape[-1]
    transform = ref.rio.transform()
    land, _ = _land_mask(cfg, ref, ny, nx, transform)
    extent = (
        transform.c, transform.c + nx * transform.a,
        transform.f + ny * transform.e, transform.f,
    )

    # Z lives on the base (production) 27km grid regardless of this machine's
    # local_data_config.json override, and comes pre-masked over water -- so it
    # gets its own extent/box_bounds and an all-True "land" mask (a no-op,
    # since Z's own NaNs already do the masking) rather than reusing the
    # HYDE/LUH-3/climr grid above. Shape is taken from a real Z file the first
    # time one is needed, rather than assumed, so a config/data drift is caught
    # as a shape-mismatch error instead of silently misaligning.
    base_spec = _base_grid_spec()
    z_land = None

    os.makedirs(OUT_DIR, exist_ok=True)
    for dataset, kind, source, var, resampling, label, transform_fn, cmap in VARIABLES:
        if not os.path.exists(source):
            print(f"[skip] {source} not present")
            continue
        row = (dataset, kind, source, var, resampling, label, transform_fn, cmap)
        arrs = {}
        for year_lo, year_hi in PERIODS:
            grid = period_mean(row, year_lo, year_hi, ref)
            arr = np.squeeze(_to_array(grid)).astype("float64")
            if transform_fn is not None:
                arr = transform_fn(np.clip(arr, 0, None))
            arrs[(year_lo, year_hi)] = arr

        if kind == "z_component":
            if z_land is None or z_land.shape != next(iter(arrs.values())).shape:
                z_land = np.ones(next(iter(arrs.values())).shape, dtype=bool)
            row_land, row_extent, row_box_bounds = z_land, base_spec["extent"], base_spec["box_bounds"]
        else:
            row_land, row_extent, row_box_bounds = land, extent, box_bounds

        # Shared, absolute color scale across this variable's eras (1st/99th
        # percentile over land, pooled across all three eras -- effectively
        # anchored to whichever era holds the largest/smallest value, but
        # robust to a handful of extreme outlier pixels that would otherwise
        # wash out the whole map) so a real cross-era change is visible
        # instead of getting rescaled away by each panel's own independent
        # stretch.
        pooled = np.concatenate([
            np.where(row_land, a, np.nan).ravel() for a in arrs.values()
        ])
        pooled = pooled[np.isfinite(pooled)]
        vlim = tuple(np.nanpercentile(pooled, [1, 99])) if pooled.size else None

        for year_lo, year_hi in PERIODS:
            era = f"{year_lo}-{year_hi}"
            title = f"{label}\n{era} mean"
            out_png = os.path.join(
                OUT_DIR, f"{dataset.lower().replace('-', '')}_{var}_{era}.png"
            )
            _plot(arrs[(year_lo, year_hi)], row_land, row_extent, row_box_bounds,
                  title, cmap, out_png, vlim)

        era_lo, era_hi = PERIODS[0], PERIODS[-1]
        diff = arrs[era_hi] - arrs[era_lo]
        diff_title = f"{label}\nchange: {era_hi[0]}-{era_hi[1]} minus {era_lo[0]}-{era_lo[1]}"
        diff_png = os.path.join(
            OUT_DIR, f"{dataset.lower().replace('-', '')}_{var}_change.png"
        )
        _plot_diff(diff, row_land, row_extent, row_box_bounds, diff_title, diff_png)


if __name__ == "__main__":
    main()
