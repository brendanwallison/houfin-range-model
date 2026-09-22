"""Route-year and site x replicate-year frames for the SDM benchmark.

WHY THIS EXISTS. The correlative baselines must be scored against the SAME
observations the dynamic model is fit to, so the response comes from the same
ingest path (src/data/preprocess/bbs.py) rather than a parallel reader. What
differs from the dynamic model's training set is deliberate and is enforced here:

  * NO pseudo-zeros. The dynamic model prepends pre-invasion absences
    (1902-1939) as evidence for a low-permeability barrier. A correlative SDM
    that saw them would be getting the invasion history it is supposed to lack.
  * NO observer covariates. `ObsN` / first-year flags exist in bbs.py but the
    dynamic model ignores them, so neither side gets them.
  * POST-INVASION ONLY. Fitting or scoring inside the invasion transient is
    indefensible for an equilibrium model; the window defaults to 2000-2025.

Two shapes are produced from one screened frame:
  route_years()      -> long, one row per route-year   (BRT)
  site_replicates()  -> sites x years count matrix     (biolith, years as
                        replicates -- NOT stop bins, which are contiguous along
                        a single route and not independent)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio

from src.config_utils import load_data_config
from src.data.preprocess import bbs as _bbs

# Post-invasion analysis window. The dynamic model runs 1902-2025; the
# correlative baselines are confined to years where an equilibrium assumption is
# defensible. 2000 is comfortably after east and west met in the 1990s.
DEFAULT_START_YEAR = 2000
DEFAULT_END_YEAR = 2025

# Hard floor for any correlative fit or score. The eastern introduction spread
# from ~1940 and met the western population in the 1990s; everything before that
# is an active invasion front, where the equilibrium assumption a correlative SDM
# rests on is violated by construction. Guarding on the pseudo-zero end year
# (1939) would let 1966 through, which is the middle of the transient.
EQUILIBRIUM_MIN_YEAR = 1995

# biolith window: years-as-replicates assumes population closure, so it is
# shorter than the BRT window. Verify the within-window trend is flat
# (see closure_check) before trusting detection estimates.
DEFAULT_REPLICATE_START = 2015
DEFAULT_REPLICATE_END = 2025

# The Plains depression is a mid-latitude phenomenon: north of ~44N the species
# is scarce on EVERY longitude (a northern range limit, not a gap), and
# collapsing over latitude dilutes the contrast with northern zeros.
PLAINS_LAT_MIN = 30.0
PLAINS_LAT_MAX = 42.0

ZONE_WEST, ZONE_BARRIER, ZONE_EAST = 1, 2, 3


def load_grid():
    """Model grid from the ocean mask: (land_mask, transform, crs, nx, ny)."""
    land_mask, _ocean, transform, crs, nx, ny = _bbs.load_grid_reference(_bbs.MASK_PATH)
    return land_mask, transform, crs, nx, ny


def route_years(start_year=DEFAULT_START_YEAR, end_year=DEFAULT_END_YEAR,
                aou=_bbs.HOUSE_FINCH_AOU):
    """One row per QC-passing route-year in [start_year, end_year].

    Columns: CountryNum/StateNum/Route/Year/count/row/col/Latitude/Longitude.
    ``count`` is the route total for ``aou``; every surveyed route-year gets a
    row, so zeros are true absences, not missing data.
    """
    if start_year < EQUILIBRIUM_MIN_YEAR:
        raise ValueError(
            f"start_year={start_year} reaches into the invasion transient "
            f"(floor is EQUILIBRIUM_MIN_YEAR={EQUILIBRIUM_MIN_YEAR}). The "
            f"correlative baselines assume equilibrium; fitting OR scoring there "
            f"is not defensible -- a model with no time dimension cannot predict "
            f"a spreading front, so failing there restates its architecture "
            f"rather than testing anything (see module docstring).")

    obs = _bbs.load_usca_observations(aou_filter=aou)
    obs = obs[obs["Year"].between(start_year, end_year)].copy()

    routes = _bbs.load_routes()
    land_mask, transform, crs, nx, ny = load_grid()
    obs = _bbs.map_routes_to_grid(obs, routes, transform, crs, nx, ny, land_mask)

    # map_routes_to_grid keeps only keys + row/col/geometry; bring lat/lon back
    # for the spatial term (Test B) and the latitude banding.
    keys = ["CountryNum", "StateNum", "Route"]
    routes_ll = routes.drop_duplicates(subset=keys)
    routes_ll[keys] = routes_ll[keys].astype(int)
    obs = obs.merge(routes_ll[keys + ["Latitude", "Longitude"]], on=keys, how="left")

    obs = obs.rename(columns={"SpeciesTotal": "count"})
    out = obs[keys + ["Year", "count", "row", "col", "Latitude", "Longitude"]]
    return out.reset_index(drop=True)


def site_replicates(df, start_year=DEFAULT_REPLICATE_START,
                    end_year=DEFAULT_REPLICATE_END):
    """Sites x years count matrix for biolith, with years as replicates.

    Returns ``(sites, counts, years)``: ``sites`` is one row per route with its
    row/col/lat/lon, ``counts`` is float (n_sites, n_years) with NaN where the
    route was not surveyed that year, and ``years`` are the years actually
    retained (seasons with no surveys anywhere are dropped -- 2020 was cancelled
    for COVID). biolith wants
    (n_species, n_sites, n_periods, n_replicates), so the caller adds the two
    leading axes -- n_periods is 1 here, the whole window being one closed
    period.

    Years, not stop bins: the 10-stop columns in the States CSVs are contiguous
    along one 40 km route and are not independent replicates.
    """
    keys = ["CountryNum", "StateNum", "Route"]
    w = df[df["Year"].between(start_year, end_year)]
    if w.empty:
        raise ValueError(f"No route-years in {start_year}-{end_year}.")

    years = np.arange(start_year, end_year + 1)
    wide = w.pivot_table(index=keys, columns="Year", values="count", aggfunc="sum")
    wide = wide.reindex(columns=years)

    # Drop years with NO surveys anywhere. The 2020 BBS season was cancelled for
    # COVID, so reindexing to a contiguous range manufactures an all-NaN
    # replicate column -- which is not a year of non-detection, it is a year that
    # did not happen, and biolith would be entitled to treat it as the former.
    surveyed = ~wide.isna().all(axis=0)
    dropped = [int(y) for y in wide.columns[~surveyed]]
    wide = wide.loc[:, surveyed]
    if dropped:
        print(f"  site_replicates: dropped unsurveyed year(s) {dropped}")

    sites = w.drop_duplicates(subset=keys).set_index(keys)
    sites = sites.loc[wide.index, ["row", "col", "Latitude", "Longitude"]].reset_index()
    return sites, wide.to_numpy(dtype=float), [int(y) for y in wide.columns]


def attach_zones(df, zone_path=None):
    """Add a ``zone`` column (1=west, 2=Great Plains barrier, 3=east, 0=nodata).

    Reads the gap-free corridor partition written by
    scripts/build_great_plains_mask.py. Returns df unchanged with zone=0 when the
    raster is absent, matching how the rest of the repo degrades on it.
    """
    if zone_path is None:
        zone_path = load_data_config()["regions"]["great_plains_zones"]
    out = df.copy()
    try:
        with rasterio.open(zone_path) as src:
            zones = src.read(1)
    except (OSError, rasterio.RasterioIOError):
        out["zone"] = 0
        return out
    out["zone"] = zones[out["row"].to_numpy(), out["col"].to_numpy()]
    return out


def plains_band(df):
    """Restrict to the latitude band where the Plains contrast is meaningful."""
    return df[df["Latitude"].between(PLAINS_LAT_MIN, PLAINS_LAT_MAX)]


def closure_check(df, start_year=DEFAULT_REPLICATE_START,
                  end_year=DEFAULT_REPLICATE_END):
    """Within-window mean count per year, and an OLS slope on it.

    Years-as-replicates assumes population closure. This does not prove it, but a
    steep slope here falsifies it, and the numbers belong in the write-up.
    """
    w = df[df["Year"].between(start_year, end_year)]
    per_year = w.groupby("Year")["count"].mean()
    yrs = per_year.index.to_numpy(dtype=float)
    slope, intercept = np.polyfit(yrs, per_year.to_numpy(), 1)
    return {
        "years": per_year.index.tolist(),
        "mean_count": per_year.round(4).tolist(),
        "slope_per_year": float(slope),
        "pct_per_year": float(100.0 * slope / per_year.mean()),
    }
