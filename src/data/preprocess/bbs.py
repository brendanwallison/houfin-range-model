"""Build the model-ready BBS observation set (House Finch counts + absences).

Reads the USGS Breeding Bird Survey release (US/Canada) and the separate Mexico
unprocessed release, maps routes onto the model grid, and writes
``bbs_data_for_python.npz`` (observations, a core/margin initialization density,
and pre-invasion pseudo-zeros).

Two provenance tiers, distinguished by a per-observation ``quality_tier``:

* **standard** (tier 0) — US/Canada. Screened to protocol-conforming runs:
  ``RunType != 0`` (0 = failed protocol / unsuitable weather) and
  ``RPID == 101`` (standard roadside survey). Pseudo-zeros are tier 0 too.
* **mx_unprocessed** (tier 1) — Mexico 2008-2018. This release has *no*
  RunType/RPID quality screening, so it is included **unscreened**; the model
  down-weights it via the quality covariate (see age_priors) rather than a
  protocol filter here. Every Mexican run contributes a real presence or a real
  absence — fixing the old bug where Mexican counts were never read and their
  routes leaked in as phantom zeros.

The timeline (first/end year, the pre-invasion pseudo-zero window) comes from
the canonical contract in src/temporal.py; nothing here hardcodes a year.
"""
import glob
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from shapely.geometry import MultiPoint

# The seed levels live in the age-model config beside the gauge they are expressed
# against (route counts), not in data_config -- keeping them next to
# population_scale_route_counts_per_relative_unit is what stops the two drifting.
from src.config_utils import load_age_model_config, load_data_config
from src.temporal import load_timeline

_CFG = load_data_config()
_DR = _CFG["datasets_root"]
_RES_KM = _CFG["grid"]["target_res_m"] // 1000
_OUT = _CFG.get("sciencebase", {}).get("out_subdirs", {})
_TL = load_timeline()

# US/Canada release (newest ScienceBase release) and Mexico unprocessed release.
BBS_PARENT_DIR = f"{_DR}/{_OUT.get('bbs', 'bbs_2026_release')}"
BBS_STATES_DIR = os.path.join(BBS_PARENT_DIR, "States")
WEATHER_FILE = os.path.join(BBS_PARENT_DIR, "Weather.csv")   # US+Canada, has RunType/RPID
ROUTES_FILE = os.path.join(BBS_PARENT_DIR, "Routes.csv")     # lat/lon
MEXICO_DIR = f"{_DR}/{_OUT.get('bbs_mexico', 'bbs_mexico_unprocessed')}"

# Model-grid ocean mask (must match Z at grid.target_res_m).
MASK_PATH = f"{_DR}/land_mask/ocean_mask_{_RES_KM}km.tif"

HOUSE_FINCH_AOU = 5190
RPID_STANDARD = 101
START_YEAR = _TL["first_year"]                 # 1902
END_YEAR = _TL["end_year"]                      # 2025
PSEUDO_ZERO_END_YEAR = _TL["invasion_year"] - 1  # last pre-invasion year (1939)
# Halo around the native hull inside which no pre-invasion zero is asserted. It was
# 1000 km, which is wider than the Great Plains themselves: measured from a CONVEX hull
# that already reaches the western Plains, a 1000 km halo left the entire barrier and
# a good part of the Midwest with NO pre-1940 constraint at all, so a westward-origin
# front could occupy the Plains from 1902 free of charge. 700 km still clears the
# native range's real fringe (the hull is a lower bound on it) while putting the
# barrier itself back inside the zero-constrained region, which is where the evidence
# for a low-permeability Plains has to come from.
BUFFER_DISTANCE_METERS = 700 * 1000             # 700 km "uninvaded east" halo
NATIVE_RANGE_MAX_YEAR = 1970                    # pre-1970 obs define the native range

QUALITY_STANDARD = 0
QUALITY_MX_UNPROCESSED = 1


def load_grid_reference(mask_path):
    """Load the model grid from the ocean mask (TIF: 1=ocean, 0=land)."""
    with rasterio.open(mask_path) as src:
        data = src.read(1)
        ocean_mask = (data == 1)
        land_mask = (data == 0)  # Python convention: True = land
        transform, crs = src.transform, src.crs
        ny, nx = data.shape
    print(f"Grid loaded: {ny}x{nx}, CRS: {crs}")
    return land_mask, ocean_mask, transform, crs, nx, ny


def load_usca_observations(aou_filter=HOUSE_FINCH_AOU, return_coverage=False):
    """US/Canada counts screened to protocol runs (RunType!=0 & RPID==101).

    ``aou_filter`` selects the species:
    - an AOU (default ``HOUSE_FINCH_AOU``) → that species' counts + true absences,
      i.e. every QC-passing run gets a row (left join → fill 0). Columns
      CountryNum/StateNum/Route/Year/SpeciesTotal/quality_tier. (Original behavior.)
    - ``None`` → **all species**, recorded (present) rows only, with the ``AOU``
      column kept, restricted to QC-passing route-years. Community absences are
      recovered downstream against the per-cell coverage, so we don't materialize
      the full species×run zero matrix here.

    With ``return_coverage=True`` also return the QC-passing route-year coverage
    frame (CountryNum/StateNum/Route/Year), i.e. which surveys happened — the
    denominator for effort weighting and absence in the community ingest.
    """
    if not os.path.isdir(BBS_STATES_DIR):
        raise FileNotFoundError(f"BBS States dir not found: {BBS_STATES_DIR}")

    count_cols = ["CountryNum", "StateNum", "Route", "RPID", "Year", "AOU", "SpeciesTotal"]
    frames = []
    for f in glob.glob(os.path.join(BBS_STATES_DIR, "*.csv")):
        try:
            frames.append(pd.read_csv(f, usecols=count_cols))
        except Exception as e:
            print(f"  Skipping state file {os.path.basename(f)}: {e}")
    if not frames:
        raise ValueError(f"No readable state count CSVs in {BBS_STATES_DIR}")
    counts = pd.concat(frames, ignore_index=True)
    for c in count_cols:
        counts[c] = pd.to_numeric(counts[c], errors="coerce")
    counts = counts.dropna(subset=count_cols[:-1]).astype({c: int for c in count_cols[:-1]})

    # Weather = quality table. RunType != 0 (0 = failed protocol/bad weather),
    # RPID == 101 (standard survey). Read explicitly (no silent except).
    w_cols = ["CountryNum", "StateNum", "Route", "RPID", "Year", "RunType"]
    qc = pd.read_csv(WEATHER_FILE, usecols=w_cols)
    for c in w_cols:
        qc[c] = pd.to_numeric(qc[c], errors="coerce")
    qc = qc.dropna().astype(int)
    qc = qc[(qc["RunType"] != 0) & (qc["RPID"] == RPID_STANDARD)]
    counts = counts[counts["RPID"] == RPID_STANDARD]

    keys = ["CountryNum", "StateNum", "Route", "RPID", "Year"]
    if aou_filter is None:
        obs = counts.merge(qc[keys], on=keys, how="inner")   # only surveyed route-years
        obs["SpeciesTotal"] = pd.to_numeric(obs["SpeciesTotal"], errors="coerce").fillna(0).astype(int)
        obs["quality_tier"] = QUALITY_STANDARD
        out = obs[["CountryNum", "StateNum", "Route", "Year", "AOU", "SpeciesTotal", "quality_tier"]]
        print(f"  US/Canada: {len(out)} species-route-years (all species, standard tier).")
    else:
        target = counts[counts["AOU"] == aou_filter]
        merged = qc.merge(target, on=keys, how="left")
        merged["SpeciesTotal"] = merged["SpeciesTotal"].fillna(0).astype(int)
        merged["quality_tier"] = QUALITY_STANDARD
        out = merged[["CountryNum", "StateNum", "Route", "Year", "SpeciesTotal", "quality_tier"]]
        print(f"  US/Canada: {len(out)} route-years (AOU {aou_filter}, standard tier).")

    if return_coverage:
        cov = qc[["CountryNum", "StateNum", "Route", "Year"]].drop_duplicates().reset_index(drop=True)
        return out, cov
    return out


def _mexico_year(run_data):
    """Year per Mexico run: from a Year column, else parsed from RunDate (M/D/YYYY)."""
    if "Year" in run_data:
        return pd.to_numeric(run_data["Year"], errors="coerce")
    return pd.to_datetime(run_data["RunDate"], errors="coerce").dt.year


# A BBS stop column, tolerant of header damage in the USGS Mexico release, which
# ships stop 46 as ``Sto 46`` -- the 'p' replaced by a space. ``\S?`` absorbs that
# character whatever it is and ``\s*`` the stray space, so ``Stop46`` and ``Sto 46``
# both resolve to 46. A stricter prefix test silently drops the damaged column and
# undercounts that route by one stop.
_STOP_RE = re.compile(r"^sto\S?\s*(\d+)$")


def _mexico_stop_columns(species):
    """``{stop_number: column}`` for the 50 per-stop count columns.

    Raises on a GAP in the numbering rather than summing what it found. That is the
    whole point: the failure this guards against is silent undercounting, so an
    unrecognised column has to stop the ingest instead of quietly contributing zero.
    A gap means the header is damaged in some way ``_STOP_RE`` does not yet absorb.
    """
    stops = {}
    for c in species.columns:
        m = _STOP_RE.match(str(c).strip().lower())
        if m:
            stops[int(m.group(1))] = c
    if not stops:
        return {}
    missing = sorted(set(range(1, max(stops) + 1)) - set(stops))
    if missing:
        raise ValueError(
            f"Mexico SpeciesData stop columns are not contiguous: found {len(stops)} "
            f"running to {max(stops)}, missing {missing}. The header is damaged in a "
            f"way the parser does not recognise -- summing the rest would silently "
            f"undercount. Columns present: {list(species.columns)}")
    return stops


def _mexico_count(species):
    """House-Finch count per Mexico record, robust to schema (SpeciesTotal or Stop sum)."""
    if "SpeciesTotal" in species:
        return pd.to_numeric(species["SpeciesTotal"], errors="coerce")
    stops = _mexico_stop_columns(species)
    if stops:
        cols = [stops[i] for i in sorted(stops)]
        return species[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    for alt in ("Count", "Total", "SpeciesCount"):
        if alt in species:
            return pd.to_numeric(species[alt], errors="coerce")
    raise ValueError(f"Cannot find a count column in Mexico SpeciesData "
                     f"(have {list(species.columns)}); verify the schema.")


def load_mexico_observations():
    """Mexico House-Finch counts + true absences, UNSCREENED (tier 1).

    RouteData.csv = runs (RunDate→year; no RunType), RouteDetails.csv = lat/lon,
    SpeciesData.csv = counts. Returns None (with a warning) if the counts file is
    absent — the USGS release currently serves it as 0 bytes, so this activates
    once a valid copy is present in MEXICO_DIR. Schema is inferred from the
    standard BBS layout and should be verified against the real file.
    """
    species_path = os.path.join(MEXICO_DIR, "SpeciesData.csv")
    run_path = os.path.join(MEXICO_DIR, "RouteData.csv")
    if not (os.path.exists(species_path) and os.path.getsize(species_path) > 0):
        print(f"[warn] Mexico counts missing/empty ({species_path}); skipping Mexico. "
              "(USGS currently serves SpeciesData.csv as 0 bytes — drop a valid copy here.)")
        return None
    if not os.path.exists(run_path):
        print(f"[warn] Mexico {run_path} missing; skipping Mexico.")
        return None

    keys = ["CountryNum", "StateNum", "Route"]
    runs = pd.read_csv(run_path)
    runs["Year"] = _mexico_year(runs)
    runs = runs.dropna(subset=keys + ["Year"]).astype({**{k: int for k in keys}, "Year": int})
    runs = runs[keys + ["Year"]].drop_duplicates()

    species = pd.read_csv(species_path)
    species["AOU"] = pd.to_numeric(species.get("AOU"), errors="coerce")
    species["count"] = _mexico_count(species)
    species["Year"] = _mexico_year(species) if ("Year" in species or "RunDate" in species) else np.nan
    finch = species[species["AOU"] == HOUSE_FINCH_AOU].copy()
    finch = finch.dropna(subset=keys).astype({k: int for k in keys})

    # Every Mexican run → presence or real absence (unscreened).
    merged = runs.merge(
        finch[keys + (["Year"] if finch["Year"].notna().any() else []) + ["count"]],
        on=keys + (["Year"] if finch["Year"].notna().any() else []), how="left",
    )
    merged["SpeciesTotal"] = merged["count"].fillna(0).astype(int)
    merged["quality_tier"] = QUALITY_MX_UNPROCESSED
    print(f"  Mexico: {len(merged)} route-years (mx_unprocessed tier).")
    return merged[["CountryNum", "StateNum", "Route", "Year", "SpeciesTotal", "quality_tier"]]


def load_routes():
    """Combined route lat/lon from US/Canada Routes.csv + Mexico RouteDetails.csv."""
    frames = []
    for path in (ROUTES_FILE, os.path.join(MEXICO_DIR, "RouteDetails.csv")):
        if not os.path.exists(path):
            continue
        try:
            frames.append(pd.read_csv(path))
        except UnicodeDecodeError:
            frames.append(pd.read_csv(path, encoding="latin1"))
    if not frames:
        raise FileNotFoundError("No route files found (Routes.csv / RouteDetails.csv).")
    routes = pd.concat(frames, ignore_index=True)
    return routes[["CountryNum", "StateNum", "Route", "Latitude", "Longitude"]]


def map_routes_to_grid(obs, routes, grid_transform, grid_crs, nx, ny, land_mask):
    """Attach (row, col) to each observation via its route lat/lon, keeping land cells."""
    gdf = gpd.GeoDataFrame(
        routes, geometry=gpd.points_from_xy(routes["Longitude"], routes["Latitude"]),
        crs="EPSG:4326",
    ).to_crs(grid_crs)
    coords = np.array([(p.x, p.y) for p in gdf.geometry])
    rows, cols = rasterio.transform.rowcol(grid_transform, coords[:, 0], coords[:, 1])
    gdf["row"], gdf["col"] = rows, cols

    inb = (gdf["row"] >= 0) & (gdf["row"] < ny) & (gdf["col"] >= 0) & (gdf["col"] < nx)
    gdf = gdf[inb].copy()
    gdf = gdf[land_mask[gdf["row"].values, gdf["col"].values]].copy()

    keys = ["CountryNum", "StateNum", "Route"]
    obs[keys] = obs[keys].astype(int)
    gdf[keys] = gdf[keys].astype(int)
    return obs.merge(gdf[keys + ["row", "col", "geometry"]], on=keys, how="inner")


def generate_core_margin_initialization(obs_df, ny, nx, transform, land_mask):
    """Native-range init density + pre-invasion pseudo-zeros.

    1. Native range = pre-1970 presences in the western two-thirds of the grid.
    2. Margin hull = all native points; core hull = points above the 75th count
       percentile.
    3. Seed map, in EXPECTED BBS ROUTE COUNTS, derived from the observed counts in
       those hulls (core overwrites margin). Previously hardcoded 0.1 / 0.001 in
       *relative density* units, which was wrong twice over: the values had no
       derivation, and relative units are gauge-dependent, so the seed silently
       changed meaning whenever pop_scalar changed. At the old gauge of 210 the core
       seed meant 21 counts against an observed native core of ~61, i.e. the 1966
       native range was seeded at a third of its actual abundance -- and against a
       fitted capacity of ~10 counts it was simultaneously 2x ABOVE capacity.
       Emitting counts and converting once, in model_inputs, fixes both.
    4. Buffer the native hull by BUFFER_DISTANCE_METERS (700 km) → the uninvaded east.
    5. Emit a zero count at every uninvaded cell for each pre-invasion year.
    """
    print("Generating core/margin map and pseudo-zeros...")
    western_limit_col = int(nx * 0.66)
    hist = obs_df[(obs_df["Year"] <= NATIVE_RANGE_MAX_YEAR)
                  & (obs_df["SpeciesTotal"] > 0)
                  & (obs_df["col"] < western_limit_col)].copy()
    if hist.empty:
        raise ValueError("No pre-1970 western presences to seed the native range.")

    locs = hist.drop_duplicates(subset=["row", "col"])
    hull_margin = MultiPoint(locs["geometry"].tolist()).convex_hull
    threshold = locs["SpeciesTotal"].quantile(0.75)
    print(f"  Core threshold (75th pct): {threshold:.1f}")
    hull_core = MultiPoint(
        locs[locs["SpeciesTotal"] > threshold]["geometry"].tolist()).convex_hull

    def _rasterize(geom):
        return rasterio.features.rasterize(
            [(geom, 1)], out_shape=(ny, nx), transform=transform,
            default_value=0, dtype=np.uint8) == 1

    mask_margin = _rasterize(hull_margin) & land_mask
    mask_core = _rasterize(hull_core) & land_mask

    # A DIMENSIONLESS shape: core = 1, margin = initpop_seed.margin_fraction_of_core.
    # The absolute scale is applied in the model as a fraction of LOCAL K_base (see
    # age_priors), because an absolute seed cannot stay coherent with a capacity level
    # that moves -- which it did, by 97x, and the seed was not rechecked. A fraction is
    # immune: it is free of both the gauge and the level.
    #
    # WHY THE MARGIN FRACTION IS NO LONGER MEASURED FROM THE COUNTS. Earlier versions
    # derived it from observed abundance -- q25/q50 = 0.32, then 4.00/28.62 = 0.14 over
    # the rasterized hull regions. Both DOUBLE-COUNT the habitat gradient: fringe cells
    # hold fewer birds largely because the fringe is poorer habitat, and K_base already
    # says so, so multiplying an observed abundance ratio ON TOP of local K penalizes
    # the same cells twice. It also has a specific bad consequence for the barrier
    # question: at initialization H_k ~ 0, so every native cell has K = k_level and the
    # ratio is applied undiluted -- at 0.14 the Plains-facing eastern MARGIN, which is
    # exactly the front that would push into the barrier, starts at N/K - 0.8 = -0.66 on
    # the emigration logit and spends its first decades filling up instead of pushing.
    # A near-vacuum front is then an alternative pathway to "no crossing" that costs the
    # fit nothing, which is precisely the distortion this rebalance is trying to remove.
    #
    # 0.5 is deliberately a round number and not an estimate: historic 1902 abundances
    # are unknown (pre-1970 BBS counts are a proxy for WHERE the native range was, not
    # how dense it was), so this is safety margin -- a fringe genuinely sparser than the
    # core, but not so sparse that emptiness substitutes for a barrier. The observed
    # quantiles are still printed so the assumption can be re-examined.
    per_cell = locs.groupby(["row", "col"])["SpeciesTotal"].mean()
    if len(per_cell):
        qs = np.percentile(per_cell, [10, 25, 50, 75, 90])
        print("  Native occupied-cell counts q10/q25/q50/q75/q90: "
              + "/".join(f"{q:.1f}" for q in qs))
    _seed_cfg = load_age_model_config()["population_model"]["initpop_seed"]
    margin_ratio = float(_seed_cfg["margin_fraction_of_core"])
    if not 0.0 < margin_ratio <= 1.0:
        raise ValueError(f"margin_fraction_of_core must be in (0, 1]; got {margin_ratio}")
    core_counts, margin_counts = 1.0, margin_ratio
    print(f"  Init SHAPE (dimensionless): core=1.0, margin={margin_ratio:.2f} "
          f"(initpop_seed.margin_fraction_of_core)")

    initpop_counts = np.zeros((ny, nx), dtype=np.float32)
    initpop_counts[mask_margin] = margin_counts
    initpop_counts[mask_core] = core_counts
    print(f"  Init map: core={mask_core.sum()} cells @ {core_counts:.1f} route counts, "
          f"margin={mask_margin.sum()} cells @ {margin_counts:.1f} route counts.")

    uninvaded = land_mask & ~_rasterize(hull_margin.buffer(BUFFER_DISTANCE_METERS))
    ui_rows, ui_cols = np.where(uninvaded)
    print(f"  {len(ui_rows)} uninvaded cells → pseudo-zeros {START_YEAR}-{PSEUDO_ZERO_END_YEAR}.")

    years = range(START_YEAR, PSEUDO_ZERO_END_YEAR + 1)
    p_rows = np.concatenate([ui_rows for _ in years]) if ui_rows.size else np.array([], int)
    p_cols = np.concatenate([ui_cols for _ in years]) if ui_cols.size else np.array([], int)
    p_years = np.concatenate([np.full(len(ui_rows), y) for y in years]) if ui_rows.size else np.array([], int)
    p_counts = np.zeros_like(p_years)
    return initpop_counts, p_rows, p_cols, p_years, p_counts


def main():
    land_mask, ocean_mask, transform, crs, nx, ny = load_grid_reference(MASK_PATH)

    frames = [load_usca_observations()]
    mexico = load_mexico_observations()
    if mexico is not None:
        frames.append(mexico)
    obs = pd.concat(frames, ignore_index=True)
    obs = obs[(obs["Year"] >= START_YEAR) & (obs["Year"] <= END_YEAR)]

    mapped = map_routes_to_grid(obs, load_routes(), transform, crs, nx, ny, land_mask)

    init_counts, p_rows, p_cols, p_years, p_counts = generate_core_margin_initialization(
        mapped, ny, nx, transform, land_mask)
    p_quality = np.full(len(p_years), QUALITY_STANDARD, dtype=int)  # derived absences

    out_path = os.path.join(BBS_PARENT_DIR, "bbs_data_for_python.npz")
    np.savez(
        out_path,
        Nx=nx, Ny=ny,
        land=land_mask.astype(int), ocean=ocean_mask.astype(int),
        obs_rows=np.concatenate([p_rows, mapped["row"].values]).astype(int),
        obs_cols=np.concatenate([p_cols, mapped["col"].values]).astype(int),
        obs_year=np.concatenate([p_years, mapped["Year"].values]).astype(int),
        observed_results=np.concatenate([p_counts, mapped["SpeciesTotal"].values]).astype(int),
        obs_quality=np.concatenate([p_quality, mapped["quality_tier"].values]).astype(int),
        # DIMENSIONLESS shape (core=1, margin=observed ratio). The model scales it by
        # LOCAL K_base at t=0 x initpop_seed.core_fraction_of_local_capacity, so the
        # seed is always a known fraction of the capacity those specific cells can
        # support. Earlier keys initpop_density (gauge-dependent) and
        # initpop_route_counts (level-dependent) both had to be rechecked by hand
        # whenever the gauge or the capacity level moved -- and were not.
        initpop_shape=init_counts,
        initpop_rows=np.where(init_counts > 0)[0],
        initpop_cols=np.where(init_counts > 0)[1],
        N_obs=len(mapped), N_pseudo=len(p_counts),
        unit_distance=1000.0,
        time=END_YEAR - START_YEAR + 1,
    )
    print(f"Done. Saved {out_path}")


if __name__ == "__main__":
    main()
