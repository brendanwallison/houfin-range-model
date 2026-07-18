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

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from shapely.geometry import MultiPoint

from src.config_utils import load_data_config
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
BUFFER_DISTANCE_METERS = 1000 * 1000            # 1000 km "uninvaded east" buffer
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


def _mexico_count(species):
    """House-Finch count per Mexico record, robust to schema (SpeciesTotal or Stop sum)."""
    if "SpeciesTotal" in species:
        return pd.to_numeric(species["SpeciesTotal"], errors="coerce")
    stops = [c for c in species.columns if c.lower().startswith("stop") and c[4:].isdigit()]
    if stops:
        return species[stops].apply(pd.to_numeric, errors="coerce").sum(axis=1)
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
    3. Density map: margin cells = 0.001, core cells = 0.1 (core overwrites margin).
    4. Buffer the native hull by 1000 km → the uninvaded east.
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
    initpop_density = np.zeros((ny, nx), dtype=np.float32)
    initpop_density[mask_margin] = 0.001
    initpop_density[mask_core] = 0.1
    print(f"  Init map: core={mask_core.sum()} margin={mask_margin.sum()} cells.")

    uninvaded = land_mask & ~_rasterize(hull_margin.buffer(BUFFER_DISTANCE_METERS))
    ui_rows, ui_cols = np.where(uninvaded)
    print(f"  {len(ui_rows)} uninvaded cells → pseudo-zeros {START_YEAR}-{PSEUDO_ZERO_END_YEAR}.")

    years = range(START_YEAR, PSEUDO_ZERO_END_YEAR + 1)
    p_rows = np.concatenate([ui_rows for _ in years]) if ui_rows.size else np.array([], int)
    p_cols = np.concatenate([ui_cols for _ in years]) if ui_cols.size else np.array([], int)
    p_years = np.concatenate([np.full(len(ui_rows), y) for y in years]) if ui_rows.size else np.array([], int)
    p_counts = np.zeros_like(p_years)
    return initpop_density, p_rows, p_cols, p_years, p_counts


def main():
    land_mask, ocean_mask, transform, crs, nx, ny = load_grid_reference(MASK_PATH)

    frames = [load_usca_observations()]
    mexico = load_mexico_observations()
    if mexico is not None:
        frames.append(mexico)
    obs = pd.concat(frames, ignore_index=True)
    obs = obs[(obs["Year"] >= START_YEAR) & (obs["Year"] <= END_YEAR)]

    mapped = map_routes_to_grid(obs, load_routes(), transform, crs, nx, ny, land_mask)

    init_density, p_rows, p_cols, p_years, p_counts = generate_core_margin_initialization(
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
        initpop_density=init_density,
        initpop_rows=np.where(init_density > 0)[0],
        initpop_cols=np.where(init_density > 0)[1],
        N_obs=len(mapped), N_pseudo=len(p_counts),
        unit_distance=1000.0,
        time=END_YEAR - START_YEAR + 1,
    )
    print(f"Done. Saved {out_path}")


if __name__ == "__main__":
    main()
