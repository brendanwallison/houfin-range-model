"""Ingest BBS community counts onto the model grid (per species × cell × year).

Feeds the amplitude-modulation step (``spacetime_community``). Uses the all-species
BBS load (``bbs.load_usca_observations(aou_filter=None)``) + QC coverage, maps AOU →
eBird ``species_code`` via the crosswalk (summing lumps), maps routes to grid cells,
and produces, per ``(cell, year)``:

- ``mean_count`` per community species = Σ SpeciesTotal over that cell-year's
  QC-passing route-years ÷ that cell-year's route-year **coverage** (so a covered
  cell where a species went unrecorded contributes a genuine 0 downstream), and
- the coverage count itself (effort), for the smoother's weighting/support.

No temporal binning here — the continuous spatiotemporal smoother (B3) replaces
decadal buckets. Output is a compact long-form ``.npz`` the smoother densifies.

    python -m src.data.preprocess.bbs_community --bbs-species <SpeciesList.csv>
"""
import argparse
import os

import numpy as np
import pandas as pd

from src.config_utils import load_config, load_data_config
from src.data.identify.bbs_crosswalk import build_crosswalk
from src.data.preprocess import bbs

_KEYS = ["CountryNum", "StateNum", "Route"]


def route_grid_map(routes, transform, crs, nx, ny, land_mask):
    """Unique-route → (row, col) table (land cells only), via ``bbs.map_routes_to_grid``."""
    uniq = routes[_KEYS].drop_duplicates().copy()
    mapped = bbs.map_routes_to_grid(uniq, routes, transform, crs, nx, ny, land_mask)
    return mapped[_KEYS + ["row", "col"]].drop_duplicates()


def ebird_stack_species(ebird_folder):
    """eBird ``species_code``s actually projected into the stack (parsed from the grid
    tif filenames ``{code}_abundance_median_*_grid.tif``). This is the authoritative
    community for amplitude modulation -- the BBS community must match it, not re-derive
    top-N-by-rank from the ranked CSV (which diverges by ~17 species)."""
    import glob
    codes = {os.path.basename(f).split("_abundance_median")[0]
             for f in glob.glob(os.path.join(ebird_folder, "*_abundance_median_*_grid.tif"))}
    return sorted(codes)


def build_community_matrix(obs_all, coverage, crosswalk, route_cells):
    """Aggregate to per-(cell,year,species) mean counts + per-(cell,year) coverage (pure).

    ``obs_all``: all-species obs ``[CountryNum,StateNum,Route,Year,AOU,SpeciesTotal]``.
    ``coverage``: QC route-years ``[CountryNum,StateNum,Route,Year]``.
    ``crosswalk``: ``[aou, species_code]`` (community; lumps = repeated species_code).
    ``route_cells``: ``[CountryNum,StateNum,Route,row,col]``.
    Returns ``(mean_df, cov_df)``:
    ``mean_df[row,col,year,species_code,mean_count]`` (present species only) and
    ``cov_df[row,col,year,n_routes]`` (effort). Absences (covered cell-year, species
    unrecorded) are implicit — recovered as 0 by the smoother against ``cov_df``.
    """
    # AOU -> eBird species_code, restricted to the community; sum lumps.
    obs = obs_all.merge(crosswalk[["aou", "species_code"]],
                        left_on="AOU", right_on="aou", how="inner")
    obs = obs.merge(route_cells, on=_KEYS, how="inner")
    # Sum lumps + multiple routes/records within a cell-year first (route-year level),
    # then aggregate to the cell.
    species_sum = (obs.groupby(["row", "col", "Year", "species_code"], as_index=False)
                      ["SpeciesTotal"].sum())

    cov = coverage.merge(route_cells, on=_KEYS, how="inner")
    cov_df = (cov.groupby(["row", "col", "Year"], as_index=False).size()
                 .rename(columns={"size": "n_routes", "Year": "year"}))

    mean_df = species_sum.merge(
        cov_df.rename(columns={"year": "Year"}), on=["row", "col", "Year"], how="left")
    mean_df["mean_count"] = mean_df["SpeciesTotal"] / mean_df["n_routes"].clip(lower=1)
    mean_df = mean_df.rename(columns={"Year": "year"})[
        ["row", "col", "year", "species_code", "mean_count"]]
    return mean_df, cov_df


def save_npz(out_path, mean_df, cov_df, species_codes, ny, nx):
    """Serialize the community matrix to a compact long-form ``.npz``."""
    code_ix = {c: i for i, c in enumerate(species_codes)}
    mean_df = mean_df[mean_df["species_code"].isin(code_ix)]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        # per (cell, year, species) present-count triples
        row=mean_df["row"].to_numpy(np.int32),
        col=mean_df["col"].to_numpy(np.int32),
        year=mean_df["year"].to_numpy(np.int32),
        species_index=mean_df["species_code"].map(code_ix).to_numpy(np.int32),
        mean_count=mean_df["mean_count"].to_numpy(np.float32),
        # per (cell, year) coverage/effort
        cov_row=cov_df["row"].to_numpy(np.int32),
        cov_col=cov_df["col"].to_numpy(np.int32),
        cov_year=cov_df["year"].to_numpy(np.int32),
        cov_n=cov_df["n_routes"].to_numpy(np.int32),
        species_codes=np.array(species_codes, dtype=object),
        dims=np.array([ny, nx], dtype=np.int32),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbs-species", default=None, help="BBS SpeciesList CSV")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-n", type=int, default=None)
    args = ap.parse_args()

    dr = load_data_config()["datasets_root"]
    cfg = load_config()
    bcfg = cfg.get("bbs", {})
    bbs_species = args.bbs_species or bcfg.get("species_list") \
        or os.path.join(bbs.BBS_PARENT_DIR, "SpeciesList.csv")

    # Authoritative community = the species the eBird stack actually projected (top-N
    # COMPLETE). Crosswalk BBS to EXACTLY those, so every BBS-community species has an eBird
    # block to modulate (matched == community). Re-deriving top-N-by-rank here diverged from
    # the eBird stack by ~17 species -- freezing common movers (starling, mockingbird, ...).
    ebird_codes = ebird_stack_species(cfg["paths"]["ebird_folder"])
    if not ebird_codes:
        raise SystemExit(f"no projected eBird grids in {cfg['paths']['ebird_folder']}; "
                         "run the ebird stage before bbs_community")
    crosswalk, _ = build_crosswalk(
        bbs_species, os.path.join(dr, "avonet", "eBird_taxonomy.csv"),
        os.path.join(dr, "avonet", "reference_community_ranked.csv"),
        community_codes=ebird_codes)
    species_codes = list(dict.fromkeys(crosswalk["species_code"]))  # community order

    obs_all, coverage = bbs.load_usca_observations(aou_filter=None, return_coverage=True)
    routes = bbs.load_routes()
    land_mask, _, transform, crs, nx, ny = bbs.load_grid_reference(bbs.MASK_PATH)
    route_cells = route_grid_map(routes, transform, crs, nx, ny, land_mask)

    mean_df, cov_df = build_community_matrix(obs_all, coverage, crosswalk, route_cells)
    out = args.out or os.path.join(dr, "bbs", "community_matrix.npz")
    save_npz(out, mean_df, cov_df, species_codes, ny, nx)
    print(f"[bbs_community] {len(species_codes)} species, {len(mean_df)} present "
          f"cell-year-species, {len(cov_df)} covered cell-years -> {out}")


if __name__ == "__main__":
    main()
