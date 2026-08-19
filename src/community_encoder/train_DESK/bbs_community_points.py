"""Raw route-level BBS community counts as the community-encoder training target.

The target this replaces was derived from published trend products, and its shape was the
problem rather than its accuracy. ``trend_community`` reconstructs each cell's community as
a closed-form function of ~4 time-invariant numbers (an anchor abundance, a BBS %/yr rate,
an eBird %/yr rate, a scale), capped, then Gaussian-smoothed across space. That makes the
target low-complexity in BOTH axes -- spatially smooth because the products are IDW/model
surfaces which we then blur again, temporally an exponential -- so an interpolator is close
to optimal on it *by construction*. Measured: Val MSE never beat an inverse-distance
baseline in any run (1.49x, 1.97x, 2.10x, 2.02x worse), and our own smoothing at sigma=2
cells was measured to retain only ~78% of real per-cell change (~65% at sigma=5; a planted
localized change collapsed 0.90 -> 0.21).

So no holdout carved from that target can answer whether the covariates carry community
signal: every split is graded against a smooth reconstruction of itself.

What this module writes instead is what the surveyors counted. No caps, no spatial
smoothing, no reconstruction. The QC gates (``RunType != 0``, ``RPID == 101``) are the only
filtering, and they live upstream in ``bbs.load_usca_observations``.

Two properties of BBS make the cell-level target honest rather than a re-smoothing:

- **1.08 routes per covered cell-year** (median 1; 79% of route-bearing cells hold exactly
  one route). Aggregating routes to 27 km cells is very nearly a no-op, not a spatial
  average. The model grid IS 27 km, so this is the target's native resolution.
- **Absences are real.** ``densify_community`` treats the coverage table as authoritative,
  so a surveyed cell-year where a species went unrecorded is a genuine zero rather than a
  gap. That is where most of the turnover signal lives.

What it costs, stated plainly: supervision covers ~4,012 cells of 17,209 land cells (23%),
against 17,205 before. The counter-argument is that the other ~13,000 cells were IDW
interpolation *from these same routes* and so were never independent information -- but Z
in unsurveyed regions is now genuinely less constrained, and that is a real scientific cost.
The cube still spans every cell because it is the encoder applied to covariates, not the
target.

Measured end-to-end on the real 2026 release with a rank-ordered 96-species proxy community
(the production community needs the eBird REST gate, so these will shift slightly):

    114,172 surveyed cell-years over 3,902 cells; 1966-2025, 59 years present, 2020 absent
    16.0 species present per cell-year (median 16, p10 8, max 34); matrix 16.7% dense
    off-diagonal Ruzicka: p50 0.197, mean 0.235
      same-cell different-year pairs  median 0.649
      different-cell pairs            median 0.197      -> 3.3x separation

That last contrast is the argument that this target is better CONDITIONED, not just more
honest. ``validate_bbs_routes`` reads a median off-diagonal similarity near 1.0 as
underpowered, and the reconstructed target sits near that failure mode: it gives every cell
a positive value for every species, so any two cell-years look alike. Raw counts are 16.7%
dense, so the kernel has structure to work with -- and it recovers the ecologically correct
ordering, where a place resembles itself across time far more than it resembles elsewhere.

Artifacts (same format as ``trend_community.build_trend_points``, so ``esk_kernel`` and
``desk_training`` need no loader changes), plus one new file:

    X_points.npy      (N, S) float32   log1p(mean_count), species in community order
    point_index.npy   (N, 3) int32     (row, col, year)
    point_weights.npy (N,)   float32   NEW: observer first-year downweight
    points_meta.json                   provenance, incl. target_source

    python -m src.community_encoder.train_DESK.bbs_community_points
"""
import json
import os

import numpy as np
import pandas as pd

_CELL_YEAR = ["row", "col", "year"]


def cell_year_weights(cov_df, route_cells, fy_flags, first_year_weight):
    """Per-(cell, year) loss weight from the observer first-year flag (pure).

    ``weight = 1 - fy_frac * (1 - first_year_weight)`` where ``fy_frac`` is the share of that
    cell-year's surveyed route-years flagged as the observer's first on that route. With 1.08
    routes per cell-year this is almost always exactly 0 or 1, so the weight is almost always
    either 1 or ``first_year_weight``; the fraction exists for the multi-route minority.

    Downweighting rather than dropping keeps the cell-year's absences, which are real
    observations and carry turnover signal even when the counts are biased low.

    ``cov_df[row,col,year,n_routes]``, ``route_cells[CountryNum,StateNum,Route,row,col]``,
    ``fy_flags[CountryNum,StateNum,Route,Year,first_year]``. Returns a frame
    ``[row, col, year, weight, fy_frac]``.

    A cell-year with no matching flag row keeps weight 1.0. That should not happen -- both
    sides derive from the same QC-passing Weather rows -- so the caller reports the count
    rather than letting it pass silently.
    """
    w = float(first_year_weight)
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"first_year_weight must be in [0, 1], got {w}")
    keys = ["CountryNum", "StateNum", "Route"]
    fy = fy_flags.rename(columns={"Year": "year"})
    joined = fy.merge(route_cells[keys + ["row", "col"]], on=keys, how="inner")
    if joined.empty:
        out = cov_df[_CELL_YEAR].copy()
        out["fy_frac"] = 0.0
        out["weight"] = 1.0
        return out
    frac = (joined.groupby(_CELL_YEAR, as_index=False)["first_year"]
                  .mean().rename(columns={"first_year": "fy_frac"}))
    out = cov_df[_CELL_YEAR].merge(frac, on=_CELL_YEAR, how="left")
    out["fy_frac"] = out["fy_frac"].fillna(0.0)
    out["weight"] = 1.0 - out["fy_frac"] * (1.0 - w)
    return out


def align_to_keys(keys, value_df, column, default=1.0):
    """Gather ``value_df[column]`` onto the densified ``keys`` order (pure).

    ``keys`` is the ``(N, 3)`` int32 ``(row, col, year)`` array ``densify_community`` returns;
    its row order is the authoritative one for every per-point artifact. Returns
    ``(values (N,) float32, n_missing)``. Missing keys take ``default`` -- reported, not hidden,
    because a per-point artifact silently misaligned against X_points would be invisible
    downstream and would attach the wrong weight to the wrong cell-year.
    """
    lut = {(int(r), int(c), int(y)): float(v) for r, c, y, v in
           zip(value_df["row"], value_df["col"], value_df["year"], value_df[column])}
    out = np.full(keys.shape[0], float(default), dtype="float32")
    n_missing = 0
    for i, (r, c, y) in enumerate(keys):
        v = lut.get((int(r), int(c), int(y)))
        if v is None:
            n_missing += 1
        else:
            out[i] = v
    return out, n_missing


def temporal_ema(X, keys, tau):
    """Gap-aware causal EMA along the YEAR axis within each cell (pure).

    Off by default (``tau <= 0`` returns ``X`` unchanged). Temporal smoothing at a FIXED
    LOCATION is the one kind that is defensible here: it borrows no statistical power across
    space, so it does not recreate the pre-smoothed-target problem that motivated this module.
    It mirrors the covariate ``ema_tau``, which the encoder inputs already carry.

    Gap-aware because BBS cell-years are irregular -- 49.5% dense, with real gaps (2020 is
    absent everywhere, a COVID cancellation). A fixed alpha would let a 1-year step and a
    12-year step smooth by the same amount, which is not an EMA of anything. Instead
    ``decay = exp(-dt/tau)`` per actual year gap, so the weight on history falls off with
    elapsed time.

    Per CELL, not per route. With 1.08 routes per cell-year the two are nearly identical; the
    difference is that a cell holding two routes surveyed in different years has them linked
    here and would not be per-route. Documented rather than hidden because the config key
    reads as a temporal choice and it is also, marginally, a spatial one.
    """
    tau = float(tau)
    if tau <= 0:
        return X
    X = np.asarray(X, dtype="float32")
    keys = np.asarray(keys)
    out = X.copy()
    cells = {}
    for i, (r, c, y) in enumerate(keys):
        cells.setdefault((int(r), int(c)), []).append((int(y), i))
    for rows in cells.values():
        rows.sort()
        prev_y, prev_i = rows[0]
        for y, i in rows[1:]:
            decay = float(np.exp(-(y - prev_y) / tau))
            out[i] = (1.0 - decay) * X[i] + decay * out[prev_i]
            prev_y, prev_i = y, i
    return out


def species_order(community_csv):
    """The community species codes, in the order that fixes the X_points column layout."""
    df = pd.read_csv(community_csv)
    return [str(c).lower() for c in df["species_code"]]


def write_points(out_dir, X, keys, weights, meta):
    """Write the four artifacts. Row order of all three arrays is ``keys``' order."""
    os.makedirs(out_dir, exist_ok=True)
    if not (X.shape[0] == keys.shape[0] == weights.shape[0]):
        raise ValueError(f"ragged artifacts: X {X.shape}, keys {keys.shape}, "
                         f"weights {weights.shape}")
    np.save(os.path.join(out_dir, "X_points.npy"), np.asarray(X, dtype="float32"))
    np.save(os.path.join(out_dir, "point_index.npy"), np.asarray(keys, dtype="int32"))
    np.save(os.path.join(out_dir, "point_weights.npy"), np.asarray(weights, dtype="float32"))
    with open(os.path.join(out_dir, "points_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return out_dir


def build_bbs_rows(community_codes, first_year_weight=0.5, temporal_ema_tau=0.0,
                   verbose=True):
    """Raw-BBS half of the target: ``(X, keys, weights, meta)`` (reads the BBS release).

    Every step delegates to an existing, tested loader -- nothing about assembling a BBS
    community matrix is reimplemented here:

        bbs.load_usca_observations(aou_filter=None, return_coverage=True)
        bbs.load_routes / load_grid_reference / map_routes_to_grid
        bbs_community.route_grid_map / build_community_matrix / densify_community
        bbs.load_run_metadata / first_year_flags        (observer downweight)

    ``densify_community`` is the load-bearing one: the COVERAGE table is authoritative, so a
    surveyed cell-year where a species went unrecorded becomes a genuine zero rather than a
    gap. Absences are most of the turnover signal.
    """
    from src.data.identify.bbs_crosswalk import build_crosswalk
    from src.data.preprocess import bbs, bbs_community
    from src.config_utils import load_data_config

    dcfg = load_data_config()
    dr = dcfg["datasets_root"]
    bbs_species = dcfg.get("bbs", {}).get("species_list") or os.path.join(
        dr, "bbs_2026_release", "SpeciesList.csv")
    ebird_tax = os.path.join(dr, "avonet", "eBird_taxonomy.csv")
    ranked = dcfg.get("species_list") or os.path.join(
        dr, "avonet", "reference_community_ranked.csv")

    codes = [str(c).lower() for c in community_codes]
    matched, diag = build_crosswalk(bbs_species, ebird_tax, ranked, top_n=None,
                                    community_codes=codes)
    if verbose:
        print(f"[bbs-points] crosswalk matched {diag['n_matched']}/{diag['n_community']} "
              f"community species; splits={diag['split_aous']}")

    obs_all, coverage = bbs.load_usca_observations(aou_filter=None, return_coverage=True)
    routes = bbs.load_routes()
    land_mask, _ocean, transform, crs, nx, ny = bbs.load_grid_reference(bbs.MASK_PATH)
    route_cells = bbs_community.route_grid_map(routes, transform, crs, nx, ny, land_mask)
    mean_df, cov_df = bbs_community.build_community_matrix(
        obs_all, coverage, matched, route_cells)

    sp_index = {c: i for i, c in enumerate(codes)}
    mean_df = mean_df[mean_df["species_code"].isin(sp_index)].copy()
    mean_df["species_index"] = mean_df["species_code"].map(sp_index).astype(int)

    X_raw, keys, n_dropped = bbs_community.densify_community(
        mean_df["row"], mean_df["col"], mean_df["year"], mean_df["species_index"],
        mean_df["mean_count"], cov_df["row"], cov_df["col"], cov_df["year"], len(codes))

    # Observer first-year downweight, aligned to the densified row order.
    runs = bbs.load_run_metadata()
    fy = bbs.first_year_flags(runs)
    wdf = cell_year_weights(cov_df, route_cells, fy, first_year_weight)
    weights, n_missing_w = align_to_keys(keys, wdf, "weight", default=1.0)

    X_raw = temporal_ema(X_raw, keys, temporal_ema_tau)
    X = bbs_community.log1p_community(X_raw)

    years = sorted({int(y) for y in keys[:, 2]}) if keys.size else []
    meta = {
        "n_rows": int(X.shape[0]), "n_species": len(codes),
        "n_cells": int(len({(int(r), int(c)) for r, c, _ in keys})) if keys.size else 0,
        "year_range": [years[0], years[-1]] if years else None,
        "n_years_present": len(years),
        "years_absent_in_range": [y for y in range(years[0], years[-1] + 1)
                                  if y not in set(years)] if years else [],
        "presence_triples_outside_coverage": int(n_dropped),
        "cell_years_without_a_first_year_flag": int(n_missing_w),
        "first_year_weight": float(first_year_weight),
        "mean_weight": float(weights.mean()) if weights.size else None,
        "n_downweighted": int((weights < 1.0).sum()),
        "temporal_ema_tau": float(temporal_ema_tau),
        "crosswalk": {k: diag[k] for k in ("n_community", "n_matched", "n_unmatched",
                                           "split_aous")},
    }
    if verbose:
        print(f"[bbs-points] {meta['n_rows']:,} surveyed cell-years over "
              f"{meta['n_cells']:,} cells; years {meta['year_range']} "
              f"({meta['n_years_present']} present, absent={meta['years_absent_in_range']})")
        print(f"[bbs-points] first-year downweight: {meta['n_downweighted']:,} rows "
              f"({meta['n_downweighted']/max(meta['n_rows'],1):.1%}), "
              f"mean weight {meta['mean_weight']:.4f}")
        if n_dropped:
            print(f"[bbs-points] dropped {n_dropped:,} presence triples outside coverage "
                  f"(failed QC)")
        if n_missing_w:
            print(f"[bbs-points] WARNING {n_missing_w:,} cell-years had no first-year flag "
                  f"row; they keep weight 1.0. Both sides derive from the same QC-passing "
                  f"Weather rows, so this should be 0 -- investigate before trusting the run.")
    return X, keys, weights, meta


def main():
    import argparse

    from src.config_utils import load_config, load_data_config

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default=None, help="community_trend.csv (default: config)")
    ap.add_argument("--out-dir", default=None, help="default: target.points_dir")
    ap.add_argument("--first-year-weight", type=float, default=None)
    ap.add_argument("--temporal-ema-tau", type=float, default=None)
    args = ap.parse_args()

    cfg, dcfg = load_config(), load_data_config()
    tcfg = cfg.get("target", {}) or {}
    bcfg = tcfg.get("bbs", {}) or {}
    community = args.community or cfg.get("trend", {}).get("community_trend_list") \
        or dcfg["community_trend_list"]
    out_dir = args.out_dir or tcfg.get("points_dir")
    if not out_dir:
        raise SystemExit("no output dir: set target.points_dir or pass --out-dir")
    fyw = args.first_year_weight if args.first_year_weight is not None \
        else float(bcfg.get("first_year_weight", 0.5))
    tau = args.temporal_ema_tau if args.temporal_ema_tau is not None \
        else float(bcfg.get("temporal_ema_tau", 0.0))

    codes = species_order(community)
    X, keys, weights, meta = build_bbs_rows(codes, first_year_weight=fyw,
                                            temporal_ema_tau=tau)
    meta.update({"target_source": "bbs_raw", "species": codes,
                 "community_csv": community, "ruzicka_log1p": True})
    write_points(out_dir, X, keys, weights, meta)
    print(f"[bbs-points] wrote {X.shape[0]:,} x {X.shape[1]} -> {out_dir}")


if __name__ == "__main__":
    main()
