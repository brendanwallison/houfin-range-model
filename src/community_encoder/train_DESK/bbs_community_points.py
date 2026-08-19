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


def write_points(out_dir, X, keys, weights, meta, source=None, supervise=None):
    """Write the point-set artifacts. Row order of every array is ``keys``' order.

    ``source`` (int8, 0 = bbs_raw, 1 = ebird_window) and ``supervise`` (bool) are written only
    when given, so a BBS-only build stays a three-array artifact. ``supervise`` is what makes
    the ESK/DESK asymmetry expressible in one file -- see ``concat_sources``.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = keys.shape[0]
    lens = {"X": X.shape[0], "keys": n, "weights": weights.shape[0]}
    if source is not None:
        lens["source"] = len(source)
    if supervise is not None:
        lens["supervise"] = len(supervise)
    if len(set(lens.values())) != 1:
        raise ValueError(f"ragged artifacts: {lens}")
    np.save(os.path.join(out_dir, "X_points.npy"), np.asarray(X, dtype="float32"))
    np.save(os.path.join(out_dir, "point_index.npy"), np.asarray(keys, dtype="int32"))
    np.save(os.path.join(out_dir, "point_weights.npy"), np.asarray(weights, dtype="float32"))
    if source is not None:
        np.save(os.path.join(out_dir, "point_source.npy"), np.asarray(source, dtype="int8"))
    if supervise is not None:
        np.save(os.path.join(out_dir, "point_supervise.npy"),
                np.asarray(supervise, dtype=bool))
    with open(os.path.join(out_dir, "points_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return out_dir


SOURCE_BBS, SOURCE_EBIRD = 0, 1


def concat_sources(X_bbs, K_bbs, W_bbs, X_eb, K_eb, W_eb):
    """Stack the two sources into one point set, resolving duplicate cell-years (pure).

    Returns ``(X, keys, weights, source, supervise)``.

    The asymmetry is deliberate and is the crux of the multi-task design:

    - **ESK sees every row.** A cell-year covered by both products is two independent
      measurements of one latent community, and the kernel should see that they resemble each
      other -- that is exactly the information tying the two scales together after calibration.
    - **DESK supervises one row per cell-year**, marked by ``supervise``. Its target grid is an
      ``(H, W, latent)`` scatter, so a duplicate ``(row, col, year)`` would silently overwrite
      -- last writer wins, with no error and no way to know which source survived.

    **BBS wins the duplicate**, because it is the measurement rather than the model output;
    eBird supervises only cell-years BBS never surveyed. That is also what makes eBird worth
    adding: its footprint is far wider than the ~3,900 BBS cells, so the modern era gets
    supervision almost everywhere while the historical era stays BBS-only.
    """
    X = np.concatenate([np.asarray(X_bbs), np.asarray(X_eb)], axis=0).astype("float32")
    keys = np.concatenate([np.asarray(K_bbs), np.asarray(K_eb)], axis=0).astype("int32")
    weights = np.concatenate([np.asarray(W_bbs), np.asarray(W_eb)]).astype("float32")
    source = np.concatenate([np.full(len(K_bbs), SOURCE_BBS, "int8"),
                             np.full(len(K_eb), SOURCE_EBIRD, "int8")])

    # First occurrence wins, and BBS rows come first by construction.
    supervise = np.zeros(keys.shape[0], dtype=bool)
    seen = set()
    for i, (r, c, y) in enumerate(keys):
        k = (int(r), int(c), int(y))
        if k not in seen:
            seen.add(k)
            supervise[i] = True
    return X, keys, weights, source, supervise


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

    # An all-zero row is a REAL observation -- a surveyed cell-year where none of the community
    # species were recorded -- and densify_community produces it on purpose. But Ruzicka cannot
    # represent it: sum(max) is 0, so the similarity is undefined, and esk_kernel's
    # `denominator > 1e-6` guard silently returns 0 for every pair INCLUDING the diagonal. Such
    # a point would enter the basis with ||z|| = 0, violating the self-similarity-1 contract
    # that desk_training.true_kernel_loss calibrates against. So it is dropped from the POINT
    # SET because the kernel cannot express it, not because it is not real -- and the count is
    # reported so the loss stays visible. (Measured: 20 of 114,172 rows.)
    nonempty = X.sum(axis=1) > 0
    n_empty = int((~nonempty).sum())
    X, keys, weights = X[nonempty], keys[nonempty], weights[nonempty]

    years = sorted({int(y) for y in keys[:, 2]}) if keys.size else []
    meta = {
        "n_rows": int(X.shape[0]), "n_species": len(codes),
        "n_cells": int(len({(int(r), int(c)) for r, c, _ in keys})) if keys.size else 0,
        "year_range": [years[0], years[-1]] if years else None,
        "n_years_present": len(years),
        "years_absent_in_range": [y for y in range(years[0], years[-1] + 1)
                                  if y not in set(years)] if years else [],
        "presence_triples_outside_coverage": int(n_dropped),
        "rows_dropped_all_zero": n_empty,
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
        if n_empty:
            print(f"[bbs-points] dropped {n_empty:,} all-zero rows (surveyed, none of the "
                  f"community species recorded): Ruzicka cannot represent them -- sum(max)=0 "
                  f"leaves self-similarity undefined, so they would enter the basis at ||z||=0")
        if n_dropped:
            print(f"[bbs-points] dropped {n_dropped:,} presence triples outside coverage "
                  f"(failed QC)")
        if n_missing_w:
            print(f"[bbs-points] WARNING {n_missing_w:,} cell-years had no first-year flag "
                  f"row; they keep weight 1.0. Both sides derive from the same QC-passing "
                  f"Weather rows, so this should be 0 -- investigate before trusting the run.")
    return X, keys, weights, meta


def ebird_window_years(start_year, end_year, window=None):
    """Integer years to sample inside a species' trend window (pure).

    Clipped to the species' OWN ``start_year``/``end_year`` from the parquet, then optionally
    further narrowed by ``window``. Never outside: the %/yr rate is a summary OF that window,
    and integrating it beyond is the closed-form extrapolation this whole target replaces.
    """
    lo, hi = int(np.ceil(start_year)), int(np.floor(end_year))
    if window is not None:
        lo, hi = max(lo, int(window[0])), min(hi, int(window[1]))
    return list(range(lo, hi + 1)) if hi >= lo else []


def ebird_window_grid(abd, ppy, mid, year):
    """One year's eBird abundance inside the window: ``abd * (1 + ppy/100)**(year - mid)``.

    Same expression as ``trend_community._trends_abd_anchor``, evaluated INSIDE the window
    instead of extrapolated to the anchor year. No caps: over an ~11-year window even a
    10%/yr rate is under 1.8x fold, so the soft caps that the extrapolated target needs have
    nothing to do here.
    """
    return abd * np.power(1.0 + ppy / 100.0, float(year) - float(mid))


def overlap_pairs(X_eb_log, K_eb, X_bbs_log, K_bbs):
    """``{species_index: (x_bbs, y_ebird)}`` over cell-years both products cover (pure).

    ``x`` is BBS and ``y`` is eBird because eBird is the common frame: it is the richer product
    (17,205 cells against BBS's ~3,900, denser per row, modelled from far more effort), so the
    fitted transform lands on the sparser data rather than on the majority of it.

    Restricted to entries where BOTH values are > 0. Zeros are excluded deliberately: log1p(0)
    is 0, so a mass of double-zero rows would pin the fit through the origin and flatten the
    slope, and a BBS zero against a positive eBird value is a detection difference rather than
    a scale difference -- calibrating on it would fold non-detection into the units.
    """
    bbs_at = {(int(r), int(c), int(y)): i for i, (r, c, y) in enumerate(K_bbs)}
    pairs = {}
    for i, (r, c, y) in enumerate(K_eb):
        j = bbs_at.get((int(r), int(c), int(y)))
        if j is None:
            continue
        ye, xb = X_eb_log[i], X_bbs_log[j]
        both = (ye > 0) & (xb > 0)
        for s in np.nonzero(both)[0]:
            xs, ys = pairs.setdefault(int(s), ([], []))
            xs.append(float(xb[s])); ys.append(float(ye[s]))
    return {s: (np.asarray(x), np.asarray(y)) for s, (x, y) in pairs.items()}


def build_ebird_window_rows(codes, window=None, weight=1.0, verbose=True):
    """eBird half of the target, in eBird's OWN units.

    No calibration here: eBird is the common frame, so it is BBS that gets transformed onto
    this scale (see ``calibrate_bbs_rows``). Returns ``(X, keys, weights, meta)``.
    """
    from src.config_utils import load_data_config
    from .trend_community import _load_trend_grid
    from src.data.preprocess import bbs as bbsmod
    from src.data.preprocess import bbs_community

    dcfg = load_data_config()
    eb_path = dcfg["trends"]["ebird_trend_grid"]
    abd, missing_abd = _load_trend_grid(eb_path, codes, "abd")          # (S, H, W)
    ppy, missing_ppy = _load_trend_grid(eb_path, codes, "abd_ppy")
    z = np.load(eb_path, allow_pickle=True)
    gc = {str(c): i for i, c in enumerate(z["species_code"])}
    sy, ey = z["start_year"], z["end_year"]

    land_mask, _o, _t, _c, _nx, _ny = bbsmod.load_grid_reference(bbsmod.MASK_PATH)
    valid = np.isfinite(abd).any(axis=0) & land_mask
    rr, cc = np.nonzero(valid)

    # Per-species windows differ, so build the union of sampled years and mask per species.
    per_species_years = {}
    for s, c in enumerate(codes):
        if c not in gc:
            continue
        i = gc[c]
        per_species_years[s] = (ebird_window_years(sy[i], ey[i], window),
                                0.5 * (float(sy[i]) + float(ey[i])))
    all_years = sorted({y for ys, _ in per_species_years.values() for y in ys})
    if not all_years:
        raise SystemExit(
            f"no eBird trend years inside window={window}. The product's own start_year/"
            f"end_year bound this, and extrapolating outside is refused by design.")

    # ``have`` distinguishes "no data" from "zero abundance", and it is load-bearing. A
    # species outside its OWN trend window, or a cell outside its footprint, has nothing to
    # say -- but log1p(0) is 0, and the calibration is affine, so a + b*0 = a would hand those
    # entries the intercept as if it were a measured abundance. Caught by a synthetic-grid
    # smoke test where one species had a narrow 2014-2016 window and came out at 0.373 in
    # 2012. The mask forces them back to exactly 0 after calibration.
    rows, keys, have = [], [], []
    for y in all_years:
        grid = np.zeros((len(codes), rr.size), dtype="float64")
        ok = np.zeros((len(codes), rr.size), dtype=bool)
        for s, (ys, mid) in per_species_years.items():
            if y not in ys:
                continue                                   # outside this species' own window
            a_s, p_s = abd[s][rr, cc], ppy[s][rr, cc]
            ok[s] = np.isfinite(a_s) & np.isfinite(p_s)
            grid[s] = np.where(ok[s], ebird_window_grid(a_s, p_s, mid, y), 0.0)
        rows.append(np.nan_to_num(grid.T, nan=0.0))                    # (M, S)
        have.append(ok.T)
        keys.append(np.stack([rr, cc, np.full(rr.size, y)], axis=1))
    X_raw = np.concatenate(rows, axis=0)
    have = np.concatenate(have, axis=0)
    K = np.concatenate(keys, axis=0).astype("int32")
    X = bbs_community.log1p_community(X_raw)
    X = np.where(have, X, 0.0).astype("float32")
    W = np.full(K.shape[0], float(weight), dtype="float32")

    # A row with no data for ANY species is not an observation of an empty community -- it is
    # not an observation. Drop it rather than feeding an all-zero vector to Ruzicka, where it
    # would have an undefined similarity to everything.
    keep = have.any(axis=1)
    n_empty = int((~keep).sum())
    X, K, W, have = X[keep], K[keep], W[keep], have[keep]

    meta = {
        "n_rows": int(X.shape[0]), "n_cells": int(rr.size),
        "years": all_years, "window": list(window) if window else None,
        "weight": float(weight),
        "species_missing_abd": missing_abd, "species_missing_abd_ppy": missing_ppy,
        "n_species_with_a_window": len(per_species_years),
        "n_rows_dropped_no_data": n_empty,
        "mean_species_with_data_per_row": float(have.sum(1).mean()) if have.size else 0.0,
    }
    if verbose:
        print(f"[ebird-window] {meta['n_rows']:,} rows over {rr.size:,} cells x "
              f"{len(all_years)} years {all_years[0]}..{all_years[-1]}")
        print(f"[ebird-window] {meta['mean_species_with_data_per_row']:.1f} species with data "
              f"per row; dropped {n_empty:,} rows with no data for any species")
    return X, K, W, meta


def calibrate_bbs_rows(X_bbs_log, K_bbs, X_eb_log, K_eb, codes, prior_slope=1.0,
                       prior_intercept=0.0, prior_log_slope_sd=0.5, prior_intercept_sd=1.0,
                       verbose=True):
    """Transform the BBS rows onto the eBird scale. Returns ``(X_bbs_calibrated, meta)``.

    eBird is the common frame because it is the richer product, so the fitted transform lands
    on the sparser data. Every species gets its own slope and intercept, shrunk toward a
    population relationship in proportion to how much its own overlapping data actually says --
    so a species with little overlap, or with no real agreement between the two products, is
    pulled to the population estimate rather than fitted to noise, and one with no overlap at
    all lands exactly on it.
    """
    from .bbs_ebird_calibration import (apply_calibration, calibration_meta,
                                        fit_hierarchical_calibration, report_zero_effect)

    pairs = overlap_pairs(X_eb_log, K_eb, X_bbs_log, K_bbs)
    cal = fit_hierarchical_calibration(
        pairs, len(codes), prior_slope=prior_slope, prior_intercept=prior_intercept,
        prior_log_slope_sd=prior_log_slope_sd, prior_intercept_sd=prior_intercept_sd,
        verbose=verbose)
    X = apply_calibration(X_bbs_log, cal)
    # What the calibration does to ABSENCES. Cannot be measured off-cluster (it needs the real
    # eBird grid), and it decides whether the affine map needs anything doing about zeros at
    # all -- see apply_calibration. Reported into the log and the meta rather than acted on.
    meta = calibration_meta(cal, codes)
    meta["zero_effect"] = report_zero_effect(X_bbs_log, cal, verbose=verbose)
    return X, meta


def main():
    import argparse

    from src.config_utils import load_config, load_data_config

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default=None, help="community_trend.csv (default: config)")
    ap.add_argument("--out-dir", default=None, help="default: target.points_dir")
    ap.add_argument("--first-year-weight", type=float, default=None)
    ap.add_argument("--temporal-ema-tau", type=float, default=None)
    ap.add_argument("--no-ebird", action="store_true",
                    help="BBS rows only (skip the eBird window half and the calibration)")
    args = ap.parse_args()

    cfg, dcfg = load_config(), load_data_config()
    tcfg = cfg.get("target", {}) or {}
    bcfg = tcfg.get("bbs", {}) or {}
    ecfg = tcfg.get("ebird_window", {}) or {}
    ccfg = tcfg.get("calibration", {}) or {}
    community = args.community or cfg.get("trend", {}).get("community_trend_list") \
        or dcfg["community_trend_list"]
    out_dir = args.out_dir or tcfg.get("points_dir")
    if not out_dir:
        raise SystemExit("no output dir: set target.points_dir or pass --out-dir")
    fyw = args.first_year_weight if args.first_year_weight is not None \
        else float(bcfg.get("first_year_weight", 0.5))
    tau = args.temporal_ema_tau if args.temporal_ema_tau is not None \
        else float(bcfg.get("temporal_ema_tau", 0.0))
    use_ebird = (not args.no_ebird) and bool(ecfg.get("enabled", True))

    codes = species_order(community)
    X, keys, weights, meta = build_bbs_rows(codes, first_year_weight=fyw,
                                            temporal_ema_tau=tau)
    meta = {"bbs": meta}
    source = supervise = None

    if use_ebird:
        win = None
        if ecfg.get("start_year") and ecfg.get("end_year"):
            win = (int(ecfg["start_year"]), int(ecfg["end_year"]))
        Xe, Ke, We, emeta = build_ebird_window_rows(
            codes, window=win, weight=float(ecfg.get("weight", 1.0)))
        # BBS is what gets transformed: eBird is the common frame, so the fitted transform
        # lands on the sparser product rather than on the majority of the data.
        X, cmeta = calibrate_bbs_rows(
            X, keys, Xe, Ke, codes,
            prior_slope=float(ccfg.get("prior_slope", 1.0)),
            prior_intercept=float(ccfg.get("prior_intercept", 0.0)),
            prior_log_slope_sd=float(ccfg.get("prior_log_slope_sd", 0.5)),
            prior_intercept_sd=float(ccfg.get("prior_intercept_sd", 1.0)))
        meta["calibration"] = cmeta
        X, keys, weights, source, supervise = concat_sources(X, keys, weights, Xe, Ke, We)
        meta["ebird_window"] = emeta
        n_sup_eb = int((supervise & (source == SOURCE_EBIRD)).sum())
        meta["combined"] = {
            "n_rows": int(X.shape[0]),
            "n_supervised": int(supervise.sum()),
            "n_supervised_bbs": int((supervise & (source == SOURCE_BBS)).sum()),
            "n_supervised_ebird": n_sup_eb,
            "n_duplicate_cell_years": int(X.shape[0] - supervise.sum()),
            "n_supervised_cells": int(len({(int(r), int(c))
                                           for r, c, _ in keys[supervise]})),
        }
        cb = meta["combined"]
        print(f"[points] combined {cb['n_rows']:,} rows; supervised {cb['n_supervised']:,} "
              f"({cb['n_supervised_bbs']:,} BBS + {n_sup_eb:,} eBird) over "
              f"{cb['n_supervised_cells']:,} cells; "
              f"{cb['n_duplicate_cell_years']:,} duplicate cell-years kept for the kernel")

    meta.update({"target_source": "bbs_raw+ebird_window" if use_ebird else "bbs_raw",
                 "species": codes, "n_species": len(codes),
                 "community_csv": community, "ruzicka_log1p": True,
                 "n_rows": int(X.shape[0]),
                 "n_recent": int(supervise.sum()) if supervise is not None
                 else int(X.shape[0])})
    write_points(out_dir, X, keys, weights, meta, source=source, supervise=supervise)
    print(f"[points] wrote {X.shape[0]:,} x {X.shape[1]} -> {out_dir}")


if __name__ == "__main__":
    main()
