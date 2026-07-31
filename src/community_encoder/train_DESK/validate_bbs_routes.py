"""Route-level BBS validation: DESK vs a no-change null, in similarity space.

WHY THIS EXISTS. Every other DESK validation grades against an IDW-interpolated target: the
BBS trend product is USGS's published inverse-distance surface (see ``bbs_trend.py``), further
smoothed by ``trend.smooth_sigma_cells``. So the IDW baseline that beats DESK on held-out cells
is approximately the target's own construction operator -- the metric rewards mimicking a
smoother, and no number computed against it can settle whether DESK has real temporal skill.

Raw BBS route counts escape that circularity: they are the observations the IDW surface was
interpolated FROM. Restricting evaluation to cell-years that were *actually surveyed* cannot be
won by reproducing an interpolator.

WHAT IT MEASURES. One question: does DESK beat a no-change null at reproducing observed
community turnover at surveyed sites -- overall, and separately in a modern and an early window.

    S_true = Ruzicka(log1p observed community)          <- raw BBS, routes averaged per cell-year
    S_desk = Z_ema @ Z_ema.T                            <- the DOT product: the kernel contract
    S_nc   = same dot, but DESK's z at each cell's MODERN year copied to all its years

    PRIMARY   rmse_skill = 1 - rmse(S_desk, S_true) / rmse(S_nc, S_true)
    SECONDARY cka_gain   = CKA(S_true, S_desk) - CKA(S_true, S_nc)

WHY THE DOT PRODUCT. ESK/DESK promise ``Z(x) dot Z(x') ~= uncentered Ruzicka(x, x')`` (recorded as
``kernel_contract`` in every ESK/cube meta), and ``desk_training.true_kernel_loss`` is the MSE
between that dot product and the Ruzicka similarity of the raw communities. The dot product is
therefore both the model's stated output and the quantity it was optimized for. Because the
contract makes the two directly comparable rather than merely correlated, they can be compared
ELEMENTWISE -- which is strictly more informative than CKA, and catches a norm-calibration error
that CKA is provably blind to (see tests). Cosine is reported as a secondary column only, to
separate an angular failure from a ||z|| scaling failure.

BOTH MODEL-SIDE MATRICES USE THE SAME FUNCTIONAL, so the ONLY difference between them is temporal
variation. That is load-bearing. An earlier version built the null as Ruzicka on the observed
modern community; because truth is also Ruzicka, that null was scored in truth's own metric while
DESK was not, and the difference between the two similarity functions swamped the temporal signal
(measured: a temporally-neutral model scored -0.28 purely from the mismatch). The observed-space
null is still reported, as ``cka_nochange_observed`` / ``rmse_nochange_observed``, but only as the
achievable-ceiling reference -- never differenced against the DESK columns.

THE GAIN IS THE READOUT, NOT ABSOLUTE CKA. A mixed similarity matrix is dominated by
between-cell (spatial) structure -- the same reason ``Val`` MSE says little about temporal skill.
The no-change null has ZERO temporal variation by construction, so differencing against it
isolates exactly the temporal component. A large ``cka_desk`` with ``cka_gain <= 0`` means DESK
reproduced the map and nothing about change.

WHY SIMILARITY SPACE AND NOT Z SPACE. The ESK landmarks are frozen at grid-product magnitude, so
projecting route counts into ``esk/spacetime`` would require inventing a per-species BBS->eBird
scale factor, and if it were off that artifact would dominate. Comparing similarities needs no
such factor and no new estimated quantity.

READ ``rmse_skill``, NOT the absolute ``rmse_desk``. Bare Ruzicka is invariant when both arguments
are rescaled, but ``log1p`` is NOT (``log1p(c*x) != c*log1p(x)``), so the absolute count scale does
survive into ``S_true``. Training applied log1p to grid-scale abundances; this applies it to
per-route mean counts, and those magnitudes differ. That inflates ``rmse_desk`` and ``bias_desk``
by an unknown offset, so their ABSOLUTE values are diagnostic only.

``rmse_skill`` is unaffected: it is a ratio of two errors measured against the SAME ``S_true``, so
whatever the units mismatch does, it does to DESK and the null alike. Same for ``cka_gain``. The
comparison between the two models is sound; the calibration of either one against observed
Ruzicka in absolute terms is not.

    python -m src.community_encoder.train_DESK.validate_bbs_routes
"""
import argparse
import json
import os
import time

import numpy as np

from src.config_utils import load_config

from .desk_training import apply_output_ema
from .validate_spacetime import linear_cka, mantel_r, ruzicka_rect

# Window definitions. 1966, not 1965: BBS begins in 1966, so a 1965 bound would silently be a
# 1966 bound and misreport the window in the output.
MODERN_WINDOW = (2010, 2025)
EARLY_WINDOW = (1966, 1980)


# ----------------------------- pure: observed community -----------------------------

def densify_community(row, col, year, species_index, mean_count,
                      cov_row, cov_col, cov_year, n_species):
    """Long-form BBS community triples → dense per-surveyed-cell-year matrix (pure).

    ``build_community_matrix`` emits PRESENT species only; absences are implicit. The
    authoritative row set is therefore the COVERAGE table (``cov_*``), not the presence
    triples: a surveyed cell-year where a species went unrecorded is a genuine zero, and a
    surveyed cell-year where *nothing* was recorded is a real all-zero row, not a missing one.
    Building rows from the presence triples instead would silently drop exactly the cell-years
    that carry the strongest turnover signal.

    Returns ``(X (N, n_species) float32 raw counts, keys (N, 3) int32 [row, col, year],
    n_dropped)`` ordered by ``(row, col, year)``. Presence triples for cell-years absent from
    coverage are dropped (they failed QC) and reported as ``n_dropped``.
    """
    keys = np.stack([np.asarray(cov_row, "int64"),
                     np.asarray(cov_col, "int64"),
                     np.asarray(cov_year, "int64")], axis=1)
    if keys.shape[0] == 0:
        return np.zeros((0, n_species), "float32"), np.zeros((0, 3), "int32")
    # Deduplicate + sort the coverage rows so the output order is deterministic and a repeated
    # cell-year in cov_* cannot create two rows for one site-year.
    keys = np.unique(keys, axis=0)
    order = {(int(r), int(c), int(y)): i for i, (r, c, y) in enumerate(keys)}

    X = np.zeros((keys.shape[0], int(n_species)), dtype="float32")
    dropped = 0
    for r, c, y, s, v in zip(np.asarray(row), np.asarray(col), np.asarray(year),
                             np.asarray(species_index), np.asarray(mean_count)):
        i = order.get((int(r), int(c), int(y)))
        if i is None:                       # present species in an uncovered cell-year
            dropped += 1
            continue
        si = int(s)
        if 0 <= si < X.shape[1]:
            X[i, si] += float(v)            # += so crosswalk lumps accumulate, matching bbs_community
    return X, keys.astype("int32"), dropped


def log1p_community(X):
    """``log1p(clip(x, 0, None))`` -- the transform the spacetime ESK was fit with.

    Matches ``trend_community.py`` (``ruzicka_log1p``, default True). Ruzicka is scale-sensitive
    per-argument, so the transform must match what training used or the similarity structure is
    not the same quantity.
    """
    return np.log1p(np.clip(np.asarray(X, "float64"), 0.0, None)).astype("float32")


def modern_reference_rows(keys, modern_window=MODERN_WINDOW):
    """Map each cell to its most recent surveyed row inside ``modern_window`` (pure).

    Returns ``(nc_src (N,) int64, keep (N,) bool)`` where ``nc_src[i]`` is the row index holding
    cell ``i``'s modern observed community and ``keep[i]`` is False for every row of a cell with
    NO surveyed year in the window. Those cells are dropped from EVERY matrix (truth included) --
    the same ``has_rec`` gate ``zspace_reconstruction`` applies, at cell granularity. Without the
    gate a no-change null cannot be defined for those sites, and silently keeping them in truth
    while excluding them from the null would compare two different row sets.
    """
    keys = np.asarray(keys)
    n = keys.shape[0]
    lo, hi = int(modern_window[0]), int(modern_window[1])
    best = {}
    for i in range(n):
        r, c, y = int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])
        if lo <= y <= hi:
            cell = (r, c)
            if cell not in best or y > best[cell][0]:
                best[cell] = (y, i)
    nc_src = np.full(n, -1, dtype="int64")
    for i in range(n):
        hit = best.get((int(keys[i, 0]), int(keys[i, 1])))
        if hit is not None:
            nc_src[i] = hit[1]
    return nc_src, nc_src >= 0


def dot_gram(Z):
    """``Z @ Z.T`` → ``(n, n)``. THE prediction, per the kernel contract.

    ESK/DESK promise ``Z(x) dot Z(x') ~= uncentered Ruzicka(x, x')`` (recorded as
    ``kernel_contract`` in every ESK/cube meta), and ``desk_training.true_kernel_loss`` is
    literally the MSE between this dot product and the Ruzicka similarity of the raw communities.
    So the dot product is what the model was optimized to produce and what must be graded.

    Using cosine here instead would discard the vector norms, which are part of the prediction --
    the contract fixes the SCALE, not just the direction. Cosine is the right choice for
    per-cell TURNOVER (``1 - similarity``), where a norm deficit masquerades as change; it is the
    wrong choice for asking whether the kernel reproduces observed similarity.
    """
    Z = np.asarray(Z, "float64")
    return Z @ Z.T


def cosine_gram(Z):
    """Row-normalized ``Z @ Z.T`` -- SECONDARY diagnostic only, not the graded quantity.

    Reported alongside the dot product because the norm deficit is real and measured (``||Z||^2``
    medians 0.73-0.81 rather than the contract's 1.0). If the dot-product result is poor while
    cosine is good, the failure is calibration of ``||Z||``; if both are poor, it is the angular
    structure. Keeping both separates those two explanations.
    """
    Z = np.asarray(Z, "float64")
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    Zn = Z / np.where(nrm > 0, nrm, 1.0)
    return Zn @ Zn.T


def _offdiag(S):
    """Upper-triangle off-diagonal entries. The diagonal is self-similarity (trivially 1 for
    Ruzicka and ``||z||^2`` for the dot) and would flatter both models identically."""
    return np.asarray(S)[np.triu_indices_from(np.asarray(S), k=1)]


def kernel_error(S_true, S_pred):
    """Elementwise agreement between a predicted kernel and observed Ruzicka (pure).

    Only meaningful for the DOT product, where the contract says the two should be EQUAL rather
    than merely correlated -- so this is a direct calibration check that CKA cannot give. Returns
    RMSE, mean signed bias (negative = predicting too little similarity) and Pearson r.
    """
    t, p = _offdiag(S_true), _offdiag(S_pred)
    d = p - t
    r = float(np.corrcoef(t, p)[0, 1]) if t.size > 1 and t.std() > 0 and p.std() > 0 else 0.0
    return {"rmse": float(np.sqrt(np.mean(d ** 2))), "bias": float(np.mean(d)), "pearson_r": r}


def bucket_metrics(S_true, S_desk, S_nc, S_nc_obs=None, S_desk_cos=None, S_nc_cos=None):
    """Grade DESK's kernel against observed Ruzicka, versus the frozen-modern null (pure).

    ``S_desk``/``S_nc`` are DOT-product Grams -- the quantity the contract and the training loss
    are both stated in. ``S_desk`` and ``S_nc`` must use the same functional as each other so
    their difference isolates temporal variation.

    PRIMARY metric is ``rmse_skill``: how much of the null's kernel error DESK removes, on the
    same scale as the similarities themselves. This is available only because the contract makes
    dot and Ruzicka directly comparable; CKA can only say the two structures covary.

    ``S_nc_obs`` (Ruzicka on the modern observed community) is reported as the achievable-ceiling
    reference. ``S_desk_cos``/``S_nc_cos`` are the cosine variants -- secondary, for separating a
    norm-calibration failure from an angular one.
    """
    cka_d, cka_n = linear_cka(S_true, S_desk), linear_cka(S_true, S_nc)
    man_d, man_n = mantel_r(S_true, S_desk), mantel_r(S_true, S_nc)
    e_d, e_n = kernel_error(S_true, S_desk), kernel_error(S_true, S_nc)
    out = {
        "n_rows": int(np.asarray(S_true).shape[0]),
        # --- primary: elementwise kernel agreement (dot product vs Ruzicka) ---
        "rmse_desk": e_d["rmse"], "rmse_nochange": e_n["rmse"],
        "rmse_skill": (1.0 - e_d["rmse"] / e_n["rmse"]) if e_n["rmse"] > 0 else 0.0,
        "bias_desk": e_d["bias"], "bias_nochange": e_n["bias"],
        "pearson_desk": e_d["pearson_r"], "pearson_nochange": e_n["pearson_r"],
        # --- secondary: structure-only agreement ---
        "cka_desk": cka_d, "cka_nochange": cka_n, "cka_gain": cka_d - cka_n,
        "mantel_desk": man_d, "mantel_nochange": man_n, "mantel_gain": man_d - man_n,
    }
    if S_nc_obs is not None:
        out["cka_nochange_observed"] = linear_cka(S_true, S_nc_obs)
        out["rmse_nochange_observed"] = kernel_error(S_true, S_nc_obs)["rmse"]
    if S_desk_cos is not None and S_nc_cos is not None:
        ck_d, ck_n = linear_cka(S_true, S_desk_cos), linear_cka(S_true, S_nc_cos)
        out.update({"cka_desk_cosine": ck_d, "cka_nochange_cosine": ck_n,
                    "cka_gain_cosine": ck_d - ck_n})
    return out


def stratified_sample(keys, n_sample, rng, windows=(MODERN_WINDOW, EARLY_WINDOW), is_heldout=None):
    """Sample row indices, giving each (year-window x holdout) cell a guaranteed share (pure).

    Two separate imbalances make a uniform sample useless here, and BOTH have to be corrected or
    the decisive bucket is estimated from a handful of rows:

    - BBS coverage grows roughly monotonically, so a uniform sample is swamped by the modern
      decades and the early window ends up too thin to report.
    - Held-out cells are a small fraction of rows (measured: ~7%), so they land in the sample by
      luck. In the first run this left ``heldout/early`` with 94 rows and ``heldout/modern`` with
      73 -- and held-out is the ONLY split that answers whether the model generalizes, so the
      table's most important cells were its least reliable.

    Crossing the two axes fixes the second: strata are (modern, early, other) x (train, heldout),
    each drawing an equal share of the budget, with short strata backfilled from the remainder.
    Because the similarity matrices are n^2, raising ``n_sample`` is a quadratic-cost way to buy
    held-out rows; stratifying buys them for free.

    ``is_heldout`` (``(N,)`` bool, optional) omitted reduces this to year-window stratification.
    """
    keys = np.asarray(keys)
    n = keys.shape[0]
    if n <= n_sample:
        return np.arange(n)
    yrs = keys[:, 2]

    year_strata, claimed = [], np.zeros(n, bool)
    for lo, hi in windows:
        m = (yrs >= lo) & (yrs <= hi) & ~claimed
        year_strata.append(m)
        claimed |= m
    year_strata.append(~claimed)

    if is_heldout is None:
        splits = [np.ones(n, bool)]
    else:
        ho = np.asarray(is_heldout, bool)
        splits = [~ho, ho]
    strata = [np.where(y & s)[0] for y in year_strata for s in splits]
    strata = [ix for ix in strata if ix.size]
    if not strata:
        return np.arange(min(n, n_sample))

    per = max(1, n_sample // len(strata))
    picked = [rng.choice(ix, size=min(per, ix.size), replace=False) for ix in strata]
    out = np.concatenate(picked)
    if out.size < n_sample:                                  # backfill from whatever is left
        rest = np.setdiff1d(np.arange(n), out, assume_unique=False)
        if rest.size:
            extra = rng.choice(rest, size=min(n_sample - out.size, rest.size), replace=False)
            out = np.concatenate([out, extra])
    return np.sort(np.unique(out))


# ----------------------------- IO / driver -----------------------------

def load_observed(config):
    """Build the observed route-level community from RAW BBS → ``(X_log, keys, meta)``.

    The community is ``community_trend_list`` (``community_trend.csv``) -- the same list
    ``trend_community.build_trend_points`` uses -- so truth is defined over exactly the community
    DESK trains on. Nothing here is reimplemented: the route QC filter, AOU->species_code
    crosswalk, route->cell mapping and per-cell-year averaging are all the existing functions,
    with the community passed in via ``build_crosswalk(community_codes=...)``.

    Reads no precomputed community artifact: it goes straight from the raw BBS release, so there
    is no separate build step and no stale intermediate to fall out of sync with the trend
    community.
    """
    from src.config_utils import load_data_config
    from src.data.identify.bbs_crosswalk import build_crosswalk
    from src.data.preprocess import bbs
    from src.data.preprocess.bbs_community import build_community_matrix, route_grid_map
    import pandas as pd

    dcfg = load_data_config()
    tc = config.get("trend", {}) or {}
    community_csv = tc.get("community_trend_list") or dcfg["community_trend_list"]
    codes = [str(c) for c in pd.read_csv(community_csv)["species_code"].tolist()]
    if not codes:
        raise ValueError(f"no species_code rows in {community_csv}")

    dr = dcfg["datasets_root"]
    bbs_species = config.get("bbs", {}).get("species_list") or \
        os.path.join(bbs.BBS_PARENT_DIR, "SpeciesList.csv")
    crosswalk, _ = build_crosswalk(
        bbs_species, os.path.join(dr, "avonet", "eBird_taxonomy.csv"),
        os.path.join(dr, "avonet", "reference_community_ranked.csv"),
        community_codes=codes)                      # <- DESK's community, not the weekly stack
    species = list(dict.fromkeys(crosswalk["species_code"]))
    if len(species) < 3:
        raise ValueError(
            f"only {len(species)} of {len(codes)} community_trend species crosswalked to a BBS "
            f"AOU (from {bbs_species}); cannot build a route-level community")

    obs_all, coverage = bbs.load_usca_observations(aou_filter=None, return_coverage=True)
    routes = bbs.load_routes()
    land_mask, _, transform, crs, nx, ny = bbs.load_grid_reference(bbs.MASK_PATH)
    route_cells = route_grid_map(routes, transform, crs, nx, ny, land_mask)
    mean_df, cov_df = build_community_matrix(obs_all, coverage, crosswalk, route_cells)

    code_ix = {c: i for i, c in enumerate(species)}
    mean_df = mean_df[mean_df["species_code"].isin(code_ix)]
    X_raw, keys, dropped = densify_community(
        mean_df["row"].to_numpy(), mean_df["col"].to_numpy(), mean_df["year"].to_numpy(),
        mean_df["species_code"].map(code_ix).to_numpy(), mean_df["mean_count"].to_numpy(),
        cov_df["row"].to_numpy(), cov_df["col"].to_numpy(), cov_df["year"].to_numpy(),
        len(species))

    # points_meta.json records the log1p flag and the species DESK trained on. Neither is in the
    # ESK meta.json. Cross-check rather than assume: a raw-count basis would make a log1p
    # similarity structure the wrong quantity, and a species set that does not match
    # community_trend.csv means this module and the trainer disagree about the community.
    zt = config.get("bbs", {}).get("z_dir", "")
    pm_path = os.path.join(zt, "points_meta.json") if zt else ""
    log1p_flag, trained = True, None
    if pm_path and os.path.exists(pm_path):
        with open(pm_path, "r", encoding="utf-8") as fh:
            pm = json.load(fh)
        log1p_flag = bool(pm.get("ruzicka_log1p", True))
        trained = [str(s) for s in (pm.get("species") or [])]
        if not log1p_flag:
            raise ValueError(
                f"{pm_path} reports ruzicka_log1p=false; the ESK basis was fit on RAW counts. "
                "Comparing a log1p similarity structure against it is not like-for-like.")
    else:
        print(f"[bbs-routes] WARNING: no points_meta.json at {pm_path or '<bbs.z_dir unset>'}; "
              "assuming ruzicka_log1p=true and skipping the species cross-check")

    sp = {"n_community_trend": len(codes), "n_bbs_matched": len(species)}
    if trained:
        # Both sides now derive from community_trend.csv, so a shortfall here is only species BBS
        # cannot survey -- not a definitional mismatch. A LARGE shortfall means the trained points
        # were built from a different community list and the comparison is not like-for-like.
        shared = [s for s in species if s in set(trained)]
        sp.update({"n_trained": len(trained), "n_shared_with_trained": len(shared)})
        print(f"[bbs-routes] community: {len(codes)} community_trend -> {len(species)} BBS-matched; "
              f"{len(shared)}/{len(trained)} of the trained community observable")
        if len(shared) < 0.5 * len(species):
            print(f"[bbs-routes] WARNING: only {len(shared)}/{len(species)} BBS-matched species "
                  f"appear in {pm_path}; verify the trained points used {community_csv}")
    else:
        print(f"[bbs-routes] community: {len(codes)} community_trend -> {len(species)} BBS-matched")

    meta = {"n_species": len(species), "n_surveyed_cell_years": int(keys.shape[0]),
            "presence_triples_outside_coverage": int(dropped),
            "ruzicka_log1p": log1p_flag, "community_csv": community_csv, **sp,
            "year_range": [int(keys[:, 2].min()), int(keys[:, 2].max())] if keys.size else []}
    return log1p_community(X_raw), keys, meta


def desk_z_ema(config, keys):
    """DESK ``z_ema`` for every row of ``keys`` → ``(N, latent)``.

    The EMA is a CAUSAL scan, so a single (cell, year) cannot be smoothed in isolation. Encodes
    the involved cells over the contiguous span ``ema_warmup_start .. max(year)``, runs the scan
    along the year axis, then gathers the requested (cell, year) rows. Grades ``z_ema`` because
    that is the quantity the trainer supervised -- the cube deliberately exports raw z (the
    population model below it supplies demographic lag), so raw is the wrong comparand here.
    """
    from .validate_spacetime import encode_points

    out_dir = config["paths"]["desk_output_dir"]
    dm = np.load(os.path.join(out_dir, "desk_meta.npz"), allow_pickle=True)
    ema_on = bool(dm["output_ema"]) if "output_ema" in dm.files else False
    hl = float(dm["ema_half_life"]) if "ema_half_life" in dm.files else float("nan")
    warm = int(dm["ema_warmup_start"]) if "ema_warmup_start" in dm.files else 1940

    keys = np.asarray(keys)
    cells = np.unique(keys[:, :2], axis=0)
    y_max = int(keys[:, 2].max())
    years = list(range(min(warm, int(keys[:, 2].min())), y_max + 1))

    # One (cell, year) request per cell per year in the span; encode_points batches by year, so
    # this costs one whole-grid forward per year regardless of how many cells we ask for.
    grid = np.empty((len(cells) * len(years), 3), dtype="int32")
    for t, y in enumerate(years):
        s = t * len(cells)
        grid[s:s + len(cells), :2] = cells
        grid[s:s + len(cells), 2] = y
    t0 = time.perf_counter()
    Z, ok = encode_points(config, grid)
    print(f"[bbs-routes] encoded {len(cells)} cells x {len(years)} yr "
          f"({years[0]}-{years[-1]}) in {time.perf_counter() - t0:.1f}s", flush=True)

    L = Z.shape[1]
    stack = Z.reshape(len(years), len(cells), L)
    valid = ok.reshape(len(years), len(cells))
    if ema_on and np.isfinite(hl):
        stack = apply_output_ema(stack, hl, valid=valid)
        print(f"[bbs-routes] applied output-EMA (half-life={hl:.2f} yr, "
              f"alpha={1.0 - 2.0 ** (-1.0 / hl):.3f}) over {len(years)} years")
    else:
        print(f"[bbs-routes] WARNING: desk_meta has output_ema={ema_on}, ema_half_life={hl}; "
              "grading RAW z (no EMA available in this checkpoint)")

    cell_ix = {(int(r), int(c)): i for i, (r, c) in enumerate(cells)}
    year_ix = {y: t for t, y in enumerate(years)}
    Zout = np.full((keys.shape[0], L), np.nan, dtype="float32")
    for i in range(keys.shape[0]):
        Zout[i] = stack[year_ix[int(keys[i, 2])], cell_ix[(int(keys[i, 0]), int(keys[i, 1]))]]
    return Zout, {"output_ema_applied": bool(ema_on and np.isfinite(hl)),
                  "ema_half_life": hl if np.isfinite(hl) else None,
                  "ema_warmup_start": warm, "encode_years": [years[0], years[-1]]}


def run(config=None, n_sample=4000, seed=0):
    """Driver: observed vs no-change vs DESK, bucketed, written to ``desk_output_dir``."""
    config = config or load_config()
    rng = np.random.default_rng(seed)

    X_log, keys, meta = load_observed(config)
    print(f"[bbs-routes] {meta['n_surveyed_cell_years']} surveyed cell-years, "
          f"{meta['n_species']} species, years {meta['year_range']}")

    # Site gate: drop every row of any cell lacking a modern survey (no definable null).
    nc_src, keep = modern_reference_rows(keys, MODERN_WINDOW)
    n_cells_all = np.unique(keys[:, :2], axis=0).shape[0]
    X_log, keys, nc_src = X_log[keep], keys[keep], nc_src[keep]
    remap = -np.ones(len(keep), dtype="int64")
    remap[np.where(keep)[0]] = np.arange(int(keep.sum()))
    nc_src = remap[nc_src]                                   # reindex into the kept rows
    n_cells_kept = np.unique(keys[:, :2], axis=0).shape[0]
    print(f"[bbs-routes] site gate: kept {int(keep.sum())}/{len(keep)} rows, "
          f"{n_cells_kept}/{n_cells_all} cells "
          f"({n_cells_all - n_cells_kept} dropped for no {MODERN_WINDOW[0]}-{MODERN_WINDOW[1]} survey)")
    if keys.shape[0] < 3:
        raise SystemExit("[bbs-routes] fewer than 3 rows survive the site gate; nothing to compare")

    # Holdout must be resolved BEFORE sampling so it can be a stratum. Held-out cells are a small
    # minority of rows, and held-out is the only split that answers whether the model generalizes,
    # so leaving it to chance makes the table's most important cells its least reliable.
    ho_path = os.path.join(config["paths"]["desk_output_dir"], "holdout_cells.npy")
    has_ho = os.path.exists(ho_path)
    if has_ho:
        ho_grid = np.load(ho_path)
        is_ho_all = ho_grid[keys[:, 0], keys[:, 1]]
        print(f"[bbs-routes] holdout mask: {int(is_ho_all.sum())}/{len(is_ho_all)} rows "
              f"({100.0 * is_ho_all.mean():.1f}%) are held-out cells")
    else:
        print(f"[bbs-routes] WARNING: no {ho_path}; train/heldout split unavailable, "
              "reporting pooled only (every prior metric was pooled at ~83% training points)")
        is_ho_all = None

    sel = stratified_sample(keys, int(n_sample), rng, is_heldout=is_ho_all)
    X_s, keys_s, nc_s = X_log[sel], keys[sel], nc_src[sel]
    keys_nc = keys[nc_s]                                     # each row's MODERN (cell, year)
    X_nc_s = X_log[nc_s]                                     # for the observed-space diagnostic
    is_ho = is_ho_all[sel] if is_ho_all is not None else np.zeros(len(sel), bool)
    print(f"[bbs-routes] sampled {len(sel)} rows ({int(is_ho.sum())} held-out) "
          f"for the {len(sel)}x{len(sel)} matrices")

    # Encode the sampled rows AND their modern references together: same cells, same year span,
    # so this is still one whole-grid forward per year. A sampled row's modern reference is
    # usually NOT itself in the sample, so it has to be requested explicitly.
    n_s = keys_s.shape[0]
    Z_both, zmeta = desk_z_ema(config, np.vstack([keys_s, keys_nc]))
    Z_s, Z_nc = Z_both[:n_s], Z_both[n_s:]

    finite = np.isfinite(Z_s).all(1) & np.isfinite(Z_nc).all(1)
    if finite.sum() < 3:
        raise SystemExit("[bbs-routes] DESK returned non-finite z for nearly every row "
                         "(covariate footprint mismatch?); nothing to compare")
    if not finite.all():
        print(f"[bbs-routes] dropping {int((~finite).sum())} rows with non-finite DESK z")
        X_s, keys_s, X_nc_s = X_s[finite], keys_s[finite], X_nc_s[finite]
        Z_s, Z_nc, is_ho = Z_s[finite], Z_nc[finite], is_ho[finite]

    S_true = ruzicka_rect(X_s, X_s)
    S_desk, S_nc = dot_gram(Z_s), dot_gram(Z_nc)             # THE contract: dot ~= Ruzicka
    S_desk_cos, S_nc_cos = cosine_gram(Z_s), cosine_gram(Z_nc)   # secondary, norm-free
    S_nc_obs = ruzicka_rect(X_nc_s, X_nc_s)                  # achievable-ceiling reference

    # Is the kernel contract even holding on this data? ||z||^2 should be ~1 and the dot should
    # land on the same [0,1] scale as Ruzicka. A large gap here means the headline RMSE is
    # dominated by calibration, and the cosine columns are the ones to read.
    z2 = float(np.median((Z_s ** 2).sum(1)))
    print(f"[bbs-routes] contract check: median ||z||^2 = {z2:.4f} (contract 1.0); "
          f"median dot = {np.median(_offdiag(S_desk)):.4f} vs observed Ruzicka "
          f"{np.median(_offdiag(S_true)):.4f}")

    # Underpowered-comparison guard: if observed communities are all near-identical there is no
    # turnover to predict and every CKA will be ~1 regardless of model quality.
    iu = np.triu_indices_from(S_true, k=1)
    med_off = float(np.median(S_true[iu])) if iu[0].size else float("nan")
    print(f"[bbs-routes] median off-diagonal observed similarity {med_off:.4f} "
          f"(near 1.0 => underpowered)")

    yrs = keys_s[:, 2]
    windows = {"all": np.ones(len(yrs), bool),
               "modern": (yrs >= MODERN_WINDOW[0]) & (yrs <= MODERN_WINDOW[1]),
               "early": (yrs >= EARLY_WINDOW[0]) & (yrs <= EARLY_WINDOW[1])}
    splits = {"pooled": np.ones(len(yrs), bool)}
    if has_ho:
        splits.update({"train": ~is_ho, "heldout": is_ho})

    report = {"config": {"n_sample": int(n_sample), "seed": int(seed),
                         "modern_window": list(MODERN_WINDOW), "early_window": list(EARLY_WINDOW)},
              "observed": meta, "desk": zmeta,
              "site_gate": {"rows_kept": int(keep.sum()), "rows_total": int(len(keep)),
                            "cells_kept": int(n_cells_kept), "cells_total": int(n_cells_all),
                            "cells_dropped_no_modern": int(n_cells_all - n_cells_kept)},
              "median_offdiag_observed_similarity": med_off,
              "buckets": {}}

    for sname, smask in splits.items():
        for wname, wmask in windows.items():
            m = smask & wmask
            if m.sum() < 3:
                report["buckets"][f"{sname}/{wname}"] = {"n_rows": int(m.sum()),
                                                         "skipped": "fewer than 3 rows"}
                continue
            ix = np.where(m)[0]
            g = np.ix_(ix, ix)
            report["buckets"][f"{sname}/{wname}"] = bucket_metrics(
                S_true[g], S_desk[g], S_nc[g], S_nc_obs=S_nc_obs[g],
                S_desk_cos=S_desk_cos[g], S_nc_cos=S_nc_cos[g])

    out_dir = config["paths"]["desk_output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "bbs_route_validation.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    np.savez_compressed(os.path.join(out_dir, "bbs_route_validation.npz"),
                        keys=keys_s, S_true=S_true.astype("float32"),
                        S_desk=S_desk.astype("float32"), S_nc=S_nc.astype("float32"),
                        S_nc_obs=S_nc_obs.astype("float32"), is_heldout=is_ho)

    print(f"\nPRIMARY -- dot product vs observed Ruzicka (the kernel contract, and the quantity "
          f"true_kernel_loss trains on):")
    print(f"{'bucket':<18}{'n':>7}{'rmse_desk':>11}{'rmse_nc':>10}{'skill':>9}"
          f"{'bias_desk':>11}{'r_desk':>9}{'r_nc':>8}")
    for k, v in report["buckets"].items():
        if "skipped" in v:
            print(f"{k:<18}{v['n_rows']:>7}  skipped ({v['skipped']})")
        else:
            print(f"{k:<18}{v['n_rows']:>7}{v['rmse_desk']:>11.4f}{v['rmse_nochange']:>10.4f}"
                  f"{v['rmse_skill']:>+9.3f}{v['bias_desk']:>+11.4f}"
                  f"{v['pearson_desk']:>9.3f}{v['pearson_nochange']:>8.3f}")
    print(f"\nSECONDARY -- structure-only (CKA/Mantel), and the norm-free cosine variant:")
    print(f"{'bucket':<18}{'n':>7}{'cka_gain':>11}{'mantel_gain':>13}{'cka_gain_cos':>14}")
    for k, v in report["buckets"].items():
        if "skipped" not in v:
            print(f"{k:<18}{v['n_rows']:>7}{v['cka_gain']:>+11.4f}{v['mantel_gain']:>+13.4f}"
                  f"{v.get('cka_gain_cosine', float('nan')):>+14.4f}")
    print(f"\n[bbs-routes] report -> {out}")
    print("[bbs-routes] rmse_skill > 0 means DESK's kernel is closer to observed Ruzicka than the "
          "frozen-modern null is, on GENUINELY OBSERVED data. <= 0 is a real negative result.")
    print("[bbs-routes] If rmse_skill is poor but cka_gain_cos is healthy, the failure is ||z|| "
          "calibration, not the angular structure -- check the contract line above.")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sample", type=int, default=4000,
                    help="rows in the similarity matrices (n^2 memory; default 4000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(n_sample=args.n_sample, seed=args.seed)


if __name__ == "__main__":
    main()
