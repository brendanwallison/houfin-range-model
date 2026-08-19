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

# densify_community / log1p_community moved to src/data/preprocess/bbs_community.py: they are
# now the shared core of BOTH the raw-BBS training target and this validation, and the two have
# to densify identically or they are not measuring the same community. Re-exported here so this
# module's public surface (and its tests) are unchanged.
from src.data.preprocess.bbs_community import (            # noqa: E402  (deliberate: see above)
    densify_community, log1p_community,
)


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


def vector_error(t, p):
    """Elementwise agreement between flat arrays of observed and predicted similarity (pure).

    The shared core of ``kernel_error`` (which feeds it matrix off-diagonals) and the epoch
    analysis (which feeds it focal-to-neighbour pair vectors), so both report identical
    definitions. See ``kernel_error`` for what each field means.
    """
    t, p = np.asarray(t, "float64").ravel(), np.asarray(p, "float64").ravel()
    d = p - t
    sd = float(t.std())
    r = float(np.corrcoef(t, p)[0, 1]) if t.size > 1 and sd > 0 and p.std() > 0 else 0.0
    rmse = float(np.sqrt(np.mean(d ** 2))) if t.size else float("nan")
    return {"rmse": rmse, "bias": float(np.mean(d)) if t.size else float("nan"),
            "pearson_r": r, "r2": (1.0 - (rmse / sd) ** 2) if sd > 0 else 0.0,
            "sd_true": sd, "mean_true": float(t.mean()) if t.size else float("nan"),
            "n": int(t.size)}


def pair_metrics(obs, desk, null):
    """DESK vs the frozen-modern null on flat pair vectors (pure).

    The same primary family ``bucket_metrics`` reports, minus CKA/Mantel (which need a full
    matrix and have no meaning for a focal-to-neighbour vector). ``rmse_skill`` and
    ``error_variance_removed`` are the readouts; absolute rmse/bias stay confounded by the log1p
    scale mismatch, as everywhere in this module.
    """
    e_d, e_n = vector_error(obs, desk), vector_error(obs, null)
    ratio = (e_d["rmse"] / e_n["rmse"]) if e_n["rmse"] > 0 else 1.0
    return {
        "n_pairs": e_d["n"],
        "observed_mean": e_d["mean_true"], "observed_sd": e_d["sd_true"],
        "rmse_desk": e_d["rmse"], "rmse_null": e_n["rmse"],
        "rmse_skill": 1.0 - ratio, "error_variance_removed": 1.0 - ratio ** 2,
        "r2_desk": e_d["r2"], "r2_null": e_n["r2"], "r2_gain": e_d["r2"] - e_n["r2"],
        "bias_desk": e_d["bias"], "bias_null": e_n["bias"],
        "pearson_desk": e_d["pearson_r"], "pearson_null": e_n["pearson_r"],
    }


def kernel_error(S_true, S_pred):
    """Elementwise agreement between a predicted kernel and observed Ruzicka (pure).

    Only meaningful for the DOT product, where the contract says the two should be EQUAL rather
    than merely correlated -- so this is a direct calibration check that CKA cannot give.

    Returns RMSE, mean signed bias (positive = predicting MORE similarity than observed, i.e.
    under-predicting turnover), Pearson r, and ``r2``: the fraction of the variance in observed
    pairwise similarity that the prediction explains, measured against the honest know-nothing
    floor -- predicting the same average similarity for every pair, whose RMSE is exactly
    ``sd(S_true)``. So ``r2 = 1 - (rmse / sd_true)^2``, zero for that trivial model and negative
    for anything worse.

    ``r2`` and ``pearson_r ** 2`` are BOTH reported because their gap is the calibration loss:
    Pearson is scale-free and only asks whether the prediction ranks pairs correctly, while r2
    also charges for bias and wrong scale. Measured on real data the two diverge sharply
    (pearson^2 0.78 vs r2 0.55), and that gap is where the model's miscalibration lives.
    """
    return vector_error(_offdiag(S_true), _offdiag(S_pred))


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
    # Error-variance reduction: the squared RMSE ratio, which is what rmse_skill means in variance
    # units. "DESK removes X% of the error the no-change assumption leaves."
    ratio = (e_d["rmse"] / e_n["rmse"]) if e_n["rmse"] > 0 else 1.0
    out = {
        "n_rows": int(np.asarray(S_true).shape[0]),
        # --- scale of the target, so every error below is interpretable ---
        "observed_mean": e_d["mean_true"], "observed_sd": e_d["sd_true"],
        # --- primary: elementwise kernel agreement (dot product vs Ruzicka) ---
        "rmse_desk": e_d["rmse"], "rmse_nochange": e_n["rmse"],
        "rmse_skill": 1.0 - ratio,
        "error_variance_removed": 1.0 - ratio ** 2,
        # r2 vs the predict-the-mean floor; r2_gain is the headline in variance-explained units
        "r2_desk": e_d["r2"], "r2_nochange": e_n["r2"], "r2_gain": e_d["r2"] - e_n["r2"],
        "bias_desk": e_d["bias"], "bias_nochange": e_n["bias"],
        # bias^2 / rmse^2: how much of the error is SYSTEMATIC under-prediction of turnover
        "bias_share_desk": ((e_d["bias"] ** 2) / (e_d["rmse"] ** 2)) if e_d["rmse"] > 0 else 0.0,
        "pearson_desk": e_d["pearson_r"], "pearson_nochange": e_n["pearson_r"],
        # pearson^2 - r2 = the calibration loss (scale-free ranking minus absolute accuracy)
        "calibration_loss_desk": e_d["pearson_r"] ** 2 - e_d["r2"],
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


# ------------------- epoch x local-neighbourhood analysis -------------------
# Motivation: the pooled analysis above compares every site-year pair, which spends nearly all its
# power on two things we do not care about -- year-to-year sampling noise, and cross-continental
# pairs (Arizona vs Maine differ enormously in any year, which is why cka_desk was 0.857 while the
# gain over the null was 0.017). This section removes both: average each cell over two 21-year
# epochs, and compare each cell only against its nearest neighbours.

EPOCH_EARLY = (1966, 1986)
EPOCH_MODERN = (2005, 2025)
MIN_EPOCH_YEARS = 3
GRID_RES_M = 27000.0


def epoch_gate(keys, early=EPOCH_EARLY, modern=EPOCH_MODERN, min_years=MIN_EPOCH_YEARS):
    """Cells with >= ``min_years`` DISTINCT surveyed years in BOTH epochs (pure).

    Distinct YEARS, not route-years: three routes run in one calendar year say nothing about the
    other twenty, and the whole point of epoch-averaging is to beat down year-to-year sampling
    noise. (``raw_bbs_turnover.py`` thresholds summed route coverage instead -- a different, weaker
    criterion.) Both epochs are required because a cell missing either one has no usable
    early-vs-modern contrast.

    Returns ``(cells (M,2) int32 sorted, early_rows, modern_rows, stats)`` where ``early_rows[m]``
    / ``modern_rows[m]`` are lists of row indices into ``keys`` for cell ``m``'s surveyed years in
    that epoch. Epoch bounds are INCLUSIVE on both ends.
    """
    keys = np.asarray(keys)
    e_lo, e_hi = int(early[0]), int(early[1])
    m_lo, m_hi = int(modern[0]), int(modern[1])
    per = {}
    for i in range(keys.shape[0]):
        r, c, y = int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])
        if e_lo <= y <= e_hi:
            per.setdefault((r, c), ({}, {}))[0].setdefault(y, []).append(i)
        elif m_lo <= y <= m_hi:
            per.setdefault((r, c), ({}, {}))[1].setdefault(y, []).append(i)

    kept, e_rows, m_rows = [], [], []
    n_fail_early = n_fail_modern = n_fail_both = 0
    for cell in sorted(per):
        e_by_year, m_by_year = per[cell]
        ok_e, ok_m = len(e_by_year) >= min_years, len(m_by_year) >= min_years
        if ok_e and ok_m:
            kept.append(cell)
            # Flatten year -> rows. A cell-year appears once (densify dedups), so this is just the
            # row list, but going through the year dict is what makes the COUNT a distinct-year one.
            e_rows.append([i for y in sorted(e_by_year) for i in e_by_year[y]])
            m_rows.append([i for y in sorted(m_by_year) for i in m_by_year[y]])
        elif not ok_e and not ok_m:
            n_fail_both += 1
        elif not ok_e:
            n_fail_early += 1
        else:
            n_fail_modern += 1

    stats = {"min_years": int(min_years), "early_window": [e_lo, e_hi],
             "modern_window": [m_lo, m_hi], "cells_kept": len(kept),
             "cells_seen": len(per), "cells_failed_early_only": n_fail_early,
             "cells_failed_modern_only": n_fail_modern, "cells_failed_both": n_fail_both}
    cells = (np.array(kept, dtype="int32") if kept
             else np.zeros((0, 2), dtype="int32"))
    return cells, e_rows, m_rows, stats


def epoch_mean_observed(X_raw, row_lists):
    """Mean RAW counts over each cell's surveyed years, then log1p (pure).

    Order matters: ``log1p`` is concave, so ``mean(log1p(x)) < log1p(mean(x))`` whenever abundance
    varies across years, and the gap grows with the spread. The epoch summary we want is "the
    average community over this period", so average the counts and transform once.
    """
    X_raw = np.asarray(X_raw)
    out = np.zeros((len(row_lists), X_raw.shape[1]), dtype="float64")
    for m, rows in enumerate(row_lists):
        out[m] = X_raw[np.asarray(rows, dtype="int64")].mean(axis=0)
    return log1p_community(out)


def epoch_mean_z(Z, row_lists):
    """Mean of ``z_ema`` over each cell's rows (pure). Averages z itself, per the request.

    NaN-aware: a cell-year outside the covariate footprint comes back as NaN from
    ``encode_points``, and one such year must not wipe out the whole epoch mean. A cell with no
    finite year at all yields NaN and is caught by the caller's finite filter.
    """
    Z = np.asarray(Z, dtype="float64")
    out = np.full((len(row_lists), Z.shape[1]), np.nan, dtype="float64")
    for m, rows in enumerate(row_lists):
        block = Z[np.asarray(rows, dtype="int64")]
        if np.isfinite(block).any():
            with np.errstate(invalid="ignore"):
                out[m] = np.nanmean(block, axis=0)
    return out.astype("float32")


def knn_neighbours(xy, k=99):
    """``k`` nearest OTHER cells for each row of ``xy`` (pure) → ``(idx (N,k), dist (N,k))``.

    ``xy`` is in metres (Albers), so distances come out in metres. Queries ``k+1`` and drops the
    self hit rather than assuming it is column 0 -- with exactly coincident points (two cells at
    the same centre should be impossible, but a duplicated row would do it) the self is not
    guaranteed to sort first, and silently keeping it would inject a similarity-1 pair.

    Returns ``k_eff = min(k, N-1)`` columns; fewer cells than requested is not an error.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(xy, dtype="float64")
    n = xy.shape[0]
    k_eff = int(min(k, max(n - 1, 0)))
    if k_eff == 0:
        return np.zeros((n, 0), dtype="int64"), np.zeros((n, 0), dtype="float64")
    dist, idx = cKDTree(xy).query(xy, k=k_eff + 1)
    dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
    out_i = np.empty((n, k_eff), dtype="int64")
    out_d = np.empty((n, k_eff), dtype="float64")
    for i in range(n):
        keep = idx[i] != i                                  # drop self wherever it landed
        take = np.where(keep)[0][:k_eff]
        if take.size < k_eff:                               # self absent (duplicate coords)
            take = np.arange(idx.shape[1])[:k_eff]
        out_i[i], out_d[i] = idx[i][take], dist[i][take]
    return out_i, out_d


def quantile_distance_bins(dist, n_bins=10):
    """Quantile edges over all finite distances (pure) → ``(edges_m, labels_km)``.

    Quantile rather than fixed-width: k-NN distances are strongly right-skewed (dense East, sparse
    West), so equal-width bins would pile most pairs into the first one or two and leave the rest
    with too few pairs to estimate anything. Equal-n bins give comparably stable estimates.
    Degenerate input (all distances equal) collapses to a single bin instead of raising.
    """
    d = np.asarray(dist, dtype="float64").ravel()
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.array([0.0, np.inf]), ["all"]
    edges = np.unique(np.quantile(d, np.linspace(0.0, 1.0, int(n_bins) + 1)))
    if edges.size < 2:
        edges = np.array([edges[0], edges[0] + 1.0])
    edges[-1] = np.nextafter(edges[-1], np.inf)             # make the last bin right-inclusive
    labels = [f"{edges[i] / 1000:.0f}-{edges[i + 1] / 1000:.0f}km" for i in range(edges.size - 1)]
    return edges, labels


def _pairwise_ruzicka_neighbours(A, B, idx, device=None, chunk=None):
    """Ružička between each focal row of ``A`` and its neighbours ``B[idx]`` (batched, GPU).

    ``A (N,S)``, ``B (N,S)``, ``idx (N,k)`` → ``(N,k)``. Computes only the focal-to-neighbour
    entries; routing this through ``ruzicka_rect`` would build a full N x N matrix and then throw
    away all but k columns per row.

    Sizing note (measured, not guessed): the gathered ``(N,k,S)`` tensor at N=3000, k=99, S=100 is
    ~119 MB in fp32 and a handful of those are live, so a single pass fits comfortably. ``chunk``
    exists only as a guard for much larger N; the real cost of this stage is ``desk_z_ema``'s
    whole-grid forwards, not this arithmetic.
    """
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N, k = idx.shape
    if k == 0:
        return np.zeros((N, 0), dtype="float64")
    if chunk is None:
        chunk = N if dev == "cpu" else max(1, min(N, int(4e8 // max(k * A.shape[1], 1))))
    ta = torch.as_tensor(np.asarray(A), dtype=torch.float32, device=dev)
    tb = torch.as_tensor(np.asarray(B), dtype=torch.float32, device=dev)
    ti = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=dev)
    out = np.empty((N, k), dtype="float64")
    with torch.no_grad():
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            a = ta[s:e].unsqueeze(1)                        # (n,1,S)
            b = tb[ti[s:e]]                                 # (n,k,S)
            mn = torch.minimum(a, b).sum(-1)
            mx = torch.maximum(a, b).sum(-1)
            r = torch.where(mx > 0, mn / mx, torch.ones_like(mx))
            out[s:e] = r.double().cpu().numpy()
    return out


def _pairwise_dot_neighbours(A, B, idx, device=None, chunk=None):
    """``A[i] . B[idx[i,j]]`` for each focal/neighbour pair (batched, GPU) → ``(N,k)``.

    The dot product, per the kernel contract -- same reasoning as ``dot_gram``.
    """
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N, k = idx.shape
    if k == 0:
        return np.zeros((N, 0), dtype="float64")
    if chunk is None:
        chunk = N if dev == "cpu" else max(1, min(N, int(4e8 // max(k * A.shape[1], 1))))
    ta = torch.as_tensor(np.asarray(A), dtype=torch.float32, device=dev)
    tb = torch.as_tensor(np.asarray(B), dtype=torch.float32, device=dev)
    ti = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=dev)
    out = np.empty((N, k), dtype="float64")
    with torch.no_grad():
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            out[s:e] = (ta[s:e].unsqueeze(1) * tb[ti[s:e]]).sum(-1).double().cpu().numpy()
    return out


def _rowwise_ruzicka(A, B):
    """Ružička between matching rows of ``A`` and ``B`` (pure) → ``(N,)``. For self-change."""
    A, B = np.asarray(A, "float64"), np.asarray(B, "float64")
    mn, mx = np.minimum(A, B).sum(1), np.maximum(A, B).sum(1)
    return np.where(mx > 0, mn / mx, 1.0)


def epoch_neighborhood_analysis(Xe, Xm, Ze, Zm, xy, k=99, n_bins=10, is_heldout=None,
                                device=None):
    """Focal-to-neighbour comparisons across space AND time, resolved by distance (pure-ish).

    Four comparison types per focal/neighbour pair, all graded against the SAME frozen-modern null
    (``Zm_i . Zm_j``) so differences isolate what varies:

    ==================  ===========================  ====================
    type                observed                     DESK
    ==================  ===========================  ====================
    ``spatial_early``   Ruzicka(Xe_i, Xe_j)          Ze_i . Ze_j
    ``spatial_modern``  Ruzicka(Xm_i, Xm_j)          Zm_i . Zm_j
    ``cross_time``      Ruzicka(Xe_i, Xm_j)          Ze_i . Zm_j
    ``self_change``     Ruzicka(Xe_i, Xm_i)  (d=0)   Ze_i . Zm_i
    ==================  ===========================  ====================

    ``spatial_modern`` is IDENTICAL to its null by construction, so its skill must be exactly 0.
    That is deliberate: it is this analysis's built-in null test. If it ever comes back non-zero
    the harness is wrong and no other row in the table can be trusted.

    Returns ``(report, per_cell)``.
    """
    idx, dist = knn_neighbours(xy, k=k)
    edges, labels = quantile_distance_bins(dist, n_bins=n_bins)
    bin_of = np.clip(np.digitize(dist, edges) - 1, 0, len(labels) - 1) if idx.size else dist

    null_pair = _pairwise_dot_neighbours(Zm, Zm, idx, device=device)
    types = {
        "spatial_early": (_pairwise_ruzicka_neighbours(Xe, Xe, idx, device=device),
                          _pairwise_dot_neighbours(Ze, Ze, idx, device=device), null_pair),
        "spatial_modern": (_pairwise_ruzicka_neighbours(Xm, Xm, idx, device=device),
                           _pairwise_dot_neighbours(Zm, Zm, idx, device=device), null_pair),
        "cross_time": (_pairwise_ruzicka_neighbours(Xe, Xm, idx, device=device),
                       _pairwise_dot_neighbours(Ze, Zm, idx, device=device), null_pair),
    }

    n = Xe.shape[0]
    ho = np.zeros(n, bool) if is_heldout is None else np.asarray(is_heldout, bool)
    splits = {"pooled": np.ones(n, bool)}
    if is_heldout is not None:
        splits.update({"train": ~ho, "heldout": ho})

    report = {
        "config": {"k": int(idx.shape[1]), "n_bins": len(labels), "n_focal_cells": int(n),
                   "distance_bin_edges_km": [float(e / 1000.0) for e in edges],
                   "distance_bin_labels": labels},
        "distance_summary": {
            "min_km": float(dist.min() / 1000) if idx.size else None,
            "median_km": float(np.median(dist) / 1000) if idx.size else None,
            "max_km": float(dist.max() / 1000) if idx.size else None},
        "types": {},
    }

    for tname, (obs, desk, null) in types.items():
        report["types"][tname] = {}
        for sname, smask in splits.items():
            rows = np.where(smask)[0]
            per_bin = {"all_distances": pair_metrics(obs[rows], desk[rows], null[rows])}
            for b, lab in enumerate(labels):
                sel = bin_of[rows] == b
                if sel.sum() < 10:
                    per_bin[lab] = {"n_pairs": int(sel.sum()), "skipped": "fewer than 10 pairs"}
                    continue
                per_bin[lab] = pair_metrics(obs[rows][sel], desk[rows][sel], null[rows][sel])
            report["types"][tname][sname] = per_bin

    # self_change: one value per focal cell, no distance axis (d=0 by definition).
    sc_obs = _rowwise_ruzicka(Xe, Xm)
    sc_desk = (np.asarray(Ze, "float64") * np.asarray(Zm, "float64")).sum(1)
    sc_null = (np.asarray(Zm, "float64") ** 2).sum(1)
    report["types"]["self_change"] = {
        s: {"all_distances": pair_metrics(sc_obs[np.where(m)[0]], sc_desk[np.where(m)[0]],
                                          sc_null[np.where(m)[0]])}
        for s, m in splits.items()}

    # Per-focal-cell arrays, so the result can be mapped without recomputing.
    per_cell = {"neighbour_idx": idx.astype("int32"), "neighbour_dist_m": dist.astype("float32"),
                "self_change_obs": sc_obs.astype("float32"),
                "self_change_desk": sc_desk.astype("float32"),
                "self_change_null": sc_null.astype("float32"),
                "is_heldout": ho}
    for tname, (obs, desk, null) in types.items():
        # RMSE of each focal cell's own 99 pairs -- a per-cell skill field for mapping.
        rd = np.sqrt(((desk - obs) ** 2).mean(1)) if idx.size else np.full(n, np.nan)
        rn = np.sqrt(((null - obs) ** 2).mean(1)) if idx.size else np.full(n, np.nan)
        per_cell[f"{tname}_rmse_desk"] = rd.astype("float32")
        per_cell[f"{tname}_rmse_null"] = rn.astype("float32")
        with np.errstate(divide="ignore", invalid="ignore"):
            per_cell[f"{tname}_skill"] = np.where(rn > 0, 1.0 - rd / rn, 0.0).astype("float32")
    return report, per_cell


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
    zt = config.get("trend", {}).get("points_dir", "")
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
        print(f"[bbs-routes] WARNING: no points_meta.json at {pm_path or '<trend.points_dir unset>'}; "
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
    # RAW counts returned alongside the log1p version: the epoch analysis must average raw counts
    # and THEN take log1p (log1p is concave, so mean-of-log1p understates abundant species), and it
    # cannot recover the raw values from the transformed ones without an expm1 round trip.
    return log1p_community(X_raw), keys, meta, X_raw


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


def _run_epoch_analysis(config, keys, X_raw_all, cells, e_rows, m_rows, gate_stats, gather_z,
                        zmeta, out_dir, k=99, n_bins=10):
    """Epoch means -> kNN neighbourhoods -> distance-resolved comparisons. Writes json + npz."""
    if cells.shape[0] < 5:
        print(f"[bbs-routes] epoch analysis SKIPPED: only {cells.shape[0]} cells passed the gate")
        return None

    Xe = epoch_mean_observed(X_raw_all, e_rows)               # mean raw counts THEN log1p
    Xm = epoch_mean_observed(X_raw_all, m_rows)
    Ze = epoch_mean_z(gather_z(keys[np.array([i for r in e_rows for i in r])]),
                      _reindex_row_lists(e_rows))
    Zm = epoch_mean_z(gather_z(keys[np.array([i for r in m_rows for i in r])]),
                      _reindex_row_lists(m_rows))

    finite = np.isfinite(Ze).all(1) & np.isfinite(Zm).all(1)
    if not finite.all():
        print(f"[bbs-routes] epoch: dropping {int((~finite).sum())} cells with non-finite DESK z")
        cells, Xe, Xm, Ze, Zm = cells[finite], Xe[finite], Xm[finite], Ze[finite], Zm[finite]
    if cells.shape[0] < 5:
        print("[bbs-routes] epoch analysis SKIPPED: too few cells after the finite-z filter")
        return None

    # cell_xy reopens the ref raster on every call, so call it ONCE for all cells.
    from src.config_utils import load_data_config
    from .validate_spacetime import cell_xy
    xy = cell_xy(cells[:, 0], cells[:, 1], load_data_config()["grid"]["ref_raster"])

    ho_path = os.path.join(config["paths"]["desk_output_dir"], "holdout_cells.npy")
    is_ho = np.load(ho_path)[cells[:, 0], cells[:, 1]] if os.path.exists(ho_path) else None

    t0 = time.perf_counter()
    rep, per_cell = epoch_neighborhood_analysis(Xe, Xm, Ze, Zm, xy, k=k, n_bins=n_bins,
                                                is_heldout=is_ho)
    rep["gate"] = gate_stats
    rep["desk"] = zmeta
    rep["epochs"] = {"early": list(EPOCH_EARLY), "modern": list(EPOCH_MODERN),
                     "min_distinct_years": MIN_EPOCH_YEARS,
                     "desk_averaging": "matched (only each cell's surveyed years)",
                     "observed_averaging": "mean RAW counts then log1p (log1p is concave)"}

    # all_years variant: average DESK over EVERY year of each epoch rather than only the surveyed
    # ones. Better-estimated but NOT matched to what the observed mean summarizes, so if BBS
    # sampling clusters inside a window the two diverge -- and that divergence measures the
    # sampling bias. Reported at all_distances only; the matched variant stays primary.
    try:
        Ze_all, Zm_all = _epoch_mean_z_all_years(cells, gather_z)
        fin2 = np.isfinite(Ze_all).all(1) & np.isfinite(Zm_all).all(1)
        if fin2.sum() >= 5:
            rep_all, _ = epoch_neighborhood_analysis(
                Xe[fin2], Xm[fin2], Ze_all[fin2], Zm_all[fin2], xy[fin2], k=k, n_bins=1,
                is_heldout=None if is_ho is None else is_ho[fin2])
            rep["variant_all_years"] = {
                t: v["pooled"]["all_distances"] for t, v in rep_all["types"].items()}
            rep["variant_all_years"]["_divergence"] = {
                "mean_abs_diff_Ze": float(np.nanmean(np.abs(Ze_all[fin2] - Ze[fin2]))),
                "mean_abs_diff_Zm": float(np.nanmean(np.abs(Zm_all[fin2] - Zm[fin2]))),
                "note": "large divergence => BBS sampling years are not representative of the "
                        "epoch, so the matched variant is the one to trust"}
    except Exception as exc:                                  # diagnostic only, never fatal
        print(f"[bbs-routes] all_years variant skipped: {type(exc).__name__}: {exc}")
    print(f"[bbs-routes] epoch neighbourhood analysis: {cells.shape[0]} cells x "
          f"{rep['config']['k']} neighbours in {time.perf_counter() - t0:.1f}s")

    with open(os.path.join(out_dir, "bbs_epoch_neighborhood.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    np.savez_compressed(os.path.join(out_dir, "bbs_epoch_neighborhood.npz"),
                        cells=cells, rows=cells[:, 0], cols=cells[:, 1],
                        Xe=Xe.astype("float32"), Xm=Xm.astype("float32"),
                        Ze=Ze, Zm=Zm, **per_cell)

    # --- console summary
    d = rep["distance_summary"]
    print(f"\nEPOCH x NEIGHBOURHOOD -- {cells.shape[0]} focal cells, k={rep['config']['k']}, "
          f"neighbour distance {d['min_km']:.0f}/{d['median_km']:.0f}/{d['max_km']:.0f} km "
          f"(min/median/max)")
    for tname, per_split in rep["types"].items():
        print(f"\n  {tname}")
        print(f"    {'split/bin':<20}{'n':>9}{'r2_desk':>9}{'r2_null':>9}{'r2_gain':>9}"
              f"{'skill':>8}{'errVar%':>9}{'bias':>9}")
        for sname, per_bin in per_split.items():
            for bname, mm in per_bin.items():
                if "skipped" in mm:
                    continue
                print(f"    {sname + '/' + bname:<20}{mm['n_pairs']:>9}{mm['r2_desk']:>9.3f}"
                      f"{mm['r2_null']:>9.3f}{mm['r2_gain']:>+9.3f}{mm['rmse_skill']:>+8.3f}"
                      f"{100 * mm['error_variance_removed']:>8.1f}%{mm['bias_desk']:>+9.4f}")
    sm = rep["types"]["spatial_modern"]["pooled"]["all_distances"]["rmse_skill"]
    print(f"\n  [null test] spatial_modern skill = {sm:+.2e} -- MUST be ~0: DESK and the null are "
          "the same quantity there by construction. Non-zero means the harness is broken.")
    print(f"[bbs-routes] epoch report -> {os.path.join(out_dir, 'bbs_epoch_neighborhood.json')}")
    return rep


def _epoch_mean_z_all_years(cells, gather_z):
    """DESK epoch means over EVERY year of each epoch (not just surveyed ones)."""
    e_years = list(range(EPOCH_EARLY[0], EPOCH_EARLY[1] + 1))
    m_years = list(range(EPOCH_MODERN[0], EPOCH_MODERN[1] + 1))
    out = []
    for years in (e_years, m_years):
        kk = np.array([[r, c, y] for r, c in cells for y in years], dtype="int32")
        Z = gather_z(kk)
        out.append(epoch_mean_z(Z, _reindex_row_lists([years] * len(cells))))
    return out[0], out[1]


def _reindex_row_lists(row_lists):
    """Row lists renumbered against their own concatenation (pure).

    ``gather_z`` is handed the flattened rows, so the per-cell groupings have to be expressed as
    offsets into that flat result rather than into the original ``keys``.
    """
    out, n = [], 0
    for rows in row_lists:
        out.append(list(range(n, n + len(rows))))
        n += len(rows)
    return out


def run(config=None, n_sample=4000, seed=0):
    """Driver: observed vs no-change vs DESK, bucketed, written to ``desk_output_dir``."""
    config = config or load_config()
    rng = np.random.default_rng(seed)

    X_log, keys, meta, X_raw_all = load_observed(config)
    print(f"[bbs-routes] {meta['n_surveyed_cell_years']} surveyed cell-years, "
          f"{meta['n_species']} species, years {meta['year_range']}")

    # ---- epoch analysis prep, done BEFORE the pooled site gate so it sees every surveyed row.
    # Its gate (>=3 distinct years in both epochs) is stricter and different from the pooled one,
    # so the two must be derived independently from the full row set.
    #
    # KEEP THE UNFILTERED KEYS. The pooled site gate below rebinds `keys = keys[keep]`, so the row
    # indices in ep_e_rows/ep_m_rows -- which index the FULL array -- would silently point into the
    # wrong rows afterwards (and did: IndexError at 110878 vs size 110839). X_raw_all is already
    # unfiltered, so the epoch analysis must be handed keys_all to match it.
    keys_all = keys
    ep_cells, ep_e_rows, ep_m_rows, ep_stats = epoch_gate(keys_all)
    print(f"[bbs-routes] epoch gate: {ep_stats['cells_kept']}/{ep_stats['cells_seen']} cells have "
          f">={MIN_EPOCH_YEARS} distinct surveyed years in BOTH "
          f"{EPOCH_EARLY[0]}-{EPOCH_EARLY[1]} and {EPOCH_MODERN[0]}-{EPOCH_MODERN[1]} "
          f"(failed: early-only {ep_stats['cells_failed_early_only']}, "
          f"modern-only {ep_stats['cells_failed_modern_only']}, "
          f"both {ep_stats['cells_failed_both']})")

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

    # ONE encode pass for BOTH analyses. encode_points batches by year, so desk_z_ema costs one
    # whole-grid forward PER YEAR regardless of how many cells are requested -- calling it once per
    # analysis would double the ~86 forwards for no benefit. So request the union and slice.
    # Sampled rows' modern references are usually not themselves in the sample, so they must be
    # asked for explicitly.
    ep_rows_flat = np.array([i for rows in (ep_e_rows + ep_m_rows) for i in rows], dtype="int64")
    need = [keys_s, keys_nc]
    if ep_rows_flat.size:
        need.append(keys_all[ep_rows_flat])          # keys_all: epoch rows index the FULL array
        # The all_years DESK variant needs z at EVERY year of each epoch, not just surveyed ones.
        # Free: desk_z_ema already encodes every requested cell across every year of the span, so
        # these extra keys add gathers, not forwards.
        ep_all = np.array([[r, c, y] for r, c in ep_cells
                           for y in list(range(EPOCH_EARLY[0], EPOCH_EARLY[1] + 1))
                           + list(range(EPOCH_MODERN[0], EPOCH_MODERN[1] + 1))], dtype="int32")
        need.append(ep_all)
    want = np.unique(np.vstack(need), axis=0)
    Z_want, zmeta = desk_z_ema(config, want)
    want_ix = {(int(r), int(c), int(y)): i for i, (r, c, y) in enumerate(want)}

    def _gather(kk):
        return Z_want[[want_ix[(int(r), int(c), int(y))] for r, c, y in np.asarray(kk)]]

    Z_s, Z_nc = _gather(keys_s), _gather(keys_nc)

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

    # ---- epoch x local-neighbourhood analysis (separate outputs, same encode pass)
    ep_report = _run_epoch_analysis(config, keys_all, X_raw_all, ep_cells, ep_e_rows, ep_m_rows,
                                    ep_stats, _gather, zmeta, out_dir)

    b0 = next((v for v in report["buckets"].values() if "observed_sd" in v), None)
    if b0:
        print(f"\nSCALE of the target: observed pairwise similarity mean {b0['observed_mean']:.4f}, "
              f"sd {b0['observed_sd']:.4f} (similarities are in [0,1]).")
        print(f"  sd is the RMSE of the know-nothing model (predict the mean for every pair), so "
              f"it is the r2=0 floor.")
    print(f"\nPRIMARY -- dot product vs observed Ruzicka (the kernel contract, and the quantity "
          f"true_kernel_loss trains on).")
    print(f"  r2 = variance of observed similarity explained; errVar% = share of the NULL's error "
          f"variance that DESK removes.")
    print(f"{'bucket':<18}{'n':>7}{'r2_desk':>9}{'r2_nc':>8}{'r2_gain':>9}{'errVar%':>9}"
          f"{'rmse_desk':>11}{'rmse_nc':>9}{'bias':>9}{'bias%':>8}")
    for k, v in report["buckets"].items():
        if "skipped" in v:
            print(f"{k:<18}{v['n_rows']:>7}  skipped ({v['skipped']})")
        else:
            print(f"{k:<18}{v['n_rows']:>7}{v['r2_desk']:>9.3f}{v['r2_nochange']:>8.3f}"
                  f"{v['r2_gain']:>+9.3f}{100 * v['error_variance_removed']:>8.1f}%"
                  f"{v['rmse_desk']:>11.4f}{v['rmse_nochange']:>9.4f}"
                  f"{v['bias_desk']:>+9.4f}{100 * v['bias_share_desk']:>7.1f}%")
    print(f"\n  pearson^2 vs r2 -- their gap is calibration loss (ranking is right, scale is not):")
    for k, v in report["buckets"].items():
        if "skipped" not in v:
            print(f"  {k:<18}pearson^2 {v['pearson_desk'] ** 2:>6.3f}  r2 {v['r2_desk']:>6.3f}  "
                  f"calibration loss {v['calibration_loss_desk']:>+6.3f}")
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
