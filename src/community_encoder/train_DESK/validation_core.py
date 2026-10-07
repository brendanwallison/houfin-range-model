"""Validation primitives shared by EVERY DESK validation arm: one definition each.

Each function here exists because a second, local copy of it once encoded a different answer to
the same question -- which rows are held out, which cells are buffer, how two independent halves of
a survey record are drawn -- and the copies drifted. ``validate_bbs_routes``, ``validate_spacetime``
and ``validate_gp_species`` import from here; none of them may re-derive these.

* ``load_holdout_masks`` -- the saved held-out cells, and the buffer ring around them. Checkpoints
  that predate ``buffer_cells.npy`` (the ``desk_tempho_*`` runs) get the ring REBUILT the way the
  trainer drew it. The previous fallback, ``np.zeros_like(holdout)``, silently let interpolation
  bars and GP fits read buffer cells the model was never fitted on.
* ``row_splits`` -- training rows and the three held-out groups (space / time / space_time),
  including the withheld years of a temporal-holdout run.
* ``abba_halves`` / ``split_half_groups`` -- two disjoint, YEAR-BALANCED halves of a group of
  rows. Year-balanced because a random split can put more early years in one half and more late
  years in the other, so a trend INSIDE the window reads as disagreement between the halves and is
  counted as noise. Every split-half in the codebase -- noise floors and independent oracles alike
  -- uses this one.
"""
import os

import numpy as np


# ----------------------------- holdout masks -----------------------------

def load_holdout_masks(run_dir, buffer_floor=None):
    """``(holdout, buffer, note)`` grids for a DESK run, or ``(None, None, note)`` without one.

    The buffer is the run's saved ``buffer_cells.npy`` when present. Otherwise it is rebuilt as
    ``augment.blocked_holdout`` built it: every cell within Chebyshev distance
    ``max(spatial_kernel // 2, buffer_floor)`` of a held-out cell, minus the held-out cells. Only
    the land mask (``valid``) is not reapplied, which is harmless: a non-land cell has no rows.
    """
    ho_p = os.path.join(run_dir, "holdout_cells.npy")
    if not os.path.exists(ho_p):
        return None, None, f"no {ho_p}"
    ho = np.load(ho_p).astype(bool)
    bf_p = os.path.join(run_dir, "buffer_cells.npy")
    if os.path.exists(bf_p):
        return ho, np.load(bf_p).astype(bool), "saved buffer_cells.npy"
    from .augment import _shift2d
    k = 0
    dm_p = os.path.join(run_dir, "desk_meta.npz")
    if os.path.exists(dm_p):
        dm = np.load(dm_p, allow_pickle=True)
        k = int(dm["spatial_kernel"]) if "spatial_kernel" in dm.files else 0
    width = max(k // 2, int(buffer_floor or 0))
    near = np.zeros_like(ho)
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            near |= _shift2d(ho, dy, dx)
    return ho, near & ~ho, f"rebuilt: Chebyshev width {width} (spatial_kernel {k})"


# ----------------------------- row splits -----------------------------

#: Held-out row groups, mirroring the existing suite's holdouts. 0 = not scored.
GROUPS = {1: "space", 2: "time", 3: "space_time"}


def row_splits(keys, holdout, buffer, block_cells, withheld_years=(), common_years=()):
    """``(is_train, group, block_id)`` per ``(row, col, year)`` key. Pure.

    Training rows are cells neither held out nor buffer, in years the run did NOT withhold --
    exactly what the DESK checkpoint trained on. ``withheld_years`` is ``desk.trend.holdout_years``
    (the ``desk_tempho_*`` runs withhold the early decades from every cell); leaving those years in
    training would hand any fitted comparator the very years DESK never saw.

    Scored rows fall in three groups:

    * ``space`` (1): held-out block cells in trained years -- spatial extrapolation only. Every
      such row has training cells around it observed in the SAME year.
    * ``time`` (2): training cells in withheld years -- temporal extrapolation only. No cell is
      observed in those years.
    * ``space_time`` (3): held-out block cells in withheld years -- both at once.

    ``common_years`` (``common_holdout_years``) restricts groups 2 and 3 to the window every run
    withholds, so the tempho runs are comparable; withheld years outside it are neither trained
    on nor scored. Block ids follow ``augment.blocked_holdout``'s tiling (origin 0,0).
    """
    keys = np.asarray(keys)
    r, c, yr = keys[:, 0], keys[:, 1], keys[:, 2]
    ho = np.asarray(holdout, bool)[r, c]
    bf = np.asarray(buffer, bool)[r, c] if buffer is not None else np.zeros(len(keys), bool)
    wh = np.isin(yr, np.asarray(list(withheld_years), int))
    win = np.isin(yr, np.asarray(list(common_years), int)) if len(common_years) else wh
    is_train = ~ho & ~bf & ~wh
    group = np.zeros(len(keys), "int8")
    group[ho & ~wh] = 1
    group[~ho & ~bf & wh & win] = 2
    group[ho & wh & win] = 3
    b = max(1, int(block_cells))
    nbx = int(np.asarray(holdout).shape[1] + b - 1) // b
    block_id = (r // b) * nbx + (c // b)
    return is_train, group, block_id.astype("int64")


# ----------------------------- split-half -----------------------------

def abba_halves(rows, years):
    """Split rows into two year-balanced halves by an ABBA pattern over year order. Pure.

    Rows are sorted by year and assigned A, B, B, A, A, B, B, A, ... Plain alternation (ABAB)
    still leaves B half a step later than A; ABBA gives both halves the same mean year in every
    complete block of four, so a linear within-window trend cancels exactly and only year-to-year
    variation about it counts as noise. A trailing unpaired year is dropped from both halves; when
    the paired count is not a multiple of four, the last pair is (A, B) and the halves' mean years
    differ by one step over n/2 -- no assignment of two rows to two halves avoids that short of
    dropping the pair, which costs a third of the data at n = 6.
    """
    rows = np.asarray(rows, int)
    order = rows[np.argsort(np.asarray(years), kind="stable")]
    n = (len(order) // 2) * 2
    lab = np.array([0, 1, 1, 0])[np.arange(n) % 4]
    return order[:n][lab == 0], order[:n][lab == 1]


def split_half_groups(groups, years=None, seed=0):
    """Split each row group into two DISJOINT, year-balanced halves -> ``(half_a, half_b, ok)``.

    The one split-half every arm uses (see ``abba_halves``). ``years`` is the year of each row
    index the groups refer to; without it each group is taken to be in year order already, which
    ``epoch_gate`` and ``window_groups`` both guarantee. ``seed`` is accepted for call-site
    compatibility and unused: the split is deterministic, which is part of the point.

    ``ok[i]`` is False where a group has fewer than 2 rows and cannot be split; the caller must
    drop those rows rather than silently compare a group against itself.
    """
    a, b, ok = [], [], []
    for g in groups:
        g = tuple(int(i) for i in g)
        if len(g) < 2:
            a.append(g)
            b.append(g)
            ok.append(False)
            continue
        yrs = (np.arange(len(g)) if years is None else np.asarray(years)[list(g)])
        ha, hb = abba_halves(g, yrs)
        a.append(tuple(sorted(int(i) for i in ha)))
        b.append(tuple(sorted(int(i) for i in hb)))
        ok.append(True)
    return a, b, np.asarray(ok, bool)


# ----------------------------- change, noise, and the share of it captured -----------------------------

def epoch_values(raw, early_rows, modern_rows):
    """log1p of the epoch-MEAN abundance per cell, early and modern: ``(e, m)``, each (cells, S).

    The suite's estimand, via ``validate_bbs_routes.epoch_mean_observed``: average the counts, then
    transform. Carrying capacity and growth rate -- what the downstream model estimates -- are
    statements about AVERAGE abundance; a mean of log1p values is instead a log geometric mean,
    which for a rare species tracks occupancy. Truth and every prediction go through this one
    function, so they are also the same functional.
    """
    from .validate_bbs_routes import epoch_mean_observed
    raw = np.asarray(raw, "float64")
    return (epoch_mean_observed(raw, early_rows).astype("float64"),
            epoch_mean_observed(raw, modern_rows).astype("float64"))


def split_half_change(raw, early_rows, modern_rows, years):
    """Observed change per cell from all years, and from two DISJOINT year-balanced halves.

    ``(d_full, d_a, d_b)``, each (cells, S), on the ``epoch_values`` estimand. ``d_a`` and ``d_b``
    are two independent observations of the SAME cell's change (``abba_halves`` within each epoch),
    so their disagreement is measurement noise -- route sampling, observer and interannual
    variation about any within-epoch trend -- and their covariance is the real change.
    """
    raw = np.asarray(raw, "float64")
    yrs = np.asarray(years)
    ea, eb, ma, mb = [], [], [], []
    for e, m in zip(early_rows, modern_rows):
        e, m = np.asarray(e, int), np.asarray(m, int)
        a1, b1 = abba_halves(e, yrs[e])
        a2, b2 = abba_halves(m, yrs[m])
        ea.append(a1), eb.append(b1), ma.append(a2), mb.append(b2)
    fe, fm = epoch_values(raw, early_rows, modern_rows)
    ae, am = epoch_values(raw, ea, ma)
    be, bm = epoch_values(raw, eb, mb)
    return fm - fe, am - ae, bm - be


def change_noise(d_full, d_a, d_b, n_boot=400, seed=0):
    """How much of observed per-species change is noise. A dict of per-species and per-cell arrays.

    * ``ms_obs`` -- mean squared observed change: the no-change predictor's MSE.
    * ``noise`` -- the noise variance of ``d_full``: ``mean((d_a - d_b)^2) / 4``. Two independent
      half estimates differ by twice a half's noise variance, and the full estimate averages both
      halves, halving it again. With a log-of-mean estimand that averaging is exact only to first
      order (delta method), and an unpaired year is in ``d_full`` but not the halves, so the
      estimate is slightly conservative.
    * ``signal_share = 1 - noise/ms_obs`` and ``ceiling_skill = 1 - sqrt(noise/ms_obs)`` -- the
      best change skill any predictor could reach.
    * ``resolvable`` -- the bootstrap over cells puts ``ms_obs - noise`` above zero: the same rule
      ``validate_bbs_routes.stratum_viable`` applies to a stratum, applied to a species.
    * ``cell_sq``, ``cell_noise`` -- the per-cell terms, for stratum viability.
    """
    sq = np.asarray(d_full, "float64") ** 2
    nz = (np.asarray(d_a, "float64") - np.asarray(d_b, "float64")) ** 2 / 4.0
    ms_obs, noise = np.nanmean(sq, 0), np.nanmean(nz, 0)
    rng = np.random.default_rng(seed + 1)
    nc = sq.shape[0]
    boot = np.empty((int(n_boot), sq.shape[1]))
    for b in range(int(n_boot)):
        i = rng.integers(0, nc, nc)
        boot[b] = np.nanmean(sq[i], 0) - np.nanmean(nz[i], 0)
    lo = np.quantile(boot, 0.025, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(ms_obs > 0, 1 - noise / ms_obs, np.nan)
        ceil = np.where(ms_obs > 0, 1 - np.sqrt(np.minimum(noise / ms_obs, 1.0)), np.nan)
    return {"ms_obs": ms_obs, "noise": noise, "signal_share": share, "ceiling_skill": ceil,
            "resolvable": (ms_obs > 0) & (lo > 0), "cell_sq": sq, "cell_noise": nz}


def captured_share(d_full, d_pred, noise):
    """Each species' gain on no-change as a share of the gain AVAILABLE. ``(per_species, pooled)``.

    ``(ms_obs - MSE_pred) / (ms_obs - noise)`` over resolvable species only (NaN elsewhere), and
    pooled as the summed gain over the summed availability. The per-species analogue of the route
    suite's "share of available temporal signal". Cells where the prediction is missing (an
    oracle with no independent half) are left out of that species' sums.
    """
    d_full, d_pred = np.asarray(d_full, "float64"), np.asarray(d_pred, "float64")
    ok = np.isfinite(d_pred).all(1)
    ms = np.mean(d_full[ok] ** 2, 0)
    mse = np.mean((d_pred[ok] - d_full[ok]) ** 2, 0)
    res = noise["resolvable"]
    avail = ms - np.nanmean(noise["cell_noise"][ok], 0)       # noise over the SAME cells
    with np.errstate(invalid="ignore", divide="ignore"):
        per = np.where(res, (ms - mse) / avail, np.nan)
    pooled = float(np.sum(np.where(res, ms - mse, 0)) / max(np.sum(np.where(res, avail, 0)),
                                                            1e-300))
    return per, pooled
