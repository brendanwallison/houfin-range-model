"""Baselines every DESK metric has to clear, and the multi-epoch direction panel.

One module because these are all the same question -- "is this better than assuming spatial
smoothness?" -- and because the answer turned out to matter more than the metrics themselves.
The trainer's direction diagnostic beat its permutation null (0.48 vs 0.22) and still LOST to
inverse-distance interpolation (0.51). A null that a plain interpolator clears by a wide margin
is not a bar; every "vs null" figure in the validation report is suspect for the same reason.

Used from two places, which is why the functions take plain arrays rather than reading config:
``desk_training`` prints the spatial and direction bars at setup so a run announces what it has
to beat, and ``validate_spacetime`` reports them alongside the metrics they qualify.
"""
import itertools

import numpy as np
import torch

#: Epochs for the direction panel. ~19 years apart, about two output-EMA half-lives at the
#: learned 10.5 y -- the shortest interval over which the EMA is not dominating the predicted
#: change. 1966 is deliberately NOT the start: it is BBS's launch year, 351 cells, and using it
#: left the trainer's diagnostic resting on 36 validation cells.
DEFAULT_EPOCHS = (1967, 1985, 2005, 2025)
DEFAULT_TOL = 2


def _np(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def median_dir_cos_np(dp, dt):
    from .desk_training import median_dir_cos
    return median_dir_cos(torch.as_tensor(np.asarray(dp), dtype=torch.float32),
                          torch.as_tensor(np.asarray(dt), dtype=torch.float32))


def spatial_interp_baseline(tgt, k=8, power=2.0):
    """Predict held-out cells by inverse-distance interpolation of the TARGETS themselves.

    The existing no-skill baselines (predict-mean, predict-zero) ignore geography, but Z is
    strongly spatially autocorrelated, so a model that merely interpolates its neighbours
    already scores well. This is the honest reference for a spatially BLOCKED holdout: it uses
    no covariates and no learning, only "the answer at nearby training cells". If the trained
    model's val error is not clearly below this, the covariates are adding nothing beyond
    spatial smoothness, and further capacity or regularization work is pointless.

    The blocked split plus its buffer put every val cell at least ``kernel//2 + 1`` cells from
    any training cell, so this baseline is genuinely extrapolating into the block interior --
    which is exactly the task the model faces.

    Returns ``(nearest_mse, idw_mse)`` on the same per-cell summed scale as ``_z_mse``.

    The per-year masks matter and must not be shortcut. ``_prepare_trend_targets`` builds each
    year's grid as ``np.zeros`` and marks only the cells with a trend point that year, so a cell
    absent in year Y holds a **zero**, not a NaN, and the train/val masks are ``present &
    (~holdout)`` / ``present & holdout`` -- i.e. per-year, not constant. An earlier version of
    this function reused the first year's masks throughout, which scored interpolation against
    zero-filled targets and fed zero-filled sources into the interpolation, on a different cell
    population and a different denominator from ``_z_mse``. That made the baseline
    incomparable with the model's val. So: rebuild the neighbour index per distinct mask and
    accumulate with the per-year counts, mirroring ``_z_mse`` exactly.
    """
    from scipy.spatial import cKDTree

    years = sorted(tgt)
    if not years:
        return float("nan"), float("nan")

    def _np(a):
        return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)

    sq_n = sq_i = 0.0
    cnt = 0
    cache = {}
    for y in years:
        zg, tr_m, va_m = tgt[y][0], tgt[y][1], tgt[y][2]
        tr_np, va_np = _np(tr_m), _np(va_m)
        if not va_np.any() or tr_np.sum() < k:
            continue                                       # same skip as _z_mse's m.any()
        key = (tr_np.tobytes(), va_np.tobytes())           # masks repeat across many years
        if key not in cache:
            tr_rc, va_rc = np.argwhere(tr_np), np.argwhere(va_np)
            dist, idx = cKDTree(tr_rc).query(va_rc, k=k)
            w = 1.0 / np.maximum(dist, 1e-6) ** power
            cache[key] = (tr_rc, va_rc, idx, w / w.sum(axis=1, keepdims=True))
        tr_rc, va_rc, idx, w = cache[key]

        z = _np(zg)
        src = z[tr_rc[:, 0], tr_rc[:, 1]]                  # (n_train, L), this year's values
        truth = z[va_rc[:, 0], va_rc[:, 1]]                # (n_val, L)
        pred_n = src[idx[:, 0]]                            # nearest training cell
        pred_i = np.einsum("nk,nkl->nl", w, src[idx])      # inverse-distance weighted
        sq_n += float(((pred_n - truth) ** 2).sum())
        sq_i += float(((pred_i - truth) ** 2).sum())
        cnt += len(va_rc)
    if cnt == 0:
        return float("nan"), float("nan")
    return sq_n / cnt, sq_i / cnt


def spatial_interp_dir_cos(tgt, y_deep, y_anchor, rot_va, k=8, power=2.0):
    """Direction-cosine of the INTERPOLATED change, on the same val cells as the model's.

    ``spatial_interp_baseline`` answers "do the covariates beat spatial smoothness at
    reproducing Z?". This answers the sharper question: do they beat it at reproducing the
    DIRECTION OF CHANGE? Inverse-distance interpolation is run separately in the deep year and
    the anchor year, using each year's own training cells, and the difference of the two
    interpolations is its predicted change vector. A model whose dir-cos merely matches this is
    getting its temporal signal from "what the neighbours did", not from the covariates.

    Returns ``(idw_dir_cos, n_cells)``, or ``(nan, 0)`` if either year is unusable.
    """
    from scipy.spatial import cKDTree

    from .desk_training import median_dir_cos

    va = _np(rot_va)
    if y_deep is None or y_deep not in tgt or y_anchor not in tgt or not va.any():
        return float("nan"), 0
    va_rc = np.argwhere(va)
    preds, truths = [], []
    for y in (y_deep, y_anchor):
        zg, tr_np = _np(tgt[y][0]), _np(tgt[y][1])
        if tr_np.sum() < k:
            return float("nan"), 0
        tr_rc = np.argwhere(tr_np)
        dist, idx = cKDTree(tr_rc).query(va_rc, k=k)
        w = 1.0 / np.maximum(dist, 1e-6) ** power
        w = w / w.sum(axis=1, keepdims=True)
        src = zg[tr_rc[:, 0], tr_rc[:, 1]]
        preds.append(np.einsum("nk,nkl->nl", w, src[idx]))
        truths.append(zg[va_rc[:, 0], va_rc[:, 1]])
    dp = torch.as_tensor(preds[1] - preds[0], dtype=torch.float32)
    dt = torch.as_tensor(truths[1] - truths[0], dtype=torch.float32)
    return median_dir_cos(dp, dt), int(len(va_rc))


def nearest_survey(pip, supervise, epochs, tol):
    """``{epoch: {(row,col): year}}`` -- each cell's surveyed year closest to each epoch.

    Ties break to the EARLIER year, so the choice cannot depend on dict ordering. Only rows
    flagged ``supervise`` count, since duplicates exist for the kernel only.
    """
    sup = supervise if supervise is not None else np.ones(len(pip), bool)
    out = {int(e): {} for e in epochs}
    order = np.lexsort((pip[:, 2],))                       # ascending year -> earlier wins ties
    for i in order:
        if not sup[i]:
            continue
        r, c, y = int(pip[i, 0]), int(pip[i, 1]), int(pip[i, 2])
        for e in out:
            if abs(y - e) > tol:
                continue
            prev = out[e].get((r, c))
            if prev is None or abs(y - e) < abs(prev - e):
                out[e][(r, c)] = y
    return out


def _idw_at(cells_wanted, train_years, z_of, k=8, power=2.0):
    """Inverse-distance estimate at ``cells_wanted`` from ``train_years`` (cell -> year).

    ``z_of(cell, year)`` supplies the target vector. Each training cell contributes the value at
    ITS OWN nearest year to the epoch, which is the same rule the model side uses.
    """
    from scipy.spatial import cKDTree

    if len(train_years) < k or not cells_wanted:
        return None
    tr_cells = sorted(train_years)
    tr_rc = np.array(tr_cells, dtype=float)
    src = np.stack([z_of(c, train_years[c]) for c in tr_cells])
    dist, idx = cKDTree(tr_rc).query(np.array(cells_wanted, dtype=float), k=k)
    w = 1.0 / np.maximum(dist, 1e-6) ** power
    w = w / w.sum(axis=1, keepdims=True)
    return np.einsum("nk,nkl->nl", w, src[idx])


def zspace_idw_baseline(pidx, z_obs, holdout, hist_mask, k=8, power=2.0):
    """Per-point Z-space error of interpolating the OBSERVED z from training cells, same year.

    The reconstruction metric compares DESK against a no-change null. That null is weak in the
    deep past by construction -- it assumes 60 years of stasis -- so beating it says little. This
    is the bar that matters: no covariates, no learning, just "the observed community at nearby
    cells surveyed the same year". Returns per-point error aligned to ``pidx[hist_mask]``, NaN
    where a year had fewer than ``k`` training cells.
    """
    from scipy.spatial import cKDTree

    hist_idx = np.flatnonzero(hist_mask)
    err = np.full(len(hist_idx), np.nan, dtype="float32")
    ho = _np(holdout)
    for y in np.unique(pidx[hist_idx, 2]):
        same = pidx[:, 2] == y
        tr = np.flatnonzero(same & ~ho[pidx[:, 0], pidx[:, 1]])
        want = np.flatnonzero(hist_mask & same)
        if len(tr) < k or not len(want):
            continue
        tree = cKDTree(pidx[tr, :2].astype(float))
        dist, idx = tree.query(pidx[want, :2].astype(float), k=k)
        # A held-out point is never its own neighbour (it is not in tr), but a TRAIN point is,
        # at distance 0 -- which would hand it the answer. Drop zero-distance neighbours.
        w = np.where(dist > 1e-9, 1.0 / np.maximum(dist, 1e-6) ** power, 0.0)
        ok = w.sum(axis=1) > 0
        w[ok] /= w[ok].sum(axis=1, keepdims=True)
        pred = np.einsum("nk,nkl->nl", w, z_obs[tr][idx])
        pos = np.searchsorted(hist_idx, want)
        e = np.linalg.norm(pred - z_obs[want], axis=1)
        err[pos[ok]] = e[ok]
    return err


def epoch_direction_panel(pidx, supervise, z_obs, z_model, holdout, buffer_mask,
                          epochs=DEFAULT_EPOCHS, tol=DEFAULT_TOL, verbose=True):
    """Model vs inverse-distance on the DIRECTION of change, per epoch pair, plus curvature.

    ``z_model`` is a ``{(row,col,year): vector}`` mapping -- the caller decides whether that is
    the EMA'd z (what the trainer supervised) or raw z (what the cube exports).

    Each cell uses its own nearest actual survey within ``tol`` years of an epoch, and the model
    is read at that same real year: no averaging and no interpolation in time. Pairs are reported
    SEPARATELY and never pooled -- they share cells and nest in time (1967->2025 contains
    1985->2005), so a pooled figure would overstate the evidence.

    Curvature is the test a single pair cannot run. A 1967->2025 difference is a chord and cannot
    distinguish monotone growth from rise-then-fall, which for House Finch is the eastern story:
    expansion, then decline once conjunctivitis arrives in the 1990s.
    """
    row_of = {(int(r), int(c), int(y)): i for i, (r, c, y) in enumerate(pidx)}

    def zt(cell, year):
        return z_obs[row_of[(cell[0], cell[1], year)]]

    near = nearest_survey(pidx, supervise, epochs, tol)
    ho, bf = _np(holdout), _np(buffer_mask)
    val_of = {e: {c: y for c, y in near[e].items() if ho[c]} for e in epochs}
    trn_of = {e: {c: y for c, y in near[e].items() if not ho[c] and not bf[c]} for e in epochs}
    out = {"epochs": list(epochs), "tol": tol, "pairs": {}, "curvature": {}}

    if verbose:
        print("  pair          cells   model    idw     null   verdict")
    for a, b in itertools.combinations(epochs, 2):
        # Intersect FIRST, then test z_model. The comprehension is evaluated in full before
        # the `&` runs, so filtering over val_of[a] would index val_of[b] at cells epoch b
        # never surveyed -- a KeyError, not a quiet drop.
        both = set(val_of[a]) & set(val_of[b])
        cells = sorted(c for c in both
                       if (c[0], c[1], val_of[a][c]) in z_model
                       and (c[0], c[1], val_of[b][c]) in z_model)
        if len(cells) < 10:
            if verbose:
                print(f"  {a}->{b}  {len(cells):>6}   (too few cells)")
            continue
        dt = np.stack([zt(c, val_of[b][c]) - zt(c, val_of[a][c]) for c in cells])
        dm = np.stack([z_model[(c[0], c[1], val_of[b][c])] - z_model[(c[0], c[1], val_of[a][c])]
                       for c in cells])
        ia, ib = _idw_at(cells, trn_of[a], zt), _idw_at(cells, trn_of[b], zt)
        di = (ib - ia) if (ia is not None and ib is not None) else None
        perm = np.random.default_rng(0).permutation(len(cells))
        mc = median_dir_cos_np(dm, dt)
        ic = median_dir_cos_np(di, dt) if di is not None else float("nan")
        null = median_dir_cos_np(dm, dt[perm])
        verdict = "model" if mc > ic + 0.02 else ("idw" if ic > mc + 0.02 else "tie")
        if verbose:
            print(f"  {a}->{b}  {len(cells):>6}   {mc:5.2f}   {ic:5.2f}   {null:5.2f}   {verdict}")
        out["pairs"][f"{a}_{b}"] = {"n": len(cells), "model_dir_cos": mc, "idw_dir_cos": ic,
                                    "null_dir_cos": null, "verdict": verdict}

    if verbose:
        print("  curvature (second difference, three consecutive epochs)")
    for a, b, c3 in zip(epochs, epochs[1:], epochs[2:]):
        cells = sorted(set(val_of[a]) & set(val_of[b]) & set(val_of[c3]))
        cells = [c for c in cells
                 if all((c[0], c[1], val_of[e][c]) in z_model for e in (a, b, c3))]
        if len(cells) < 10:
            if verbose:
                print(f"  {a}/{b}/{c3} {len(cells):>6}  (too few cells)")
            continue

        def second(get):
            d1 = np.stack([get(cc, b) - get(cc, a) for cc in cells])
            d2 = np.stack([get(cc, c3) - get(cc, b) for cc in cells])
            return d2 - d1

        sm = second(lambda cc, e: z_model[(cc[0], cc[1], val_of[e][cc])])
        st = second(lambda cc, e: zt(cc, val_of[e][cc]))
        si = None
        ii = {e: _idw_at(cells, trn_of[e], zt) for e in (a, b, c3)}
        if all(v is not None for v in ii.values()):
            si = (ii[c3] - ii[b]) - (ii[b] - ii[a])
        mc = median_dir_cos_np(sm, st)
        ic = median_dir_cos_np(si, st) if si is not None else float("nan")
        if verbose:
            print(f"  {a}/{b}/{c3} {len(cells):>6}  model {mc:5.2f}   idw {ic:5.2f}")
        out["curvature"][f"{a}_{b}_{c3}"] = {"n": len(cells), "model_dir_cos": mc,
                                             "idw_dir_cos": ic}
    return out


# --- The ladder: each rung is handed DIFFERENT information ------------------------
#
# What a baseline needs is what it isolates. Read as a set, they separate "DESK knows where"
# from "DESK knows when" from "DESK knows how this place changed":
#
#   no-change            the cell's modern state           zero dynamics
#   cell_temporal        the cell's other TRAINING years    time without covariates
#   spatial_interp       the target year elsewhere          space, year given free
#   borrowed_delta       modern state + neighbours' change  "it changed like its neighbours"
#   spacetime_idw        anything near in space AND time    joint interpolation
#
# Under a temporal block holdout the middle two go unavailable by construction -- there are no
# training points in those years at all. That narrowing is the point: it leaves the honest
# competitor set for backward extrapolation, and it is why spatial IDW could never test it.

_MIN_TREND_YEARS = 2          # a line needs two points
#: Space/time anisotropy candidates for the joint IDW bar, in grid cells per year. Extends
#: BELOW 1 on purpose: a cell is 27 km, and a year of community change is very unlikely to be
#: worth more than a whole cell of distance, so the interesting range is the sub-cell one. The
#: sweep picks among these on training rows; it is not a prior on the answer.
SPACETIME_RATIOS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)


def _train_rows(pidx, holdout, buffer_mask=None, exclude_years=()):
    """Rows the model actually trained on: cell not held out or buffered, year not withheld.

    ``exclude_years`` matters for the temporal experiment. A baseline that sourced from a
    withheld year would be handed information the model never saw, and would stop being a fair
    bar -- it would be scoring an interpolation of the answer.
    """
    ho = _np(holdout)
    keep = ~ho[pidx[:, 0], pidx[:, 1]]
    if buffer_mask is not None:
        keep &= ~_np(buffer_mask)[pidx[:, 0], pidx[:, 1]]
    if len(exclude_years):
        keep &= ~np.isin(pidx[:, 2], np.asarray(list(exclude_years)))
    return keep


def cell_temporal_baseline(pidx, z_obs, holdout, target_rows, mode="trend",
                           buffer_mask=None, exclude_years=()):
    """Predict a point from the SAME CELL's other training years. No covariates, no neighbours.

    ``mode="nearest"`` takes the training year closest in time; ``mode="trend"`` fits a line per
    z-dimension over that cell's training years and evaluates it at the target year, which is
    genuine temporal EXTRAPOLATION when the target lies outside their range.

    This is the rung that survives a temporal block holdout, so it is the one that matters for
    the backward-extrapolation claim. Returns per-row error aligned to ``pidx[target_rows]``,
    NaN where the cell has too few training years.
    """
    tr = _train_rows(pidx, holdout, buffer_mask, exclude_years)
    by_cell = {}
    for i in np.flatnonzero(tr):
        by_cell.setdefault((int(pidx[i, 0]), int(pidx[i, 1])), []).append(i)

    idx = np.flatnonzero(target_rows)
    err = np.full(len(idx), np.nan, dtype="float32")
    need = _MIN_TREND_YEARS if mode == "trend" else 1
    for out_i, i in enumerate(idx):
        rows = by_cell.get((int(pidx[i, 0]), int(pidx[i, 1])))
        if not rows:
            continue
        rows = [r for r in rows if r != i]                 # never its own answer
        if len(rows) < need:
            continue
        ys = pidx[rows, 2].astype("float64")
        y0 = float(pidx[i, 2])
        if mode == "nearest":
            pred = z_obs[rows[int(np.argmin(np.abs(ys - y0)))]]
        else:
            if np.ptp(ys) == 0:                            # all one year -> no slope
                continue
            A = np.vstack([ys - ys.mean(), np.ones_like(ys)]).T
            coef, *_ = np.linalg.lstsq(A, z_obs[rows], rcond=None)
            pred = coef[0] * (y0 - ys.mean()) + coef[1]
        err[out_i] = float(np.linalg.norm(pred - z_obs[i]))
    return err


def borrowed_delta_baseline(pidx, z_obs, holdout, target_rows, recent_year,
                            buffer_mask=None, k=8, power=2.0, exclude_years=()):
    """``z(c, recent) + mean over neighbours of [z(n, y) - z(n, recent)]``.

    The direct competitor to DESK's actual claim. DESK says the covariates tell you HOW a place
    changed; this says "assume it changed the way nearby places changed", using the cell's own
    modern state as the starting point. If DESK cannot beat this, its temporal contribution is a
    regional trend rather than anything covariate-driven.

    Needs training neighbours in BOTH the target year and the recent year, so it is unavailable
    under a temporal block holdout -- NaN, never a degenerate value.
    """
    from scipy.spatial import cKDTree

    tr = _train_rows(pidx, holdout, buffer_mask, exclude_years)
    idx = np.flatnonzero(target_rows)
    err = np.full(len(idx), np.nan, dtype="float32")

    # the target cell's own modern value, and each training cell's
    rec_of = {(int(r), int(c)): i for i, (r, c, y) in enumerate(pidx) if int(y) == recent_year}
    for y in np.unique(pidx[idx, 2]):
        if int(y) == recent_year:
            continue
        src = [i for i in np.flatnonzero(tr & (pidx[:, 2] == y))
               if (int(pidx[i, 0]), int(pidx[i, 1])) in rec_of]
        if len(src) < k:
            continue
        deltas = np.stack([z_obs[i] - z_obs[rec_of[(int(pidx[i, 0]), int(pidx[i, 1]))]]
                           for i in src])
        tree = cKDTree(pidx[src, :2].astype(float))
        want = [(o, i) for o, i in enumerate(idx)
                if pidx[i, 2] == y and (int(pidx[i, 0]), int(pidx[i, 1])) in rec_of]
        if not want:
            continue
        pos = np.array([o for o, _ in want])
        rows = np.array([i for _, i in want])
        dist, nb = tree.query(pidx[rows, :2].astype(float), k=k)
        w = np.where(dist > 1e-9, 1.0 / np.maximum(dist, 1e-6) ** power, 0.0)
        ok = w.sum(axis=1) > 0
        w[ok] /= w[ok].sum(axis=1, keepdims=True)
        borrowed = np.einsum("nk,nkl->nl", w, deltas[nb])
        base = np.stack([z_obs[rec_of[(int(pidx[i, 0]), int(pidx[i, 1]))]] for i in rows])
        e = np.linalg.norm(base + borrowed - z_obs[rows], axis=1)
        err[pos[ok]] = e[ok]
    return err


def _spacetime_predict(pidx, z_obs, tr_rows, target_rows, ratio, k=8, power=2.0):
    """``(pred, ok)`` -- spacetime-IDW estimate of z at ``target_rows``, at anisotropy ``ratio``.

    Zero-distance neighbours are dropped, so a training row is never handed its own answer.
    """
    from scipy.spatial import cKDTree

    def coords(rows):
        c = pidx[rows][:, [0, 1, 2]].astype("float64").copy()
        c[:, 2] *= ratio                                   # years -> grid-cell units
        return c

    src = np.flatnonzero(tr_rows)
    idx = np.flatnonzero(target_rows)
    pred = np.full((len(idx), z_obs.shape[1]), np.nan, dtype="float32")
    if len(src) < k or not len(idx):
        return pred, np.zeros(len(idx), bool)
    dist, nb = cKDTree(coords(src)).query(coords(idx), k=k)
    w = np.where(dist > 1e-9, 1.0 / np.maximum(dist, 1e-6) ** power, 0.0)
    ok = w.sum(axis=1) > 0
    w[ok] /= w[ok].sum(axis=1, keepdims=True)
    pred[ok] = np.einsum("nk,nkl->nl", w[ok], z_obs[src][nb[ok]])
    return pred, ok


def _spacetime_err(pidx, z_obs, tr_rows, target_rows, ratio, k=8, power=2.0):
    """Per-row error of spacetime IDW at a given space/time anisotropy ``ratio``."""
    idx = np.flatnonzero(target_rows)
    pred, ok = _spacetime_predict(pidx, z_obs, tr_rows, target_rows, ratio, k=k, power=power)
    err = np.full(len(idx), np.nan, dtype="float32")
    err[ok] = np.linalg.norm(pred[ok] - z_obs[idx][ok], axis=1)
    return err


def spacetime_idw_z(pidx, z_obs, holdout, ratio, buffer_mask=None, k=8, power=2.0,
                    exclude_years=()):
    """A full ``(N, L)`` stand-in for the model's Z, interpolated in spacetime from TRAINING rows.

    Lets the report re-run its EXISTING metrics -- ``directional_change_agreement``,
    ``analog_displacement`` -- on an interpolation instead of the model, which turns every
    "vs null" figure into "vs a real alternative" without writing a second copy of any metric.
    Rows the interpolation cannot reach come back NaN and the metric drops them as usual.
    """
    tr = _train_rows(pidx, holdout, buffer_mask, exclude_years)
    pred, _ok = _spacetime_predict(pidx, z_obs, tr, np.ones(len(pidx), bool), ratio,
                                   k=k, power=power)
    return pred


def spacetime_idw_baseline(pidx, z_obs, holdout, target_rows, buffer_mask=None,
                           ratios=SPACETIME_RATIOS, k=8, power=2.0, verbose=True,
                           exclude_years=()):
    """Inverse-distance interpolation in SPACE AND TIME jointly. Returns ``(err, ratio)``.

    One ``cKDTree`` over ``(row, col, year * ratio)``, so ordinary Euclidean distance in the
    scaled space IS spacetime distance -- no bespoke metric.

    ``ratio`` (grid cells per year) is chosen by holding out a random half of the TRAINING rows
    and scoring the rest, then taking the minimiser. Fitted on training data only, so it cannot
    leak; and a bar should be the strongest cheap alternative rather than a guess, which is why
    it is measured instead of hardcoded.
    """
    tr = _train_rows(pidx, holdout, buffer_mask, exclude_years)
    tr_idx = np.flatnonzero(tr)
    best, best_ratio = np.inf, float(ratios[0])
    if len(tr_idx) >= 4 * k:
        rng = np.random.default_rng(0)
        probe = rng.permutation(tr_idx)[: len(tr_idx) // 2]
        fit = np.zeros(len(pidx), bool); fit[tr_idx] = True; fit[probe] = False
        pm = np.zeros(len(pidx), bool); pm[probe] = True
        for r in ratios:
            e = _spacetime_err(pidx, z_obs, fit, pm, float(r), k=k, power=power)
            fin = np.isfinite(e)
            if fin.sum() and float(np.median(e[fin])) < best:
                best, best_ratio = float(np.median(e[fin])), float(r)
        if verbose:
            print(f"  spacetime IDW anisotropy: {best_ratio:g} grid cells per year "
                  f"(chosen on training rows only, from {list(ratios)})")
    return _spacetime_err(pidx, z_obs, tr, target_rows, best_ratio, k=k, power=power), best_ratio


def _era_of(years):
    """Decade label per year, so eras are legible without a config knob."""
    return np.array([f"{int(y) // 10 * 10}s" for y in years])


def baseline_panel(pidx, z_obs, z_desk, holdout, recent_year, buffer_mask=None,
                   heldout_only=True, verbose=True, target_rows=None, exclude_years=()):
    """The whole ladder, scored per era against DESK. Returns a nested dict.

    Reports **DESK beats it in X%** per rung per era, plus median errors. Held-out cells only by
    default -- a training cell's score is not evidence about anything.

    ``target_rows`` overrides what is graded, which is how the temporal experiment gets its
    three buckets (unseen year / unseen cell / both). ``exclude_years`` must carry the withheld
    years so no rung sources from them.

    **The n/a pattern is structural, not a defect, and the two holdouts are complementary:**

    * Under a SPATIAL holdout a held-out cell is held out in every year, so it has no training
      years of its own -- ``cell_nearest_year`` and ``cell_trend`` cannot run at all.
    * Under a TEMPORAL holdout there are no training points in the withheld years, so
      ``spatial_interp`` and ``borrowed_delta`` cannot run.

    Neither holdout alone can exercise the whole ladder. That is the argument for running both
    rather than picking one.
    """
    ho = _np(holdout)
    is_ho = ho[pidx[:, 0], pidx[:, 1]]
    if target_rows is not None:
        target = np.asarray(target_rows, bool)
    else:
        target = is_ho if heldout_only else np.ones(len(pidx), bool)
    target = target & (pidx[:, 2] != recent_year)      # the recent year IS the no-change source
    if not target.any():
        return {"note": "no target rows"}

    err_desk_all = np.linalg.norm(z_desk - z_obs, axis=1)
    rec_of = {(int(r), int(c)): i for i, (r, c, y) in enumerate(pidx) if int(y) == recent_year}
    nochange = np.full(len(pidx), np.nan, dtype="float32")
    for i in np.flatnonzero(target):
        j = rec_of.get((int(pidx[i, 0]), int(pidx[i, 1])))
        if j is not None:
            nochange[i] = np.linalg.norm(z_obs[j] - z_obs[i])

    _kw = {"buffer_mask": buffer_mask, "exclude_years": exclude_years}
    st_err, ratio = spacetime_idw_baseline(pidx, z_obs, holdout, target,
                                           verbose=verbose, **_kw)
    bars = {
        "no_change": nochange[target],
        "cell_nearest_year": cell_temporal_baseline(pidx, z_obs, holdout, target,
                                                    mode="nearest", **_kw),
        "cell_trend": cell_temporal_baseline(pidx, z_obs, holdout, target,
                                             mode="trend", **_kw),
        "borrowed_delta": borrowed_delta_baseline(pidx, z_obs, holdout, target, recent_year,
                                                  **_kw),
        "spacetime_idw": st_err,
    }
    ed = err_desk_all[target]
    eras = _era_of(pidx[target, 2])
    out = {"recent_year": int(recent_year), "spacetime_ratio_cells_per_year": ratio,
           "heldout_only": bool(heldout_only), "by_era": {}, "overall": {}}

    def _score(mask):
        row = {"n": int(mask.sum()), "median_err_desk": float(np.median(ed[mask]))}
        for name, e in bars.items():
            fin = mask & np.isfinite(e)
            row[name] = ({"n": int(fin.sum()), "median_err": float(np.median(e[fin])),
                          "desk_beats_frac": float(np.mean(ed[fin] < e[fin]))}
                         if fin.sum() >= 4 else {"n": int(fin.sum()), "median_err": float("nan"),
                                                 "desk_beats_frac": float("nan")})
        return row

    for era in sorted(set(eras)):
        out["by_era"][era] = _score(eras == era)
    out["overall"] = _score(np.ones(len(ed), bool))

    if verbose:
        names = list(bars)
        print(f"  DESK beats each bar, held-out cells, by era (n/a = bar cannot run there)")
        print("  era        n   errDESK  " + "  ".join(f"{n[:13]:>13}" for n in names))
        for era in sorted(set(eras)) + ["overall"]:
            r = out["by_era"][era] if era != "overall" else out["overall"]
            cells = []
            for n in names:
                f = r[n]["desk_beats_frac"]
                cells.append("          n/a" if not np.isfinite(f) else f"{f:12.1%} ")
            print(f"  {era:<8} {r['n']:>5}   {r['median_err_desk']:7.4f}  " + " ".join(cells))
    return out
