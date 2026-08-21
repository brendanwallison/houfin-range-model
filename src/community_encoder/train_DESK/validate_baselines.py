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


def error_decomposition(a, b, eps=1e-12):
    """Split ``||a - b||^2`` into its magnitude and angular halves. EXACT, no residual.

    ``a`` is the prediction, ``b`` the truth, both ``(n, L)``. Returns arrays
    ``(total, magnitude, angular, cos)`` with ``magnitude + angular == total`` to
    floating-point, from the identity

        ||a - b||^2 = (||a|| - ||b||)^2  +  2 ||a|| ||b|| (1 - cos t)
                      |--- magnitude ---|    |------- angular -------|

    WHY THIS EXISTS. Every angular measure in this codebase -- ``rot``, ``dcos``, the epoch
    panel's dir-cos, ``cosine_gram`` -- is precisely ONE of these two terms, and the other one is
    the norm/shrinkage deficit (measured ``||z_desk||^2 ~ 0.60`` against a contract of 1.0).
    Reporting an angle without its magnitude partner hides half the error, and hides that the two
    TRADE OFF: minimising the total over ``||a||`` at fixed ``cos = rho`` gives ``||a|| =
    rho*||b||``, so shrinkage is the MSE-optimal response to a poor angle. That is also where
    ``rot ~ dcos^2`` comes from (via ``1 - cos ~ theta^2/2``), which makes ``cal = rot/dcos^2`` a
    ratio between these two terms rather than an ad-hoc index.

    Degenerate rows (either vector ~0) get ``cos = nan``; their magnitude term is still exact and
    their angular term is 0, which is the correct split for a zero-length prediction.
    """
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    total = np.sum((a - b) ** 2, axis=-1)
    mag = (na - nb) ** 2
    # angular = total - mag identically; computing it by subtraction rather than from cos keeps
    # the identity exact even where cos is undefined, and cos is reported separately for reading.
    ang = total - mag
    den = na * nb
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(den > eps, np.sum(a * b, axis=-1) / np.maximum(den, eps), np.nan)
    return total, mag, ang, cos


def median_dir_cos_np(dp, dt):
    from .desk_training import median_dir_cos
    return median_dir_cos(torch.as_tensor(np.asarray(dp), dtype=torch.float32),
                          torch.as_tensor(np.asarray(dt), dtype=torch.float32))


def _interp_usable_years(tgt, k=8):
    """Years spatial interpolation can actually serve: some val cell AND >= ``k`` train cells.

    Factored out so ``spatial_interp_baseline`` and ``interp_year_coverage`` cannot drift. A
    temporally withheld year carries an all-zero train mask (that is how it is withheld), so it
    fails the ``>= k`` test and is skipped. Correct -- but the skip is INVISIBLE unless the count
    is reported, and that invisibility is what made "model 0.2222 vs idw 0.2143" look like a
    comparison when the two sides sat on different year sets.
    """
    out = []
    for y in sorted(tgt):
        tr_np, va_np = _np(tgt[y][1]), _np(tgt[y][2])
        if va_np.any() and tr_np.sum() >= k:
            out.append(int(y))
    return out


def interp_year_coverage(tgt, k=8):
    """``(n_usable, n_total)`` -- supervised years the spatial IDW bar actually covers.

    Reported next to the bar so the reader knows which model column it may be compared against.
    Under a temporal holdout the withheld years are missing here, so the bar is comparable to the
    trained-year val MSE (``va(sp)``) and NOT to the pooled one.
    """
    return len(_interp_usable_years(tgt, k)), len(tgt)


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

    Returns ``(nearest_mse, idw_mse)`` on the same per-cell summed scale as ``_z_mse``. The
    arity is load-bearing: seven call sites unpack it as a 2-tuple, so year coverage is reported
    by ``interp_year_coverage`` rather than appended here.

    Under a TEMPORAL holdout this bar is not merely unavailable for the withheld years, it is
    **inadmissible**. The truth is still sitting in ``zg`` (only the train mask was zeroed), so a
    value could be computed -- but it would interpolate that year's answer from that year's
    neighbours, information DESK never saw anywhere. Different information sets are not a bar.
    ``spacetime_idw_baseline`` is the admissible relative: it borrows across years as well as
    space, and its source rows honour ``exclude_years``.

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

    years = _interp_usable_years(tgt, k)
    if not years:
        return float("nan"), float("nan")

    sq_n = sq_i = 0.0
    cnt = 0
    cache = {}
    for y in years:                                        # same skip as _z_mse's m.any()
        zg, tr_m, va_m = tgt[y][0], tgt[y][1], tgt[y][2]
        tr_np, va_np = _np(tr_m), _np(va_m)
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


def surveys_in_window(pip, supervise, epochs, half_width):
    """``{epoch: {(row,col): (years,...)}}`` -- EVERY surveyed year within +/-half_width.

    The windowed sibling of :func:`nearest_survey`. ``nearest_survey`` answers "which single year
    stands for this epoch"; this answers "which years should be averaged to stand for it". The
    target is raw BBS at ~1.08 routes per cell-year, so a single-year endpoint is one observer on
    one morning: its noise inflates ``||dt||`` and randomizes its direction, attenuating every
    dir-cos toward zero by roughly ``tau/sqrt(tau^2+sigma^2)``. Worse for a cross-era sweep,
    ``sigma`` is not constant across eras -- the first-year-observer share is 25.6% in 1966-1980
    against 12.3% in 2001-2025 -- so the attenuation is DIFFERENTIAL and lands directly on the
    axis the temporal sweep varies.

    Averaging depth is deliberately unequal: inclusion is "any survey in the window", not "all
    years present", because requiring full coverage would thin the early era hardest -- exactly
    where the noise is worst. The caller reports the depth distribution so the imbalance is
    visible rather than assumed away.

    Same ``supervise`` gate and ascending-year iteration as ``nearest_survey``, so the two select
    from the same row population.
    """
    sup = supervise if supervise is not None else np.ones(len(pip), bool)
    hw = int(half_width)
    out = {int(e): {} for e in epochs}
    for i in np.lexsort((pip[:, 2],)):                     # ascending year -> stable tuples
        if not sup[i]:
            continue
        r, c, y = int(pip[i, 0]), int(pip[i, 1]), int(pip[i, 2])
        for e in out:
            if abs(y - e) <= hw:
                out[e].setdefault((r, c), []).append(y)
    return {e: {c: tuple(ys) for c, ys in d.items()} for e, d in out.items()}


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


def zspace_idw_baseline(pidx, z_obs, holdout, hist_mask, k=8, power=2.0, return_z=False):
    """Per-point Z-space error of interpolating the OBSERVED z from training cells, same year.

    The reconstruction metric compares DESK against a no-change null. That null is weak in the
    deep past by construction -- it assumes 60 years of stasis -- so beating it says little. This
    is the bar that matters: no covariates, no learning, just "the observed community at nearby
    cells surveyed the same year". Returns per-point error aligned to ``pidx[hist_mask]``, NaN
    where a year had fewer than ``k`` training cells.

    ``return_z=True`` additionally returns the interpolated ``(n_hist, L)`` latents themselves, so
    the bar can be graded by the SAME predictor table as every other predictor rather than only
    through a precomputed error. A flag, not a changed arity, matching ``return_proj`` /
    ``return_pidx`` elsewhere in this package -- the five existing callers keep working untouched.
    """
    from scipy.spatial import cKDTree

    hist_idx = np.flatnonzero(hist_mask)
    err = np.full(len(hist_idx), np.nan, dtype="float32")
    zi = np.full((len(hist_idx), z_obs.shape[1]), np.nan, dtype="float32")
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
        if return_z:
            zi[pos[ok]] = pred[ok]
    return (err, zi) if return_z else err


def epoch_direction_panel(pidx, supervise, z_obs, z_model, holdout, buffer_mask,
                          epochs=DEFAULT_EPOCHS, tol=DEFAULT_TOL, verbose=True,
                          exclude_years=(), half_width=0, z_spacetime=None,
                          x_obs=None, project=None):
    """Model vs inverse-distance on the DIRECTION of change, per epoch pair, plus curvature.

    ``z_model`` is a ``{(row,col,year): vector}`` mapping -- the caller decides whether that is
    the EMA'd z (what the trainer supervised) or raw z (what the cube exports).

    Each cell uses its own nearest actual survey within ``tol`` years of an epoch, and the model
    is read at that same real year: no averaging and no interpolation in time. Pairs are reported
    SEPARATELY and never pooled -- they share cells and nest in time (1967->2025 contains
    1985->2005), so a pooled figure would overstate the evidence.

    ``exclude_years`` MUST carry ``desk.trend.holdout_years``. The IDW source set is otherwise
    filtered on the spatial masks alone, so for an epoch inside a temporal holdout the bar would
    interpolate that year's TRUTH from that year's neighbours -- information the model never saw
    anywhere. That is not a weaker bar, it is a different information set, and it silently rigged
    every epoch inside the holdout (``DEFAULT_EPOCHS`` puts 1967 inside all three sweep runs and
    1985 inside two). Excluded years leave the bar unavailable -> ``nan`` -> printed ``n/a``,
    matching the structural n/a pattern documented on ``baseline_panel``.

    ``half_width`` > 0 replaces each single-year endpoint with the MEAN over all of a cell's
    surveys within +/-half_width of the epoch, on the model and the target and the bar alike --
    see ``surveys_in_window``. The target is raw BBS at ~1.08 routes per cell-year, so a
    single-year endpoint is one observer on one morning; that noise attenuates every dir-cos
    toward zero, and unequally by era. ``half_width=0`` is the historical single-year behaviour.

    Curvature is the test a single pair cannot run. A 1967->2025 difference is a chord and cannot
    distinguish monotone growth from rise-then-fall, which for House Finch is the eastern story:
    expansion, then decline once conjunctivitis arrives in the 1990s.
    """
    ex = set(int(y) for y in (exclude_years or ()))
    row_of = {(int(r), int(c), int(y)): i for i, (r, c, y) in enumerate(pidx)}

    # Endpoint years per cell. half_width=0 keeps the historical single-year selection exactly
    # (nearest survey within tol, ties to the earlier year); >0 averages a window. Both are
    # normalized to a TUPLE of years so everything downstream -- target, model and IDW bar alike
    # -- runs one code path and cannot treat the three asymmetrically.
    if int(half_width) > 0:
        # The two tables are only comparable if they score the SAME cells, and inclusion is
        # "has a survey within +/-tol" for the single-year path but "within +/-half_width" for
        # this one. With half_width < tol the windowed table silently loses cells (measured: a
        # set giving n=30 at tol=2 collapsed to n=0 at half_width=1), which would make the gap
        # between the tables read as a noise effect when it is a population change. Floor the
        # window at tol: equal at the default half_width == tol, a superset above it, never less.
        hw_eff = max(int(half_width), int(tol))
        if hw_eff != int(half_width) and verbose:
            print(f"  half_width raised {half_width} -> {hw_eff} to match tol={tol}, so the "
                  f"windowed and single-year tables cover the same cells")
        half_width = hw_eff
        near = surveys_in_window(pidx, supervise, epochs, half_width)
    else:
        near = {e: {c: (y,) for c, y in d.items()}
                for e, d in nearest_survey(pidx, supervise, epochs, tol).items()}

    # RAW-SPACE averaging for the truth side when it is available. Averaging raw counts and then
    # projecting is the CONSISTENT estimator of a denoised community: the mean of raw counts is the
    # MVUE for a Poisson rate, whereas averaging z converges on the noise-attenuated quantity
    # instead (simulated: 0.234 vs 0.138 against a true 0.2354). It is gated because
    # phi(mean x) is off-span unless the ESK landmark support was widened to cover window means --
    # see esk_kernel.augment_with_windowed. Ungated, this would silently project inputs the basis
    # cannot represent, which is exactly how the ceiling oracle produced a withdrawn finding.
    raw_truth = False
    if x_obs is not None and project is not None:
        try:
            _probe = project(np.log1p(np.expm1(np.maximum(np.asarray(x_obs[:512]), 0.0))))
            _ann = float(np.median((np.asarray(z_obs[:512]) ** 2).sum(1)))
            _avg = float(np.median((np.asarray(_probe) ** 2).sum(1)))
            raw_truth = _ann > 0 and (_avg / _ann) >= 0.5
            if not raw_truth and verbose:
                print(f"  truth side stays in z-space: window means project to ||z||^2 {_avg:.4f} "
                      f"against {_ann:.4f} annual, so the basis does not span them. The z-space "
                      f"average is the NOISE-ATTENUATED estimand -- read dir-cos accordingly.")
        except Exception:
            raw_truth = False

    def zt(cell, years):
        """Mean observed community over the cell's window (None if none are present).

        Averages RAW counts then projects when the basis can represent the result; otherwise falls
        back to averaging z and says so. The two are different estimands, not two routes to one.
        """
        rows = [row_of[(cell[0], cell[1], y)] for y in years
                if (cell[0], cell[1], y) in row_of]
        if not rows:
            return None
        if raw_truth:
            cnt = np.expm1(np.maximum(np.asarray(x_obs)[rows], 0.0)).mean(axis=0)
            return np.asarray(project(np.log1p(cnt)[None, :]))[0]
        return np.mean(np.stack([z_obs[r] for r in rows]), axis=0)

    def zm(cell, years):
        """Mean MODEL z over the SAME years -- symmetry is the point; see the docstring."""
        vs = [z_model[(cell[0], cell[1], y)] for y in years
              if (cell[0], cell[1], y) in z_model]
        return np.mean(np.stack(vs), axis=0) if vs else None

    def zs(cell, years):
        """Mean SPACETIME-IDW z over the same years. Same window as model and target."""
        rows = [row_of[(cell[0], cell[1], y)] for y in years
                if (cell[0], cell[1], y) in row_of]
        if not rows:
            return np.full(z_obs.shape[1], np.nan)
        return np.mean(np.stack([z_spacetime[r] for r in rows]), axis=0)

    ho, bf = _np(holdout), _np(buffer_mask)
    val_of = {e: {c: ys for c, ys in near[e].items() if ho[c]} for e in epochs}
    # Training sources: spatial masks AND the year filter. Dropping the year filter is the bug
    # described in the docstring -- the bar would be handed the withheld year's answer. A cell
    # whose every windowed year is withheld drops out entirely rather than contributing a
    # partial (and differently-smoothed) average.
    trn_of = {}
    for e in epochs:
        keep = {}
        for c, ys in near[e].items():
            if ho[c] or bf[c]:
                continue
            ok = tuple(y for y in ys if int(y) not in ex)
            if ok:
                keep[c] = ok
        trn_of[e] = keep
    # An epoch has no admissible bar when the year filter is what emptied its source set --
    # distinct from an epoch that simply has no nearby training cells, which is not a holdout
    # artifact and should not be described as one.
    _spatial_src = {int(e): any(not ho[c] and not bf[c] for c in near[e]) for e in epochs}
    withheld_epoch = {int(e): (not trn_of[e]) and _spatial_src[int(e)] for e in epochs}
    out = {"epochs": list(epochs), "tol": tol, "exclude_years": sorted(ex),
           "half_width": int(half_width),
           # Which estimand the truth side is: raw-space averaging is consistent for the denoised
           # community, z-space averaging is asymptotically biased toward the noise-attenuated one.
           "truth_averaging": "raw_counts_then_project" if raw_truth else "z_space_biased",
           "epochs_without_bar": [e for e in epochs if withheld_epoch[int(e)]],
           "pairs": {}, "curvature": {}}
    if verbose and any(withheld_epoch.values()):
        print(f"  epochs with no admissible IDW bar (inside the temporal holdout): "
              f"{[e for e in epochs if withheld_epoch[int(e)]]} -- 'n/a', not a failure")

    if verbose:
        print("  pair          cells   model    idw   st-idw    null   verdict   depth")
    for a, b in itertools.combinations(epochs, 2):
        # Intersect FIRST, then test z_model. The comprehension is evaluated in full before
        # the `&` runs, so filtering over val_of[a] would index val_of[b] at cells epoch b
        # never surveyed -- a KeyError, not a quiet drop.
        both = set(val_of[a]) & set(val_of[b])
        cells = sorted(c for c in both
                       if zm(c, val_of[a][c]) is not None and zm(c, val_of[b][c]) is not None
                       and zt(c, val_of[a][c]) is not None and zt(c, val_of[b][c]) is not None)
        if len(cells) < 10:
            if verbose:
                print(f"  {a}->{b}  {len(cells):>6}   (too few cells)")
            continue
        dt = np.stack([zt(c, val_of[b][c]) - zt(c, val_of[a][c]) for c in cells])
        dm = np.stack([zm(c, val_of[b][c]) - zm(c, val_of[a][c]) for c in cells])
        # dir-cos is the ANGULAR half of an exact two-term split of ||dm - dt||^2; the other half
        # is the magnitude of the predicted change. Reporting the angle alone cannot distinguish
        # "moved the wrong way" from "barely moved", and the two trade off -- under-moving is the
        # MSE-optimal response to a poor angle, so a good dir-cos with a tiny magnitude ratio is a
        # different model from a good dir-cos that also moves the right distance.
        _tot, _mag, _ang, _ = error_decomposition(dm, dt)
        nm, nt = np.linalg.norm(dm, axis=1), np.linalg.norm(dt, axis=1)
        mag_ratio = float(np.median(nm[nt > 1e-12] / nt[nt > 1e-12])) if (nt > 1e-12).any() \
            else float("nan")
        mag_share = float(np.mean(_mag) / max(np.mean(_tot), 1e-12))
        ia, ib = _idw_at(cells, trn_of[a], zt), _idw_at(cells, trn_of[b], zt)
        di = (ib - ia) if (ia is not None and ib is not None) else None
        # The SPACETIME bar. The spatial one above is unavailable for any epoch inside the
        # temporal holdout (no training cells that year), which is exactly the deep-past epoch
        # the experiment cares about -- so the pairs that matter most had no bar at all. This one
        # borrows across years as well as space and does reach them.
        ds = None
        if z_spacetime is not None:
            sa = np.stack([zs(c, val_of[a][c]) for c in cells])
            sb = np.stack([zs(c, val_of[b][c]) for c in cells])
            if np.isfinite(sa).all() and np.isfinite(sb).all():
                ds = sb - sa
        perm = np.random.default_rng(0).permutation(len(cells))
        mc = median_dir_cos_np(dm, dt)
        ic = median_dir_cos_np(di, dt) if di is not None else float("nan")
        sc = median_dir_cos_np(ds, dt) if ds is not None else float("nan")
        null = median_dir_cos_np(dm, dt[perm])
        # A missing bar is not a tie. Saying "tie" when the bar could not run would read as
        # evidence of parity, which is the opposite of what an absent comparison means.
        if not np.isfinite(ic):
            verdict = "no-bar"
        else:
            verdict = "model" if mc > ic + 0.02 else ("idw" if ic > mc + 0.02 else "tie")
        depth = float(np.mean([len(val_of[a][c]) + len(val_of[b][c]) for c in cells]) / 2.0)
        ic_s = "  n/a" if not np.isfinite(ic) else f"{ic:5.2f}"
        if verbose:
            sc_s = "  n/a" if not np.isfinite(sc) else f"{sc:5.2f}"
            print(f"  {a}->{b}  {len(cells):>6}   {mc:5.2f}   {ic_s}   {sc_s}   {null:5.2f}   "
                  f"{verdict:<7}  {depth:.2f}")
        out["pairs"][f"{a}_{b}"] = {"n": len(cells), "model_dir_cos": mc, "idw_dir_cos": ic,
                                    "null_dir_cos": null, "verdict": verdict,
                                    "mean_window_depth": depth,
                                    # the magnitude half, so the pair's error is attributable
                                    "change_magnitude_ratio": mag_ratio,
                                    "err_magnitude_share": mag_share,
                                    "err_angular_share": 1.0 - mag_share,
                                    # MSE-calibrated magnitude ratio is exactly dir-cos; >1 means
                                    # moving further than the direction accuracy justifies
                                    "magnitude_calibration": (mag_ratio / mc)
                                    if (np.isfinite(mc) and abs(mc) > 1e-6) else float("nan"),
                                    # available even inside the holdout, unlike idw_dir_cos
                                    "spacetime_idw_dir_cos": sc}

    if verbose:
        print("  curvature (second difference, three consecutive epochs)")
    for a, b, c3 in zip(epochs, epochs[1:], epochs[2:]):
        cells = sorted(set(val_of[a]) & set(val_of[b]) & set(val_of[c3]))
        cells = [c for c in cells
                 if all(zm(c, val_of[e][c]) is not None and zt(c, val_of[e][c]) is not None
                        for e in (a, b, c3))]
        if len(cells) < 10:
            if verbose:
                print(f"  {a}/{b}/{c3} {len(cells):>6}  (too few cells)")
            continue

        def second(get):
            d1 = np.stack([get(cc, b) - get(cc, a) for cc in cells])
            d2 = np.stack([get(cc, c3) - get(cc, b) for cc in cells])
            return d2 - d1

        sm = second(lambda cc, e: zm(cc, val_of[e][cc]))
        st = second(lambda cc, e: zt(cc, val_of[e][cc]))
        si = None
        ii = {e: _idw_at(cells, trn_of[e], zt) for e in (a, b, c3)}
        if all(v is not None for v in ii.values()):
            si = (ii[c3] - ii[b]) - (ii[b] - ii[a])
        mc = median_dir_cos_np(sm, st)
        ic = median_dir_cos_np(si, st) if si is not None else float("nan")
        if verbose:
            ic_s = "  n/a" if not np.isfinite(ic) else f"{ic:5.2f}"
            print(f"  {a}/{b}/{c3} {len(cells):>6}  model {mc:5.2f}   idw {ic_s}")
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
#:
#: Extended down to 0.01 because the first temporal-holdout sweep selected the OLD floor (0.1)
#: in all three runs -- an argmin on the grid boundary is censored, not measured, and the bar was
#: therefore weaker than the data wanted. 0.01 means a 27 km step is worth ~100 years of change,
#: which is a deliberately generous bound rather than a belief. ``spacetime_idw_baseline`` warns
#: when the winner still lands on an endpoint, so a future censoring cannot pass silently.
SPACETIME_RATIOS = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)


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

    ``ratio`` (grid cells per year) is chosen by scoring a held-back probe of the TRAINING rows
    and taking the minimiser. Fitted on training data only, so it cannot leak; and a bar should be
    the strongest cheap alternative rather than a guess, which is why it is measured.

    **The probe has to match the gap being judged.** A random half of the training rows is mostly
    SHORT temporal gaps -- a probe row usually has a training neighbour a year or two away -- so
    the ratio it selects is tuned for interpolation. Applied to a 20-30 year backward
    extrapolation that is the wrong tuning, and it makes the bar weaker (or stronger) than it
    should be for reasons unrelated to the model. So when ``exclude_years`` is non-empty the probe
    becomes the EARLIEST contiguous block of training years, sized to the withheld span: a
    synthetic temporal holdout inside the training set, extrapolated backward the same way the
    real task is. The mode and ratio are both printed, because the bar's strength depends on them.
    """
    tr = _train_rows(pidx, holdout, buffer_mask, exclude_years)
    tr_idx = np.flatnonzero(tr)
    best, best_ratio = np.inf, float(ratios[0])
    mode = "none"
    if len(tr_idx) >= 4 * k:
        tr_years = np.unique(pidx[tr_idx, 2])
        span = len(set(int(y) for y in (exclude_years or ())))
        # Cap the probe at half the training years: a probe wide enough to swallow most of
        # training would leave too little to interpolate FROM, and the ratio would be chosen
        # under a data regime the real bar never sees.
        n_probe_yr = int(min(span, max(1, len(tr_years) // 2))) if span else 0
        if n_probe_yr:
            probe_years = set(int(y) for y in tr_years[:n_probe_yr])
            probe = tr_idx[np.isin(pidx[tr_idx, 2], list(probe_years))]
            mode = f"long-gap probe ({min(probe_years)}-{max(probe_years)}, {n_probe_yr} yr)"
        else:
            probe = np.random.default_rng(0).permutation(tr_idx)[: len(tr_idx) // 2]
            mode = "random-half probe"
        # Guard both ends: too small a probe gives a noisy minimiser, too large leaves no
        # source rows. Fall back to the random half rather than silently fitting on nothing.
        if len(probe) < k or len(tr_idx) - len(probe) < 4 * k:
            probe = np.random.default_rng(0).permutation(tr_idx)[: len(tr_idx) // 2]
            mode = "random-half probe (long-gap probe too small or too large)"
        fit = np.zeros(len(pidx), bool); fit[tr_idx] = True; fit[probe] = False
        pm = np.zeros(len(pidx), bool); pm[probe] = True
        for r in ratios:
            e = _spacetime_err(pidx, z_obs, fit, pm, float(r), k=k, power=power)
            fin = np.isfinite(e)
            if fin.sum() and float(np.median(e[fin])) < best:
                best, best_ratio = float(np.median(e[fin])), float(r)
        if verbose:
            print(f"  spacetime IDW anisotropy: {best_ratio:g} grid cells per year "
                  f"({mode}, training rows only, from {list(ratios)})")
            # An argmin on the boundary is censored: the optimum may lie outside the grid, so the
            # bar is weaker than the data wants and any DESK margin over it is overstated. This
            # actually happened -- the first sweep picked the then-floor 0.1 in all three runs.
            if best_ratio in (float(ratios[0]), float(ratios[-1])):
                print(f"  WARNING: that ratio is the {'LOW' if best_ratio == float(ratios[0]) else 'HIGH'}"
                      f" end of the grid, so the optimum may lie outside it. This bar is a LOWER "
                      f"bound on the strength of spacetime interpolation; widen SPACETIME_RATIOS "
                      f"before trusting a narrow DESK win over it.")
    return _spacetime_err(pidx, z_obs, tr, target_rows, best_ratio, k=k, power=power), best_ratio


#: Gap, in years, at which the per-era long-baseline difference is measured, and its tolerance.
#: FIXED on purpose -- see ``per_era_attenuation``. An earlier version used each cell's own
#: first-to-last span, which varies systematically with era (a record starting in 1966 spans ~59
#: years, one starting in 2010 spans <=15), so the reported ordering ranked cells by RECORD LENGTH
#: rather than eras by noisiness.
ATTEN_GAP = 20
ATTEN_GAP_TOL = 2


def per_dimension_signal_noise(pidx, z_obs, min_pairs=30, gap=ATTEN_GAP,
                               gap_tol=ATTEN_GAP_TOL):
    """Where along the basis does temporal SIGNAL live, and where does survey NOISE live?

    ``{"noise_var": [..], "signal_var": [..], "total_var": [..], "snr": [..], ...}``, one entry per
    latent dimension, in eigen order.

    THE QUESTION THIS ANSWERS. ``zspace_reconstruction`` measures a monotone shrinkage profile --
    DESK reproduces the leading eigen-directions at ~1.11 of observed variance and the trailing ones
    at ~0.43. That was reported as evidence the kernel is tilted toward spatial structure, and the
    interpretation had to be withdrawn, because what the trailing directions CONTAIN was never
    measured. Noise accounts for roughly half the difference between two surveys of the same cell,
    and noise is high-dimensional and low-variance per direction, so it should land precisely in
    those trailing directions. **If the tail is mostly noise, shrinking it is correct behaviour and
    the tilt is the model denoising rather than failing.** Only this measurement separates the two.

    Same decomposition as :func:`per_era_attenuation`, resolved per dimension instead of summed:

    * ``noise_var[k]``  -- adjacent-year within-cell differences. Real community change over ONE
      year is small, so this is essentially ``2*sigma^2`` per dimension: pure measurement noise.
    * ``total_var[k]``  -- fixed-gap within-cell differences, carrying the real change over that
      gap PLUS the same noise.
    * ``signal_var[k]`` -- the difference, clipped at zero. Where temporal signal actually lives.

    The gap is FIXED for the reason recorded on ``per_era_attenuation``: using each cell's own span
    makes the baseline vary with record length, so the result ranks record length rather than the
    thing being asked about.

    Call this on ``z_obs`` -- the question is where the DATA's temporal signal sits, which then gets
    read against the model's shrinkage profile. Passing ``z_desk`` would ask a different question and
    silently answer it.
    """
    z_obs = np.asarray(z_obs, dtype="float64")
    L = z_obs.shape[1]
    lo_gap, hi_gap = int(gap) - int(gap_tol), int(gap) + int(gap_tol)
    by_cell = {}
    for i, (r, c, y) in enumerate(pidx):
        by_cell.setdefault((int(r), int(c)), []).append((int(y), i))
    adj = np.zeros(L); lng = np.zeros(L)
    n_adj = n_lng = 0
    for _cell, rows in by_cell.items():
        rows.sort()
        for a, (y0, i0) in enumerate(rows):
            for (y1, i1) in rows[a + 1:]:
                d = y1 - y0
                if d == 1:
                    adj += (z_obs[i1] - z_obs[i0]) ** 2
                    n_adj += 1
                elif lo_gap <= d <= hi_gap:
                    lng += (z_obs[i1] - z_obs[i0]) ** 2
                    n_lng += 1
                    break                     # one pair per (cell, earlier year), as elsewhere
                elif d > hi_gap:
                    break
    if n_adj < min_pairs or n_lng < min_pairs:
        return {"note": f"too few pairs (adjacent {n_adj}, gap {n_lng})",
                "n_adjacent_pairs": n_adj, "n_gap_pairs": n_lng}
    noise = adj / n_adj
    total = lng / n_lng
    signal = np.maximum(total - noise, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(noise > 1e-12, signal / noise, np.nan)
    k = np.arange(L)
    fin = np.isfinite(snr)
    return {"n_adjacent_pairs": n_adj, "n_gap_pairs": n_lng,
            "gap_years": int(gap), "gap_tol": int(gap_tol),
            "noise_var": [float(v) for v in noise],
            "total_var": [float(v) for v in total],
            "signal_var": [float(v) for v in signal],
            "snr": [None if not np.isfinite(v) else float(v) for v in snr],
            # Does temporal signal concentrate in the LEADING or the TRAILING directions? A negative
            # slope means signal-to-noise falls off along the basis, so the directions DESK shrinks
            # hardest are the ones carrying least signal -- i.e. the shrinkage is appropriate. A flat
            # or positive slope means it is discarding signal.
            "snr_slope": (float(np.polyfit(k[fin], snr[fin], 1)[0]) if fin.sum() >= 3
                          else float("nan")),
            "snr_leading_8": float(np.nanmedian(snr[:8])),
            "snr_trailing_8": float(np.nanmedian(snr[-8:])),
            "signal_share_leading_half": float(signal[:L // 2].sum()
                                               / max(signal.sum(), 1e-12))}


def per_era_attenuation(pidx, z_obs, era_width=10, min_pairs=30,
                        gap=ATTEN_GAP, gap_tol=ATTEN_GAP_TOL):
    """Per-era measurement noise and the dir-cos attenuation it causes. ``{era: {...}}``.

    Turns "the direction numbers are biased low" from a caveat into a measurement. Two
    quantities, both from the point set already in hand, and both binned on the EARLIER year of
    the pair:

    * ``adj_sq`` = mean ``||z(y+1) - z(y)||^2`` over cells surveyed in consecutive years. Real
      community change over ONE year is small, so this is essentially ``2*sigma^2`` -- pure
      measurement noise.
    * ``long_sq`` = the same over pairs separated by ``gap +/- gap_tol`` years, which carry both
      the true change over that gap and the same noise: ``tau^2 + 2*sigma^2``.

    So ``tau^2 = long_sq - adj_sq`` and the attenuation factor on a dir-cos measured against
    that noisy endpoint is ``sqrt(tau^2 / long_sq) = sqrt(1 - adj_sq/long_sq)`` -- no need to
    separate sigma at all. A factor of 0.8 means an observed dir-cos of 0.40 is consistent with
    a true 0.50.

    **The gap is FIXED, and that is the whole point of this function's shape.** An earlier
    version took ``long_sq`` from each cell's own first-to-last span while binning the era on the
    cell's FIRST survey year. Span then varies with era by construction, a longer record
    accumulates more real change, ``long_sq`` grows, and the attenuation figure rises -- so the
    table ranked record length, not era noisiness, and the shipped ordering (1960s 0.81 ->
    2010s 0.59) pointed the opposite way from the claim it was written to support. Holding the
    gap fixed makes the era the only thing that varies between rows.

    The AGGREGATE magnitude was never in doubt and is unchanged: noise accounts for a third to
    two-thirds of the difference between two surveys of the same cell, corroborated independently
    by windowed endpoints recovering ~30% of every epoch-pair dir-cos.

    This estimator is valid ONLY because the target is raw BBS (``target.source = bbs_raw``, ~1.08
    routes per cell-year), where consecutive years are independent observations. Against a
    reconstructed target -- where every year is a closed-form function of one anchor and a couple
    of rate coefficients -- adjacent-year differences are pure SIGNAL, ``adj_sq`` would collapse
    toward zero, and this would confidently report no attenuation. Do not point it at one.

    Returns ``{}`` for eras with too few pairs rather than a noisy number.
    """
    by_cell = {}
    for i, (r, c, y) in enumerate(pidx):
        by_cell.setdefault((int(r), int(c)), []).append((int(y), i))
    lo_gap, hi_gap = int(gap) - int(gap_tol), int(gap) + int(gap_tol)
    adj, lng = {}, {}
    for _cell, rows in by_cell.items():
        rows.sort()
        for a, (y0, i0) in enumerate(rows):
            era = f"{y0 // era_width * era_width}s"
            for (y1, i1) in rows[a + 1:]:
                d = y1 - y0
                if d == 1:
                    adj.setdefault(era, []).append(
                        float(np.sum((z_obs[i1] - z_obs[i0]) ** 2)))
                elif lo_gap <= d <= hi_gap:
                    # One pair per (cell, earlier year) at the fixed gap: taking every
                    # qualifying pair would weight densely-surveyed cells more heavily, and
                    # survey density is itself era-correlated.
                    lng.setdefault(era, []).append(
                        float(np.sum((z_obs[i1] - z_obs[i0]) ** 2)))
                    break
                elif d > hi_gap:
                    break
    out = {}
    for era in sorted(set(adj) & set(lng)):
        if len(adj[era]) < min_pairs or len(lng[era]) < min_pairs:
            continue
        a2, l2 = float(np.mean(adj[era])), float(np.mean(lng[era]))
        frac = 1.0 - (a2 / l2) if l2 > 1e-12 else float("nan")
        out[era] = {"n_adjacent_pairs": len(adj[era]), "n_long_pairs": len(lng[era]),
                    "gap_years": int(gap), "gap_tol": int(gap_tol),
                    "adjacent_sq": a2, "long_sq": l2,
                    "noise_share_of_long_gap": (a2 / l2) if l2 > 1e-12 else float("nan"),
                    "dir_cos_attenuation": float(np.sqrt(frac)) if frac > 0 else float("nan")}
    return out


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
        # DESK is a ROW here, not the permanent subject outside the table. It used to sit outside
        # as `median_err_desk` / `desk_beats_frac`, which made "does the spacetime bar beat the
        # no-change null" -- the question that says whether borrowing across time helps at all --
        # unaskable, and left DESK the only thing any comparison could be about.
        "desk": err_desk_all,
    }
    bars = {k: (v[target] if len(v) == len(pidx) else v) for k, v in bars.items()}
    ed = err_desk_all[target]
    eras = _era_of(pidx[target, 2])
    out = {"recent_year": int(recent_year), "spacetime_ratio_cells_per_year": ratio,
           "heldout_only": bool(heldout_only), "by_era": {}, "overall": {}}

    def _score(mask, reference="no_change"):
        """Every predictor graded on identical terms, with the subject of the comparison a NAMED
        argument. Win rates use the intersection of the pair's finite rows -- a bar reaches only
        where it has neighbours, and scoring its easy subset against a reference's full set would
        flatter whichever predictor declined the hardest rows."""
        row = {"n": int(mask.sum()), "reference": reference, "predictors": {}, "win_rate_vs": {}}
        for name, e in bars.items():
            fin = mask & np.isfinite(e)
            row["predictors"][name] = {
                "n": int(fin.sum()),
                "median_err": float(np.median(e[fin])) if fin.sum() >= 4 else float("nan")}
        ref = bars.get(reference)
        if ref is not None:
            for name, e in bars.items():
                both = mask & np.isfinite(e) & np.isfinite(ref)
                row["win_rate_vs"][name] = (float(np.mean(e[both] < ref[both]))
                                            if both.sum() >= 4 else float("nan"))
        return row

    for era in sorted(set(eras)):
        out["by_era"][era] = _score(eras == era)
    out["overall"] = _score(np.ones(len(ed), bool))

    if verbose:
        names = [n for n in bars if n != "no_change"]
        print("  each predictor vs the NO-CHANGE null, held-out cells, by era "
              "(n/a = predictor cannot run there). DESK is one row among them, so a bar beating "
              "another bar is visible too.")
        print("  era        n  " + "  ".join(f"{n[:13]:>13}" for n in names))
        for era in sorted(set(eras)) + ["overall"]:
            r = out["by_era"][era] if era != "overall" else out["overall"]
            cells = ["          n/a" if not np.isfinite(r["win_rate_vs"].get(n, np.nan))
                     else f"{r['win_rate_vs'][n]:12.1%} " for n in names]
            print(f"  {era:<8} {r['n']:>5}  " + " ".join(cells))
        ov = out["overall"]["predictors"]
        print("  median error: " + "  ".join(
            f"{n}={ov[n]['median_err']:.4f}" for n in sorted(ov) if np.isfinite(ov[n]["median_err"])))
    return out
