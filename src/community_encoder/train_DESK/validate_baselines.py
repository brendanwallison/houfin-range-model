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
        cells = sorted(set(val_of[a]) & set(val_of[b]) & {c for c in val_of[a]
                                                          if (c[0], c[1], val_of[a][c]) in z_model
                                                          and (c[0], c[1], val_of[b][c]) in z_model})
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
