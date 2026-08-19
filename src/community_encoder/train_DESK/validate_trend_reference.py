"""Grade the encoder against the published trend products as a REFERENCE, not as truth.

The trend products used to be the training target. They are not any more, because their shape
made them unusable as one: each cell was reconstructed from about four time-invariant numbers,
capped, then Gaussian-smoothed across space, which left the target low-complexity in both axes
and made an interpolator near-optimal on it by construction. Validation MSE never beat an
inverse-distance baseline in any run.

They are still worth checking against. They encode real published knowledge about which
direction communities moved, and if the model disagrees with them about direction that is worth
knowing. What they are NOT worth checking against is MAGNITUDE: the soft caps
(``soft_max_fold`` at 100x, an absolute cap at the 95th percentile of the anchor) and the
spatial blur (``smooth_sigma_cells``) both act on magnitude, and our own measurements put the
cost of the blur at roughly a fifth of real per-cell change at sigma=2 cells and a third at
sigma=5.

So this module reports five comparisons, deliberately separated rather than combined into one
score. The separation is the point: the reference is smoothed SPATIALLY and closed-form
TEMPORALLY, so the two axes are contaminated differently, and a single number would hide which.

    1  temporal direction   per cell, does the model move the same WAY as the reference?
    2  temporal rank        are the cells that changed most the same cells?
    3  species trend sign   per cell and species, does the long-run direction match the
                            published percent-per-year sign?
    4  full similarity      across all (cell, year) pairs, does the whole similarity
                            structure match?
    5  spatial similarity   within a single year, does the spatial structure match?

1 and 2 are the CLEAN comparisons -- they use the reference's direction and ordering, not its
magnitudes, and the spatial blur affects them only weakly. 5 is the CONTAMINATED one: it is
exactly the axis the blur operates on. 4 mixes them. So if the model scores well on 5 and
poorly on 1, that is evidence about the reference rather than about the model, and reporting
them apart is what makes that readable.

Comparison 3 needs something the encoder does not provide. Z is a community embedding and the
ESK has only a forward map, so there is no way to read a single species' abundance out of it.
A ridge decoder from Z back to per-species abundance is fitted here for that purpose, and its
own per-species goodness of fit is reported alongside: where the decoder is poor, comparison 3
says nothing about the model and the fit statistic is what tells you so.
"""
import numpy as np

from .desk_training import median_dir_cos
from .validate_bbs_routes import (_offdiag, cosine_gram, dot_gram, kernel_error, vector_error)
from .validate_spacetime import linear_cka, mantel_r, partial_spearman, ruzicka_rect


def cell_epoch_rows(keys, min_gap=5):
    """Per cell, the row indices of its earliest and latest year, if far enough apart (pure).

    Returns ``(early_idx, late_idx, cells)``, all length ``n_cells``. Cells whose observed span
    is shorter than ``min_gap`` years are dropped: the temporal comparisons measure change, and
    over a 1-2 year span there is almost none to measure, so including those cells would dilute
    the statistic with rows that carry no signal either way.
    """
    keys = np.asarray(keys)
    first, last = {}, {}
    for i, (r, c, y) in enumerate(keys):
        cell, y = (int(r), int(c)), int(y)
        if cell not in first or y < first[cell][0]:
            first[cell] = (y, i)
        if cell not in last or y > last[cell][0]:
            last[cell] = (y, i)
    cells, ei, li = [], [], []
    for cell, (y0, i0) in sorted(first.items()):
        y1, i1 = last[cell]
        if y1 - y0 >= int(min_gap):
            cells.append(cell); ei.append(i0); li.append(i1)
    return np.asarray(ei, dtype=int), np.asarray(li, dtype=int), cells


def temporal_direction(Zm, Zr, early_idx, late_idx, rng, n_perm=20):
    """Comparison 1. Do model and reference move the same WAY at each cell?

    Per cell, the cosine between the model's change vector and the reference's. Uses direction
    only, so the reference's capped magnitudes never enter.

    The null is a shuffle: the model's change at one cell paired with the reference's change at
    a DIFFERENT cell. Without it the number is uninterpretable, because community change is
    spatially broad -- a model that predicted one continent-wide direction would score well
    against any reference. Averaged over ``n_perm`` shuffles so the null is a stable reference
    rather than one draw's noise.
    """
    import torch

    dp = np.asarray(Zm)[late_idx] - np.asarray(Zm)[early_idx]
    dt = np.asarray(Zr)[late_idx] - np.asarray(Zr)[early_idx]
    # Reuse desk_training's median_dir_cos rather than rewriting it: it is the same statistic the
    # trainer logs each epoch as `dcos`, it is unit-tested against exact known angles, and a
    # second implementation here could drift from it silently. It takes torch tensors.
    tp = torch.as_tensor(dp, dtype=torch.float64)
    tt = torch.as_tensor(dt, dtype=torch.float64)
    obs = median_dir_cos(tp, tt)
    nulls = [median_dir_cos(tp, tt[torch.as_tensor(rng.permutation(len(dt)))])
             for _ in range(int(n_perm))]
    # All-NaN is a real outcome, not an error: a model predicting zero change has no direction,
    # so every cosine is undefined. Report NaN rather than warning and returning a number.
    finite = [v for v in nulls if np.isfinite(v)]
    null = float(np.mean(finite)) if finite else float("nan")
    return {"median_dir_cos": float(obs), "null": null,
            "null_sd": float(np.std(finite)) if len(finite) > 1 else float("nan"),
            "margin_over_null": float(obs - null), "n_cells": int(len(dp))}


def _turnover(Z, early_idx, late_idx):
    """``1 - cos`` between each cell's early and late vector: how much it changed."""
    Z = np.asarray(Z)
    a, b = Z[early_idx], Z[late_idx]
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.full(len(a), np.nan)
    ok = den > 1e-12
    out[ok] = 1.0 - np.sum(a[ok] * b[ok], axis=1) / den[ok]
    return out


def temporal_rank(Zm, Zr, early_idx, late_idx, cells=None):
    """Comparison 2. Are the cells that changed most the same cells?

    Reduces each cell's change to one number, then compares the ORDERINGS. Rank only, so the
    reference's magnitudes are discarded entirely -- which is the right call given the caps, and
    is what ``validate_spacetime.temporal_turnover_agreement`` already warns about for the
    equivalent observed-space comparison.

    Also reports a partial correlation that removes the cell's row and column, because BBS
    coverage is strongly East-heavy and both fields inherit that geography; a raw correlation
    can be carried by "east cells differ from west cells" without any agreement about change.
    """
    from scipy.stats import spearmanr

    tm, tr = _turnover(Zm, early_idx, late_idx), _turnover(Zr, early_idx, late_idx)
    ok = np.isfinite(tm) & np.isfinite(tr)
    out = {"n_cells": int(ok.sum())}
    if ok.sum() < 3:
        return {**out, "spearman": None, "spearman_partial_rowcol": None}
    out["spearman"] = float(spearmanr(tm[ok], tr[ok]).correlation)
    if cells is not None:
        rc = np.asarray([[c[0], c[1]] for c in cells], dtype="float64")[ok]
        try:
            out["spearman_partial_rowcol"] = float(partial_spearman(tm[ok], tr[ok], rc))
        except Exception:
            out["spearman_partial_rowcol"] = None
    return out


def fit_species_decoder(Z, X, ridge=1.0):
    """Least-squares map from the embedding back to per-species abundance (pure).

    Returns ``(W, r2)`` with ``W`` of shape ``(L+1, S)`` (a bias row appended) and ``r2`` the
    per-species coefficient of determination on the fitting rows.

    This exists only for comparison 3, and it is the weakest link in this module: the ESK gives
    a forward map from abundance to Z and no inverse, so reading one species out of Z requires
    fitting one. ``r2`` is reported per species precisely so a poor decoder invalidates the
    comparison for that species rather than being mistaken for a model failure.
    """
    Z = np.asarray(Z, "float64"); X = np.asarray(X, "float64")
    A = np.hstack([Z, np.ones((Z.shape[0], 1))])
    G = A.T @ A + float(ridge) * np.eye(A.shape[1])
    W = np.linalg.solve(G, A.T @ X)
    resid = X - A @ W
    var = X.var(axis=0)
    r2 = np.where(var > 1e-12, 1.0 - resid.var(axis=0) / np.maximum(var, 1e-12), np.nan)
    return W, r2


def decode_species(Z, W):
    """Apply a fitted decoder: ``(N, L) -> (N, S)``."""
    Z = np.asarray(Z, "float64")
    return np.hstack([Z, np.ones((Z.shape[0], 1))]) @ np.asarray(W, "float64")


def species_sign_agreement(dX_model, ppy_ref, min_abs_ppy=0.0, decoder_r2=None,
                           min_r2=None):
    """Comparison 3. Does the long-run direction match the published percent-per-year sign?

    ``dX_model`` is the decoded change per (cell, species); ``ppy_ref`` the published rate for
    the same entries. Both may hold NaN where a product is absent. Compares SIGNS only -- the
    single thing a percent-per-year rate states without any of the reconstruction on top.

    ``min_abs_ppy`` drops entries whose published rate is nearly zero: there the sign is a
    coin-flip in the product itself, so counting them pulls the statistic toward 0.5 regardless
    of the model. ``min_r2`` drops species the decoder cannot reconstruct, since for those the
    comparison measures the decoder rather than the encoder.
    """
    dX = np.asarray(dX_model, "float64"); pp = np.asarray(ppy_ref, "float64")
    ok = np.isfinite(dX) & np.isfinite(pp) & (np.abs(pp) >= float(min_abs_ppy))
    if min_r2 is not None and decoder_r2 is not None:
        keep_sp = np.asarray(decoder_r2, "float64") >= float(min_r2)
        ok &= keep_sp.reshape(1, -1)
    n = int(ok.sum())
    if n == 0:
        return {"frac_same_sign": None, "n": 0, "n_species_used": 0}
    agree = np.sign(dX[ok]) == np.sign(pp[ok])
    per_sp = {}
    for s in range(dX.shape[1]):
        m = ok[:, s]
        if m.any():
            per_sp[s] = float((np.sign(dX[m, s]) == np.sign(pp[m, s])).mean())
    return {"frac_same_sign": float(agree.mean()), "null": 0.5, "n": n,
            "n_species_used": len(per_sp),
            "median_per_species": float(np.median(list(per_sp.values()))) if per_sp else None,
            "per_species": per_sp}


def similarity_agreement(X_ref, Zm, Z_null=None):
    """Comparisons 4 and 5, depending on which rows the caller passes in.

    Grades the model's dot-product Gram against the reference's Ruzicka similarity. The dot
    product, not cosine, because that is what the kernel contract and the training loss are both
    stated in -- cosine would discard the norms, which are part of the prediction.

    ``Z_null`` is a no-change baseline (each cell's modern embedding copied to every year). Skill
    is measured against it rather than against zero, because a model that simply reproduced the
    static spatial pattern would otherwise score well.

    Pass all rows for comparison 4; pass one year's rows for comparison 5.
    """
    S_true = ruzicka_rect(X_ref, X_ref)
    S_pred = dot_gram(Zm)
    out = {"n_rows": int(len(Zm)),
           "kernel_error": kernel_error(S_true, S_pred),
           "cka": float(linear_cka(S_true, S_pred)),
           "mantel_r": float(mantel_r(S_true, S_pred)),
           "median_offdiag_reference": float(np.median(_offdiag(S_true)))}
    if Z_null is not None:
        S_nc = dot_gram(Z_null)
        e_d = out["kernel_error"]["rmse"]
        e_n = kernel_error(S_true, S_nc)["rmse"]
        out["null"] = {"rmse": e_n,
                       "cka": float(linear_cka(S_true, S_nc)),
                       "mantel_r": float(mantel_r(S_true, S_nc))}
        out["rmse_skill"] = float(1.0 - e_d / e_n) if e_n > 1e-12 else None
        out["cka_gain"] = float(out["cka"] - out["null"]["cka"])
        out["mantel_gain"] = float(out["mantel_r"] - out["null"]["mantel_r"])
    return out


# ----------------------------- orchestration -----------------------------

def build_reference_points(config=None):
    """Build the reference point set: the trend products WITHOUT our spatial blur.

    Same builder as the retired target (``trend_community.build_trend_points``) with two
    overrides from the ``trend_reference`` config block:

    - ``smooth_sigma_cells = 0``. The blur exists to stabilise a training target, and there is
      no target to stabilise any more. Leaving it on would mean comparison 5 grades the model's
      spatial structure against our own smoothing of the products rather than against the
      products, so a poor score there would be unattributable.
    - a separate ``points_dir``, so the reference and the training target coexist and the A/B
      against the old target stays runnable.

    The soft caps stay ON. Their stabilising purpose is gone, but their outlier-control purpose
    is not: a handful of 100x-fold extrapolations would dominate an RMSE or a rank correlation,
    and comparisons 2 and 4 are exactly those.
    """
    import copy

    from src.config_utils import load_config
    from .trend_community import build_trend_points

    config = load_config(config) if not isinstance(config, dict) else config
    rcfg = config.get("trend_reference", {}) or {}
    cfg = copy.deepcopy(config)
    cfg["trend"]["smooth_sigma_cells"] = float(rcfg.get("smooth_sigma_cells", 0.0))
    if rcfg.get("points_dir"):
        # Both keys: this cfg copy describes the REFERENCE point set, and target.points_dir
        # is what config_utils.target_points_dir prefers. Setting only trend.points_dir would
        # leave any resolver-based reader pointed at the training target instead.
        cfg["trend"]["points_dir"] = rcfg["points_dir"]
        cfg.setdefault("target", {})["points_dir"] = rcfg["points_dir"]
    print(f"[ref-build] trend products WITHOUT the spatial blur "
          f"(smooth_sigma_cells={cfg['trend']['smooth_sigma_cells']}) "
          f"-> {cfg['trend']['points_dir']}", flush=True)
    return build_trend_points(cfg)


def load_reference(points_dir):
    """The reference point set: ``(X_ref, keys)``, log1p community vectors per (cell, year).

    Built by ``trend_community.build_trend_points``. Which point set is used matters: the
    reference should be built with ``smooth_sigma_cells = 0`` so comparison 5 grades against the
    products' own spatial structure rather than against our blur of it. The soft caps stay on,
    because a handful of 100x-fold extrapolations would otherwise dominate an RMSE or a rank.
    """
    import os
    X = np.load(os.path.join(points_dir, "X_points.npy"))
    keys = np.load(os.path.join(points_dir, "point_index.npy"))
    return X, keys


def published_rates(codes):
    """Published percent-per-year per (species, cell) for comparison 3: ``(bbs, ebird)``.

    Straight from the rasters, with none of the reconstruction on top -- the sign of these rates
    is the single thing the products state most directly, which is why comparison 3 uses it
    instead of going through the reference embedding.
    """
    from src.config_utils import load_data_config
    from .trend_community import _load_trend_grid

    dcfg = load_data_config()
    bbs_rate, _ = _load_trend_grid(dcfg["trends"]["bbs_trend_grid"], codes, "rate")
    eb_ppy, _ = _load_trend_grid(dcfg["trends"]["ebird_trend_grid"], codes, "abd_ppy")
    return bbs_rate, eb_ppy


def run_panel(config=None, n_sample=3000, min_gap=5, seed=0, out_dir=None,
              spatial_year=None, verbose=True):
    """Run all five comparisons off ONE encode pass and write the results.

    One pass matters: ``desk_z_ema`` runs a causal scan, so it encodes the whole grid once per
    year over the EMA window (~86 forwards). Each comparison re-encoding would multiply that.
    """
    import json
    import os

    from src.config_utils import load_config
    from .validate_bbs_routes import desk_z_ema, stratified_sample

    config = config or load_config()
    ref_dir = (config.get("trend_reference", {}) or {}).get("points_dir") \
        or config["trend"]["points_dir"]
    out_dir = out_dir or config["paths"]["desk_output_dir"]
    X_ref, keys = load_reference(ref_dir)
    meta = json.load(open(os.path.join(ref_dir, "points_meta.json")))
    codes = [str(c) for c in meta["species"]]
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"[ref-panel] reference: {X_ref.shape[0]:,} rows x {len(codes)} species "
              f"from {ref_dir}", flush=True)
        sig = (meta.get("handoff") or {}).get("smooth_sigma_cells")
        if sig:
            print(f"[ref-panel] NOTE this reference was built with smooth_sigma_cells={sig}. "
                  f"Comparison 5 (pure spatial) grades the axis that blur acts on, so a poor "
                  f"score there is partly the reference. Rebuild with 0 to separate them.",
                  flush=True)

    # Temporal comparisons use every cell with a wide enough span; the similarity comparisons are
    # n^2 in memory, so they take a stratified sample.
    ei, li, cells = cell_epoch_rows(keys, min_gap=min_gap)
    want = np.unique(np.concatenate([ei, li]))
    samp = stratified_sample(keys, int(n_sample), rng)
    need = np.unique(np.concatenate([want, samp]))

    z_sub = desk_z_ema(config, keys[need])
    L = z_sub.shape[1]
    Zm = np.full((keys.shape[0], L), np.nan, dtype="float64")
    Zm[need] = z_sub

    # The reference embedding, in the SAME basis, so the two are comparable at all.
    from .esk_kernel import project_points_to_z
    Zr = project_points_to_z(X_ref, config["desk"]["z_dir"], L)

    ok = np.isfinite(Zm).all(axis=1)
    ei_ok = np.array([i for i, (a, b) in enumerate(zip(ei, li)) if ok[a] and ok[b]])
    ei2, li2 = ei[ei_ok], li[ei_ok]
    cells2 = [cells[i] for i in ei_ok]

    # No-change null: each cell's latest embedding copied to every one of its rows.
    latest = {}
    for i, (r, c, y) in enumerate(keys):
        k = (int(r), int(c))
        if ok[i] and (k not in latest or int(y) > latest[k][0]):
            latest[k] = (int(y), i)
    Z_null = np.full_like(Zm, np.nan)
    for i, (r, c, _y) in enumerate(keys):
        src = latest.get((int(r), int(c)))
        if src is not None:
            Z_null[i] = Zm[src[1]]

    results = {
        "reference_points_dir": ref_dir,
        "reference_smooth_sigma_cells": (meta.get("handoff") or {}).get("smooth_sigma_cells"),
        "n_reference_rows": int(X_ref.shape[0]),
        "min_gap": int(min_gap), "n_sample": int(n_sample), "seed": int(seed),
        "temporal_direction": temporal_direction(Zm, Zr, ei2, li2, rng),
        "temporal_rank": temporal_rank(Zm, Zr, ei2, li2, cells2),
    }

    # Comparison 3, via the decoder. Fitted on the sampled rows the model could encode.
    fit_rows = samp[ok[samp]]
    W, r2 = fit_species_decoder(Zm[fit_rows], X_ref[fit_rows])
    dX = decode_species(Zm[li2], W) - decode_species(Zm[ei2], W)
    bbs_rate, eb_ppy = published_rates(codes)
    rows = np.array([[c[0], c[1]] for c in cells2], dtype=int)
    ref_bbs = bbs_rate[:, rows[:, 0], rows[:, 1]].T if len(rows) else np.zeros((0, len(codes)))
    ref_eb = eb_ppy[:, rows[:, 0], rows[:, 1]].T if len(rows) else np.zeros((0, len(codes)))
    results["decoder_r2"] = {"median": float(np.nanmedian(r2)),
                             "n_species_above_0.2": int(np.nansum(r2 >= 0.2))}
    results["species_sign_bbs"] = species_sign_agreement(
        dX, ref_bbs, min_abs_ppy=0.5, decoder_r2=r2, min_r2=0.2)
    results["species_sign_ebird"] = species_sign_agreement(
        dX, ref_eb, min_abs_ppy=0.5, decoder_r2=r2, min_r2=0.2)

    # Comparisons 4 and 5.
    s_ok = samp[ok[samp]]
    results["similarity_spacetime"] = similarity_agreement(X_ref[s_ok], Zm[s_ok], Z_null[s_ok])
    yr = int(spatial_year) if spatial_year else int(np.median(keys[s_ok, 2]))
    in_year = s_ok[keys[s_ok, 2] == yr]
    if in_year.size >= 20:
        results["similarity_spatial"] = {
            "year": yr,
            **similarity_agreement(X_ref[in_year], Zm[in_year], Z_null[in_year])}
    else:
        results["similarity_spatial"] = {"year": yr, "skipped": "fewer than 20 rows"}

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "trend_reference_panel.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=str)
    if verbose:
        print_panel(results)
        print(f"[ref-panel] wrote -> {path}", flush=True)
    return results


def print_panel(r):
    """One readable block, with each comparison's null next to it."""
    d, k = r["temporal_direction"], r["temporal_rank"]
    print("[ref-panel] 1 temporal direction (clean): "
          f"dir-cos {d['median_dir_cos']:.3f} vs null {d['null']:.3f} "
          f"(margin {d['margin_over_null']:+.3f}, {d['n_cells']} cells)")
    sp = k.get("spearman")
    pp = k.get("spearman_partial_rowcol")
    print("[ref-panel] 2 temporal rank (clean):      "
          f"spearman {sp if sp is None else f'{sp:.3f}'}"
          + (f", partial(row,col) {pp:.3f}" if pp is not None else "")
          + f" ({k['n_cells']} cells)")
    for tag, key in (("BBS", "species_sign_bbs"), ("eBird", "species_sign_ebird")):
        s = r[key]
        v = s.get("frac_same_sign")
        print(f"[ref-panel] 3 species trend sign vs {tag:<5}: "
              f"{'n/a' if v is None else f'{v:.3f}'} vs null 0.500 "
              f"({s['n']:,} cell-species, {s['n_species_used']} species)")
    print(f"[ref-panel]   decoder r2 median {r['decoder_r2']['median']:.3f}, "
          f"{r['decoder_r2']['n_species_above_0.2']} species above 0.2 "
          f"-- comparison 3 means nothing where this is low")
    for tag, key in (("4 full spacetime  ", "similarity_spacetime"),
                     ("5 spatial (dirty) ", "similarity_spatial")):
        s = r[key]
        if s.get("skipped"):
            print(f"[ref-panel] {tag}: skipped ({s['skipped']})")
            continue
        sk = s.get("rmse_skill")
        print(f"[ref-panel] {tag}: rmse_skill {'n/a' if sk is None else f'{sk:+.3f}'}, "
              f"cka_gain {s.get('cka_gain', float('nan')):+.3f}, "
              f"reference median off-diagonal similarity {s['median_offdiag_reference']:.3f}")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sample", type=int, default=3000)
    ap.add_argument("--min-gap", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spatial-year", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    run_panel(n_sample=args.n_sample, min_gap=args.min_gap, seed=args.seed,
              out_dir=args.out_dir, spatial_year=args.spatial_year)


if __name__ == "__main__":
    main()
