"""E037: repair the covariate GP baseline (defect A13) -- a fast harness that reproduces the suite's covariate arm and
tests one fix at a time against acceptance criteria fixed in advance (research/experiments/E037.md).

    ESK_DESK_CONFIG=config/overlays/gp_species_base.json python research/exp/covgp_harness.py --out <dir> [variant flags]
    python research/exp/covgp_harness.py --summarize <dir>

Data: the gp_species_base research cache (DESK-normalized covariates F_cov at every key, dev-species counts, split).
Species: dev species seen in >= 1% of training rows. The shape (lengthscales) is fitted with the suite's own routine
(gp_kernels.fit_shared_shape), per-species scales with blocked_scales on all training blocks, predictions with
predict_local (exact leave-one-out for in-sample rows), exactly as validate_gp_species does -- only the pieces under test
change:
  --features raw|pca     the 302 channels as the suite uses them, or the top --n-pcs principal components, whitened
  --bounds suite|data    lengthscale bounds from the shape rows' extent (the suite; the suspected bug) or from ALL training
                         rows in units of each feature's own spread: [lo, hi] x sd
  --sample blocks|strat  ~n_fit rows as a few contiguous blocks (the suite) or as many small random blocks
  --cap-s2 C             per-species signal variance capped at C x var_y (and no 1e3 start multiplier)
  --cell                 add a same-cell term to the kernel, k = RBF + rho [same cell] (rho shared, fitted with the
                         lengthscales): a place's lasting departure from what its covariates predict is modelled as its
                         own effect instead of as covariate signal; a new cell's effect is unknown (mean 0, variance
                         s2 rho)
  --linear               add a linear term lambda x.x'/d to the kernel (a global trend the local conditioning sets
                         can otherwise only approximate)
Criteria (E037.md): fit health, prediction sanity, leave-one-out level skill on training rows, held-out level skill vs
a linear (BLR) covariate baseline on the same features. Evaluation rows come from their own seed, so every variant is
scored on the same rows.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
from lib import blr  # noqa: E402

CACHE = os.path.expanduser("~/houfin/work/houfin/research_cache/gp_species_base")


def skill(y, p, ybar):
    """Per-species 1 - SSE / SSE(training mean)."""
    sse = ((y - p) ** 2).sum(0)
    sse0 = ((y - ybar) ** 2).sum(0)
    return np.where(sse0 > 0, 1.0 - sse / np.maximum(sse0, 1e-12), np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--features", default="raw", choices=("raw", "pca"))
    ap.add_argument("--n-pcs", type=int, default=32)
    ap.add_argument("--bounds", default="suite", choices=("suite", "data"))
    ap.add_argument("--bound-lo", type=float, default=0.05)
    ap.add_argument("--bound-hi", type=float, default=20.0)
    ap.add_argument("--sample", default="blocks", choices=("blocks", "strat", "random", "randcells"))
    ap.add_argument("--scale-sample", default="blocks", choices=("blocks", "random", "randcells"))
    ap.add_argument("--years-per-cell", type=int, default=4)
    ap.add_argument("--resid-linear", action="store_true")
    ap.add_argument("--n-fit", type=int, default=3000)
    ap.add_argument("--cap-s2", type=float, default=None)
    ap.add_argument("--cell", action="store_true")
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--shape-iters", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=3000)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--k-max", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from src.community_encoder.train_DESK import gp_kernels as gk
    from src.community_encoder.train_DESK.validate_gp_species import spatial_blocks
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    keys = np.load(os.path.join(CACHE, "keys.npy"))
    split = np.load(os.path.join(CACHE, "split.npz"))
    F = np.asarray(np.load(os.path.join(CACHE, "F_cov.npy"), mmap_mode="r"), "float64")
    Xd = np.load(os.path.join(CACHE, "X_dev.npy")).astype("float64")
    held = split["holdout"][keys[:, 0], keys[:, 1]]
    trc = ~held & ~split["buffer"][keys[:, 0], keys[:, 1]]
    fin = np.isfinite(F).all(1)
    tr = np.where(trc & fin)[0]
    te_all = np.where(held & fin)[0]
    Y = np.log1p(Xd)
    prev = (Xd[tr] > 0).mean(0)
    sp = prev >= 0.01
    Y = Y[:, sp]
    Ytr = Y[tr]
    ybar = Ytr.mean(0)
    var_y = Ytr.var(0) * len(tr) / max(len(tr) - 1, 1)
    if a.features == "pca":
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        Fs = (F - mu) / sd
        sub = tr[:: max(1, len(tr) // 40000)]
        _, sv, Vt = np.linalg.svd(Fs[sub] - Fs[sub].mean(0), full_matrices=False)
        Pc = Fs @ Vt[: a.n_pcs].T
        Pc = (Pc - Pc[tr].mean(0)) / (Pc[tr].std(0) + 1e-12)                 # whitened on training rows
        X = Pc
    else:
        X = F
    d = X.shape[1]
    # linear covariate model: one BLR block on 64 whitened PCs of the covariates (a BLR on the 302 raw, collinear
    # channels fails its Cholesky -- E015). It is the fixed reference for every variant, and with --resid-linear the
    # GP's mean function: the GP then models only what the linear model leaves
    mu_, sd_ = F[tr].mean(0), F[tr].std(0) + 1e-9
    Fs_ = (F - mu_) / sd_
    sub_ = tr[:: max(1, len(tr) // 40000)]
    _, _, Vt_ = np.linalg.svd(Fs_[sub_] - Fs_[sub_].mean(0), full_matrices=False)
    L = Fs_ @ Vt_[:64].T
    L = (L - L[tr].mean(0)) / (L[tr].std(0) + 1e-12)
    try:
        bl = blr.fit(L[tr], Ytr, [(0, 64)])
        lin_all = (L - bl["xbar"]) @ bl["coef"].T + bl["ybar"]
    except Exception as e:  # noqa: BLE001 -- the GP results must not be lost to the reference
        if a.resid_linear:
            raise
        print(f"[covgp] linear baseline failed: {e}", flush=True)
        lin_all = np.full_like(Y, np.nan)
    Yw = Y - lin_all if a.resid_linear else Y                  # what the GP models
    Ywtr = Yw[tr]
    ybar_w = Ywtr.mean(0)
    var_w = Ywtr.var(0) * len(tr) / max(len(tr) - 1, 1)
    # --- the pieces under test, patched into the suite's own routines ---------------------------------------------
    kind = "covariate"
    n_extra = int(a.cell) + int(a.linear)
    if n_extra:
        # extra kernel terms ride in theta after the d log-lengthscales; the cell id rides as the last input column
        kind = "covariate_x"
        cid = (keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]).astype("float64")
        X = np.column_stack([X, cid])

        def kern_x(A, B, theta):
            import torch
            ell = torch.exp(theta[:d])
            K = torch.exp(-0.5 * torch.cdist(A[:, :d] / ell, B[:, :d] / ell) ** 2)
            j = d
            if a.cell:
                K = K + torch.exp(theta[j]) * (A[:, d:d + 1] == B[:, d:d + 1].T).to(A.dtype)
                j += 1
            if a.linear:
                K = K + torch.exp(theta[j]) * (A[:, :d] @ B[:, :d].T) / d
            return K
        gk.KERNELS[kind] = kern_x
        orig_sc = gk.scaled_coords
        gk.scaled_coords = lambda kind_, F_, th_: (np.asarray(F_, "float64")[:, :d] / np.exp(np.asarray(th_)[:d])
                                                   if kind_ == kind else orig_sc(kind_, F_, th_))
    orig_bounds = gk.theta_bounds
    sdx = X[tr][:, :d].std(0) + 1e-12

    def bounds_any(kind_, Fb, n_theta):
        if a.bounds == "data":
            b = [(float(np.log(a.bound_lo * s)), float(np.log(a.bound_hi * s))) for s in sdx]
        else:
            b = orig_bounds("covariate", np.asarray(Fb)[:, :d], d)
        return b + [(-9.0, 5.0)] * n_extra
    gk.theta_bounds = bounds_any
    if a.cap_s2 is not None:
        gk.SCALE_MULTIPLIERS = (1.0,)
        orig_core = gk._fit_scales_core

        def capped_core(nll_parts, var_y_, kappa, live, floor, max_iter=500):
            from scipy.optimize import minimize
            kl = int(live.sum())
            lo = np.log(floor[live])
            hi_s = np.log(a.cap_s2 * np.maximum(var_y_[live], 1e-12))
            hi_n = np.log(np.maximum(var_y_[live], 1e-12) * 10.0)
            bounds = [(l, max(h, l + 1.0)) for l, h in zip(lo, hi_s)] + [(l, max(h, l + 1.0)) for l, h in zip(lo, hi_n)]

            def f(x):
                nll, gs, gn = nll_parts(x[:kl], x[kl:])
                return float(nll.sum()), np.concatenate([gs, gn])
            best_x, best_nll = None, None
            for share in gk.SCALE_STARTS:
                s2_0 = np.clip(share * var_y_[live] / max(kappa, 1e-12), floor[live], np.exp(hi_s) * 0.99)
                n2_0 = np.maximum((1.0 - share) * var_y_[live], floor[live])
                res = minimize(f, np.concatenate([np.log(s2_0), np.log(n2_0)]), jac=True, method="L-BFGS-B",
                               bounds=bounds, options={"maxiter": int(max_iter)})
                nll = nll_parts(res.x[:kl], res.x[kl:])[0]
                if best_x is None:
                    best_x, best_nll = res.x.copy(), nll
                else:
                    better = nll < best_nll - 1e-9
                    best_x[:kl][better] = res.x[:kl][better]
                    best_x[kl:][better] = res.x[kl:][better]
                    best_nll = np.where(better, nll, best_nll)
            return np.exp(best_x[:kl]), np.exp(best_x[kl:]), best_nll
        gk._fit_scales_core = capped_core
    # --- shape sample ------------------------------------------------------------------------------------------------
    blocks = spatial_blocks(keys, tr, 1000)
    pos = {r_: i for i, r_ in enumerate(tr)}
    if a.sample == "blocks":
        shape_blocks, n_sel = [], 0
        for i in rng.permutation(len(blocks)):
            shape_blocks.append(blocks[i])
            n_sel += len(blocks[i])
            if n_sel >= a.n_fit:
                break
    elif a.sample == "strat":
        per = 20
        shape_blocks = []
        for i in rng.permutation(len(blocks))[: max(1, a.n_fit // per)]:
            b = blocks[i]
            shape_blocks.append(np.sort(rng.choice(b, min(per, len(b)), replace=False)))
    elif a.sample == "random":
        # one block of rows drawn from the whole continent: the global mean is then the right centring (a contiguous
        # block sits at its own regional level, which only a flat, inflated kernel can absorb) and a cell rarely
        # appears twice, so a place's lasting quirk falls to the noise rather than to the covariate signal
        shape_blocks = [np.sort(rng.choice(tr, min(a.n_fit, len(tr)), replace=False))]
    # cells of the training rows, for the random-cell samplers: continental like "random", but a cell's rows come
    # together, so the same-cell pairs that identify a cell effect are in the fit
    cell_tr = keys[tr, 0].astype(np.int64) * 100000 + keys[tr, 1]
    o_ = np.argsort(cell_tr, kind="stable")
    ucell, st_ = np.unique(cell_tr[o_], return_index=True)
    cell_rows = np.split(tr[o_], st_[1:])
    if a.sample == "randcells":
        sel, n_sel = [], 0
        for i in rng.permutation(len(ucell)):
            rr = cell_rows[i]
            rr = rng.choice(rr, min(a.years_per_cell, len(rr)), replace=False)
            sel.append(rr)
            n_sel += len(rr)
            if n_sel >= a.n_fit:
                break
        shape_blocks = [np.sort(np.concatenate(sel))]
    if a.scale_sample == "random":
        perm = rng.permutation(tr)
        scale_blocks = [np.sort(b) for b in np.array_split(perm, max(1, len(perm) // 1000))]
    elif a.scale_sample == "randcells":
        scale_blocks, cur, n_cur = [], [], 0
        for i in rng.permutation(len(ucell)):
            cur.append(cell_rows[i])
            n_cur += len(cell_rows[i])
            if n_cur >= 1000:
                scale_blocks.append(np.sort(np.concatenate(cur)))
                cur, n_cur = [], 0
        if cur:
            scale_blocks.append(np.sort(np.concatenate(cur)))
    else:
        scale_blocks = blocks
    th0 = np.concatenate([np.full(d, np.log(np.sqrt(d))), np.zeros(n_extra)])
    t0 = time.perf_counter()
    shape = gk.fit_shared_shape(kind, [X[b] for b in shape_blocks], [Yw[b] for b in shape_blocks], th0,
                                n_iter=a.shape_iters, verbose=True, ybar=ybar_w, var_y=var_w)
    t_shape = time.perf_counter() - t0
    sc = gk.blocked_scales(kind, shape["theta"], [X[b] for b in scale_blocks], [Yw[b] for b in scale_blocks], ybar_w,
                           var_w)
    shape.update({"s2": sc["s2"], "n2": sc["n2"], "ok": sc["ok"], "ybar": ybar_w})
    rho = float(np.exp(shape["theta"][d])) if a.cell else 0.0
    lam_lin = float(np.exp(shape["theta"][d + int(a.cell)])) if a.linear else 0.0
    # --- evaluation rows (their own seed: the same rows for every variant) --------------------------------------------
    erng = np.random.default_rng(1000 + a.seed)
    te = np.sort(erng.choice(te_all, min(a.n_eval, len(te_all)), replace=False))
    loo = np.sort(erng.choice(tr, min(a.n_eval, len(tr)), replace=False))
    grp = lambda rows: ((keys[rows, 0] // 6) * 1000 + keys[rows, 1] // 6) * 10 + (keys[rows, 2] - 1960) // 10
    mte, vte = gk.predict_local(shape, X[tr], Ywtr, X[te], grp(te), k=a.k, k_max=a.k_max)
    # predict_local takes k(x, x) = 1; the extra terms add to a held-out row's prior variance (its cell effect is
    # unknown, so all of s2 rho applies)
    kd_extra = rho + lam_lin * (X[te][:, :d] ** 2).sum(1) / d
    vte = vte + shape["s2"][None, :] * kd_extra[:, None]
    self_idx = np.array([pos[r_] for r_ in loo])
    mlo, vlo = gk.predict_local(shape, X[tr], Ywtr, X[loo], grp(loo), k=a.k, k_max=a.k_max, self_idx=self_idx)
    lin = lin_all[te]
    if a.resid_linear:
        mte, mlo = mte + lin_all[te], mlo + lin_all[loo]
        vratio_den = var_w                                     # the GP's variance is of the residual
    else:
        vratio_den = var_y
    ymax = Ytr.max(0)
    out_range = float(((mte < -1.0) | (mte > ymax[None, :] + 1.0)).mean())
    vratio = np.nanmax(vte, axis=0) / np.maximum(vratio_den, 1e-12)
    bnds = gk.theta_bounds(kind, np.concatenate([X[b] for b in shape_blocks]), d + n_extra)
    lo_b, hi_b = np.array([b[0] for b in bnds]), np.array([b[1] for b in bnds])
    th_all = np.asarray(shape["theta"])
    th = th_all[:d]
    vt = Y[te].var(0)
    res = {"variant": vars(a), "n_species": int(sp.sum()), "n_train": int(len(tr)), "dims": int(d),
           "shape_seconds": round(t_shape, 1), "converged": shape["converged"], "message": shape["message"],
           "n_iterations": shape["n_iterations"],
           "frac_at_bound": float(np.mean(np.isclose(th_all, lo_b, atol=1e-6) | np.isclose(th_all, hi_b, atol=1e-6))),
           "lengthscale_quantiles": [float(np.exp(np.percentile(th, q))) for q in (0, 10, 50, 90, 100)],
           "rho_cell": rho, "lambda_linear": lam_lin,
           "s2_over_var_median": float(np.median(shape["s2"] / np.maximum(var_w, 1e-12))),
           "n2_over_var_median": float(np.median(shape["n2"] / np.maximum(var_w, 1e-12))),
           "resid_var_over_var_median": float(np.median(var_w / np.maximum(var_y, 1e-12))),
           "pred_mean_spread_ratio_median": float(np.nanmedian(mte.var(0) / np.maximum(vt, 1e-12))),
           "linear_mean_spread_ratio_median": float(np.nanmedian(lin.var(0) / np.maximum(vt, 1e-12))),
           "pred_out_of_range_frac": out_range, "pred_var_ratio_median": float(np.nanmedian(vratio)),
           "loo_level_skill_median": float(np.nanmedian(skill(Y[loo], mlo, ybar))),
           "heldout_level_skill_median": float(np.nanmedian(skill(Y[te], mte, ybar))),
           "heldout_linear_skill_median": float(np.nanmedian(skill(Y[te], lin, ybar))),
           "heldout_gp_minus_linear_median": float(np.nanmedian(skill(Y[te], mte, ybar) - skill(Y[te], lin, ybar)))}
    tag = a.tag or (f"{a.features}_{a.bounds}_{a.sample}_cap{a.cap_s2}{'_cell' * a.cell}{'_lin' * a.linear}"
                    f"_scales{a.scale_sample}{'_resid' * a.resid_linear}")
    with open(os.path.join(a.out, f"covgp_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    for f in sorted(os.listdir(out)):
        if not (f.startswith("covgp_") and f.endswith(".json")):
            continue
        r = json.load(open(os.path.join(out, f), encoding="utf-8"))
        extra = (f" rho {r['rho_cell']:.3g}" if r.get("rho_cell") else "") + \
                (f" lam {r['lambda_linear']:.3g}" if r.get("lambda_linear") else "")
        spread = (f" | mean spread GP {r['pred_mean_spread_ratio_median']:.2f} lin {r['linear_mean_spread_ratio_median']:.2f}"
                  if "pred_mean_spread_ratio_median" in r else "")
        n2 = f" n2/var {r['n2_over_var_median']:.2f}" if "n2_over_var_median" in r else ""
        print(f"{f[6:-5]:32s} dims {r['dims']:3d} | {'converged' if r['converged'] else 'NOT conv'} it {r['n_iterations']:3d}"
              f" at-bound {r['frac_at_bound']:.2f} ls q0/50/100 {r['lengthscale_quantiles'][0]:.3g}/"
              f"{r['lengthscale_quantiles'][2]:.3g}/{r['lengthscale_quantiles'][4]:.3g} s2/var {r['s2_over_var_median']:.2f}"
              f"{n2}{extra} | out-of-range {r['pred_out_of_range_frac']:.4f} var-ratio {r['pred_var_ratio_median']:.2f}"
              f"{spread} | LOO skill {r['loo_level_skill_median']:+.3f} held-out {r['heldout_level_skill_median']:+.3f}"
              f" (linear {r['heldout_linear_skill_median']:+.3f}; GP-linear {r['heldout_gp_minus_linear_median']:+.3f})")


if __name__ == "__main__":
    main()
