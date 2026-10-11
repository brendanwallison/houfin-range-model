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
Criteria (E037.md): fit health, prediction sanity, leave-one-out level skill on training rows, held-out level skill vs
a linear (BLR) covariate baseline on the same features.
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
    ap.add_argument("--sample", default="blocks", choices=("blocks", "strat"))
    ap.add_argument("--n-fit", type=int, default=3000)
    ap.add_argument("--cap-s2", type=float, default=None)
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
    # --- the pieces under test, patched into the suite's own routines ---------------------------------------------
    if a.bounds == "data":
        sdx = X[tr].std(0) + 1e-12

        def bounds_data(kind, Fb, n_theta):
            return [(float(np.log(a.bound_lo * s)), float(np.log(a.bound_hi * s))) for s in sdx[:n_theta]]
        gk.theta_bounds = bounds_data
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
    else:
        per = 20
        shape_blocks = []
        for i in rng.permutation(len(blocks))[: max(1, a.n_fit // per)]:
            b = blocks[i]
            shape_blocks.append(np.sort(rng.choice(b, min(per, len(b)), replace=False)))
    th0 = np.full(d, np.log(np.sqrt(d)))
    t0 = time.perf_counter()
    shape = gk.fit_shared_shape("covariate", [X[b] for b in shape_blocks], [Y[b] for b in shape_blocks], th0,
                                n_iter=a.shape_iters, verbose=True, ybar=ybar, var_y=var_y)
    t_shape = time.perf_counter() - t0
    sc = gk.blocked_scales("covariate", shape["theta"], [X[b] for b in blocks], [Y[b] for b in blocks], ybar, var_y)
    shape.update({"s2": sc["s2"], "n2": sc["n2"], "ok": sc["ok"], "ybar": ybar})
    # --- evaluation rows -----------------------------------------------------------------------------------------------
    te = np.sort(rng.choice(te_all, min(a.n_eval, len(te_all)), replace=False))
    loo = np.sort(rng.choice(tr, min(a.n_eval, len(tr)), replace=False))
    grp = lambda rows: ((keys[rows, 0] // 6) * 1000 + keys[rows, 1] // 6) * 10 + (keys[rows, 2] - 1960) // 10
    mte, vte = gk.predict_local(shape, X[tr], Ytr, X[te], grp(te), k=a.k, k_max=a.k_max)
    self_idx = np.array([pos[r_] for r_ in loo])
    mlo, vlo = gk.predict_local(shape, X[tr], Ytr, X[loo], grp(loo), k=a.k, k_max=a.k_max, self_idx=self_idx)
    # linear covariate baseline on the same features (one BLR block)
    bl = blr.fit(X[tr], Ytr, [(0, d)])
    lin = (X[te] - bl["xbar"]) @ bl["coef"].T + bl["ybar"]
    ymax = Ytr.max(0)
    out_range = float(((mte < -1.0) | (mte > ymax[None, :] + 1.0)).mean())
    vratio = np.nanmax(vte, axis=0) / np.maximum(var_y, 1e-12)
    lo_b = np.array([b[0] for b in gk.theta_bounds("covariate", np.concatenate([X[b] for b in shape_blocks]), d)])
    hi_b = np.array([b[1] for b in gk.theta_bounds("covariate", np.concatenate([X[b] for b in shape_blocks]), d)])
    th = np.asarray(shape["theta"])
    res = {"variant": vars(a), "n_species": int(sp.sum()), "n_train": int(len(tr)), "dims": int(d),
           "shape_seconds": round(t_shape, 1), "converged": shape["converged"], "message": shape["message"],
           "n_iterations": shape["n_iterations"],
           "frac_at_bound": float(np.mean(np.isclose(th, lo_b, atol=1e-6) | np.isclose(th, hi_b, atol=1e-6))),
           "lengthscale_quantiles": [float(np.exp(np.percentile(th, q))) for q in (0, 10, 50, 90, 100)],
           "s2_over_var_median": float(np.median(shape["s2"] / np.maximum(var_y, 1e-12))),
           "pred_out_of_range_frac": out_range, "pred_var_ratio_median": float(np.nanmedian(vratio)),
           "loo_level_skill_median": float(np.nanmedian(skill(Y[loo], mlo, ybar))),
           "heldout_level_skill_median": float(np.nanmedian(skill(Y[te], mte, ybar))),
           "heldout_linear_skill_median": float(np.nanmedian(skill(Y[te], lin, ybar))),
           "heldout_gp_minus_linear_median": float(np.nanmedian(skill(Y[te], mte, ybar) - skill(Y[te], lin, ybar)))}
    tag = a.tag or f"{a.features}_{a.bounds}_{a.sample}_cap{a.cap_s2}"
    with open(os.path.join(a.out, f"covgp_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    for f in sorted(os.listdir(out)):
        if not (f.startswith("covgp_") and f.endswith(".json")):
            continue
        r = json.load(open(os.path.join(out, f), encoding="utf-8"))
        print(f"{f[6:-5]:32s} dims {r['dims']:3d} | {'converged' if r['converged'] else 'NOT conv'} it {r['n_iterations']:3d}"
              f" at-bound {r['frac_at_bound']:.2f} ls q0/50/100 {r['lengthscale_quantiles'][0]:.3g}/"
              f"{r['lengthscale_quantiles'][2]:.3g}/{r['lengthscale_quantiles'][4]:.3g} s2/var {r['s2_over_var_median']:.2f}"
              f" | out-of-range {r['pred_out_of_range_frac']:.4f} var-ratio {r['pred_var_ratio_median']:.2f}"
              f" | LOO skill {r['loo_level_skill_median']:+.3f} held-out {r['heldout_level_skill_median']:+.3f}"
              f" (linear {r['heldout_linear_skill_median']:+.3f}; GP-linear {r['heldout_gp_minus_linear_median']:+.3f})")


if __name__ == "__main__":
    main()
