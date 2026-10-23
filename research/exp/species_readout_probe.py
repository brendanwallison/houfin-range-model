"""E059c: is DESK's out-of-community species readout held back by its REGULARIZATION, its RANK, or its annual TARGET?

    python research/exp/species_readout_probe.py --cache <dir> [--esk-dir <dir>] --out <dir>

The species change estimand of fair_rivals --target dev: held-out cells, early withheld epoch vs 2005-25, the 273 dev
species, log1p of ABBA half-epoch means; pooled noise-free centred corr and k. DESK's within-place readout maps z
deviations to species deviations on training cells' trained years. Here the readout is a closed-form ridge
(all species at once through one eigendecomposition), varied along three axes:
    lambda   shrinkage strength, a grid of 1e-4..1e4 x trace(X'X)/p  (is the BLR's marginal-likelihood choice far
             from the best? fair_rivals' BLR result is the reference)
    rank     24 (what the age model reads) vs 64 (all of DESK)
    target   annual deviations (as fair_rivals) vs 5-year window means of the deviations (closer to decadal change)
E059d (the user: amplitude and lengthscale trade off; Ruzicka fixes the kernel's shape, so ONE amplitude cannot change
its smoothness, but RELATIVE amplitudes across z dimensions can): the prior is reshaped with --alphas. Each z
dimension's prior variance contribution is var_d^alpha (features scaled by sd_d^(alpha - 1)):
    alpha 1   the Ruzicka kernel z.z' (iid prior on raw z, what the age model does)
    alpha 0   whitened z (every eigen-direction equal; fine-scale turnover up-weighted, a shorter effective lengthscale)
    alpha <0  trailing dimensions favoured further
plus a per-dimension ARD readout (lib.blr with one block per dimension, amplitudes by marginal likelihood per species).
Also an IN-SAMPLE ORACLE: the same ridge fitted directly on the held-out cells' observed change, 5-fold over spatial
blocks. That is an upper bound on how much species change DESK's change vector carries linearly, whatever the readout
learns on trained years. Caches are only read.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from rrr_backcast import wc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--esk-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    ap.add_argument("--targets", nargs="+", default=["annual", "window5"])
    ap.add_argument("--ard", action="store_true")
    ap.add_argument("--level", action="store_true",
                    help="E059e: PRODUCTION-form readouts fitted on LEVELS (H = z.beta pooled over places and years): "
                         "iid prior (Ruzicka) vs level-variance reweighting (alphas) vs per-dimension ARD vs the split "
                         "kernel (between-place z mean and within-place deviation, each its own amplitude)")
    a = ap.parse_args()
    from scipy import sparse
    from change_oracle import desk_epoch_z, group_rows
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
    from src.data.preprocess.bbs_community import log1p_community
    c = lambda f: os.path.join(a.cache, f)
    ed = a.esk_dir or a.cache
    keys = np.load(c("keys.npy"))
    N = len(keys)
    split = np.load(c("split.npz"))
    withheld = [int(y) for y in split["withheld"]]
    E, full, half_a, half_b = group_rows(a.cache, keys, withheld, epochs_file=os.path.join(ed, "esk_epochs.npz"))
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh):
            idx.setdefault(cell, [None, None])[0] = j
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = j
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    cells = E["cells"][pairs[:, 0]]
    ev = np.where(split["holdout"][cells[:, 0], cells[:, 1]])[0]
    Xdev = np.asarray(np.load(c("X_dev.npy")), "float64")
    je, jm = pairs[ev, 0], pairs[ev, 1]
    Ta = epoch_mean_observed(Xdev, [half_a[j] for j in jm]) - epoch_mean_observed(Xdev, [half_a[j] for j in je])
    Tb = epoch_mean_observed(Xdev, [half_b[j] for j in jm]) - epoch_mean_observed(Xdev, [half_b[j] for j in je])
    early_rows = [np.asarray(full[pairs[j, 0]]) for j in ev]
    mod_rows = [np.asarray(full[pairs[j, 1]]) for j in ev]
    zr_all = np.load(c("z_raw_all.npy"), mmap_mode="r")
    ci_, y0 = np.load(c("key_cell_index.npy")), int(np.load(c("years.npy"))[0])
    Y = np.asarray(log1p_community(Xdev), "float64")
    ho = split["holdout"][keys[:, 0], keys[:, 1]]
    bf = split["buffer"][keys[:, 0], keys[:, 1]]
    ty = ~np.isin(keys[:, 2], withheld)
    F_raw = np.asarray(np.load(c("F_cov_raw.npy"), mmap_mode="r"), "float64")
    trn = np.where(~ho & ~bf & ty & np.isfinite(F_raw).all(1))[0]
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    uc, inv = np.unique(cid, return_inverse=True)
    w_ty = ty.astype(float)

    def cmean(X, w, groups=inv, ng=len(uc)):
        S = sparse.csr_matrix((w, (groups, np.arange(N))), shape=(ng, N))
        cnt = np.asarray(S.sum(1)).ravel()
        return (np.asarray(S @ X) / np.maximum(cnt, 1e-12)[:, None])[groups]
    # 5-year windows inside each cell (trained years): window id = cell x floor((year - 1966) / 5)
    wid = cid * 100 + (keys[:, 2] - 1966) // 5
    uw, winv = np.unique(wid, return_inverse=True)
    Yd = Y - cmean(Y, w_ty)
    Yd_w = cmean(Yd, w_ty, winv, len(uw))                         # 5-year window mean of the deviation
    blk = (cells[ev, 0] // 6) * 100000 + cells[ev, 1] // 6
    ub, binv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(11)
    W = np.vstack([np.ones(len(ev)), rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)[:, binv]])

    def score(P):
        out = []
        for w in W:
            vp, vt = wc(P, P, w), wc(Ta, Tb, w)
            cv = 0.5 * (wc(P, Ta, w) + wc(P, Tb, w))
            out.append((cv / np.sqrt(vp * vt), cv / vp))
        out = np.array(out)
        return {"corr": float(out[0, 0]), "k": float(out[0, 1]),
                "ci": [float(np.nanpercentile(out[1:, 0], q)) for q in (2.5, 97.5)]}
    res = {"cache": a.cache, "n_cells": int(len(ev)), "n_train_rows": int(len(trn)), "grid": {}}
    lams = [10.0 ** e for e in range(-4, 5)]
    for r in (24, 64):
        zk = np.asarray(np.load(c("z_raw.npy"), mmap_mode="r")[:, :r], "float64")
        Zdev0 = zk - cmean(zk, w_ty)
        dz0 = desk_epoch_z(zr_all, ci_, y0, keys, mod_rows, r) - desk_epoch_z(zr_all, ci_, y0, keys, early_rows, r)
        sd = Zdev0[trn].std(0)
        res.setdefault("dim_sd", {})[f"r{r}"] = sd.tolist()
        for alpha in a.alphas:
            sc = sd ** (alpha - 1.0)                                    # prior contribution per dim ~ var_d^alpha
            Zdev, dz = Zdev0 * sc, dz0 * sc
            Xt = Zdev[trn] - Zdev[trn].mean(0)
            evals, V = np.linalg.eigh(Xt.T @ Xt)
            base = np.trace(Xt.T @ Xt) / r
            tg = {"annual": Yd, "window5": Yd_w}
            for tname in a.targets:
                T = tg[tname]
                XtY = Xt.T @ (T[trn] - T[trn].mean(0))
                VtXtY = V.T @ XtY
                for lam in lams:
                    B = V @ (VtXtY / (evals + lam * base)[:, None])      # (r, S), every species at once
                    tag = f"r{r}_{tname}_lam{lam:g}" if alpha == 1.0 else f"r{r}_a{alpha:g}_{tname}_lam{lam:g}"
                    res["grid"][tag] = score(dz @ B)
        if a.ard:
            from lib import blr
            m = blr.fit(Zdev0[trn], Yd[trn], [(d, d + 1) for d in range(r)])
            res["grid"][f"r{r}_ARD_blr"] = score(dz0 @ m["coef"].T)
            mi = blr.fit(Zdev0[trn], Yd[trn], [(0, r)])
            res["grid"][f"r{r}_iid_blr"] = score(dz0 @ mi["coef"].T)
            res.setdefault("ard_median_log_amp", {})[f"r{r}"] = np.median(np.log(m["a"]), 0).tolist()
        dz = dz0
        # in-sample oracle on the held-out change itself, 5-fold by spatial block (upper bound)
        Tm = 0.5 * (Ta + Tb)
        fold = np.random.default_rng(3).integers(5, size=len(ub))[binv]
        P = np.zeros_like(Tm)
        for f in range(5):
            tr, te = fold != f, fold == f
            Xc = dz[tr] - dz[tr].mean(0)
            lam = 1e-1 * np.trace(Xc.T @ Xc) / r
            Bo = np.linalg.solve(Xc.T @ Xc + lam * np.eye(r), Xc.T @ (Tm[tr] - Tm[tr].mean(0)))
            P[te] = (dz[te] - dz[tr].mean(0)) @ Bo
        res["grid"][f"r{r}_ORACLE_cv5"] = score(P)
    if a.level:
        from lib import blr
        Yc_tr = Y[trn] - Y[trn].mean(0)
        for r in (24, 64):
            zk = np.asarray(np.load(c("z_raw.npy"), mmap_mode="r")[:, :r], "float64")
            dz0 = desk_epoch_z(zr_all, ci_, y0, keys, mod_rows, r) - desk_epoch_z(zr_all, ci_, y0, keys, early_rows, r)
            sdl = zk[trn].std(0)
            res.setdefault("level_dim_sd", {})[f"r{r}"] = sdl.tolist()
            for alpha in a.alphas:
                sc = sdl ** (alpha - 1.0)
                Xt = zk[trn] * sc
                Xt = Xt - Xt.mean(0)
                evals, V = np.linalg.eigh(Xt.T @ Xt)
                base = np.trace(Xt.T @ Xt) / r
                VtXtY = V.T @ (Xt.T @ Yc_tr)
                for lam in lams:
                    B = V @ (VtXtY / (evals + lam * base)[:, None])
                    res["grid"][f"LEVEL_r{r}_a{alpha:g}_lam{lam:g}"] = score((dz0 * sc) @ B)
            m = blr.fit(zk[trn], Y[trn], [(0, r)])
            res["grid"][f"LEVEL_r{r}_iid_blr (production form)"] = score(dz0 @ m["coef"].T)
            m = blr.fit(zk[trn], Y[trn], [(d, d + 1) for d in range(r)])
            res["grid"][f"LEVEL_r{r}_ARD_blr"] = score(dz0 @ m["coef"].T)
            res.setdefault("level_ard_median_log_amp", {})[f"r{r}"] = np.median(np.log(m["a"]), 0).tolist()
            zbar = cmean(zk, w_ty)
            m = blr.fit(np.hstack([zbar, zk - zbar])[trn], Y[trn], [(0, r), (r, 2 * r)])
            res["grid"][f"LEVEL_r{r}_split_blr"] = score(dz0 @ m["coef"][:, r:].T)
            res.setdefault("split_median_log_amp_between_within", {})[f"r{r}"] = np.median(np.log(m["a"]), 0).tolist()
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "species_readout_probe.json"), "w"), indent=1)
    print(f"[species-readout-probe] {os.path.basename(a.cache)}: {len(ev)} held-out cells, 273 dev species")
    for k_, v in res["grid"].items():
        print(f"  {k_:28s} corr {v['corr']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] k {v['k']:.2f}", flush=True)


if __name__ == "__main__":
    main()
