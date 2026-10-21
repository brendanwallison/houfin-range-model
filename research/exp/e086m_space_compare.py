"""E086m: compare the representation SPACES directly (not through species readouts), on held-out cells, at ranks k.

    python research/exp/e086m_space_compare.py --cache <tempho research cache> --out <dir> [--ks 8 16 24 32 48 64]

Representations at each held-out cell-epoch (withheld early epoch, modern epoch):
  desk     DESK raw z, mean over the epoch's surveyed years, first k dims
  rrr      reduced-rank ridge from standardized covariates (F_cov_raw) to the OBSERVED ESK coordinates (esk_annual, 64),
           fitted on training cells x trained years; rank-k directions = top right singular vectors of the fitted values
           (a linear map INTO the Mercer feature space); epoch means of covariates mapped through it
  covpca   top-k principal components of the standardized covariates (unsupervised; its own space)
  esk      the observed ESK coordinates themselves, first k dims (odd-year half): the truncation ceiling
Targets (noise kept apart: target from the EVEN-year half, the odd-year half only for the observed-ESK arm and for
noise-free denominators):
  spatial kernel fidelity   corr over random pairs of held-out cells (modern epoch) of the representation's dot product
                            with the observed Ruzicka similarity Ruz(odd_i, even_j) of log1p epoch-mean communities
  change fidelity           corr over held-out cells of the representation's squared distance between a cell's two epochs
                            with the observed kernel distance Ruz(o_e,e_e) + Ruz(o_m,e_m) - 2 Ruz(o_e,e_m)
  CKA                       linear CKA of the representation with the observed ESK-64 coordinates (even half), over
                            held-out cell-epochs (scale / rotation invariant: works for covpca's own space)
Uses corr (scale-free), so representations not calibrated to Ruzicka (covpca) are comparable.
"""
import argparse
import json
import os

import numpy as np


def ruz(A, B):
    return np.minimum(A, B).sum(-1) / np.maximum(np.maximum(A, B).sum(-1), 1e-9)


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hs = np.linalg.norm(X.T @ Y) ** 2
    return float(hs / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64]); ap.add_argument("--pairs", type=int, default=40000)
    a = ap.parse_args()
    j = lambda f: os.path.join(a.cache, f)
    keys = np.load(j("keys.npy")); E = dict(np.load(j("esk_epochs.npz"))); ptr, flat = E["rows_ptr"], E["rows_flat"]
    sp = np.load(j("split.npz")); ho, bf = sp["holdout"].astype(bool), sp["buffer"].astype(bool)
    withheld = set(int(v) for v in sp["withheld"])
    r, c, y = keys[:, 0], keys[:, 1], keys[:, 2]
    Eann = np.load(j("esk_annual.npy")).astype(np.float64)
    Lc = np.log1p(np.clip(np.load(j("X_comm.npy")).astype(np.float64), 0, None))
    F = np.asarray(np.load(j("F_cov_raw.npy"), mmap_mode="r"), np.float64)
    Zr = np.load(j("z_raw_all.npy"), mmap_mode="r"); cells = np.load(j("cells.npy")); yrs = np.load(j("years.npy"))
    cix = {(int(p), int(q)): i for i, (p, q) in enumerate(cells)}
    # held-out cell-epoch pairs
    idx = {}
    for i, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh):
            idx.setdefault(cell, [None, None])[0] = i
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = i
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    pc = E["cells"][pairs[:, 0]]; hp = pairs[ho[pc[:, 0], pc[:, 1]]]
    rows = lambda i: flat[ptr[i]:ptr[i + 1]]
    half = lambda V, i, h: (lambda rr: V[rr].mean(0) if len(rr) else np.full(V.shape[1], np.nan))(rows(i)[(y[rows(i)] % 2) == h])
    allm = lambda V, i: V[rows(i)].mean(0)
    # covariate models, fitted on training cells x trained years
    trn = np.where(~ho[r, c] & ~bf[r, c] & ~np.isin(y, list(withheld)) & np.isfinite(F).all(1) & np.isfinite(Eann).all(1))[0]
    mu, sd = F[trn].mean(0), F[trn].std(0); keep = sd > 1e-6
    X = np.where(np.isfinite(F[:, keep]), (F[:, keep] - mu[keep]) / sd[keep], 0.0)
    Xc = X[trn] - X[trn].mean(0); Yc = Eann[trn] - Eann[trn].mean(0)
    lam = 1e-2 * np.trace(Xc.T @ Xc) / Xc.shape[1]
    B = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Yc)
    _, _, Vt = np.linalg.svd(Xc @ B, full_matrices=False)                       # RRR directions in ESK space
    _, _, Pt = np.linalg.svd(Xc[:: max(1, len(Xc) // 40000)], full_matrices=False)   # covariate PCs
    xm = X[trn].mean(0)
    def desk_epoch(i):
        rr = rows(i)
        return np.asarray(Zr[[cix[(int(r[k]), int(c[k]))] for k in rr], y[rr] - yrs[0]], np.float64).mean(0)
    ce = np.array([e for e, m in hp]); cm = np.array([m for e, m in hp])
    rep_e, rep_m = {}, {}
    rep_e["desk"] = np.stack([desk_epoch(i) for i in ce]); rep_m["desk"] = np.stack([desk_epoch(i) for i in cm])
    Xe = np.stack([allm(X, i) for i in ce]) - xm; Xm = np.stack([allm(X, i) for i in cm]) - xm
    rep_e["rrr"] = Xe @ B @ Vt.T; rep_m["rrr"] = Xm @ B @ Vt.T              # coordinates in the RRR-ordered basis
    rep_e["covpca"] = Xe @ Pt.T; rep_m["covpca"] = Xm @ Pt.T
    rep_e["esk"] = np.stack([half(Eann, i, 1) for i in ce]); rep_m["esk"] = np.stack([half(Eann, i, 1) for i in cm])
    Oe, Om = np.stack([half(Lc, i, 1) for i in ce]), np.stack([half(Lc, i, 1) for i in cm])   # odd-year communities
    Ee, Em = np.stack([half(Lc, i, 0) for i in ce]), np.stack([half(Lc, i, 0) for i in cm])   # even-year communities
    Te = np.stack([half(Eann, i, 0) for i in ce]); Tm = np.stack([half(Eann, i, 0) for i in cm])
    ok = np.isfinite(Oe).all(1) & np.isfinite(Om).all(1) & np.isfinite(Ee).all(1) & np.isfinite(Em).all(1) \
        & np.isfinite(rep_e["esk"]).all(1) & np.isfinite(rep_m["esk"]).all(1) & np.isfinite(Te).all(1) & np.isfinite(Tm).all(1)
    for d in (rep_e, rep_m):
        for kk in d:
            d[kk] = d[kk][ok]
    Oe, Om, Ee, Em, Te, Tm = Oe[ok], Om[ok], Ee[ok], Em[ok], Te[ok], Tm[ok]
    n = len(Oe)
    rng = np.random.default_rng(86)
    i1, i2 = rng.integers(0, n, (2, a.pairs)); keep_p = i1 != i2; i1, i2 = i1[keep_p], i2[keep_p]
    obs_sim = ruz(Om[i1], Em[i2])
    obs_d2 = ruz(Oe, Ee) + ruz(Om, Em) - 2 * ruz(Oe, Em)
    T_all = np.vstack([Te, Tm])
    res = {"n_heldout_cells": int(n), "ranks": a.ks, "spatial_kernel_fidelity": {}, "change_fidelity": {}, "cka": {}}
    for name in ("desk", "rrr", "covpca", "esk"):
        for d in ("spatial_kernel_fidelity", "change_fidelity", "cka"):
            res[d][name] = []
        for k in a.ks:
            Re, Rm = rep_e[name][:, :k], rep_m[name][:, :k]
            sim = (Rm[i1] * Rm[i2]).sum(1)
            res["spatial_kernel_fidelity"][name].append(float(np.corrcoef(sim, obs_sim)[0, 1]))
            d2 = ((Rm - Re) ** 2).sum(1)
            res["change_fidelity"][name].append(float(np.corrcoef(d2, obs_d2)[0, 1]))
            res["cka"][name].append(cka(np.vstack([Re, Rm]), T_all))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e086m.json"), "w"), indent=1)
    print(f"[e086m] {os.path.basename(a.cache.rstrip('/'))}: {n} held-out cells; ranks {a.ks}")
    for d in ("spatial_kernel_fidelity", "change_fidelity", "cka"):
        print(f"  {d}")
        for name in ("desk", "rrr", "covpca", "esk"):
            print(f"    {name:7s} " + "  ".join(f"k{k}:{v:+.3f}" for k, v in zip(a.ks, res[d][name])))


if __name__ == "__main__":
    main()
