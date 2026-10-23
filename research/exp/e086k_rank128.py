"""E086k: is there anything worth having beyond ESK rank 64? Three checks that need no DESK retraining, plus DESK's
training fit against the target noise floor.

    python research/exp/e086k_rank128.py --z-dir <ESK basis dir> --cache <tempho research cache> --run <DESK run desk/>
        --similarity <species_similarity.csv> --out <dir> [--rank 128]

0. BASIS: the same 16,000 landmarks, the same uncentred Ruzicka kernel; top --rank eigenpairs of K_mm (Lanczos). Dims
   1-64 must reproduce the saved basis (per-dim |corr| with the cache's esk_annual); signs aligned to it.
A. RELIABILITY of the observed community's coordinates per dim (1..rank), held-out cells: split-half (odd / even
   surveyed years) reliability (Spearman-Brown) of cell-epoch LEVEL and of epoch CHANGE (withheld early -> modern).
B. ORACLE species relevance by rank k (24 .. rank): observed community change (k ESK dims, half A) -> dev species change
   (half B), ridge fitted on training cells' trained-era window change, scored noise-free on held-out long change;
   sedentary common / sedentary / all dev.
C. LINEAR COVARIATE PREDICTABILITY per dim: a supervised ridge from standardized covariates (F_cov_raw) to the observed
   annual coordinates, fitted on training cells x trained years (levels) and, separately, on within-cell deviations;
   per-dim held-out corr of cell-epoch level (level map) and of change (within map), noise-free on the change.
D. NOISE FLOOR: DESK's per-element z-MSE on training cell-years (trajectory zmse_train at the selected and last epoch)
   against the annual targets' noise: half the mean squared difference between a training cell's consecutive surveyed
   years in the observed ESK (dims 1-64; an upper-ish bound -- it includes one year of true change).
"""
import argparse
import json
import os

import numpy as np


def ruz_kernel(A, B, chunk=2000):
    """Uncentred Ruzicka kernel between rows of A (n, d) and B (m, d), non-negative inputs; float32 output."""
    sa, sb = A.sum(1), B.sum(1)
    out = np.empty((A.shape[0], B.shape[0]), np.float32)
    import torch
    Bt = torch.as_tensor(B, dtype=torch.float32)
    for s in range(0, A.shape[0], chunk):
        At = torch.as_tensor(A[s:s + chunk], dtype=torch.float32)
        l1 = torch.cdist(At, Bt, p=1).numpy()
        ss = sa[s:s + chunk, None] + sb[None, :]
        num, den = 0.5 * (ss - l1), 0.5 * (ss + l1)
        out[s:s + chunk] = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
    return out


def sb(r):
    r = np.asarray(r, float)
    return np.where(r > 0, 2 * r / (1 + r), np.nan)


def colcorr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0) + 1e-300)


def nf_corr(p, a, b):
    p = p - p.mean(0); a = a - a.mean(0); b = b - b.mean(0)
    return float((p * b).sum() / np.sqrt((p * p).sum() * max((a * b).sum(), 1e-12)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--z-dir", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--similarity", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=128)
    a = ap.parse_args()
    import pandas as pd
    from scipy.sparse.linalg import eigsh
    os.makedirs(a.out, exist_ok=True)
    res = {}
    # ---- 0. basis
    lm = np.load(os.path.join(a.z_dir, "esk_landmarks.npy")).astype(np.float32)
    K = ruz_kernel(lm, lm)
    ev, U = eigsh(K.astype(np.float64), k=a.rank + 2, which="LA")
    o = np.argsort(ev)[::-1][: a.rank]; ev, U = ev[o], U[:, o]
    proj = (U / np.sqrt(np.clip(ev, 1e-10, None))).astype(np.float32)
    del K
    j = lambda f: os.path.join(a.cache, f)
    keys = np.load(j("keys.npy")); Xc = np.load(j("X_comm.npy")).astype(np.float64)
    Lc = np.log1p(np.clip(Xc, 0, None)).astype(np.float32)
    Z = np.zeros((len(keys), a.rank), np.float32)
    for s in range(0, len(keys), 20000):
        Z[s:s + 20000] = ruz_kernel(Lc[s:s + 20000], lm) @ proj
    E64 = np.load(j("esk_annual.npy")).astype(np.float64)
    ok = np.isfinite(E64).all(1)
    cc = colcorr(Z[ok, :64].astype(np.float64), E64[ok])
    Z[:, :64] *= np.sign(cc)[None, :]
    res["basis"] = {"eigvals": ev.tolist(), "energy_cum": (np.cumsum(ev) / ev.sum()).tolist(),
                    "dims1_64_abs_corr_min": float(np.abs(cc).min()), "dims1_64_abs_corr_median": float(np.median(np.abs(cc))),
                    "eig_share_65_128_of_1_128": float(ev[64:].sum() / ev.sum())}
    np.save(os.path.join(a.out, f"esk{a.rank}_annual.npy"), Z)
    print(f"[e086k] basis: eig share of dims 65-{a.rank} = {res['basis']['eig_share_65_128_of_1_128']:.3f}; dims 1-64 "
          f"reproduce the saved basis |corr| min {res['basis']['dims1_64_abs_corr_min']:.4f}", flush=True)
    # ---- shared structures
    sp = np.load(j("split.npz")); ho, bf = sp["holdout"].astype(bool), sp["buffer"].astype(bool)
    withheld = set(int(y) for y in sp["withheld"])
    E = dict(np.load(j("esk_epochs.npz")))
    r, c, y = keys[:, 0], keys[:, 1], keys[:, 2]
    ptr, flat = E["rows_ptr"], E["rows_flat"]
    def half_mean(V, i, h):
        rr = flat[ptr[i]:ptr[i + 1]]; rr = rr[(y[rr] % 2) == h]
        return V[rr].mean(0) if len(rr) else np.full(V.shape[1], np.nan)
    idx = {}
    for i, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh):
            idx.setdefault(cell, [None, None])[0] = i
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = i
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    pc = E["cells"][pairs[:, 0]]; hp = pairs[ho[pc[:, 0], pc[:, 1]]]
    ZA = {i: half_mean(Z, i, 0) for i in np.unique(hp)}; ZB = {i: half_mean(Z, i, 1) for i in np.unique(hp)}
    LA = np.stack([np.concatenate([ZA[e], ZA[m]]) for e, m in hp]); LB = np.stack([np.concatenate([ZB[e], ZB[m]]) for e, m in hp])
    R = a.rank
    okp = np.isfinite(LA).all(1) & np.isfinite(LB).all(1)
    hp, LA, LB = hp[okp], LA[okp], LB[okp]
    lvA = np.vstack([LA[:, :R], LA[:, R:]]); lvB = np.vstack([LB[:, :R], LB[:, R:]])
    dA, dB = LA[:, R:] - LA[:, :R], LB[:, R:] - LB[:, :R]
    # ---- A. reliability per dim
    res["A_reliability"] = {"level": sb(colcorr(lvA, lvB)).tolist(), "change": sb(colcorr(dA, dB)).tolist(),
                            "n_heldout_cells": int(len(hp))}
    band = lambda v, lo, hi: float(np.nanmean(np.asarray(v)[lo:hi]))
    for nm in ("level", "change"):
        v = res["A_reliability"][nm]
        print(f"[e086k] A {nm} reliability: 1-24 {band(v, 0, 24):.2f} 25-64 {band(v, 24, 64):.2f} 65-96 {band(v, 64, 96):.2f} "
              f"97-{R} {band(v, 96, R):.2f}", flush=True)
    # ---- B. oracle species relevance by rank
    sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    meta = json.load(open(j("meta.json"))); dev = list(meta["dev_species"])
    Ld = np.log1p(np.asarray(np.load(j("X_dev.npy"), mmap_mode="r"), np.float64))
    SA = np.stack([half_mean(Ld, m, 0) - half_mean(Ld, e, 0) for e, m in hp])
    SB = np.stack([half_mean(Ld, m, 1) - half_mean(Ld, e, 1) for e, m in hp])
    is_tr = ~ho[r, c] & ~bf[r, c] & ~np.isin(y, list(withheld))
    trained = sorted(set(int(v) for v in np.unique(y)) - withheld); mid = trained[len(trained) // 2]
    cid = r.astype(np.int64) * 100000 + c
    def win_change(V):
        out = {}
        for h in (0, 1):
            m = is_tr & (y % 2 == h)
            df = pd.DataFrame(V[m]); df["cell"] = cid[m]; df["w"] = (y[m] >= mid).astype(int)
            g = df.groupby(["cell", "w"]).mean(); w0 = g.xs(0, level="w"); w1 = g.xs(1, level="w")
            cm = w0.index.intersection(w1.index); out[h] = (w1.loc[cm] - w0.loc[cm])
        cm = out[0].index.intersection(out[1].index)
        return out[0].loc[cm].to_numpy(), out[1].loc[cm].to_numpy()
    cA, _ = win_change(Z.astype(np.float64)); _, sB_tr = win_change(Ld); sA_tr, _ = win_change(Ld)
    ok2 = np.isfinite(SA).all(1) & np.isfinite(SB).all(1)
    prev = (np.asarray(np.load(j("X_dev.npy"), mmap_mode="r"))[is_tr] > 0).mean(0)
    sed = np.array([mig.get(s_) == 1.0 for s_ in dev])
    strata = {"sed_common": sed & (prev >= 0.08), "sedentary": sed, "all_dev": np.ones(len(dev), bool)}
    def fit_pred(Ftr, Ytr, Fte, lam):
        mu = Ftr.mean(0); Fc_ = Ftr - mu
        W = np.linalg.solve(Fc_.T @ Fc_ + lam * np.eye(Fc_.shape[1]), Fc_.T @ (Ytr - Ytr.mean(0)))
        return (Fte - mu) @ W + Ytr.mean(0)
    res["B_oracle"] = {}
    for k in [24, 32, 48, 64, 96, R]:
        Ftr, Fte = cA[:, :k], dA[ok2][:, :k]
        best = None
        for lam in (1e-3, 1e-2, 1e-1, 1.0):
            h = np.arange(len(Ftr)) % 2 == 0
            sc = nf_corr(fit_pred(Ftr[h], sB_tr[h], Ftr[~h], lam), sA_tr[~h], sB_tr[~h])
            best = (sc, lam) if best is None or sc > best[0] else best
        pred = fit_pred(Ftr, sB_tr, Fte, best[1])
        res["B_oracle"][f"k{k}"] = {s_: nf_corr(pred[:, m], SA[ok2][:, m], SB[ok2][:, m]) for s_, m in strata.items()}
        print(f"[e086k] B oracle k={k:3d}: " + "  ".join(f"{s_} {v:+.3f}" for s_, v in res["B_oracle"][f'k{k}'].items()), flush=True)
    # ---- C. linear covariate predictability per dim
    F = np.asarray(np.load(j("F_cov_raw.npy"), mmap_mode="r"), np.float64)
    trn = np.where(is_tr & np.isfinite(F).all(1))[0]
    mu, sd = F[trn].mean(0), F[trn].std(0); keep = sd > 1e-6
    X = np.where(np.isfinite(F[:, keep]), (F[:, keep] - mu[keep]) / sd[keep], 0.0)
    Zf = Z.astype(np.float64)
    u, inv = np.unique(cid, return_inverse=True)
    wt = np.zeros(len(keys)); wt[trn] = 1.0
    def cmean(V):
        s_ = np.zeros((len(u), V.shape[1])); np.add.at(s_, inv, V * wt[:, None]); n_ = np.bincount(inv, weights=wt, minlength=len(u))
        return (s_ / np.maximum(n_, 1e-12)[:, None])[inv]
    def ridge(Xt, Yt, lam_rel=1e-2):
        Xm, Ym = Xt.mean(0), Yt.mean(0); Xc_ = Xt - Xm
        lam = lam_rel * np.trace(Xc_.T @ Xc_) / Xc_.shape[1]
        return np.linalg.solve(Xc_.T @ Xc_ + lam * np.eye(Xc_.shape[1]), Xc_.T @ (Yt - Ym)), Xm, Ym
    Bl, Xm, Ym = ridge(X[trn], Zf[trn])
    Xd, Zd = X - cmean(X), Zf - cmean(Zf)
    Bw, _, _ = ridge(Xd[trn], Zd[trn])
    XA = {i: half_mean(X, i, 0) for i in np.unique(hp)}; XB = {i: half_mean(X, i, 1) for i in np.unique(hp)}
    Xe = np.stack([(XA[e] + XB[e]) / 2 for e, m in hp]); Xm_ = np.stack([(XA[m] + XB[m]) / 2 for e, m in hp])
    lvl_pred = np.vstack([(Xe - Xm) @ Bl + Ym, (Xm_ - Xm) @ Bl + Ym])
    chg_pred = (Xm_ - Xe) @ Bw
    lvl_true = (lvA + lvB) / 2
    res["C_linear"] = {"level_corr": colcorr(lvl_pred, lvl_true).tolist(),
                       "change_corr_nf": [nf_corr(chg_pred[:, [q]], dA[:, [q]], dB[:, [q]]) for q in range(R)]}
    for nm, key in (("level", "level_corr"), ("change", "change_corr_nf")):
        v = res["C_linear"][key]
        print(f"[e086k] C linear covariate {nm} corr: 1-24 {band(v, 0, 24):+.2f} 25-64 {band(v, 24, 64):+.2f} "
              f"65-96 {band(v, 64, 96):+.2f} 97-{R} {band(v, 96, R):+.2f}", flush=True)
    # ---- D. DESK training fit vs the target noise floor
    rows = [json.loads(l) for l in open(os.path.join(a.run, "train_trajectory.jsonl"))]
    dm = np.load(os.path.join(a.run, "desk_meta.npz"), allow_pickle=True)
    be = int(dm["best_epoch"]) if "best_epoch" in dm.files else None
    pick = lambda ep: next((r_ for r_ in rows if r_.get("epoch") == ep), None)
    tr_rows = np.where(is_tr & np.isfinite(E64).all(1))[0]
    o_ = np.lexsort((y[tr_rows], cid[tr_rows])); tr_rows = tr_rows[o_]
    same = (cid[tr_rows][1:] == cid[tr_rows][:-1]) & (y[tr_rows][1:] - y[tr_rows][:-1] == 1)
    dd = E64[tr_rows][1:][same] - E64[tr_rows][:-1][same]
    noise = float(0.5 * (dd ** 2).mean())
    res["D_noise_floor"] = {"annual_target_noise_per_element": noise, "n_consecutive_pairs": int(same.sum()), "best_epoch": be}
    for lab, ep in (("selected", be), ("last", rows[-1].get("epoch"))):
        rr = pick(ep) if ep is not None else None
        if rr:
            res["D_noise_floor"][lab] = {k: rr.get(k) for k in ("epoch", "zmse_train", "zmse_val", "zmse_val_sp", "loss_stab")}
    print(f"[e086k] D noise floor (annual targets, per element) {noise:.5f}; DESK " +
          "; ".join(f"{lab} ep{v['epoch']} train {v['zmse_train']} val {v['zmse_val']}" for lab, v in res["D_noise_floor"].items()
                    if isinstance(v, dict)), flush=True)
    json.dump(res, open(os.path.join(a.out, "e086k.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
