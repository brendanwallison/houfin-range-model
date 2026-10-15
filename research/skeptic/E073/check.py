"""E073 B2 skeptic (verification d) + independent re-derivation (b) of the claim that the clock-free varying-coefficient
delta (vc_desk4_no_clock_h25) beats DESK's readout on sedentary out-of-community species.

    python research/lib/launch.py research/skeptic/E073/check.py --cache <tempho cache> --saved <B2 out dir> \
        --out <scratch dir> --tag t1985

Controls (BLR arms reuse anchored_delta's own classes and fitting path; env BLR_MAX_ITER=100 BLR_STARTS=0.5 as B2):
  a. vc_desk4_no_clock_h25 with DESK-zbar context permuted across CELLS (2 permutations);
  b. capacity-matched DESK: vc_blocks(zw, ctx_desk4) on within deviations;
  c. independent re-derivation: closed-form ridge (lambda by 5-fold CV over 6x6 blocks of training cells), own
     features (own within-centering, own per-group PCA/whitening, own context standardization), own pooled corr.
"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "4"
os.environ.setdefault("BLR_MAX_ITER", "100")
os.environ.setdefault("BLR_STARTS", "0.5")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

REPO = "/mnt/c/Dev/houfin-range-model"
for p in (REPO, REPO + "/src", REPO + "/research", REPO + "/research/exp"):
    if p not in sys.path:
        sys.path.insert(0, p)
import anchored_delta as ad  # noqa: E402

SIM = "/home/breallis/houfin/work/houfin/research/E067/composition-d2e6a98f0eb7/species_similarity.csv"
RESULTS = os.path.join(REPO, "research/skeptic/E073/check_results.json")
T0 = time.time()


def log(msg):
    print(f"[skeptic {time.time() - T0:7.0f}s] {msg}", flush=True)


def own_corr(P, Ta, Tb):
    """Noise-free pooled corr and k, centred across held-out cells, pooled over species (columns)."""
    Pc, Ac, Bc = P - P.mean(0), Ta - Ta.mean(0), Tb - Tb.mean(0)
    vp = (Pc * Pc).sum()
    vt = (Ac * Bc).sum()
    cv = 0.5 * ((Pc * Ac).sum() + (Pc * Bc).sum())
    return float(cv / np.sqrt(vp * vt)), float(cv / vp)


def score(est, P, strata, pairs):
    out = {}
    for sname in strata:
        m = est.strata[sname]
        D = {k: est.draws(v, "dev", m) for k, v in P.items()}
        o = {"n_species": int(m.sum()),
             "arms": {k: {"corr": float(v[0, 0]), "ci": ad.ci(v[:, 0]), "k": float(v[0, 1])} for k, v in D.items()},
             "diffs": {}}
        for x, y in pairs:
            if x in D and y in D:
                d = D[x][:, 0] - D[y][:, 0]
                o["diffs"][f"{x} - {y}"] = {"diff": float(d[0]), "ci": ad.ci(d),
                                            "p_le0": float((d[1:] <= 0).mean())}
        out[sname] = o
    return out


def ridge_cv(F_trn, Y_trn, blocks_id, alphas, n_folds=5, seed=0):
    """Closed-form ridge (no intercept: features and targets are within-cell deviations). Lambda = alpha * mean diag
    of X'X. CV over spatial blocks of training cells. Returns per-alpha CV MSE (n_alpha, S) and the full-data coefs."""
    ub = np.unique(blocks_id)
    rng = np.random.default_rng(seed)
    fold_of_block = rng.permutation(len(ub)) % n_folds
    fold = fold_of_block[np.searchsorted(ub, blocks_id)]
    p = F_trn.shape[1]
    XtX = F_trn.T @ F_trn
    XtY = F_trn.T @ Y_trn
    scale = np.trace(XtX) / p
    mse = np.zeros((len(alphas), Y_trn.shape[1]))
    for f in range(n_folds):
        te = fold == f
        Xf, Yf = F_trn[te], Y_trn[te]
        XtXf, XtYf = Xf.T @ Xf, Xf.T @ Yf
        A, b = XtX - XtXf, XtY - XtYf
        for i, al in enumerate(alphas):
            B = np.linalg.solve(A + al * scale * np.eye(p), b)
            R = Yf - Xf @ B
            mse[i] += (R * R).sum(0)
    coefs = [np.linalg.solve(XtX + al * scale * np.eye(p), XtY) for al in alphas]
    return mse, coefs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--saved", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    _inv = torch.linalg.inv

    def safe_inv(A):
        """blr.fit's final A is SPD: invert by Cholesky (batched MKL getrf/getri failed here with a DLASWP error);
        fall back to the original inverse per bad item, and log non-finite inputs."""
        nf = ~torch.isfinite(A).reshape(A.shape[0], -1).all(1) if A.dim() == 3 else None
        if nf is not None and bool(nf.any()):
            log(f"safe_inv: {int(nf.sum())} items with non-finite A")
        L, info = torch.linalg.cholesky_ex(A)
        out = torch.cholesky_inverse(L)
        bad = info > 0
        if bool(bad.any()):
            log(f"safe_inv: {int(bad.sum())} items not PD; per-item inverse")
            for i in torch.where(bad)[0].tolist():
                out[i] = torch.linalg.pinv(A[i])
        return out
    torch.linalg.inv = safe_inv
    est = ad.Estimand(a.cache, a.n_boot, SIM)
    log("estimand built")
    cov = ad.Cov(est.keys, ~est.ho & ~est.bf & est.ty, est.ty)
    log("covariates built")
    keys, fin = est.keys, cov.fin
    z = np.asarray(np.load(os.path.join(a.cache, "z_raw.npy"), mmap_mode="r")[:, :24], "float64")
    z = np.where(np.isfinite(z), z, 0.0)
    zbar = ad.cell_mean(keys, z, est.ty & fin)
    zw = z - zbar
    extra = {"z": z, "zbar": zbar, "Zesk": est.Zesk}
    trn = np.where(~est.ho & ~est.bf & est.ty & fin)[0]
    Yw = est.Y - ad.cell_mean(keys, est.Y, est.ty & fin)
    info = {"n_rows": int(len(keys)), "n_train_rows": int(len(trn)), "n_eval_cells": int(len(est.early)),
            "withheld": [int(min(est.withheld)), int(max(est.withheld))] if est.withheld else None}
    # sanity: leakage / coverage facts
    ev_rows = np.concatenate(est.early + est.modern)
    info["eval_rows_nonfinite_cov_share"] = float((~fin[ev_rows]).mean())
    info["eval_rows_in_holdout_share"] = float(est.ho[ev_rows].mean())
    info["train_rows_in_withheld_years"] = int(np.isin(keys[trn, 2], est.withheld).sum())
    info["train_rows_in_holdout_or_buffer"] = int((est.ho[trn] | est.bf[trn]).sum())
    early_rows = np.concatenate(est.early)
    info["early_rows_years"] = [int(keys[early_rows, 2].min()), int(keys[early_rows, 2].max())]
    info["early_rows_withheld_share"] = float(np.isin(keys[early_rows, 2], est.withheld).mean())
    log(f"info {info}")

    P = {}

    def within(name, blocks):
        t = time.time()
        m = ad.fit_blocks(blocks, Yw, trn)
        P[name] = np.hstack([est.delta(B) for B in blocks]) @ m["coef"].T
        log(f"{name}: {sum(B.shape[1] for B in blocks)} features, {time.time() - t:.0f}s, "
            f"median log amp {np.median(np.log(m['a']), 0).round(2).tolist()}")

    # references, refit with the same path
    within("desk_within", [zw])
    ms = ad.fit_blocks([zbar, zw], est.Y, trn)
    P["desk_split_level"] = est.delta(z) @ ms["coef"][:, zbar.shape[1]:].T
    log("desk_split_level")
    W25 = cov.w(25, "no_clock")
    ctx = ad.ctx_feats(cov, extra, "desk", 4, trn)
    within("vc_desk4_no_clock_h25", ad.vc_blocks(W25, ctx))
    # saved B2 preds, for reproduction and extra references
    sv = np.load(os.path.join(a.saved, "preds_b2_dev.npz"))
    for k in ("vc_cov4_no_clock_h25", "within_no_clock_h25", "placebo"):
        P[f"saved:{k}"] = np.asarray(sv[k], "float64")
    info["reproduction_max_abs_diff_vs_saved"] = {
        k: float(np.abs(P[k] - np.asarray(sv[k], "float64")).max() / (np.abs(sv[k]).max() + 1e-12))
        for k in ("desk_within", "desk_split_level", "vc_desk4_no_clock_h25")}
    log(f"reproduction (relative max abs diff vs saved): {info['reproduction_max_abs_diff_vs_saved']}")

    # (a) permuted context: permute DESK-zbar context across CELLS (cell-constant feature, one perm for all its rows)
    ncell = len(cov.cells)
    first = np.zeros(ncell, int)
    first[cov.cidx[::-1]] = np.arange(len(keys))[::-1]
    raw_ctx_cell = zbar[first, :4]
    assert np.allclose(zbar[:, :4], raw_ctx_cell[cov.cidx])
    perms = []
    for s in (1, 2):
        perm = np.random.default_rng(100 + s).permutation(ncell)
        C = raw_ctx_cell[perm][cov.cidx]
        C = (C - C[trn].mean(0)) / (C[trn].std(0) + 1e-9)
        perms.append(C)
        within(f"vc_desk4_no_clock_h25_permctx#{s}", ad.vc_blocks(W25, C))
    # (b) capacity-matched DESK: DESK within deviation with the same place-dependent slopes
    within("vc_zw_desk4ctx", ad.vc_blocks(zw, ctx))
    within("vc_zw_desk4ctx_permctx#1", ad.vc_blocks(zw, perms[0]))

    # (c) independent re-derivation: ridge with own features
    cid = cov.cidx
    trow = est.ty & fin                         # rows used for each cell's mean (trained years, finite covariates)
    cnt = np.bincount(cid, weights=trow.astype(float), minlength=ncell)

    def own_cell_center(X):
        s = np.zeros((ncell, X.shape[1]))
        np.add.at(s, cid[trow], X[trow])
        return X - (s / np.maximum(cnt, 1)[:, None])[cid]

    Xs = cov.X[25]                              # standardized (affine) raw EMA covariates, h25
    feats = []
    for g in ("climate", "cover", "human"):
        cols = ad.group_cols(cov.stream, cov.names, g)
        D = own_cell_center(Xs[:, cols])
        _, _, Vt = np.linalg.svd(D[trn] - D[trn].mean(0), full_matrices=False)   # full SVD, no subsampling
        V = Vt[: ad.NPC[g]].T
        Fg = D @ V
        feats.append(Fg / Fg[trn].std(0))
    Wown = np.hstack(feats)
    zbar_own = own_cell_center(z)               # -> z - own mean; recover own mean
    zmean_own = z - zbar_own
    trn_cells = np.unique(cid[trn])
    cc = zmean_own[first, :4]                   # cell-level context, standardized over TRAINING CELLS (not rows)
    cc = (cc - cc[trn_cells].mean(0)) / cc[trn_cells].std(0)
    ctx_own = cc[cid]
    zw_own = zbar_own
    Yw_own = own_cell_center(est.Y)
    cells_xy = keys[:, :2]
    blk = (cells_xy[trn, 0] // 6).astype(np.int64) * 100000 + cells_xy[trn, 1] // 6
    alphas = np.logspace(-4, 2, 13)

    def vc(Wf, C):
        return np.hstack([Wf] + [Wf * C[:, [j]] for j in range(C.shape[1])])

    def ep_delta(F):
        return np.stack([F[m].mean(0) for m in est.modern]) - np.stack([F[e].mean(0) for e in est.early])

    perm_own = np.random.default_rng(101).permutation(ncell)
    ctx_own_perm = cc[perm_own][cid]
    ridge_arms = {"ridge:desk_within": zw_own, "ridge:vc_desk4_no_clock_h25": vc(Wown, ctx_own),
                  "ridge:within_no_clock_h25": Wown, "ridge:vc_zw_desk4ctx": vc(zw_own, ctx_own),
                  "ridge:vc_desk4_no_clock_h25_permctx": vc(Wown, ctx_own_perm)}
    ridge_info = {}
    Ta, Tb = est.T["dev"]
    for name, F in ridge_arms.items():
        t = time.time()
        mse, coefs = ridge_cv(F[trn], Yw_own[trn], blk, alphas)
        dF = ep_delta(F)
        ri = {}
        for sname in ("sedentary", "all"):
            m = est.strata[sname]
            i_sh = int(np.argmin(mse[:, m].sum(1)))                     # one lambda shared by the stratum's species
            i_sp = np.argmin(mse[:, m], 0)                              # per-species lambda
            Psh = dF @ coefs[i_sh][:, m]
            Psp = np.stack([dF @ coefs[i][:, j] for i, j in zip(i_sp, np.where(m)[0])], 1)
            P[f"{name}|shared|{sname}"] = (Psh, m)
            P[f"{name}|perspecies|{sname}"] = (Psp, m)
            ri[sname] = {"alpha_shared": float(alphas[i_sh]),
                         "alpha_perspecies_median": float(np.median(alphas[i_sp])),
                         "own_corr_shared": own_corr(Psh, Ta[:, m], Tb[:, m]),
                         "own_corr_perspecies": own_corr(Psp, Ta[:, m], Tb[:, m])}
        ridge_info[name] = ri
        log(f"{name}: p {F.shape[1]}, {time.time() - t:.0f}s, {ri}")

    # scoring of the BLR arms (est.draws, paired bootstrap)
    blr_P = {k: v for k, v in P.items() if not isinstance(v, tuple)}
    pairs = [("vc_desk4_no_clock_h25", "desk_split_level"), ("vc_desk4_no_clock_h25", "desk_within"),
             ("vc_desk4_no_clock_h25", "saved:within_no_clock_h25"),
             ("vc_desk4_no_clock_h25_permctx#1", "vc_desk4_no_clock_h25"),
             ("vc_desk4_no_clock_h25_permctx#2", "vc_desk4_no_clock_h25"),
             ("vc_desk4_no_clock_h25_permctx#1", "saved:within_no_clock_h25"),
             ("vc_desk4_no_clock_h25_permctx#2", "saved:within_no_clock_h25"),
             ("vc_desk4_no_clock_h25_permctx#1", "desk_split_level"),
             ("vc_desk4_no_clock_h25_permctx#2", "desk_split_level"),
             ("vc_zw_desk4ctx", "desk_within"), ("vc_zw_desk4ctx", "desk_split_level"),
             ("vc_desk4_no_clock_h25", "vc_zw_desk4ctx"), ("vc_zw_desk4ctx_permctx#1", "vc_zw_desk4ctx"),
             ("saved:vc_cov4_no_clock_h25", "desk_split_level"), ("saved:placebo", "vc_desk4_no_clock_h25")]
    res = {"info": info, "blr": score(est, blr_P, ("sedentary", "all"), pairs), "ridge": {"lambda": ridge_info}}
    # ridge arms: corr via est.draws (bootstrap) and paired diffs, per stratum and lambda mode
    for sname in ("sedentary", "all"):
        m = est.strata[sname]
        for mode in ("shared", "perspecies"):
            D = {}
            for name in ridge_arms:
                Pm, _ = P[f"{name}|{mode}|{sname}"]
                D[name] = _draws_sub(est, Pm, m)
            o = {"arms": {k: {"corr": float(v[0, 0]), "ci": ad.ci(v[:, 0]), "k": float(v[0, 1])} for k, v in D.items()},
                 "diffs": {}}
            for x, y in (("ridge:vc_desk4_no_clock_h25", "ridge:desk_within"),
                         ("ridge:vc_desk4_no_clock_h25", "ridge:within_no_clock_h25"),
                         ("ridge:vc_desk4_no_clock_h25", "ridge:vc_zw_desk4ctx"),
                         ("ridge:vc_zw_desk4ctx", "ridge:desk_within"),
                         ("ridge:vc_desk4_no_clock_h25_permctx", "ridge:vc_desk4_no_clock_h25")):
                d = D[x][:, 0] - D[y][:, 0]
                o["diffs"][f"{x} - {y}"] = {"diff": float(d[0]), "ci": ad.ci(d), "p_le0": float((d[1:] <= 0).mean())}
            res["ridge"][f"{sname}|{mode}"] = o
    json.dump(res, open(os.path.join(a.out, f"check_{a.tag}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, f"preds_{a.tag}.npz"),
                        **{k.replace("+", "_PLUS_").replace(":", "__").replace("#", "_").replace("|", "__"):
                           (v[0] if isinstance(v, tuple) else v).astype("float32") for k, v in P.items()})
    allres = {}
    if os.path.exists(RESULTS):
        allres = json.load(open(RESULTS))
    allres[a.tag] = res
    json.dump(allres, open(RESULTS, "w"), indent=1)
    for sname in ("sedentary", "all"):
        o = res["blr"][sname]
        log(f"== BLR {sname}: " + "  ".join(f"{k} {v['corr']:.3f}(k{v['k']:.2f})" for k, v in o["arms"].items()))
        for k, v in o["diffs"].items():
            log(f"     {k:60s} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")
        for mode in ("shared", "perspecies"):
            o = res["ridge"][f"{sname}|{mode}"]
            log(f"== ridge {sname} {mode}: " + "  ".join(f"{k} {v['corr']:.3f}(k{v['k']:.2f})"
                                                       for k, v in o["arms"].items()))
            for k, v in o["diffs"].items():
                log(f"     {k:60s} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")


def _draws_sub(est, Pm, m):
    """est.draws for a prediction that already holds only the stratum's species (columns of m)."""
    Ta, Tb = est.T["dev"]
    saved = est.T["dev"]
    est.T["dev"] = [Ta[:, m], Tb[:, m]]
    try:
        return est.draws(Pm, "dev", None)
    finally:
        est.T["dev"] = saved


if __name__ == "__main__":
    main()
