"""E044d: within an environment TYPE, did places whose covariates changed MORE than is typical for their type also see
their bird community change more? (the M26 "where it acts" test, generalized; the E044 skeptic's point 1)

    python research/exp/type_dose.py --cache <tempho research cache> --out <dir>
    python research/exp/type_dose.py --summarize <dir>

Per cell (both epochs present): epoch-mean standardized covariates (F_cov_raw, early withheld epoch vs 2005-25), their
change dX, and the observed ESK change dZ (r24). TYPE = 32 PCs of the cell's mean covariate levels over trained years.
On TRAINING cells (not held out, not buffer): dX and dZ regressed on [1, type] -> type-typical change; residuals are
change beyond what the cell's type did. Ridge maps (lambda by 6x6-block CV on training cells), applied to HELD-OUT
cells and graded by noise-free centred corr (ABBA halves; 6x6-block bootstrap):
    type        [1, type] -> dZ                       what "this kind of place" did
    full:<fam>  dX[fam] -> dZ                         covariate change, all of it
    dose:<fam>  dX_res[fam] -> dZ_res, graded against the held-out truth with ITS type-typical change removed
                (the same training-fitted subtraction applied to both halves): within-type dose-response
Families: all channels, land (LUH classes + management + HYDE + BUI), luh_class, luh_mgmt, climate.
NOTE: the training cells' change uses their observed early-epoch communities (years withheld from DESK); this is a
cross-sectional diagnostic of a relationship, not a backcast evaluation of any model.
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


def wc(x, y, w):
    xm = (w[:, None] * x).sum(0) / w.sum()
    ym = (w[:, None] * y).sum(0) / w.sum()
    return (w[:, None] * (x - xm) * (y - ym)).sum(0)


# LUH channels 240-272 in schema order (alphabetical); classes incl. secondary age (secma/secmb) vs management rates
LUH = ["c3ann", "c3nfx", "c3per", "c4ann", "c4per", "fertl_c3ann", "fertl_c3nfx", "fertl_c3per", "fertl_c4ann",
       "fertl_c4per", "fharv_c3per", "fharv_c4per", "flood", "irrig_c3ann", "irrig_c3nfx", "irrig_c3per", "irrig_c4ann",
       "irrig_c4per", "pastr", "pltns_wdprd", "primf", "primn", "prtct_pltns", "prtct_primf", "prtct_primn",
       "prtct_secdf", "prtct_secdn", "range", "secdf", "secdn", "secma", "secmb", "urban"]


def families(n_ch):
    fam = np.empty(n_ch, object)
    fam[0:240] = "climate"
    fam[273:276] = "hyde"
    fam[276:283] = "bui"
    fam[283:302] = "static"
    for i, v in enumerate(LUH):
        fam[240 + i] = "luh_mgmt" if v.startswith(("fertl", "fharv", "flood", "irrig", "pltns", "prtct")) else "luh_class"
    return fam


def loo_local_op(cells_rc, ell_km, cell_km=27.0):
    """V -> V - K V, K the row-normalized leave-one-out Gaussian kernel over the given cells (no fitted basis)."""
    xy = np.asarray(cells_rc, float) * cell_km
    D2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    K = np.exp(-D2 / (2.0 * ell_km ** 2))
    np.fill_diagonal(K, 0.0)
    K /= np.maximum(K.sum(1, keepdims=True), 1e-300)
    return lambda V: np.asarray(V, float) - K @ np.asarray(V, float)


def ridge_cv(X, Y, groups, lams):
    """Ridge with lambda (relative to trace(X'X)/p) chosen by leave-group-out CV; returns (B, intercepts, lam)."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    sc = np.trace(Xc.T @ Xc) / Xc.shape[1]
    ug = np.unique(groups)
    rng = np.random.default_rng(3)
    fold_of = dict(zip(ug.tolist(), rng.integers(5, size=len(ug)).tolist()))
    fold = np.array([fold_of[g] for g in groups])
    best, best_lam = np.inf, None
    for lam in lams:
        err = 0.0
        for f in range(5):
            tr, te = fold != f, fold == f
            a, b = Xc[tr], Yc[tr]
            Bf = np.linalg.solve(a.T @ a + lam * sc * np.eye(a.shape[1]), a.T @ b)
            err += ((Yc[te] - Xc[te] @ Bf) ** 2).sum()
        if err < best:
            best, best_lam = err, lam
    B = np.linalg.solve(Xc.T @ Xc + best_lam * sc * np.eye(Xc.shape[1]), Xc.T @ Yc)
    return B, xm, ym, best_lam


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--n-type", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--kloc-km", type=float, default=150.0, help="LOO Gaussian bandwidth of the kernel-local mode")
    ap.add_argument("--conus-only", action="store_true", help="keep cells with BUI available in 2015 (CONUS)")
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from scipy import sparse
    from change_oracle import group_rows
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    N = len(keys)
    split = np.load(c("split.npz"))
    withheld = [int(y) for y in split["withheld"]]
    r = a.rank
    E, full, _, _ = group_rows(a.cache, keys, withheld)
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh):
            idx.setdefault(cell, [None, None])[0] = j
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = j
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    cells = E["cells"][pairs[:, 0]]
    held = split["holdout"][cells[:, 0], cells[:, 1]].astype(bool)
    buf = split["buffer"][cells[:, 0], cells[:, 1]].astype(bool)
    keep_cell = np.ones(len(cells), bool)
    if a.conus_only:
        from src.config_utils import load_config
        sdir = os.path.join(load_config()["paths"]["hist_dir"], "yearly_states")
        bui_avail = np.load(os.path.join(sdir, "state_2015.npz"))["bui"][..., 0]
        keep_cell = np.nan_to_num(bui_avail[cells[:, 0], cells[:, 1]]) > 0.5
    ev, tc = np.where(held & keep_cell)[0], np.where(~held & ~buf & keep_cell)[0]
    dZ = (E["z_full"][pairs[:, 1], :r] - E["z_full"][pairs[:, 0], :r]).astype("float64")
    Ta = (E["z_a"][pairs[:, 1], :r] - E["z_a"][pairs[:, 0], :r]).astype("float64")
    Tb = (E["z_b"][pairs[:, 1], :r] - E["z_b"][pairs[:, 0], :r]).astype("float64")

    def gmat(groups):
        rows = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
        cols = np.concatenate([np.asarray(g, int) for g in groups])
        vals = np.concatenate([np.full(len(g), 1.0 / len(g)) for g in groups])
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(groups), N))
    Ge = gmat([np.asarray(full[p]) for p in pairs[:, 0]])
    Gm = gmat([np.asarray(full[p]) for p in pairs[:, 1]])
    F = np.asarray(np.load(c("F_cov_raw.npy"), mmap_mode="r"), "float64")
    ho = split["holdout"][keys[:, 0], keys[:, 1]]
    bfk = split["buffer"][keys[:, 0], keys[:, 1]]
    ty = ~np.isin(keys[:, 2], withheld)
    trn = np.where(~ho & ~bfk & ty & np.isfinite(F).all(1))[0]
    mu, sd = F[trn].mean(0), F[trn].std(0)
    keep = sd > 1e-6
    X = np.where(np.isfinite(F[:, keep]), (F[:, keep] - mu[keep]) / sd[keep], 0.0)
    colidx = np.where(keep)[0]
    dX = np.asarray(Gm @ X) - np.asarray(Ge @ X)
    # type: 32 PCs of the cell's mean covariate levels over trained years (modern epoch rows are trained years)
    _, _, Vt = np.linalg.svd(X[trn][:: max(1, len(trn) // 40000)] - X[trn].mean(0), full_matrices=False)
    lev = np.asarray(Gm @ X) @ Vt[:a.n_type].T
    Tm = np.hstack([np.ones((len(pairs), 1)), lev])
    gX, *_ = np.linalg.lstsq(Tm[tc], dX[tc], rcond=None)
    gZ, *_ = np.linalg.lstsq(Tm[tc], dZ[tc], rcond=None)
    dX_res, dZ_res = dX - Tm @ gX, dZ - Tm @ gZ
    type_pred = Tm @ gZ
    # secondary (added after the T0 1996 smoke run): residualize on type AND position (96 RFF fields at 300 km),
    # so only change that is local within a type and a region counts
    rgp = np.random.default_rng(1)
    om, ph = rgp.normal(size=(2, 96)) / 300.0, rgp.uniform(0, 2 * np.pi, 96)
    Rpos = np.sqrt(2.0 / 96) * np.cos(cells.astype(float) * 27.0 @ om + ph)
    Tm2 = np.hstack([Tm, Rpos])
    gX2, *_ = np.linalg.lstsq(Tm2[tc], dX[tc], rcond=None)
    gZ2, *_ = np.linalg.lstsq(Tm2[tc], dZ[tc], rcond=None)
    dX_loc, dZ_loc = dX - Tm2 @ gX2, dZ - Tm2 @ gZ2
    loc_pred = Tm2 @ gZ2
    share_type_X = 1 - ((dX_res[tc] - dX_res[tc].mean(0)) ** 2).sum() / ((dX[tc] - dX[tc].mean(0)) ** 2).sum()
    famX = families(F.shape[1])[colidx]
    FAMS = {"all": None, "land": ("luh_class", "luh_mgmt", "hyde", "bui"), "luh_class": ("luh_class",),
            "luh_mgmt": ("luh_mgmt",), "climate": ("climate",)}
    blk_tc = (cells[tc, 0] // 6) * 100000 + cells[tc, 1] // 6
    lams = [10.0 ** e for e in np.arange(-3, 3.01, 0.5)]
    preds_full, preds_dose, preds_loc, preds_kloc, lam_used = {"type": type_pred[ev]}, {}, {}, {}, {}
    # E047 amendment: a LOCAL mode with no fitted extrapolation -- each cell minus the leave-one-out Gaussian (ell
    # 150 km) average of its OWN set (training cells among training cells, held-out among held-out), applied after
    # the type residualization to features, targets, predictions' inputs and both truth halves alike
    kl_tr, kl_ev = loo_local_op(cells[tc], a.kloc_km), loo_local_op(cells[ev], a.kloc_km)
    dZk_tr = kl_tr(dZ_res[tc])
    for fn, fs in FAMS.items():
        sel = np.ones(len(colidx), bool) if fs is None else np.isin(famX, fs)
        if sel.sum() == 0:
            continue
        B, xm, ym, lam = ridge_cv(dX[tc][:, sel], dZ[tc], blk_tc, lams)
        preds_full[f"full:{fn}"] = (dX[ev][:, sel] - xm) @ B + ym
        Bd, xmd, ymd, lamd = ridge_cv(dX_res[tc][:, sel], dZ_res[tc], blk_tc, lams)
        preds_dose[f"dose:{fn}"] = (dX_res[ev][:, sel] - xmd) @ Bd + ymd
        Bl, xml, yml, laml = ridge_cv(dX_loc[tc][:, sel], dZ_loc[tc], blk_tc, lams)
        preds_loc[f"local:{fn}"] = (dX_loc[ev][:, sel] - xml) @ Bl + yml
        Bk, xmk, ymk, lamk = ridge_cv(kl_tr(dX_res[tc][:, sel]), dZk_tr, blk_tc, lams)
        preds_kloc[f"kloc:{fn}"] = (kl_ev(dX_res[ev][:, sel]) - xmk) @ Bk + ymk
        lam_used[fn] = {"full": lam, "dose": lamd, "local": laml, "kloc": lamk, "n_channels": int(sel.sum())}
    # grading
    Ta_e, Tb_e = Ta[ev], Tb[ev]
    Ta_r, Tb_r = Ta_e - type_pred[ev], Tb_e - type_pred[ev]
    blk = (cells[ev, 0] // 6) * 100000 + cells[ev, 1] // 6
    ub, binv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(11)
    W = np.vstack([np.ones(len(ev)), rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)[:, binv]])

    def corr(D, A_, B_, w):
        vp, vt = wc(D, D, w).sum(), wc(A_, B_, w).sum()
        cv = 0.5 * (wc(D, A_, w).sum() + wc(D, B_, w).sum())
        return cv / np.sqrt(vp * vt) if vp > 0 and vt > 0 else np.nan
    res = {"cache": a.cache, "n_heldout": int(len(ev)), "n_train_cells": int(len(tc)),
           "share_of_dX_variance_typical_for_type": float(share_type_X),
           "truth_noise_free_share_typical_for_type": float(1 - wc(Ta_r, Tb_r, W[0]).sum() / wc(Ta_e, Tb_e, W[0]).sum()),
           "truth_noise_free_share_typical_for_type_and_region": None,
           "lambda": lam_used, "vs_truth": {}, "vs_within_type_truth": {}, "vs_local_truth": {}}
    for nm, D in {**preds_full, **preds_dose}.items():
        g = np.array([corr(D, Ta_e, Tb_e, w) for w in W])
        res["vs_truth"][nm] = {"corr": float(g[0]), "ci": [float(np.nanpercentile(g[1:], q)) for q in (2.5, 97.5)]}
    for nm, D in preds_dose.items():
        g = np.array([corr(D, Ta_r, Tb_r, w) for w in W])
        res["vs_within_type_truth"][nm] = {"corr": float(g[0]),
                                           "ci": [float(np.nanpercentile(g[1:], q)) for q in (2.5, 97.5)]}
    Ta_l, Tb_l = Ta_e - loc_pred[ev], Tb_e - loc_pred[ev]
    res["truth_noise_free_share_typical_for_type_and_region"] = float(
        1 - wc(Ta_l, Tb_l, W[0]).sum() / wc(Ta_e, Tb_e, W[0]).sum())
    for nm, D in preds_loc.items():
        g = np.array([corr(D, Ta_l, Tb_l, w) for w in W])
        res["vs_local_truth"][nm] = {"corr": float(g[0]), "ci": [float(np.nanpercentile(g[1:], q)) for q in (2.5, 97.5)]}
    Ta_k, Tb_k = kl_ev(Ta_r), kl_ev(Tb_r)
    res["truth_noise_free_share_kernel_local"] = float(wc(Ta_k, Tb_k, W[0]).sum() / wc(Ta_e, Tb_e, W[0]).sum())
    res["vs_kernel_local_truth"] = {}
    for nm, D in preds_kloc.items():
        g = np.array([corr(D, Ta_k, Tb_k, w) for w in W])
        res["vs_kernel_local_truth"][nm] = {"corr": float(g[0]),
                                            "ci": [float(np.nanpercentile(g[1:], q)) for q in (2.5, 97.5)]}
    res["kloc_km"], res["conus_only"] = a.kloc_km, bool(a.conus_only)
    with open(os.path.join(a.out, "type_dose.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "type_dose.json"), encoding="utf-8"))
    print(f"{os.path.basename(os.path.normpath(r['cache']))}: {r['n_heldout']} held-out / {r['n_train_cells']} training "
          f"cells; share of covariate-change variance typical for the type {r['share_of_dX_variance_typical_for_type']:.2f}; "
          f"of the truth's noise-free change {r['truth_noise_free_share_typical_for_type']:.2f}")
    for nm, v in r["vs_truth"].items():
        print(f"  vs truth          {nm:16s} {v['corr']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")
    for nm, v in r["vs_within_type_truth"].items():
        lam = r["lambda"][nm.split(":")[1]]
        print(f"  vs WITHIN-TYPE    {nm:16s} {v['corr']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]  "
              f"({lam['n_channels']} ch, lambda {lam['dose']:.3g})")
    if r.get("vs_local_truth"):
        print(f"  (truth's noise-free change typical for type AND region: "
              f"{r['truth_noise_free_share_typical_for_type_and_region']:.2f})")
    for nm, v in r.get("vs_local_truth", {}).items():
        print(f"  vs WITHIN-TYPE+REGION {nm:13s} {v['corr']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")
    if r.get("vs_kernel_local_truth"):
        print(f"  (kernel-local, LOO {r['kloc_km']:.0f} km within each set; conus_only {r['conus_only']}; truth's "
              f"noise-free change left {r['truth_noise_free_share_kernel_local']:.2f})")
    for nm, v in r.get("vs_kernel_local_truth", {}).items():
        print(f"  vs WITHIN-TYPE, KERNEL-LOCAL {nm:12s} {v['corr']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
