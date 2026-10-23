"""E045: does a covariate-CHANGE -> community-CHANGE map, fitted across places on change WITHIN the trained era,
transfer to the withheld decades? (a rung-1 surrogate for change-supervised representations; from E044d)

    python research/exp/change_map.py --cache <tempho research cache> --out <dir> --w1 1996 2005 --w2 2016 2025
    python research/exp/change_map.py --summarize <dir>

Training cells (not held out, not buffer) with surveyed rows in both trained-era windows w1 and w2: dX = mean
standardized covariates (F_cov_raw) in w2 - in w1; dZ = mean observed ESK (esk_annual, r24) in w2 - in w1. No
withheld year is used. Ridge maps (lambda by 6x6-block CV), applied to HELD-OUT cells' backcast covariate change
(early withheld epoch 1966-86 vs 2005-25; the E039h estimand) and graded by noise-free centred corr, paired 6x6-block
bootstrap against DESK's raw change on the same cells:
    chg:<fam>     dX[fam] -> dZ                (cross-place change supervision)
    type          [1, type] -> dZ              (what "this kind of place" did in the trained era)
    type+chg:all  [type, dX] -> dZ
Families: all, land (LUH classes + management + HYDE + BUI), luh_class, climate.
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
from type_dose import families, ridge_cv, wc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--w1", type=int, nargs=2, default=[1996, 2005])
    ap.add_argument("--w2", type=int, nargs=2, default=[2016, 2025])
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--n-type", type=int, default=32)
    ap.add_argument("--min-rows", type=int, default=2, help="surveyed rows required in each window")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from scipy import sparse
    from change_oracle import desk_epoch_z, group_rows
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    N = len(keys)
    split = np.load(c("split.npz"))
    withheld = [int(y) for y in split["withheld"]]
    assert a.w1[0] > max(withheld) and a.w2[0] > max(withheld), "windows must lie in trained years"
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
    ev = np.where(split["holdout"][cells[:, 0], cells[:, 1]])[0]
    Ta = (E["z_a"][pairs[:, 1], :r] - E["z_a"][pairs[:, 0], :r]).astype("float64")[ev]
    Tb = (E["z_b"][pairs[:, 1], :r] - E["z_b"][pairs[:, 0], :r]).astype("float64")[ev]
    early_rows = [np.asarray(full[pairs[j, 0]]) for j in ev]
    mod_rows = [np.asarray(full[pairs[j, 1]]) for j in ev]

    def gmat(groups, n_rows):
        rows = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
        cols = np.concatenate([np.asarray(g, int) for g in groups])
        vals = np.concatenate([np.full(len(g), 1.0 / len(g)) for g in groups])
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(groups), n_rows))
    F = np.asarray(np.load(c("F_cov_raw.npy"), mmap_mode="r"), "float64")
    Eann = np.load(c("esk_annual.npy")).astype("float64")[:, :r]
    ho = split["holdout"][keys[:, 0], keys[:, 1]]
    bf = split["buffer"][keys[:, 0], keys[:, 1]]
    ty = ~np.isin(keys[:, 2], withheld)
    fin = np.isfinite(F).all(1) & np.isfinite(Eann).all(1)
    trn = np.where(~ho & ~bf & ty & fin)[0]
    mu, sd = F[trn].mean(0), F[trn].std(0)
    keep = sd > 1e-6
    X = np.where(np.isfinite(F[:, keep]), (F[:, keep] - mu[keep]) / sd[keep], 0.0)
    famX = families(F.shape[1])[np.where(keep)[0]]
    # training cells' trained-era change
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    yr = keys[:, 2]
    in1 = (yr >= a.w1[0]) & (yr <= a.w1[1]) & ~ho & ~bf & fin
    in2 = (yr >= a.w2[0]) & (yr <= a.w2[1]) & ~ho & ~bf & fin
    c1, n1 = np.unique(cid[in1], return_counts=True)
    c2, n2 = np.unique(cid[in2], return_counts=True)
    both = np.intersect1d(c1[n1 >= a.min_rows], c2[n2 >= a.min_rows])
    g1 = [np.where(in1 & (cid == cc))[0] for cc in both]
    g2 = [np.where(in2 & (cid == cc))[0] for cc in both]
    G1, G2 = gmat(g1, N), gmat(g2, N)
    dX_tr = np.asarray(G2 @ X) - np.asarray(G1 @ X)
    dZ_tr = np.asarray(G2 @ Eann) - np.asarray(G1 @ Eann)
    _, _, Vt = np.linalg.svd(X[trn][:: max(1, len(trn) // 40000)] - X[trn].mean(0), full_matrices=False)
    V = Vt[:a.n_type].T
    lev_tr = np.asarray(G2 @ X) @ V
    tr_rc = np.stack([both // 100000, both % 100000], 1)
    blk_tr = (tr_rc[:, 0] // 6) * 100000 + tr_rc[:, 1] // 6
    # held-out backcast change (same rows as E039h)
    Ge, Gm = gmat(early_rows, N), gmat(mod_rows, N)
    dX_ev = np.asarray(Gm @ X) - np.asarray(Ge @ X)
    lev_ev = np.asarray(Gm @ X) @ V
    zr_all = np.load(c("z_raw_all.npy"), mmap_mode="r")
    ci_, y0 = np.load(c("key_cell_index.npy")), int(np.load(c("years.npy"))[0])
    D_desk = desk_epoch_z(zr_all, ci_, y0, keys, mod_rows, r) - desk_epoch_z(zr_all, ci_, y0, keys, early_rows, r)
    lams = [10.0 ** e for e in np.arange(-3, 3.01, 0.5)]
    preds, lam_used = {"desk": D_desk}, {}
    FAMS = {"all": None, "land": ("luh_class", "luh_mgmt", "hyde", "bui"), "luh_class": ("luh_class",),
            "climate": ("climate",)}
    for fn, fs in FAMS.items():
        sel = np.ones(len(famX), bool) if fs is None else np.isin(famX, fs)
        B, xm, ym, lam = ridge_cv(dX_tr[:, sel], dZ_tr, blk_tr, lams)
        preds[f"chg:{fn}"] = (dX_ev[:, sel] - xm) @ B + ym
        lam_used[f"chg:{fn}"] = lam
    Tt = np.hstack([np.ones((len(both), 1)), lev_tr])
    gt, *_ = np.linalg.lstsq(Tt, dZ_tr, rcond=None)
    preds["type"] = np.hstack([np.ones((len(ev), 1)), lev_ev]) @ gt
    B, xm, ym, lam = ridge_cv(np.hstack([lev_tr, dX_tr]), dZ_tr, blk_tr, lams)
    preds["type+chg:all"] = (np.hstack([lev_ev, dX_ev]) - xm) @ B + ym
    lam_used["type+chg:all"] = lam
    blk = (cells[ev, 0] // 6) * 100000 + cells[ev, 1] // 6
    ub, binv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(11)
    W = np.vstack([np.ones(len(ev)), rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)[:, binv]])

    def corr(D, w):
        vp, vt = wc(D, D, w).sum(), wc(Ta, Tb, w).sum()
        cv = 0.5 * (wc(D, Ta, w).sum() + wc(D, Tb, w).sum())
        return cv / np.sqrt(vp * vt) if vp > 0 and vt > 0 else np.nan
    boots = {k: np.array([corr(D, w) for w in W]) for k, D in preds.items()}
    res = {"cache": a.cache, "windows": [a.w1, a.w2], "n_train_cells": int(len(both)), "n_heldout": int(len(ev)),
           "lambda": lam_used, "arms": {}}
    for k, v in boots.items():
        o = {"corr": float(v[0]), "ci": [float(np.nanpercentile(v[1:], q)) for q in (2.5, 97.5)]}
        if k != "desk":
            d = v - boots["desk"]
            o["minus_desk"] = float(d[0])
            o["minus_desk_ci"] = [float(np.nanpercentile(d[1:], q)) for q in (2.5, 97.5)]
        res["arms"][k] = o
    with open(os.path.join(a.out, "change_map.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "change_map.json"), encoding="utf-8"))
    print(f"{os.path.basename(os.path.normpath(r['cache']))}: maps fitted on {r['n_train_cells']} training cells' change "
          f"{r['windows'][0]} -> {r['windows'][1]}; graded on {r['n_heldout']} held-out cells' backcast change")
    for k, v in r["arms"].items():
        extra = (f" | - desk {v['minus_desk']:+.3f} [{v['minus_desk_ci'][0]:+.3f},{v['minus_desk_ci'][1]:+.3f}]"
                 if "minus_desk" in v else "")
        print(f"  {k:14s} {v['corr']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]" + extra)


if __name__ == "__main__":
    main()
