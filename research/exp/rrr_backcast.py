"""E039h: does DESK's network add anything in TIME over a bird-supervised linear map of its covariates?

    python research/exp/rrr_backcast.py --cache <tempho research cache> --out <dir> [--rank 24]
    python research/exp/rrr_backcast.py --summarize <dir>

On a model with withheld decades, held-out cells' community change between the early withheld epoch (1966-86) and the
modern epoch (2005-25) -- the backcast_attribution / E035 estimand: noise-free centred corr with the observed change
(ESK r24, ABBA halves) and k, the slope of truth on prediction. Predictions compared on the same cells:
  desk         raw z window means (z_raw_all), first `rank` dims
  linear       ESK coordinates predicted by a ridge-regularized linear map of the standardized raw-matched covariates,
               fitted on TRAINING cells x TRAINED years (DESK's own supervision rows), first `rank` coordinates
  linear_rrr   the same map reduced to rank `rank` (reduced-rank regression), first `rank` coordinates
Paired 6x6-block bootstrap of desk - linear for corr. A linear map fitted across places is the pure space-for-time
predictor; E016 found such maps predict none of the decadal change.
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
    return float((w[:, None] * (x - xm) * (y - ym)).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from change_oracle import desk_epoch_z, group_rows
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    split = np.load(c("split.npz"))
    withheld = [int(y) for y in split["withheld"]]
    r = a.rank
    E, full, half_a, half_b = group_rows(a.cache, keys, withheld)
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh):
            idx.setdefault(cell, [None, None])[0] = j
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = j
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    cells = E["cells"][pairs[:, 0]]
    held = split["holdout"][cells[:, 0], cells[:, 1]]
    ev = np.where(held)[0]
    Za, Zb = E["z_a"][:, :r].astype("float64"), E["z_b"][:, :r].astype("float64")
    Ta = (Za[pairs[:, 1]] - Za[pairs[:, 0]])[ev]
    Tb = (Zb[pairs[:, 1]] - Zb[pairs[:, 0]])[ev]
    # desk
    zr = np.load(c("z_raw_all.npy"), mmap_mode="r")
    ci, y0 = np.load(c("key_cell_index.npy")), int(np.load(c("years.npy"))[0])
    early_rows = [np.asarray(full[pairs[j, 0]]) for j in ev]
    mod_rows = [np.asarray(full[pairs[j, 1]]) for j in ev]
    D_desk = desk_epoch_z(zr, ci, y0, keys, mod_rows, r) - desk_epoch_z(zr, ci, y0, keys, early_rows, r)
    # linear map: standardized raw-matched covariates -> observed ESK coordinates, DESK's supervision rows
    F = np.asarray(np.load(c("F_cov_raw.npy"), mmap_mode="r"), "float64")
    Eann = np.load(c("esk_annual.npy")).astype("float64")
    yr = keys[:, 2]
    ho = split["holdout"][keys[:, 0], keys[:, 1]]
    bf = split["buffer"][keys[:, 0], keys[:, 1]]
    trn = np.where(~ho & ~bf & ~np.isin(yr, withheld) & np.isfinite(F).all(1) & np.isfinite(Eann).all(1))[0]
    mu, sd = F[trn].mean(0), F[trn].std(0)
    keep = sd > 1e-6
    S = lambda rows: np.where(np.isfinite(F[rows][:, keep]), (F[rows][:, keep] - mu[keep]) / sd[keep], 0.0)
    X = S(trn)
    Yt = Eann[trn, :64]
    xm, ym = X.mean(0), Yt.mean(0)
    Xc, Yc = X - xm, Yt - ym
    lam = 1e-3 * np.trace(Xc.T @ Xc) / Xc.shape[1]
    B = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Yc)
    _, _, Vt = np.linalg.svd(Xc @ B, full_matrices=False)
    Brr = B @ Vt[:r].T @ Vt[:r]                                    # rank-r reduced-rank map
    wmean = lambda rows_list: np.stack([S(np.asarray(g)).mean(0) for g in rows_list])
    dX = wmean(mod_rows) - wmean(early_rows)
    preds = {"desk": D_desk, "linear": (dX @ B)[:, :r], "linear_rrr": (dX @ Brr)[:, :r]}
    blk = (cells[ev, 0] // 6) * 100000 + cells[ev, 1] // 6
    ub, inv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(11)
    W = np.vstack([np.ones(len(ev)), rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)[:, inv]])
    def corr_k(D, w):
        vp, vt = wc(D, D, w), wc(Ta, Tb, w)
        cv = 0.5 * (wc(D, Ta, w) + wc(D, Tb, w))
        return (cv / np.sqrt(vp * vt) if vp > 0 and vt > 0 else np.nan), (cv / vp if vp > 0 else np.nan)
    res = {"cache": a.cache, "n_cells": int(len(ev)), "rank": r, "withheld": [min(withheld), max(withheld)],
           "n_fit_rows": int(len(trn)), "fit_r2": float(1 - ((Yc - Xc @ B) ** 2).sum() / (Yc ** 2).sum()),
           "models": {}}
    boots = {}
    for name, D in preds.items():
        ck = np.array([corr_k(D, w) for w in W])
        boots[name] = ck[:, 0]
        res["models"][name] = {"corr": float(ck[0, 0]), "k": float(ck[0, 1]),
                               "corr_ci": [float(np.nanpercentile(ck[1:, 0], 2.5)), float(np.nanpercentile(ck[1:, 0], 97.5))],
                               "size": float(np.linalg.norm(D, axis=1).mean())}
    for name in ("linear", "linear_rrr"):
        d = boots["desk"] - boots[name]
        res["models"][name]["desk_minus_corr"] = float(d[0])
        res["models"][name]["desk_minus_ci"] = [float(np.nanpercentile(d[1:], 2.5)), float(np.nanpercentile(d[1:], 97.5))]
    with open(os.path.join(a.out, "rrr_backcast.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "rrr_backcast.json"), encoding="utf-8"))
    print(f"{os.path.basename(os.path.normpath(r['cache']))}: {r['n_cells']} held-out cells, withheld {r['withheld']}, "
          f"r{r['rank']}; linear map fitted on {r['n_fit_rows']:,} training rows (R^2 {r['fit_r2']:.3f})")
    for name, v in r["models"].items():
        extra = (f" | desk - {name} {v['desk_minus_corr']:+.3f} [{v['desk_minus_ci'][0]:+.3f},{v['desk_minus_ci'][1]:+.3f}]"
                 if "desk_minus_corr" in v else "")
        print(f"  {name:11s} corr {v['corr']:+.3f} [{v['corr_ci'][0]:+.3f},{v['corr_ci'][1]:+.3f}]  k {v['k']:.2f}  "
              f"size {v['size']:.3f}" + extra)


if __name__ == "__main__":
    main()
