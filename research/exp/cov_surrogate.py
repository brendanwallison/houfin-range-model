"""E016 (T4, rung 1): do covariate -> community coefficients learned from WITHIN-cell variation transfer to the
withheld decades better than spatial ones? A linear surrogate of DESK, fitted inside the trained era only.

    python research/exp/cov_surrogate.py --cache <dir> --out <dir> [--population trained|withheld] [--rank 24]
    python research/exp/cov_surrogate.py --summarize <out>

Rows: TRAINING rows (training cells x trained years; split.npz is_train). Target: the observed community's ESK z
per cell-year (esk_annual.npy, first r). Covariates: the 302 channels at each key (F_cov), standardized and reduced
to 64 PCs on training rows. Maps (one BLR per component, research/lib/blr, an iid block per feature group):
    pooled   z ~ cov                               one set of coefficients, mostly SPATIAL (as DESK is trained)
    mundlak  z ~ [cell mean cov, cov - cell mean]  between- (beta_b) and within-cell (beta_w) blocks (T4)
A held-out cell's predicted early -> modern change: pooled beta . dcov; within beta_w . dcov; between beta_b . dcov
(the cell-mean block cancels in a change, so "mundlak" change = within); DESK's own change for reference. Grade:
noise-free CENTERED correlation with the true ESK change (half covariance as the true variance), 6x6-block bootstrap,
paired differences on the same draws. within > between = space-for-time mismatch with temporal coefficients that
TRANSFER (T4, and a temporal block worth having); within ~ between ~ DESK = the covariates' change cannot be read
better linearly either.
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
from lib import blr  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--population", default="trained", choices=("trained", "withheld"))
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--n-pcs", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from change_oracle import group_rows
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    split = np.load(c("split.npz"))
    withheld = [int(y) for y in split["withheld"]]
    E, full, _, _ = group_rows(a.cache, keys, withheld)
    r = a.rank
    is_train = split["is_train"].astype(bool)
    trained_year = ~np.isin(keys[:, 2], withheld)

    # covariates: standardize + PCA on training rows
    F = np.asarray(np.load(c("F_cov.npy"), mmap_mode="r"), "float64")
    mu, sd = F[is_train].mean(0), F[is_train].std(0) + 1e-9
    F = (F - mu) / sd
    _, _, Vt = np.linalg.svd(F[is_train][:: max(1, is_train.sum() // 40000)], full_matrices=False)
    P = F @ Vt[: a.n_pcs].T
    # cell means over each cell's TRAINED-YEAR rows (known for held-out cells too: covariates are everywhere)
    cell_id = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    uc, inv = np.unique(cell_id, return_inverse=True)
    w = trained_year.astype(float)
    sums = np.zeros((len(uc), P.shape[1]))
    np.add.at(sums, inv, P * w[:, None])
    cnt = np.bincount(inv, weights=w, minlength=len(uc))
    Pbar = (sums / np.maximum(cnt, 1)[:, None])[inv]
    Z = np.load(c("esk_annual.npy"))[:, :r].astype("float64")

    k = P.shape[1]
    m_pool = blr.fit(P[is_train], Z[is_train], [(0, k)])
    Xm = np.hstack([Pbar, P - Pbar])
    m_mund = blr.fit(Xm[is_train], Z[is_train], [(0, k), (k, 2 * k)])
    beta_pool = m_pool["coef"].T                        # (k, r)
    beta_b, beta_w = m_mund["coef"][:, :k].T, m_mund["coef"][:, k:].T

    # held-out cell pairs for the population
    want_wh = a.population == "withheld"
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh) == want_wh:
            idx.setdefault(cell, [None, None])[0] = j
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = j
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    cells = E["cells"][pairs[:, 0]]
    ev = np.where(split["holdout"][cells[:, 0], cells[:, 1]])[0]
    gm = lambda A, gs: np.stack([A[np.asarray(g)].mean(0) for g in gs])
    Pg = gm(P, full)
    dP = (Pg[pairs[:, 1]] - Pg[pairs[:, 0]])[ev]
    Zd = gm(np.asarray(np.load(c("z_raw.npy"), mmap_mode="r")[:, :r], "float64"), full)
    preds = {"pooled": dP @ beta_pool, "within": dP @ beta_w, "between": dP @ beta_b,
             "desk": (Zd[pairs[:, 1]] - Zd[pairs[:, 0]])[ev]}
    dd = {h: (E[f"z_{h}"][pairs[:, 1], :r] - E[f"z_{h}"][pairs[:, 0], :r]).astype("float64")[ev]
          for h in ("full", "a", "b")}
    blocks = (cells[ev, 0] // 6) * 100000 + cells[ev, 1] // 6
    ub, binv = np.unique(blocks, return_inverse=True)
    cen = lambda v, w_: v - (w_[:, None] * v).sum(0) / w_.sum()

    def corr(pred, w_):
        p, f_, a_, b_ = cen(pred, w_), cen(dd["full"], w_), cen(dd["a"], w_), cen(dd["b"], w_)
        num, vp, vt = (w_[:, None] * p * f_).sum(), (w_[:, None] * p * p).sum(), (w_[:, None] * a_ * b_).sum()
        return float(num / np.sqrt(vp * vt)) if vp > 0 and vt > 0 else np.nan

    def slope(pred, w_):        # truth on prediction: the calibration factor (1 = calibrated)
        p, f_ = cen(pred, w_), cen(dd["full"], w_)
        return float((w_[:, None] * p * f_).sum() / (w_[:, None] * p * p).sum())

    rng = np.random.default_rng(0)
    W = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)
    ones = np.ones(len(ev))
    res = {"cache": a.cache, "population": a.population, "rank": r, "n_pcs": k,
           "n_train_rows": int(is_train.sum()), "n_eval_cells": int(len(ev)),
           "amplitudes_mundlak": [float(np.median(m_mund["a"][:, 0])), float(np.median(m_mund["a"][:, 1]))],
           "arms": {}}
    for name, p in preds.items():
        b = np.array([corr(p, w_[binv].astype(float)) for w_ in W])
        b = b[np.isfinite(b)]
        res["arms"][name] = {"corr": corr(p, ones), "ci": [float(np.percentile(b, 2.5)),
                                                           float(np.percentile(b, 97.5))],
                             "truth_on_pred": slope(p, ones)}
    for A_, B_ in (("within", "between"), ("within", "desk"), ("pooled", "desk"), ("within", "pooled")):
        b = np.array([corr(preds[A_], w_[binv].astype(float)) - corr(preds[B_], w_[binv].astype(float))
                      for w_ in W])
        b = b[np.isfinite(b)]
        res["arms"][f"{A_} - {B_}"] = {"corr": res["arms"][A_]["corr"] - res["arms"][B_]["corr"],
                                       "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    with open(os.path.join(a.out, "cov_surrogate.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "cov_surrogate.json"), encoding="utf-8"))
    print(f"cov surrogate [{r['population']}] r{r['rank']}: linear covariate->ESK maps fitted on {r['n_train_rows']:,} "
          f"training cell-years; {r['n_eval_cells']} held-out cells' change; Mundlak amplitudes between/within "
          f"{r['amplitudes_mundlak'][0]:.3g}/{r['amplitudes_mundlak'][1]:.3g}")
    for k, v in r["arms"].items():
        cal = f"  truth-on-pred {v['truth_on_pred']:.2f}" if "truth_on_pred" in v else ""
        print(f"  {k:18s} corr {v['corr']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]{cal}")


if __name__ == "__main__":
    main()
