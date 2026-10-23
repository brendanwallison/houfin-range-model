"""E068b: are early CBC lists notable-first? Inclusion-vs-list-length slopes per species, early vs modern, same circles.

    python research/exp/e068b_reporting_slopes.py --data cbc_circles_v2.npz --completeness completeness_v3.npz --out DIR

Per era (< 1940; >= 1966), in circles counted in both eras: x = log list length centred within circle and era; y_s =
species s recorded (count-day > 0). Linear-probability slope cov(x, y_s) / var(x) from per-circle sufficient statistics,
put on the logit scale by dividing by pbar_s (1 - pbar_s). d_s = slope_early - slope_modern; Spearman of d_s with modern
prevalence over species with >= 30 early inclusions. Bootstrap over circles (multinomial weights on the per-circle
statistics, 500 resamples).
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True); ap.add_argument("--completeness", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    D = np.load(a.data, allow_pickle=False)
    Y, yr, ci = (D["Y"] > 0).astype(float), D["year_end"], D["circle_idx"]
    L = np.load(a.completeness, allow_pickle=False)["list_len"]
    prev = Y[yr >= 1966].mean(0)
    early, modern = (yr < 1940) & (L > 0), (yr >= 1966) & (L > 0)
    both = np.intersect1d(np.unique(ci[early]), np.unique(ci[modern]))
    nC = len(both); cidx = {c: i for i, c in enumerate(both)}

    def stats(mask):
        rows = np.where(mask & np.isin(ci, both))[0]
        k = np.array([cidx[c] for c in ci[rows]])
        x = np.log(L[rows])
        xm = np.bincount(k, x, nC) / np.maximum(np.bincount(k, minlength=nC), 1)
        xc = x - xm[k]
        S = Y.shape[1]
        n = np.bincount(k, minlength=nC).astype(float)
        sxx = np.bincount(k, xc ** 2, nC)
        sy = np.zeros((nC, S)); sxy = np.zeros((nC, S))
        np.add.at(sy, k, Y[rows]); np.add.at(sxy, k, xc[:, None] * Y[rows])
        return n, sxx, sy, sxy

    def slopes(st, w):
        n, sxx, sy, sxy = st
        N = (w * n).sum(); p = (w[:, None] * sy).sum(0) / N
        b = (w[:, None] * sxy).sum(0) / (w * sxx).sum()
        return b / np.clip(p * (1 - p), 1e-4, None), p, (w[:, None] * sy).sum(0)

    se, sm = stats(early), stats(modern)
    w1 = np.ones(nC)
    be, pe, inc_e = slopes(se, w1); bm, pm, _ = slopes(sm, w1)
    keep = inc_e >= 30
    d = be - bm
    rho = float(spearmanr(d[keep], prev[keep]).correlation)
    rng = np.random.default_rng(68)
    boots = []
    for _ in range(a.boot):
        w = rng.multinomial(nC, np.full(nC, 1 / nC)).astype(float)
        b1, _, _ = slopes(se, w); b2, _, _ = slopes(sm, w)
        boots.append(spearmanr((b1 - b2)[keep], prev[keep]).correlation)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    res = {"circles": int(nC), "early_lists": int(se[0].sum()), "modern_lists": int(sm[0].sum()),
           "species": int(keep.sum()), "spearman_d_prevalence": rho, "ci95": [float(lo), float(hi)],
           "median_slope_early": float(np.median(be[keep])), "median_slope_modern": float(np.median(bm[keep])),
           "spearman_slope_prevalence_early": float(spearmanr(be[keep], prev[keep]).correlation),
           "spearman_slope_prevalence_modern": float(spearmanr(bm[keep], prev[keep]).correlation)}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e068b.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
