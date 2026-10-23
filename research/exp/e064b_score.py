"""E064b: held-out Spearman on post-1966 rows for the species scored before 1940, full vs density-thinned counters fits.

    python research/exp/e064b_score.py --data cbc_circles_v2.npz --fit-full fit_counters.npz --fit-thin fit_thin.npz --out DIR
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr


def era_scores(D, fit, y0, y1, species_idx=None, min_nz=20):
    Y, yr = D["Y"], D["year_end"]
    rows, pred = fit["hold_rows"], fit["hold_pred"]
    m = (yr[rows] >= y0) & (yr[rows] <= y1)
    out = {}
    for s in (range(Y.shape[1]) if species_idx is None else species_idx):
        obs = Y[rows[m], s]
        if (obs > 0).sum() >= min_nz:
            out[int(s)] = float(spearmanr(pred[m, s], obs).correlation)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data"); ap.add_argument("--fit-full"); ap.add_argument("--fit-thin"); ap.add_argument("--out")
    a = ap.parse_args()
    D = np.load(a.data, allow_pickle=False)
    full, thin = np.load(a.fit_full, allow_pickle=False), np.load(a.fit_thin, allow_pickle=False)
    pre = era_scores(D, full, 1901, 1939)
    sp = sorted(pre)
    post_full = era_scores(D, full, 1966, 2025, sp, min_nz=1)
    post_thin = era_scores(D, thin, 1966, 2025, sp, min_nz=1)
    common = [s for s in sp if s in post_full and s in post_thin]
    med = lambda d: float(np.nanmedian([d[s] for s in common]))
    res = {"species": len(common), "pre1940_full": med(pre), "post1966_full_same_species": med(post_full),
           "post1966_thinned_same_species": med(post_thin)}
    res["gap_thinned_minus_pre1940"] = res["post1966_thinned_same_species"] - res["pre1940_full"]
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e064b.json"), "w"), indent=1)
    for k, v in res.items():
        print(f"  {k:34s} {v:.3f}" if isinstance(v, float) else f"  {k:34s} {v}")


if __name__ == "__main__":
    main()
