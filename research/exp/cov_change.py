"""E015: is a cell's 40-year community change predictable from its COVARIATES' change at all -- and does DESK
extract what is there? (T5 vs T1/T3/T4.)

    python research/exp/cov_change.py --cache <dir> --out <dir> [--population trained|withheld] [--rank 24]
    python research/exp/cov_change.py --summarize <out>

Target: the observed community's ESK change (early 1966-1986 -> modern 2005-2025) on held-out cells, first r
components. Every arm is a map fitted on TRAINING cells' change and applied to held-out cells, except DESK (used
as is). Graded by the noise-free, CENTERED (place-specific) correlation with the true change, pooled over
components: cov(pred, dz_full) / sqrt(var(pred) * cov(dz_A, dz_B)), the halves' covariance being the true change's
variance. Arms:
    dcov        covariate change (epoch means of the 302 covariate channels, as the covariate GPs see them)
    dcov_raw    the same on un-smoothed covariates
    xy          position only (random Fourier features, 300 km): the spatial-interpolation placebo
    dcov+xy     both (two blocks)
    cov_static  the covariates' MODERN level (where, not what changed)
    desk        DESK's own change, unfitted (raw z epoch means over the surveyed years)
dcov ~ xy -> covariate change carries nothing beyond where a cell is (T5: the drivers are not in the covariates).
dcov > desk -> the information is in the covariates and DESK does not extract it (T1/T3/T4).
Fitted maps are per-component BLRs with one iid block per feature group (research/lib/blr).
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


def group_means(A, groups):
    return np.stack([np.asarray(A[np.asarray(g)], "float64").mean(0) for g in groups])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--population", default="trained", choices=("trained", "withheld"))
    ap.add_argument("--rank", type=int, default=24)
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
    E, full, _, _ = group_rows(a.cache, keys, [int(y) for y in split["withheld"]])
    want_wh = a.population == "withheld"
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh) == want_wh:
            idx.setdefault(cell, [None, None])[0] = j
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = j
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    cells = E["cells"][pairs[:, 0]]
    held = split["holdout"][cells[:, 0], cells[:, 1]]
    buf = split["buffer"][cells[:, 0], cells[:, 1]]
    tr, ev = np.where(~held & ~buf)[0], np.where(held)[0]
    r = a.rank
    d = {h: (E[f"z_{h}"][pairs[:, 1], :r] - E[f"z_{h}"][pairs[:, 0], :r]).astype("float64")
         for h in ("full", "a", "b")}

    feats = {}
    def pcs(M, k=64):
        """Top-k principal components of M, fitted on the training cells: 302 channels are collinear
        (rank ~255) and outnumber nothing useful at ~1400 training cells."""
        mu = M[tr].mean(0)
        _, _, Vt = np.linalg.svd(M[tr] - mu, full_matrices=False)
        return (M - mu) @ Vt[:k].T
    for name, f in (("dcov", "F_cov.npy"), ("dcov_raw", "F_cov_raw.npy")):
        G = group_means(np.load(c(f), mmap_mode="r"), full)
        mu, sd = G.mean(0), G.std(0) + 1e-9
        G = (G - mu) / sd
        feats[name] = [pcs(G[pairs[:, 1]] - G[pairs[:, 0]])]
        if name == "dcov":
            feats["cov_static"] = [pcs(G[pairs[:, 1]])]
    rng = np.random.default_rng(0)
    xy = cells.astype("float64") * 27.0
    Rxy = np.sqrt(2.0 / r) * np.cos(xy @ (rng.normal(size=(2, r)) / 300.0) + rng.uniform(0, 2 * np.pi, r))
    feats["xy"] = [Rxy]
    feats["dcov+xy"] = [feats["dcov"][0], Rxy]
    zr = np.load(c("z_raw.npy"), mmap_mode="r")
    Zd = group_means(zr, full)[:, :r]
    desk = Zd[pairs[:, 1]] - Zd[pairs[:, 0]]

    preds = {"desk": desk[ev]}
    for name, blocks in feats.items():
        X = np.hstack(blocks)
        bl, p = [], 0
        for B in blocks:
            bl.append((p, p + B.shape[1]))
            p += B.shape[1]
        m = blr.fit(X[tr], d["b"][tr], bl)                  # fit on half B: no shared noise with grading
        preds[name] = blr.predict(m, X[ev])[0]

    dA, dB, dF = d["a"][ev], d["b"][ev], d["full"][ev]
    blocks = (cells[ev, 0] // 6) * 100000 + cells[ev, 1] // 6
    ub, inv = np.unique(blocks, return_inverse=True)
    cen = lambda v, w: v - (w[:, None] * v).sum(0) / w.sum()

    def corr(pred, w):
        p, f_, a_, b_ = cen(pred, w), cen(dF, w), cen(dA, w), cen(dB, w)
        num = (w[:, None] * p * f_).sum()
        vp = (w[:, None] * p * p).sum()
        vt = (w[:, None] * a_ * b_).sum()
        return float(num / np.sqrt(vp * vt)) if vp > 0 and vt > 0 else np.nan

    ones = np.ones(len(ev))
    res = {"cache": a.cache, "population": a.population, "rank": r, "n_train": int(len(tr)),
           "n_eval": int(len(ev)), "arms": {}}
    W = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)
    for name, pred in preds.items():
        boots = np.array([corr(pred, w[inv].astype(float)) for w in W])
        boots = boots[np.isfinite(boots)]
        res["arms"][name] = {"corr": corr(pred, ones),
                             "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    # paired: dcov - xy, dcov+xy - xy, dcov - desk on the same draws
    for A_, B_ in (("dcov", "xy"), ("dcov+xy", "xy"), ("dcov", "desk"), ("xy", "desk")):
        diffs = np.array([corr(preds[A_], w[inv].astype(float)) - corr(preds[B_], w[inv].astype(float))
                          for w in W])
        diffs = diffs[np.isfinite(diffs)]
        res["arms"][f"{A_} - {B_}"] = {"corr": res["arms"][A_]["corr"] - res["arms"][B_]["corr"],
                                       "ci": [float(np.percentile(diffs, 2.5)),
                                              float(np.percentile(diffs, 97.5))]}
    with open(os.path.join(a.out, "cov_change.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "cov_change.json"), encoding="utf-8"))
    print(f"cov change [{r['population']}] r{r['rank']}: fitted on {r['n_train']} training cells' change, graded on "
          f"{r['n_eval']} held-out cells; noise-free centered corr with the true ESK change [95% block CI]")
    for k, v in r["arms"].items():
        print(f"  {k:16s} {v['corr']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
