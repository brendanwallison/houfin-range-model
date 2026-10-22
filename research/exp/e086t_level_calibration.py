"""E086t: does a per-species TEMPORAL CALIBRATION of DESK's readout -- one slope fitted on an observed change inside the
trained years -- improve its predicted change at held-out cells? (E086r-b exploratory: the level readout gained
+0.05 to +0.10 on 55-70 CBC cells.)

    python research/exp/e086t_level_calibration.py --cache <tempho research cache> --similarity species_similarity.csv
        --out <dir> [--ranks 24 64] [--thin 0.03] [--thin-seeds 1 2]

Per dev species, readouts on DESK z (first k dims), target data only at TRAINING cells x TRAINED years, rows thinned to
--thin per seed (full data too):
  level   iid-prior BLR (research/lib/blr) of cell-mean log1p counts on cell-mean z (the production form), applied
          to the held-out cells' z change between epochs
  within  the same BLR on per-cell demeaned z -> demeaned counts (deviations on deviations)
  *_cal   the readout's predicted change x c_s, c_s = least squares (no intercept, cell-weighted) of the observed
          BBS change between the two HALVES of the trained span on the readout's predicted change, over training cells
          with >= 1 kept row in each half
Scored on all held-out cell-epoch pairs (withheld early -> modern): noise-free pooled corr (odd / even-year halves of
the BBS change), 6x6-block bootstrap, paired differences cal - uncal; strata sedentary common (prevalence >= 0.08),
sedentary, sedentary rare/mid, migrants, all dev.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from lib import blr  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--similarity", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--ranks", type=int, nargs="+", default=[24, 64])
    ap.add_argument("--thin", type=float, default=0.03); ap.add_argument("--thin-seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    import pandas as pd
    j = lambda f: os.path.join(a.cache, f)
    meta = json.load(open(j("meta.json"))); dev = list(meta["dev_species"]); assert "houfin" not in dev
    keys = np.load(j("keys.npy")); r, c, y = keys[:, 0], keys[:, 1], keys[:, 2]
    sp = np.load(j("split.npz")); ho, bf = sp["holdout"].astype(bool), sp["buffer"].astype(bool)
    wh = np.array(sorted(int(v) for v in sp["withheld"]))
    Xd = np.asarray(np.load(j("X_dev.npy"), mmap_mode="r"))
    Ld = np.log1p(Xd.astype(np.float64)); S = Ld.shape[1]
    Zall = np.load(j("z_ema.npy")).astype(np.float64)
    sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    is_tr = ~ho[r, c] & ~bf[r, c] & ~np.isin(y, wh)
    prev = (Xd[is_tr] > 0).mean(0)
    sed = np.array([mig.get(s) == 1.0 for s in dev])
    strata = {"sed_common": sed & (prev >= 0.08), "sedentary": sed, "sed_rare_mid": sed & (prev < 0.08),
              "migrants": ~sed, "all_dev": np.ones(S, bool)}
    ty = np.unique(y[is_tr]); ycut = ty[0] + len(ty) // 2; half1 = y < ycut
    cid = r.astype(np.int64) * 1000 + c
    # held-out cell-epoch pairs, as e086l / e086r
    E = dict(np.load(j("esk_epochs.npz"))); ptr, flat = E["rows_ptr"], E["rows_flat"]
    idx = {}
    for i, (cell, ep, w) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(w):
            idx.setdefault(cell, [None, None])[0] = i
        elif ep == 1 and not w:
            idx.setdefault(cell, [None, None])[1] = i
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    pc = E["cells"][pairs[:, 0]]; hp = pairs[ho[pc[:, 0], pc[:, 1]]]
    def hm(V, i, h):
        rr = flat[ptr[i]:ptr[i + 1]]; rr = rr[(y[rr] % 2) == h] if h is not None else rr
        return V[rr].mean(0) if len(rr) else np.full(V.shape[1], np.nan)
    SA = np.stack([hm(Ld, m, 1) - hm(Ld, e, 1) for e, m in hp]); SB = np.stack([hm(Ld, m, 0) - hm(Ld, e, 0) for e, m in hp])
    dZall = np.stack([hm(Zall, m, None) - hm(Zall, e, None) for e, m in hp])
    ok = np.isfinite(SA).all(1) & np.isfinite(SB).all(1) & np.isfinite(dZall).all(1)
    hp, SA, SB, dZall = hp[ok], SA[ok], SB[ok], dZall[ok]
    hcells = E["cells"][hp[:, 1]]
    blk = (hcells[:, 0] // 6) * 100000 + hcells[:, 1] // 6
    ub, binv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(86)
    W = np.vstack([np.ones(len(blk)), rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub)), size=a.n_boot)[:, binv]])
    def nfcorr(P, m):
        out = []
        for w in W:
            sw = w.sum(); mean = lambda X: (w @ X) / sw
            Pc, Ac, Bc = P[:, m] - mean(P[:, m]), SA[:, m] - mean(SA[:, m]), SB[:, m] - mean(SB[:, m])
            out.append((w @ (Pc * Bc)).sum() / np.sqrt((w @ (Pc * Pc)).sum() * max((w @ (Ac * Bc)).sum(), 1e-12)))
        return np.array(out)
    def means(V, rows, inv, n):
        s = np.zeros((n.size, V.shape[1])); np.add.at(s, inv, V[rows]); return s / n[:, None]
    res = {"cache": a.cache, "n_heldout_cells": int(len(hp)), "n_blocks": int(len(ub)), "halves": [int(ty[0]), int(ycut), int(ty[-1])],
           "strata_n": {k: int(v.sum()) for k, v in strata.items()}, "results": {}}
    print(f"[e086t] {os.path.basename(a.cache.rstrip('/'))}: {len(hp)} held-out cells ({len(ub)} blocks), halves "
          f"{ty[0]}-{ycut - 1} / {ycut}-{ty[-1]}; strata {res['strata_n']}", flush=True)
    regimes = [("full", 1.0, 0)] + [(f"row{int(round(a.thin * 100)):02d}t{s}", a.thin, s) for s in a.thin_seeds]
    for rname, frac, seed in regimes:
        keep = is_tr & (np.random.default_rng(seed).random(len(y)) < frac if frac < 1 else True)
        rows = np.where(keep)[0]
        u, inv = np.unique(cid[rows], return_inverse=True); n = np.bincount(inv)
        # half means at training cells with >= 1 kept row in each half
        hcell = {}
        for h, hmask in ((0, half1), (1, ~half1)):
            rr = np.where(keep & hmask)[0]; uh, ih = np.unique(cid[rr], return_inverse=True); nh = np.bincount(ih)
            hcell[h] = (uh, rr, ih, nh)
        both = np.intersect1d(hcell[0][0], hcell[1][0])
        for k in a.ranks:
            Z = Zall[:, :k]; dZ = dZall[:, :k]
            Zm, Ym = means(Z, rows, inv, n), means(Ld, rows, inv, n)
            g2 = n[inv] >= 2
            Zd, Yd = (Z[rows] - Zm[inv])[g2], (Ld[rows] - Ym[inv])[g2]
            Bl = np.asarray(blr.fit(Zm, Ym, [(0, k)])["coef"]); Bw = np.asarray(blr.fit(Zd, Yd, [(0, k)])["coef"])
            hz = {h: means(Z, hcell[h][1], hcell[h][2], hcell[h][3])[np.searchsorted(hcell[h][0], both)] for h in (0, 1)}
            hy = {h: means(Ld, hcell[h][1], hcell[h][2], hcell[h][3])[np.searchsorted(hcell[h][0], both)] for h in (0, 1)}
            dzf, dyf = hz[1] - hz[0], hy[1] - hy[0]
            P, cal = {}, {}
            for nm, B in (("level", Bl), ("within", Bw)):
                pf = dzf @ B.T
                cal[nm] = (pf * dyf).sum(0) / np.maximum((pf * pf).sum(0), 1e-12)
                P[nm] = dZ @ B.T; P[nm + "_cal"] = P[nm] * cal[nm]
            out = {"train_rows": int(len(rows)), "cal_cells": int(len(both))}
            for st, m in strata.items():
                dr = {nm: nfcorr(v, m) for nm, v in P.items()}
                o = {"skill": {nm: float(v[0]) for nm, v in dr.items()}}
                for nm in ("level", "within"):
                    d = dr[nm + "_cal"] - dr[nm]
                    o[f"{nm}_cal_minus_{nm}"] = [float(d[0])] + [float(x) for x in np.percentile(d[1:], [2.5, 97.5])]
                    o[f"{nm}_cal_slope_median"] = float(np.median(cal[nm][m]))
                    o[f"{nm}_cal_slope_neg_share"] = float((cal[nm][m] < 0).mean())
                d = dr["level_cal"] - dr["within"]
                o["level_cal_minus_within"] = [float(d[0])] + [float(x) for x in np.percentile(d[1:], [2.5, 97.5])]
                out[st] = o
            res["results"][f"{rname}|r{k}"] = out
            sc = out["sed_common"]
            print(f"  {rname:8s} r{k:<3d} cal cells {len(both):5d} | sed_common level {sc['skill']['level']:+.3f} -> cal "
                  f"{sc['skill']['level_cal']:+.3f} ({sc['level_cal_minus_level'][0]:+.3f} [{sc['level_cal_minus_level'][1]:+.3f},"
                  f"{sc['level_cal_minus_level'][2]:+.3f}]; slopes med {sc['level_cal_slope_median']:+.2f}, neg "
                  f"{sc['level_cal_slope_neg_share']:.2f}) | within {sc['skill']['within']:+.3f} -> cal {sc['skill']['within_cal']:+.3f}",
                  flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e086t.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
