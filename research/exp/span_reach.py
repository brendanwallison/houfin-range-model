"""E023: does more training HISTORY improve a backcast at a FIXED reach? Training span vs extrapolation reach.

    ESK_DESK_CONFIG=<overlay> python research/exp/span_reach.py --out <dir> [--rank 24]
    python research/exp/span_reach.py --summarize <out>

The tempho DESK models confound the two: a model reaching 21-30 years back (tempho1995) has also trained on only 30
years, against 40 / 50 for the shorter reaches, and production trains on 60. Here the cheap surrogates are refitted
on training spans [T0, T0 + L - 1] of chosen length L (training cells only), and every fit is graded on the SAME
held-out cells and the SAME backcast windows W before T0, so that at fixed (T0, W) -- fixed reach -- only the span
changes. The change scored is W against the first ten trained years [T0, T0 + 9], which lie inside every span.

Surrogates (within-cell: the target is each training cell's observed ESK z minus its own mean over the span's rows):
    trend     smooth position fields (24 random Fourier features, 300 km) x (year - the cell's mean year): regional
              trend persistence, no covariates (the skeptic's placebo, which matched every covariate map)
    cov       64 PCs of the covariates (as DESK sees them), cell-demeaned
    cov+trend both blocks
    desk      where a tempho DESK model has exactly this span (T1 = 2025): its raw change, and recalibrated by a
              within-cell BLR fitted on the span's rows (desk_recal)
Grade: noise-free (ABBA halves of W and of the reference window) centered correlation with the true ESK change over
held-out cells, and calibration (truth on prediction). Grid: T0 = 1996 (L 10/20/30; reach 3.5-27.5), 1986 (L 10-40;
5.5-17.5), 1976 (L 10-50; 7.5).
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
from lib import blr, covfeat  # noqa: E402

CACHE_ROOT = os.path.expanduser("~/houfin/work/houfin/research_cache")
WINDOWS = [(1966, 1971), (1972, 1977), (1978, 1983), (1984, 1989), (1990, 1995)]
GRID = {1996: (10, 20, 30), 1986: (10, 20, 30, 40), 1976: (10, 20, 30, 40, 50)}
DESK_OF_T0 = {1996: "desk_tempho_1995", 1986: "desk_tempho_1985", 1976: "desk_tempho_1975"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--min-years", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--balance-years", action="store_true",
                    help="subsample training rows to equal counts per year (BBS coverage grows ~7x 1966-1995, so a long"
                         " span is otherwise dominated by its late years)")
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from src.config_utils import load_config
    from src.community_encoder.train_DESK.esk_kernel import project_points_to_z
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
    from src.community_encoder.train_DESK.validation_core import split_half_groups
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(CACHE_ROOT, "gp_species_base")
    keys = np.load(os.path.join(base, "keys.npy"))
    split = np.load(os.path.join(base, "split.npz"))
    yr = keys[:, 2]
    held = split["holdout"][keys[:, 0], keys[:, 1]]
    train_cell = ~held & ~split["buffer"][keys[:, 0], keys[:, 1]]
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    r = a.rank

    # truth: held-out (cell, window) groups for the backcast windows and the three reference windows
    wins = WINDOWS + [(t0, t0 + 9) for t0 in GRID]
    groups, gkey = [], []
    for (lo, hi) in wins:
        sel = np.where(held & (yr >= lo) & (yr <= hi))[0]
        sel = sel[np.lexsort((yr[sel], cid[sel]))]
        cs, st = np.unique(cid[sel], return_index=True)
        for c_, s, e in zip(cs, st, list(st[1:]) + [len(sel)]):
            if e - s >= a.min_years:
                groups.append(sel[s:e])
                gkey.append((int(c_), lo, hi))
    A, B, ok = split_half_groups(groups, years=yr)
    Xc = np.load(os.path.join(base, "X_comm.npy")).astype("float64")
    cfg = load_config()
    proj = lambda v: np.asarray(project_points_to_z(np.asarray(v, "float32"), cfg["desk"]["z_dir"], 64),
                                "float64")[:, :r]
    T = {"full": proj(epoch_mean_observed(Xc, groups)), "a": proj(epoch_mean_observed(Xc, A)),
         "b": proj(epoch_mean_observed(Xc, B))}
    gidx = {k: j for j, k in enumerate(gkey) if ok[j]}

    # features at every key
    P = covfeat.cov_pcs(np.load(os.path.join(base, "F_cov.npy"), mmap_mode="r"), train_cell)
    rng = np.random.default_rng(0)
    xy = keys[:, :2].astype("float64") * 27.0
    Rpos = np.sqrt(2.0 / 24) * np.cos(xy @ (rng.normal(size=(2, 24)) / 300.0) + rng.uniform(0, 2 * np.pi, 24))
    Zobs = np.load(os.path.join(base, "esk_annual.npy"))[:, :r].astype("float64")
    gmean = lambda M, g: np.asarray(M[g], "float64").mean(0)

    res = {"rank": r, "balance_years": bool(a.balance_years), "rows": []}
    for t0, spans in GRID.items():
        ref = (t0, t0 + 9)
        cells_w = {}
        for (lo, hi) in WINDOWS:
            if hi >= t0:
                continue
            cells_w[(lo, hi)] = [c_ for (c_, l_, h_) in gidx if (l_, h_) == (lo, hi) and (c_, *ref) in gidx]
        desk = None
        if t0 in DESK_OF_T0:
            desk = np.asarray(np.load(os.path.join(CACHE_ROOT, DESK_OF_T0[t0], "z_raw.npy"), mmap_mode="r")[:, :r],
                              "float64")
        for L in spans:
            t1 = t0 + L - 1
            rows = np.where(train_cell & (yr >= t0) & (yr <= t1))[0]
            if a.balance_years:
                per = np.bincount(yr[rows] - t0, minlength=L)
                k = int(per[per > 0].min())
                rows = np.sort(np.concatenate([rng.choice(rows[yr[rows] == y], k, replace=False)
                                               for y in range(t0, t1 + 1) if (yr[rows] == y).any()]))
            uc, inv = np.unique(cid[rows], return_inverse=True)
            cnt = np.bincount(inv).astype(float)

            def demean(M):
                s = np.zeros((len(uc), M.shape[1]))
                np.add.at(s, inv, M)
                return M - (s / cnt[:, None])[inv]
            y = demean(Zobs[rows])
            tdev = demean(yr[rows, None].astype(float))[:, 0]
            Xt = Rpos[rows] * tdev[:, None]
            Xp = demean(P[rows])
            fits = {"trend": blr.fit(Xt, y, [(0, 24)]),
                    "cov": blr.fit(Xp, y, [(0, Xp.shape[1])]),
                    "cov+trend": blr.fit(np.hstack([Xp, Xt]), y, [(0, Xp.shape[1]), (Xp.shape[1], Xp.shape[1] + 24)])}
            if desk is not None:
                # the DESK model itself always trained on [T0, 2025]; only its RECALIBRATION uses the span's rows --
                # a post-hoc test of weighting the trained years nearest the target
                fits["desk_recal"] = blr.fit(demean(desk[rows]), y, [(0, r)])
            for w, cl in cells_w.items():
                if len(cl) < 20:
                    continue
                jw = np.array([gidx[(c_, *w)] for c_ in cl])
                jr = np.array([gidx[(c_, *ref)] for c_ in cl])
                gw, gr = [groups[j] for j in jw], [groups[j] for j in jr]
                dyear = np.array([yr[g1].mean() - yr[g2].mean() for g1, g2 in zip(gw, gr)])
                rpos = np.stack([Rpos[g1[0]] for g1 in gw])
                dP = np.stack([gmean(P, g1) - gmean(P, g2) for g1, g2 in zip(gw, gr)])
                preds = {"trend": (rpos * dyear[:, None]) @ fits["trend"]["coef"].T,
                         "cov": dP @ fits["cov"]["coef"].T,
                         "cov+trend": np.hstack([dP, rpos * dyear[:, None]]) @ fits["cov+trend"]["coef"].T}
                if "desk_recal" in fits:
                    dD = np.stack([gmean(desk, g1) - gmean(desk, g2) for g1, g2 in zip(gw, gr)])
                    if t1 == 2025:
                        preds["desk"] = dD
                    preds["desk_recal"] = dD @ fits["desk_recal"]["coef"].T
                dT = {h: T[h][jw] - T[h][jr] for h in T}
                cells_rc = np.array([(c_ // 100000, c_ % 100000) for c_ in cl])
                blk = (cells_rc[:, 0] // 6) * 100000 + cells_rc[:, 1] // 6
                ub, binv = np.unique(blk, return_inverse=True)
                W = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)

                def stats(p, w):
                    cen = lambda v: v - (w[:, None] * v).sum(0) / w.sum()
                    p_, f_, a_, b_ = cen(p), cen(dT["full"]), cen(dT["a"]), cen(dT["b"])
                    cov = (w[:, None] * p_ * f_).sum()
                    vp, vt = (w[:, None] * p_ * p_).sum(), (w[:, None] * a_ * b_).sum()
                    if vp <= 0 or vt <= 0:
                        return np.nan, np.nan
                    return cov / np.sqrt(vp * vt), cov / vp
                row = {"t0": t0, "span": L, "t1": t1, "window": list(w), "reach": float(t0 - (w[0] + w[1]) / 2),
                       "n_cells": int(len(cl)), "arms": {}}
                for name, p in preds.items():
                    c0, k0 = stats(p, np.ones(len(cl)))
                    bs = np.array([stats(p, ww[binv].astype(float))[0] for ww in W])
                    row["arms"][name] = {"corr": float(c0), "calib": float(k0),
                                         "corr_ci": [float(np.nanpercentile(bs, 2.5)),
                                                     float(np.nanpercentile(bs, 97.5))]}
                res["rows"].append(row)
    with open(os.path.join(a.out, "span_reach.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "span_reach.json"), encoding="utf-8"))
    print(f"span x reach (r{r['rank']}{', years balanced' if r.get('balance_years') else ''}): change of held-out cells between a backcast window and the first ten trained "
          f"years; noise-free centered corr (calibration). At fixed T0 and window, only the span changes.")
    arms = ("trend", "cov", "cov+trend", "desk", "desk_recal")
    print(f"  {'T0':>5s} {'window':>10s} {'reach':>6s} {'span':>5s} {'cells':>5s}  " + "  ".join(f"{a_:>16s}" for a_ in arms))
    for row in sorted(r["rows"], key=lambda x: (-x["t0"], x["reach"], x["span"])):
        cells = []
        for a_ in arms:
            v = row["arms"].get(a_)
            cells.append(f"{v['corr']:+.3f} ({v['calib']:.2f})" if v else f"{'':>16s}")
        print(f"  {row['t0']:5d} {row['window'][0]}-{row['window'][1] % 100:02d} {row['reach']:+6.1f} {row['span']:5d} "
              f"{row['n_cells']:5d}  " + "  ".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    main()
