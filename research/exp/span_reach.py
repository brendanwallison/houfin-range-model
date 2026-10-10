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
# DESK models: (name, cache subdir, first trained year, last trained year); --desk-models adds more (E024 span runs)
PAIRS = [("cov+trend", "trend"), ("lags+trend", "cov+trend"), ("rff+trend", "cov+trend"), ("lags", "cov"),
         ("rff", "cov")]
DESK_MODELS = [("t1995", "desk_tempho_1995", 1996, 2025), ("t1985", "desk_tempho_1985", 1986, 2025),
               ("t1975", "desk_tempho_1975", 1976, 2025)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--min-years", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--plus-year", type=int, default=None,
                    help="also train every span on this year (DESK cannot withhold its label year, 2025: this is what a "
                         "'span' DESK run actually trains on)")
    ap.add_argument("--desk-models", nargs="*", default=(),
                    help="more DESK models as name=cache_subdir:T0:T1 (e.g. a span run)")
    ap.add_argument("--balance-years", action="store_true",
                    help="subsample training rows to equal counts per year (BBS coverage grows ~7x 1966-1995, so a long"
                         " span is otherwise dominated by its late years)")
    ap.add_argument("--lag-hls", type=float, nargs="*", default=(),
                    help="E027: also covariates EMA'd at these half-lives (years; 0 = no EMA beyond the states' own), 32 "
                         "PCs each: arms lag<hl>, lags (all, one prior block each), lags+trend")
    ap.add_argument("--rff", type=int, default=0,
                    help="E027: also a NONLINEAR covariate surrogate, this many random Fourier features of the 64 "
                         "covariate PCs (RBF, lengthscale = median pairwise distance): arms rff, rff+trend")
    ap.add_argument("--no-default-desk", action="store_true", help="skip the archived tempho models")
    ap.add_argument("--desk-plus", nargs="*", default=(),
                    help="E027b: for every DESK model also recalibrate on its z PLUS these blocks (lags, lag<hl>, rff, "
                         "trend), each with its own prior: does the block add anything DESK's z does not already carry?")
    ap.add_argument("--skip-surrogates", action="store_true", help="fit only what the DESK arms need")
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

    extra = {}                                   # E027 feature blocks: name -> (N, k) at every key
    if a.lag_hls:
        for hl, F in covfeat.multi_ema_rows(cfg, keys, a.lag_hls).items():
            extra[f"lag{hl:g}"] = np.nan_to_num(covfeat.cov_pcs(F, train_cell & np.isfinite(F).all(1), n_pcs=32))
    if a.rff:
        sub = np.where(train_cell)[0][:: max(1, int(train_cell.sum()) // 2000)]
        Ps = np.asarray(P[sub], "float64")
        sq = (Ps * Ps).sum(1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * Ps @ Ps.T, 0.0)
        ls = float(np.sqrt(np.median(d2[np.triu_indices(len(sub), 1)])))
        extra["rff"] = covfeat.rff(P, a.rff, ls, seed=1)
        print(f"[span-reach] rff: {a.rff} features, lengthscale {ls:.2f} (median distance of 64 covariate PCs)", flush=True)
    lag_names = [k for k in extra if k.startswith("lag")]

    models = [] if a.no_default_desk else list(DESK_MODELS)
    for spec in a.desk_models:
        name, rest = spec.split("=")
        sub, m0, m1 = rest.split(":")
        models.append((name, sub, int(m0), int(m1)))
    res = {"rank": r, "balance_years": bool(a.balance_years), "plus_year": a.plus_year, "desk_models": models,
           "rows": []}
    for t0, spans in GRID.items():
        ref = (t0, t0 + 9)
        cells_w = {}
        for (lo, hi) in WINDOWS:
            if hi >= t0:
                continue
            cells_w[(lo, hi)] = [c_ for (c_, l_, h_) in gidx if (l_, h_) == (lo, hi) and (c_, *ref) in gidx]
        desks = {name: (np.asarray(np.load(os.path.join(CACHE_ROOT, sub, "z_raw.npy"), mmap_mode="r")[:, :r], "float64"), m1)
                 for (name, sub, m0, m1) in models if m0 == t0}
        for L in spans:
            t1 = t0 + L - 1
            in_span = (yr >= t0) & (yr <= t1)
            if a.plus_year is not None:
                in_span |= yr == a.plus_year
            rows = np.where(train_cell & in_span)[0]
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
            fits = {"trend": blr.fit(Xt, y, [(0, 24)])}
            if not a.skip_surrogates:
                fits["cov"] = blr.fit(Xp, y, [(0, Xp.shape[1])])
                fits["cov+trend"] = blr.fit(np.hstack([Xp, Xt]), y, [(0, Xp.shape[1]), (Xp.shape[1], Xp.shape[1] + 24)])
            Xe = {k: demean(v[rows]) for k, v in extra.items()}

            def plus_blocks(blk, M, tr):
                """The feature blocks a --desk-plus name stands for, from the per-row block dict ``M`` (training rows
                or change rows) and the trend block ``tr``."""
                if blk == "lags":
                    return [M[k] for k in lag_names]
                if blk == "trend":
                    return [tr]
                return [M[blk]]
            for name, (dz, _m1) in desks.items():
                Xd = demean(dz[rows])
                for blk in a.desk_plus:
                    parts = plus_blocks(blk, Xe, Xt)
                    edges = np.cumsum([r] + [q.shape[1] for q in parts])
                    fits[f"recal+{blk}:{name}"] = blr.fit(np.hstack([Xd] + parts), y,
                                                          [(0, r)] + [(int(edges[i]), int(edges[i + 1]))
                                                                      for i in range(len(parts))])
            for k in (lag_names if not a.skip_surrogates else []):
                fits[k] = blr.fit(Xe[k], y, [(0, Xe[k].shape[1])])
            if lag_names and not a.skip_surrogates:
                XL = np.hstack([Xe[k] for k in lag_names])
                bl = [(32 * i, 32 * (i + 1)) for i in range(len(lag_names))]
                fits["lags"] = blr.fit(XL, y, bl)
                fits["lags+trend"] = blr.fit(np.hstack([XL, Xt]), y, bl + [(XL.shape[1], XL.shape[1] + 24)])
            if "rff" in extra and not a.skip_surrogates:
                k_ = Xe["rff"].shape[1]
                fits["rff"] = blr.fit(Xe["rff"], y, [(0, k_)])
                fits["rff+trend"] = blr.fit(np.hstack([Xe["rff"], Xt]), y, [(0, k_), (k_, k_ + 24)])
            for name, (dz, _m1) in desks.items():
                # recal:<model> -- the DESK model's own training span is fixed; only its RECALIBRATION uses this span's
                # rows (a post-hoc test of weighting the trained years nearest the target)
                fits["recal:" + name] = blr.fit(demean(dz[rows]), y, [(0, r)])
            for w, cl in cells_w.items():
                if len(cl) < 20:
                    continue
                jw = np.array([gidx[(c_, *w)] for c_ in cl])
                jr = np.array([gidx[(c_, *ref)] for c_ in cl])
                gw, gr = [groups[j] for j in jw], [groups[j] for j in jr]
                dyear = np.array([yr[g1].mean() - yr[g2].mean() for g1, g2 in zip(gw, gr)])
                rpos = np.stack([Rpos[g1[0]] for g1 in gw])
                dP = np.stack([gmean(P, g1) - gmean(P, g2) for g1, g2 in zip(gw, gr)])
                preds = {"trend": (rpos * dyear[:, None]) @ fits["trend"]["coef"].T}
                if "cov" in fits:
                    preds["cov"] = dP @ fits["cov"]["coef"].T
                    preds["cov+trend"] = np.hstack([dP, rpos * dyear[:, None]]) @ fits["cov+trend"]["coef"].T
                dE = {k: np.stack([gmean(v, g1) - gmean(v, g2) for g1, g2 in zip(gw, gr)]) for k, v in extra.items()}
                for k in (lag_names if not a.skip_surrogates else []):
                    preds[k] = dE[k] @ fits[k]["coef"].T
                if lag_names and not a.skip_surrogates:
                    dL = np.hstack([dE[k] for k in lag_names])
                    preds["lags"] = dL @ fits["lags"]["coef"].T
                    preds["lags+trend"] = np.hstack([dL, rpos * dyear[:, None]]) @ fits["lags+trend"]["coef"].T
                if "rff" in fits:
                    preds["rff"] = dE["rff"] @ fits["rff"]["coef"].T
                    preds["rff+trend"] = np.hstack([dE["rff"], rpos * dyear[:, None]]) @ fits["rff+trend"]["coef"].T
                for name, (dz, m1) in desks.items():
                    dD = np.stack([gmean(dz, g1) - gmean(dz, g2) for g1, g2 in zip(gw, gr)])
                    if m1 == t1:
                        preds["raw:" + name] = dD              # the model's own span: its unrecalibrated change
                    preds["recal:" + name] = dD @ fits["recal:" + name]["coef"].T
                    for blk in a.desk_plus:
                        parts = plus_blocks(blk, dE, rpos * dyear[:, None])
                        preds[f"recal+{blk}:{name}"] = np.hstack([dD] + parts) @ fits[f"recal+{blk}:{name}"]["coef"].T
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
                # paired differences (same bootstrap draws): what a block adds over the arm it extends
                row["diffs"] = {}
                pairs = list(PAIRS) + [(f"recal+{blk}:{name}", f"recal:{name}") for name in desks for blk in a.desk_plus]
                for p1, p0 in pairs:
                    if p1 in preds and p0 in preds:
                        d0 = stats(preds[p1], np.ones(len(cl)))[0] - stats(preds[p0], np.ones(len(cl)))[0]
                        bd = np.array([stats(preds[p1], ww[binv].astype(float))[0]
                                       - stats(preds[p0], ww[binv].astype(float))[0] for ww in W])
                        row["diffs"][f"{p1} - {p0}"] = {"diff": float(d0), "ci": [float(np.nanpercentile(bd, 2.5)),
                                                                                  float(np.nanpercentile(bd, 97.5))]}
                res["rows"].append(row)
    with open(os.path.join(a.out, "span_reach.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "span_reach.json"), encoding="utf-8"))
    print(f"span x reach (r{r['rank']}{', years balanced' if r.get('balance_years') else ''}"
          f"{', + year ' + str(r['plus_year']) if r.get('plus_year') else ''}): change of held-out cells between a backcast window and the first ten trained "
          f"years; noise-free centered corr (calibration). At fixed T0 and window, only the span changes.")
    for row in sorted(r["rows"], key=lambda x: (-x["t0"], x["reach"], x["span"])):
        parts = [f"{k} {v['corr']:+.3f} ({v['calib']:.2f})" for k, v in row["arms"].items()]
        print(f"  T0 {row['t0']} W {row['window'][0]}-{row['window'][1] % 100:02d} reach {row['reach']:+5.1f} span "
              f"{row['span']:2d} cells {row['n_cells']:3d} | " + " | ".join(parts))
        if row.get("diffs"):
            print("      paired: " + " | ".join(f"{k} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]"
                                                for k, v in row["diffs"].items()))


if __name__ == "__main__":
    main()
