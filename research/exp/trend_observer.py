"""E034b: is the "kind of place x year" trend -- the strongest predictor of change found so far (E027c, E033b) --
ecology, or a change of observer?

    ESK_DESK_CONFIG=<overlay> python research/exp/trend_observer.py --cache <dir> --out <dir>
    python research/exp/trend_observer.py --summarize <out>

A smooth position x year trend (24 random Fourier features of cell position, 300 km, times the year's deviation from
the cell's mean year) is fitted within cells over 1996-2025 on four of five folds of 6x6-cell blocks and predicts the
held-out fold's change 2005-14 -> 2016-25. Every cell is predicted out of fold. The predictions are then graded
SEPARATELY in cells whose observers were unchanged, partly changed, or replaced between the two windows (BBS ObsN of
the cell's QC runs). If the trend's skill comes from observer turnover, it collapses where the observer stayed; if it
is ecological (or a continent-wide observation drift common to all observers), it holds.
Two targets: the community (ESK r24 of epoch-mean counts; noise-free cross-half covariances; centred over cells) and
the graded dev species (log1p epoch means; one trend per species; pooled over resolvable species, centred).
Caveat: within-observer drift (an observer's skill or hearing changing over a decade) is NOT excluded.
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

CLASSES = ("unchanged", "partial", "replaced")


def ccov(x, y):
    return ((x - x.mean(0)) * (y - y.mean(0))).mean(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    import pandas as pd
    from shared_change import groups_for
    from src.config_utils import load_config
    from src.community_encoder.train_DESK.esk_kernel import project_points_to_z
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
    from src.community_encoder.train_DESK.validation_core import change_noise
    from src.data.preprocess import bbs
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    yr = keys[:, 2]
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    r = a.rank
    Zobs = np.load(c("esk_annual.npy"))[:, :r].astype("float64")
    Xc = np.load(c("X_comm.npy")).astype("float64")
    Xd = np.load(c("X_dev.npy")).astype("float64")
    Ld = np.log1p(Xd)
    cfg = load_config()
    proj = lambda v: np.asarray(project_points_to_z(np.asarray(v, "float32"), cfg["desk"]["z_dir"], 64),
                                "float64")[:, :r]
    rng = np.random.default_rng(0)
    Wf, bf = rng.normal(size=(2, 24)) / 300.0, rng.uniform(0, 2 * np.pi, 24)
    pos = lambda rc: np.sqrt(2.0 / 24) * np.cos(rc.astype("float64") * 27.0 @ Wf + bf)
    # folds of 6x6 blocks
    blk = (keys[:, 0] // 6) * 100000 + keys[:, 1] // 6
    ub = np.unique(blk)
    fold_of_blk = dict(zip(ub, rng.permutation(len(ub)) % 5))
    fold = np.array([fold_of_blk[b] for b in blk])
    # evaluation groups: every cell with >= 4 years in both modern windows
    G = groups_for(keys, cid, yr != 2015, 2005, 2014, 2016, 2025)
    gcid = np.array([g[0] for g in G])
    gfold = np.array([fold[g[1][0]] for g in G])
    grc = np.stack([gcid // 100000, gcid % 100000], 1)
    m = lambda X, rl: epoch_mean_observed(X, rl)
    Tc = {h: proj(m(Xc, [g[2] if h == "full" else g[4][i] for g in G]))
             - proj(m(Xc, [g[1] if h == "full" else g[3][i] for g in G]))
          for h, i in (("full", 0), ("a", 0), ("b", 1))}
    Ts = {h: m(Xd, [g[2] if h == "full" else g[4][i] for g in G]) - m(Xd, [g[1] if h == "full" else g[3][i] for g in G])
          for h, i in (("full", 0), ("a", 0), ("b", 1))}
    dyear = np.array([yr[g[2]].mean() - yr[g[1]].mean() for g in G])
    Pc = np.zeros_like(Tc["full"])
    Ps = np.zeros_like(Ts["full"])
    rows_all = np.where((yr >= 1996) & (yr <= 2025))[0]
    for f in range(5):
        rows = rows_all[fold[rows_all] != f]
        uc, inv = np.unique(cid[rows], return_inverse=True)
        cnt = np.bincount(inv).astype(float)

        def demean(M):
            s = np.zeros((len(uc), M.shape[1]))
            np.add.at(s, inv, M)
            return M - (s / cnt[:, None])[inv]
        tdev = demean(yr[rows, None].astype(float))[:, 0]
        X = pos(keys[rows, :2]) * tdev[:, None]
        bc = blr.fit(X, demean(Zobs[rows]), [(0, 24)])["coef"].T
        bs = blr.fit(X, demean(Ld[rows]), [(0, 24)])["coef"].T
        sel = gfold == f
        Xe = pos(grc[sel]) * dyear[sel, None]
        Pc[sel] = Xe @ bc
        Ps[sel] = Xe @ bs
    # observer classes
    runs = bbs.load_run_metadata()
    ry = pd.read_csv(c("route_years.csv"))
    mm = ry.merge(runs, left_on=["country", "state", "route", "year"], right_on=["CountryNum", "StateNum", "Route", "Year"])
    obs = mm.groupby(["row", "col", "year"])["ObsN"].apply(lambda s_: frozenset(s_.astype(int))).to_dict()
    obs_of = lambda rows: frozenset().union(*[obs.get((int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])),
                                                      frozenset()) for i in rows])
    kind = []
    for g in G:
        o1, o2 = obs_of(g[1]), obs_of(g[2])
        kind.append("none" if not o1 or not o2 else
                    ("unchanged" if o1 == o2 else ("replaced" if not (o1 & o2) else "partial")))
    kind = np.array(kind)
    res_sp = change_noise(Ts["full"], Ts["a"], Ts["b"])["resolvable"]
    gblk = (grc[:, 0] // 6) * 100000 + grc[:, 1] // 6

    def stats(mask, w=None):
        w = np.ones(int(mask.sum())) if w is None else w
        cw = lambda x, y: (((x - np.average(x, 0, w)) * (y - np.average(y, 0, w))) * w[:, None]).sum(0) / w.sum()
        pc, ta, tb, tf = Pc[mask], Tc["a"][mask], Tc["b"][mask], Tc["full"][mask]
        cc = cw(pc, tf).sum() / np.sqrt(cw(pc, pc).sum() * cw(ta, tb).sum())
        kc = cw(pc, tf).sum() / cw(pc, pc).sum()
        ps, sa, sb, sf = Ps[mask][:, res_sp], Ts["a"][mask][:, res_sp], Ts["b"][mask][:, res_sp], Ts["full"][mask][:, res_sp]
        cs = cw(ps, sf).sum() / np.sqrt(cw(ps, ps).sum() * cw(sa, sb).sum())
        return float(cc), float(kc), float(cs)
    res = {"cache": a.cache, "n_cells": int(len(G)), "n_species": int(res_sp.sum()), "classes": {}}
    for k in CLASSES:
        msk = kind == k
        if msk.sum() < 30:
            continue
        cc, kc, cs = stats(msk)
        u, binv = np.unique(gblk[msk], return_inverse=True)
        bc, bs_ = [], []
        for _ in range(a.n_boot):
            wb = rng.multinomial(len(u), np.full(len(u), 1.0 / len(u)))[binv].astype(float)
            if wb.sum() == 0:
                continue
            x = stats(msk, wb)
            bc.append(x[0])
            bs_.append(x[2])
        res["classes"][k] = {"n_cells": int(msk.sum()), "community_corr": cc, "community_k": kc,
                             "community_corr_ci": [float(np.nanpercentile(bc, 2.5)), float(np.nanpercentile(bc, 97.5))],
                             "species_corr": cs,
                             "species_corr_ci": [float(np.nanpercentile(bs_, 2.5)), float(np.nanpercentile(bs_, 97.5))]}
    with open(os.path.join(a.out, "trend_observer.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "trend_observer.json"), encoding="utf-8"))
    print(f"position x year trend (fit 1996-2025 out of fold) vs the 2005-14 -> 2016-25 change, by observer continuity "
          f"({r['n_cells']} cells, {r['n_species']} resolvable species); noise-free centred corr:")
    for k, v in r["classes"].items():
        print(f"  observers {k:9s} cells {v['n_cells']:4d}: community corr {v['community_corr']:+.3f} "
              f"[{v['community_corr_ci'][0]:+.3f}, {v['community_corr_ci'][1]:+.3f}] (k {v['community_k']:.2f}) | species "
              f"corr {v['species_corr']:+.3f} [{v['species_corr_ci'][0]:+.3f}, {v['species_corr_ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
