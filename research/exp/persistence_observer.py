"""E028 (O1, A14, B5): is "damped persistence" -- a cell's deviation from its regional surface fading over time, the
behaviour that makes the spacetime_sum GP beat no change on trained cells (E006) -- habitat, or a change of observer?

    ESK_DESK_CONFIG=<overlay> python research/exp/persistence_observer.py --cache <dir> --out <dir>
    python research/exp/persistence_observer.py --summarize <out>

Within the modern era (2005-2014 vs 2016-2025, as E021), cells are classed by observer continuity (BBS ObsN of their
QC runs: unchanged / partial / replaced). For each dev species, a cell's epoch-1 deviation d1 = (its epoch-1 mean
log1p count, from ONE ABBA half of epoch 1) minus a regional surface (Gaussian-weighted mean of OTHER cells' epoch-1
means, 150 km scale); its change = epoch-2 mean minus the OTHER half of epoch 1. The halves make the noise in d1
independent of the change, so a negative slope of change on d1 is not noise regression -- but they share the epoch's
observers, so an observer effect that leaves with the observer DOES produce one. Slope by continuity class, pooled over
resolvable species (sum of covariances over sum of variances), block-bootstrapped (6x6 cells); and the change skill vs
no change of the damped-persistence predictor -k*d1 with k fitted on the other classes' cells.
Observational: cells that change observers may differ otherwise (their deviation sizes are reported).
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
CLASSES = ("unchanged", "partial", "replaced")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--scale-km", type=float, default=150.0)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    import pandas as pd
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
    from src.community_encoder.train_DESK.validation_core import abba_halves
    from src.data.preprocess import bbs
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    yr = keys[:, 2]
    runs = bbs.load_run_metadata()
    ry = pd.read_csv(c("route_years.csv"))
    mm = ry.merge(runs, left_on=["country", "state", "route", "year"],
                  right_on=["CountryNum", "StateNum", "Route", "Year"])
    obs = mm.groupby(["row", "col", "year"])["ObsN"].apply(lambda s: frozenset(s.astype(int))).to_dict()
    obs_of = lambda rows: frozenset().union(*[obs.get((int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])),
                                                      frozenset()) for i in rows])
    Xd = np.load(c("X_dev.npy")).astype("float64")
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    rows_by = {}
    for i in np.where((yr >= 2005) & (yr <= 2025) & (yr != 2015))[0]:
        rows_by.setdefault(cid[i], []).append(i)
    g = {"h1a": [], "h1b": [], "e1": [], "e2": []}
    kind, cells = [], []
    for cc, rows in rows_by.items():
        rows = np.array(sorted(rows, key=lambda i: yr[i]))
        r1, r2 = rows[yr[rows] < 2015], rows[yr[rows] > 2015]
        if len(r1) < 4 or len(r2) < 4:
            continue
        o1, o2 = obs_of(r1), obs_of(r2)
        if not o1 or not o2:
            continue
        kind.append("unchanged" if o1 == o2 else ("replaced" if not (o1 & o2) else "partial"))
        h = abba_halves(r1, yr[r1])
        g["h1a"].append(h[0]), g["h1b"].append(h[1]), g["e1"].append(r1), g["e2"].append(r2)
        cells.append((cc // 100000, cc % 100000))
    kind, cells = np.array(kind), np.array(cells)
    Y = {k: epoch_mean_observed(Xd, v).astype("float64") for k, v in g.items()}
    # regional surface of epoch-1 means, leaving each cell out
    xy = cells.astype("float64") * 27.0
    D2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    W = np.exp(-0.5 * D2 / a.scale_km ** 2)
    np.fill_diagonal(W, 0.0)
    W /= W.sum(1, keepdims=True)
    S1 = W @ Y["e1"]
    d1 = Y["h1a"] - S1                              # deviation, from half A of epoch 1
    dc = Y["e2"] - Y["h1b"]                         # change, against the OTHER half of epoch 1
    # resolvable: species whose within-modern change has positive cross-half signal somewhere (E021's rule, simplified)
    sp = (Y["e1"].std(0) > 0) & (Y["e2"].std(0) > 0) & ((Xd[np.concatenate(g["e1"])] > 0).mean(0) > 0.01)
    blk = (cells[:, 0] // 6) * 100000 + cells[:, 1] // 6
    ub, binv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(0)

    def slope(m, w=None):
        w = np.ones(int(m.sum())) if w is None else w
        x, y = d1[m][:, sp], dc[m][:, sp]
        xc = x - (w[:, None] * x).sum(0) / w.sum()
        yc = y - (w[:, None] * y).sum(0) / w.sum()
        return float((w[:, None] * xc * yc).sum() / (w[:, None] * xc * xc).sum())

    def skill(m, k):
        """1 - SSE(-k*d1) / SSE(0) on the change, pooled over resolvable species (centered per species)."""
        x, y = d1[m][:, sp], dc[m][:, sp]
        y = y - y.mean(0)
        p = -k * (x - x.mean(0))
        return float(1.0 - ((y - p) ** 2).sum() / (y ** 2).sum())

    res = {"cache": a.cache, "scale_km": a.scale_km, "n_species": int(sp.sum()),
           "n_cells": {k: int((kind == k).sum()) for k in CLASSES}, "classes": {}}
    boots = {k: [] for k in CLASSES}
    for _ in range(a.n_boot):
        wb = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)))[binv].astype(float)
        for k in CLASSES:
            m = kind == k
            if wb[m].sum() > 0:
                boots[k].append(slope(m, wb[m]))
    for k in CLASSES:
        m = kind == k
        others = (kind != k)
        kfit = -slope(others)                      # damping fitted on the other classes (no self-fit)
        res["classes"][k] = {"slope": slope(m), "slope_ci": [float(np.percentile(boots[k], 2.5)),
                                                             float(np.percentile(boots[k], 97.5))],
                             "dev_sd": float(d1[m][:, sp].std()), "change_sd": float(dc[m][:, sp].std()),
                             "skill_damped_vs_no_change": skill(m, max(kfit, 0.0)), "k_from_others": kfit}
    diff = np.array(boots["replaced"]) - np.array(boots["unchanged"])
    res["slope_replaced_minus_unchanged"] = {"diff": res["classes"]["replaced"]["slope"]
                                             - res["classes"]["unchanged"]["slope"],
                                             "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
    with open(os.path.join(a.out, "persistence_observer.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "persistence_observer.json"), encoding="utf-8"))
    print(f"damped persistence by observer continuity, 2005-14 -> 2016-25 ({r['n_species']} species, surface "
          f"{r['scale_km']:.0f} km): slope of change on the epoch-1 deviation (noise-independent halves); -1 = the "
          f"deviation vanishes, 0 = it persists")
    for k in CLASSES:
        c = r["classes"][k]
        print(f"  {k:9s} cells {r['n_cells'][k]:4d}  slope {c['slope']:+.3f} [{c['slope_ci'][0]:+.3f}, "
              f"{c['slope_ci'][1]:+.3f}]  dev sd {c['dev_sd']:.3f}  change sd {c['change_sd']:.3f}  "
              f"skill of damping (k={c['k_from_others']:+.3f} from the other classes) {c['skill_damped_vs_no_change']:+.4f}")
    d = r["slope_replaced_minus_unchanged"]
    print(f"  replaced - unchanged slope {d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
