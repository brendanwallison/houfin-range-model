"""E031: is a species' change SHARED with the species it lives with (habitat-like), or its own (species dynamics)?

    ESK_DESK_CONFIG=<overlay> python research/exp/shared_change.py --cache <dir> --out <dir> [--k 5]
    python research/exp/shared_change.py --summarize <out>

E030 found the community's ~40-year change carries only ~5-7% of a graded species' change. That could mean the
species' change is its own story (spread, disease, wintering grounds -- the population model's job), or that the
community summary misses habitat change the species shares with its particular associates. Habitat change should
move the species that use that habitat TOGETHER; a species' own dynamics move it alone. Regional vs local cannot
separate them (habitat change is regional too), shared vs unique can.

For each graded (dev) species: its K associates are the other species (reference community + other dev species)
whose abundance it tracks most closely across TRAINING cells, using the average of the early and modern epochs (the
time-averaged "where it lives", roughly orthogonal to the change itself). Its change (log1p epoch means) is then
compared with the mean standardized change of its associates, on HELD-OUT cells, noise-free via ABBA cross-halves
(target from half A with associates from half B and vice versa, so a survey's shared weather/day effects cancel):
    corr_true = [cov(t_A, a_B) + cov(t_B, a_A)] / 2 / sqrt(cov(t_A, t_B) cov(a_A, a_B))
pooled over resolvable species, centred per species (where it changed more, did its associates?). Null: K random
other species, 30 draws. Also, ACROSS species: does a species' continent-wide mean change follow its associates'?
Observer check (O1): the same statistic within the modern era (2005-14 vs 2016-25) by observer continuity of the
cell (unchanged / replaced), since a change of observer moves every species on a route together.
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


def ccov(x, y):
    return ((x - x.mean(0)) * (y - y.mean(0))).mean(0)


def groups_for(keys, cid, rows_mask, lo1, hi1, lo2, hi2, min_years=4):
    """Per cell with >= min_years rows in both windows: (cell id, rows1, rows2) in year order."""
    from src.community_encoder.train_DESK.validation_core import abba_halves
    yr = keys[:, 2]
    by = {}
    for i in np.where(rows_mask)[0]:
        y = yr[i]
        if lo1 <= y <= hi1:
            by.setdefault(cid[i], ([], []))[0].append(i)
        elif lo2 <= y <= hi2:
            by.setdefault(cid[i], ([], []))[1].append(i)
    out = []
    for c, (r1, r2) in by.items():
        if len(r1) >= min_years and len(r2) >= min_years:
            r1 = np.array(sorted(r1, key=lambda i: yr[i]))
            r2 = np.array(sorted(r2, key=lambda i: yr[i]))
            out.append((c, r1, r2, abba_halves(r1, yr[r1]), abba_halves(r2, yr[r2])))
    return out


def changes(X, G):
    """log1p epoch-mean change per cell: full, half A, half B -> (n_cells, S) each."""
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
    m = lambda rl: epoch_mean_observed(X, rl)
    full = m([g[2] for g in G]) - m([g[1] for g in G])
    a = m([g[4][0] for g in G]) - m([g[3][0] for g in G])
    b = m([g[4][1] for g in G]) - m([g[3][1] for g in G])
    return full, a, b


def sharing(t, A, assoc, mask_species):
    """Noise-free centred correlation of each target's change with its associates' mean standardized change, pooled
    over ``mask_species`` (sum of numerators over sum of denominators, each target standardized)."""
    tf, ta, tb = t
    Af, Aa, Ab = A
    sd = Af.std(0) + 1e-12
    num = vt = va = 0.0
    per = []
    for s in np.where(mask_species)[0]:
        j = assoc[s]
        aa = (Aa[:, j] / sd[j]).mean(1)
        ab = (Ab[:, j] / sd[j]).mean(1)
        x_a, x_b = ta[:, s] / (tf[:, s].std() + 1e-12), tb[:, s] / (tf[:, s].std() + 1e-12)
        n = 0.5 * (ccov(x_a[:, None], ab[:, None])[0] + ccov(x_b[:, None], aa[:, None])[0])
        v1 = ccov(x_a[:, None], x_b[:, None])[0]
        v2 = ccov(aa[:, None], ab[:, None])[0]
        num, vt, va = num + n, vt + v1, va + v2
        per.append(n / np.sqrt(v1 * v2) if v1 > 0 and v2 > 0 else np.nan)
    pooled = float(num / np.sqrt(vt * va)) if vt > 0 and va > 0 else float("nan")
    return pooled, float(np.nanmedian(per)) if per else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-null", type=int, default=30)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    import pandas as pd
    from src.community_encoder.train_DESK.validation_core import change_noise
    from src.data.preprocess import bbs
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    split = np.load(c("split.npz"))
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    held_cell = split["holdout"][keys[:, 0], keys[:, 1]]
    train_cell = ~held_cell & ~split["buffer"][keys[:, 0], keys[:, 1]]
    Xc = np.load(c("X_comm.npy")).astype("float64")
    Xd = np.load(c("X_dev.npy")).astype("float64")
    X = np.hstack([Xd, Xc])                                    # targets first (dev), then reference species
    n_dev = Xd.shape[1]
    rng = np.random.default_rng(0)
    res = {"cache": a.cache, "k": a.k, "n_dev": n_dev, "n_all": X.shape[1]}

    # associates from TRAINING cells, time-averaged abundance (early + modern epochs averaged)
    Gtr = groups_for(keys, cid, train_cell, 1966, 1986, 2005, 2025)
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
    lev = 0.5 * (epoch_mean_observed(X, [g[1] for g in Gtr]) + epoch_mean_observed(X, [g[2] for g in Gtr]))
    Ls = (lev - lev.mean(0)) / (lev.std(0) + 1e-12)
    C = Ls.T @ Ls / len(Ls)
    np.fill_diagonal(C, -np.inf)
    present = (X > 0).mean(0) > 0.005
    C[:, ~present] = -np.inf
    assoc = np.argsort(-C, axis=1)[:, : a.k]
    res["assoc_corr_median"] = float(np.median(np.take_along_axis(C, assoc, 1)[:n_dev]))

    def analyse(G, label):
        t = changes(X, G)
        nz = change_noise(t[0][:, :n_dev], t[1][:, :n_dev], t[2][:, :n_dev])
        mask = np.zeros(X.shape[1], bool)
        mask[:n_dev] = nz["resolvable"] & present[:n_dev]
        pooled, med = sharing(t, t, assoc, mask)
        nulls = []
        others = np.where(present)[0]
        for _ in range(a.n_null):
            rnd = np.array([rng.choice(others[others != s], a.k, replace=False) for s in range(X.shape[1])])
            nulls.append(sharing(t, t, rnd, mask)[0])
        nulls = np.array(nulls)
        return {"n_cells": len(G), "n_species": int(mask.sum()), "associates_corr": pooled,
                "associates_corr_median_species": med, "random_corr_mean": float(nulls.mean()),
                "random_corr_p95": float(np.percentile(nulls, 95)),
                "associates_shared_share": pooled ** 2, "random_shared_share": float(nulls.mean()) ** 2}

    # 40-year change, held-out cells, and all cells
    res["change40_heldout"] = analyse(groups_for(keys, cid, held_cell, 1966, 1986, 2005, 2025), "40y held-out")
    res["change40_all"] = analyse(groups_for(keys, cid, np.ones(len(keys), bool), 1966, 1986, 2005, 2025),
                                  "40y all")
    # across species: continent-wide mean change vs associates' mean change
    Gall = groups_for(keys, cid, np.ones(len(keys), bool), 1966, 1986, 2005, 2025)
    tf = changes(X, Gall)[0]
    mean_ch = tf.mean(0)
    sds = tf.std(0) + 1e-12
    sp = np.where(present[:n_dev])[0]
    ma = np.array([(mean_ch[assoc[s]] / sds[assoc[s]]).mean() for s in sp])
    ms = mean_ch[sp] / sds[sp]
    across = float(np.corrcoef(ms, ma)[0, 1])
    nul = []
    others = np.where(present)[0]
    for _ in range(200):
        draws = [rng.choice(others[others != s], a.k, replace=False) for s in sp]
        mr = np.array([(mean_ch[j] / sds[j]).mean() for j in draws])
        nul.append(np.corrcoef(ms, mr)[0, 1])
    res["across_species"] = {"n_species": int(len(sp)), "corr_with_associates": across,
                             "random_mean": float(np.mean(nul)), "random_p95": float(np.percentile(nul, 95))}

    # observer check: within-modern change by observer continuity (all cells)
    runs = bbs.load_run_metadata()
    ry = pd.read_csv(c("route_years.csv"))
    mm = ry.merge(runs, left_on=["country", "state", "route", "year"], right_on=["CountryNum", "StateNum", "Route", "Year"])
    obs = mm.groupby(["row", "col", "year"])["ObsN"].apply(lambda s_: frozenset(s_.astype(int))).to_dict()
    obs_of = lambda rows: frozenset().union(*[obs.get((int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])),
                                                      frozenset()) for i in rows])
    Gm = groups_for(keys, cid, keys[:, 2] != 2015, 2005, 2014, 2016, 2025)
    kind = []
    for g in Gm:
        o1, o2 = obs_of(g[1]), obs_of(g[2])
        kind.append("none" if not o1 or not o2 else
                    ("unchanged" if o1 == o2 else ("replaced" if not (o1 & o2) else "partial")))
    kind = np.array(kind)
    res["observer_11y"] = {}
    for kname in ("unchanged", "replaced"):
        sel = [g for g, k in zip(Gm, kind) if k == kname]
        res["observer_11y"][kname] = analyse(sel, f"11y {kname}")
    with open(os.path.join(a.out, "shared_change.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "shared_change.json"), encoding="utf-8"))
    print(f"shared vs unique change: {r['n_dev']} graded species, K={r['k']} associates (median spatial corr of "
          f"time-averaged abundance with the target {r['assoc_corr_median']:.2f}); noise-free centred corr of a "
          f"species' change with its associates' change, vs K random species")
    for name in ("change40_heldout", "change40_all"):
        v = r[name]
        print(f"  {name:18s} cells {v['n_cells']:4d} species {v['n_species']:3d}: associates {v['associates_corr']:+.3f}"
              f" (median per species {v['associates_corr_median_species']:+.3f}) | random {v['random_corr_mean']:+.3f}"
              f" (p95 {v['random_corr_p95']:+.3f}) | shared share {v['associates_shared_share']:.3f} vs "
              f"{v['random_shared_share']:.3f}")
    v = r["across_species"]
    print(f"  across species (continent-wide mean change, {v['n_species']} species): corr with associates "
          f"{v['corr_with_associates']:+.3f} | random {v['random_mean']:+.3f} (p95 {v['random_p95']:+.3f})")
    for k, v in r["observer_11y"].items():
        print(f"  11-year change, observers {k:9s} cells {v['n_cells']:4d} species {v['n_species']:3d}: associates "
              f"{v['associates_corr']:+.3f} | random {v['random_corr_mean']:+.3f} (p95 {v['random_corr_p95']:+.3f})")


if __name__ == "__main__":
    main()
