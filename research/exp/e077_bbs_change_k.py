"""E077: does the CBC presence term shrink DESK's BBS species-change magnitude toward calibration (k -> 1) where land
cover changed most? (peer question after E076: representation fix, or only the winter readout?)

    python research/exp/e077_bbs_change_k.py --arm <cache> --control <cache> --cover cover_change_1920_1966.npz
        --similarity species_similarity.csv --out DIR --label NAME

Test cells: DESK held-out + buffer cells (never in BBS training nor in the presence term) with >= 2 surveyed years in
each window. OBSERVED change per cell and dev species: mean log1p count over surveyed years 2005-2024 minus 1966-1985
(the withheld years). PREDICTED change: (mean z_ema 2005-2024 - mean z_ema 1966-1985 at the cell, dense cube) @ W,
W = a WITHIN-cell ridge readout fitted on training cell-years from trained years (>= 1996): deviations of log1p dev
counts from the cell's trained-year mean on deviations of z_ema from its cell mean, lambda by 5-fold CV over spatial
blocks -- fitted per cache (arm and control each get their own). Dev species with >= 30 non-zero test cell-years.
k = slope of observed on predicted change, pooled over species x cells after removing each species' mean change over
the test cells (the regional component; k < 1 = over-prediction); corr likewise. Strata: cover-change TOP tercile vs
rest, Great Plains vs rest. 95% bootstrap over 6x6-cell blocks for arm - control.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e069_presence_score import ridge_cv  # noqa: E402

W0, W1 = (1966, 1985), (2005, 2024)


def load(cache, sp_cols=None):
    meta = json.load(open(os.path.join(cache, "meta.json")))
    keys = np.load(os.path.join(cache, "keys.npy"))
    X = np.load(os.path.join(cache, "X_dev.npy"), mmap_mode="r")
    zk = np.load(os.path.join(cache, "z_ema.npy")).astype(float)
    Z = np.load(os.path.join(cache, "z_ema_all.npy"), mmap_mode="r")
    cells = np.load(os.path.join(cache, "cells.npy")); years = np.load(os.path.join(cache, "years.npy"))
    kci = np.load(os.path.join(cache, "key_cell_index.npy")); spl = np.load(os.path.join(cache, "split.npz"))
    return meta, keys, X, zk, Z, cells, years, kci, spl


def prepare(cache, cover, zones, t_top):
    meta, keys, X, zk, Z, cells, years, kci, spl = load(cache)
    dev = list(meta["dev_species"])
    assert "houfin" not in dev and not set(dev) & set(meta["sealed_species"])
    L = np.log1p(np.asarray(X, float))
    tr = spl["is_train"] & (keys[:, 2] >= 1996)
    nC = len(cells)
    cnt = np.bincount(kci[tr], minlength=nC).astype(float)
    zbar = np.zeros((nC, zk.shape[1])); np.add.at(zbar, kci[tr], zk[tr]); zbar /= np.maximum(cnt, 1)[:, None]
    lbar = np.zeros((nC, L.shape[1])); np.add.at(lbar, kci[tr], L[tr]); lbar /= np.maximum(cnt, 1)[:, None]
    dZ, dL = zk[tr] - zbar[kci[tr]], L[tr] - lbar[kci[tr]]
    ro, lam = ridge_cv(dZ, dL, spl["block_id"][tr], [1e-3, 1e-2, 1e-1, 1, 10])
    test_grid = spl["holdout"] | spl["buffer"]
    is_test = test_grid[cells[:, 0], cells[:, 1]]
    yk = keys[:, 2]
    m0, m1 = (yk >= W0[0]) & (yk <= W0[1]), (yk >= W1[0]) & (yk <= W1[1])
    n0 = np.bincount(kci[m0], minlength=nC); n1 = np.bincount(kci[m1], minlength=nC)
    use = is_test & (n0 >= 2) & (n1 >= 2)
    s0 = np.zeros((nC, L.shape[1])); np.add.at(s0, kci[m0], L[m0])
    s1 = np.zeros((nC, L.shape[1])); np.add.at(s1, kci[m1], L[m1])
    obs = s1 / np.maximum(n1, 1)[:, None] - s0 / np.maximum(n0, 1)[:, None]
    ya = lambda a, b: (years >= a) & (years <= b)
    dz = np.asarray(Z[:, ya(*W1)].mean(1), float) - np.asarray(Z[:, ya(*W0)].mean(1), float)
    pred = ro(dz) - ro(np.zeros_like(dz))         # a change: intercept removed
    rows = (m0 | m1) & use[kci]
    nz = (L[rows] > 0).sum(0)                     # non-zero test cell-years per species
    ix = np.where(use)[0]
    cov = cover[cells[ix, 0], cells[ix, 1]]
    return {"cells": ix, "obs": obs[ix], "pred": pred[ix], "nz": nz, "lam": lam,
            "top": cov > t_top, "covok": np.isfinite(cov), "gp": zones[cells[ix, 0], cells[ix, 1]] == 2,
            "block": (cells[ix, 0] // 6) * 1000 + cells[ix, 1] // 6, "dev": dev}


def k_corr(o, p, cmask, smask):
    o, p = o[cmask][:, smask], p[cmask][:, smask]
    o = o - o.mean(0); p = p - p.mean(0)
    return float((o * p).sum() / max((p * p).sum(), 1e-12)), float((o * p).sum() / np.sqrt((o * o).sum() * (p * p).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True); ap.add_argument("--control", required=True)
    ap.add_argument("--cover", required=True); ap.add_argument("--out", required=True); ap.add_argument("--label", default="arm")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--similarity", default=None,
                    help="E077b: species_similarity.csv; species are then split by AVONET migration class")
    a = ap.parse_args()
    M = np.load(a.cover)
    A = prepare(a.arm, M["change"], M["zones"], M["terciles"][1])
    C = prepare(a.control, M["change"], M["zones"], M["terciles"][1])
    assert (A["cells"] == C["cells"]).all()
    base = A["nz"] >= 30
    classes = {"all_species": base}
    if a.similarity:
        import pandas as pd
        sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
        mv = np.array([mig.get(s, np.nan) for s in A["dev"]])
        classes = {"sedentary": base & (mv == 1), "residents": base & np.isin(mv, (1, 2)),
                   "migratory": base & (mv == 3), "all_species": base}
    res = {"label": a.label, "test_cells": int(len(A["cells"])), "lambda_arm": A["lam"], "lambda_control": C["lam"]}
    blocks = A["block"]; ub = np.unique(blocks)
    rng = np.random.default_rng(77)
    boots = [np.concatenate([np.where(blocks == b)[0] for b in rng.choice(ub, len(ub))]) for _ in range(a.boot)]
    f = lambda o, p: ((o - o.mean(0)) * (p - p.mean(0))).sum()
    g = lambda p: ((p - p.mean(0)) ** 2).sum()
    h = lambda o: ((o - o.mean(0)) ** 2).sum()
    for cname, smask in classes.items():
        rc = {"species": int(smask.sum())}
        for name, m in (("all", np.ones(len(blocks), bool)), ("cover_top", A["top"] & A["covok"]),
                        ("cover_rest", ~A["top"] & A["covok"]), ("great_plains", A["gp"]), ("not_gp", ~A["gp"])):
            ka, ca = k_corr(A["obs"], A["pred"], m, smask)
            kc, cc = k_corr(C["obs"], C["pred"], m, smask)
            dk, dc = [], []
            for bi in boots:
                idx = bi[m[bi]]                   # resampled blocks, with multiplicity, restricted to the stratum
                if len(idx) < 5:
                    continue
                o_a, p_a = A["obs"][idx][:, smask], A["pred"][idx][:, smask]
                o_c, p_c = C["obs"][idx][:, smask], C["pred"][idx][:, smask]
                dk.append(f(o_a, p_a) / g(p_a) - f(o_c, p_c) / g(p_c))
                dc.append(f(o_a, p_a) / np.sqrt(h(o_a) * g(p_a)) - f(o_c, p_c) / np.sqrt(h(o_c) * g(p_c)))
            rc[name] = {"cells": int(m.sum()), "k_arm": ka, "k_control": kc, "corr_arm": ca, "corr_control": cc,
                        "pred_sd_ratio_arm_over_control": float(np.std(A["pred"][m][:, smask]) /
                                                                np.std(C["pred"][m][:, smask])),
                        "dk": ka - kc, "dk_ci95": [float(x) for x in np.percentile(dk, [2.5, 97.5])],
                        "dcorr": ca - cc, "dcorr_ci95": [float(x) for x in np.percentile(dc, [2.5, 97.5])]}
        res[cname] = rc
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"e077{'b' if a.similarity else ''}_{a.label}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
