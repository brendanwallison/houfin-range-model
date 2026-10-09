"""E017 (S2): where do DESK's backcast moves point -- along the SPACE-FOR-TIME direction?

    python research/exp/analog_pull.py --cache <dir> --out <dir> [--population withheld] [--ranks 12 24 64]
        [--exclude-km 200]
    python research/exp/analog_pull.py --summarize <out>

The worry (user, S2): in a readout crowded with modern cell-years, a backcast state that leaves its own modern state
lands near someone else's, and the readout then borrows that place's habitat -- confidently (S1). If DESK reads a
cell's past covariates through spatial associations, its backcast moves along the space-for-time direction.

For each held-out cell x, in ESK coordinates (first r):
    space-for-time direction  v = T(analogs of x's EARLY covariates) - T(analogs of x's MODERN covariates)
        analogs = the k modern cell-epochs of OTHER cells (optionally > --exclude-km away) whose modern covariates
        (64 PCs) are nearest; x's own idiosyncrasy cancels (the first version anchored v at x's own modern state, and
        ~80% of it was that idiosyncrasy -- skeptic, 2026-10-09). Halves A, B of the analog cells give v_A, v_B.
    a predictor's backcast move  m = state(x, early) - state(x, modern): DESK; the linear covariate maps (E016:
        between = space-for-time, the positive control; within; pooled); the TRUTH (halves of x itself)
    align = sum <m, v_B> / sqrt(sum |m|^2 . sum <v_A, v_B>)   (truth: cross halves both ways)
    pull  = sum <m, v_B> / sum <v_A, v_B>                      fraction of the space-for-time move made
CENTERED across cells by default (a shared continental shift must not count), uncentered also reported. Null: the
same with two RANDOM sets of modern cell-epochs in place of the two analog sets (~0 for everything).
Neighbour misfit: the noise-free true squared distance from x's real past to the readout's 20 nearest modern
cell-epochs (DESK space), for DESK's backcast point, the no-change point and the maps' (no winner's-curse reference).
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

PREDICTORS = ("truth", "desk", "between", "within", "pooled")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--population", default="withheld", choices=("trained", "withheld"))
    ap.add_argument("--ranks", type=int, nargs="+", default=(12, 24, 64))
    ap.add_argument("--k-analogs", type=int, default=10)
    ap.add_argument("--k-neighbours", type=int, default=20)
    ap.add_argument("--exclude-km", type=float, default=0.0)
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
    withheld = [int(y) for y in split["withheld"]]
    E, full, _, _ = group_rows(a.cache, keys, withheld)
    is_train = split["is_train"].astype(bool)
    trained_year = ~np.isin(keys[:, 2], withheld)
    want_wh = a.population == "withheld"
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh) == want_wh:
            idx.setdefault(cell, [None, None])[0] = j
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = j
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    cells = E["cells"][pairs[:, 0]]
    ev = np.where(split["holdout"][cells[:, 0], cells[:, 1]])[0]
    modern = np.where((E["epoch"] == 1) & ~E["withheld"])[0]

    P = covfeat.cov_pcs(np.load(c("F_cov.npy"), mmap_mode="r"), is_train)
    Pbar = covfeat.cell_means(keys, P, trained_year)
    gm = lambda A: np.stack([np.asarray(A[np.asarray(g)], "float64").mean(0) for g in full])
    Pg = gm(P)
    rng = np.random.default_rng(0)
    e_ix, m_ix = pairs[ev, 0], pairs[ev, 1]
    an_e, an_m, rn_e, rn_m, an_km = [], [], [], [], []
    for i, x in enumerate(ev):
        dist_km = np.hypot(*(E["cells"][modern] - cells[x]).T.astype(float)) * 27.0
        bad = (dist_km <= max(a.exclude_km, 1e-9)) | np.all(E["cells"][modern] == cells[x], axis=1)
        for src, dst in ((e_ix[i], an_e), (m_ix[i], an_m)):
            d = ((Pg[modern] - Pg[src]) ** 2).sum(1)
            d[bad] = np.inf
            pick = np.argsort(d)[: a.k_analogs]
            dst.append(modern[pick])
            if dst is an_e:
                an_km.append(float(np.median(dist_km[pick])))
        ok = modern[~bad]
        rn_e.append(rng.choice(ok, a.k_analogs, replace=False))
        rn_m.append(rng.choice(ok, a.k_analogs, replace=False))
    an_e, an_m, rn_e, rn_m = map(np.array, (an_e, an_m, rn_e, rn_m))

    Z = np.load(c("esk_annual.npy")).astype("float64")
    R = max(a.ranks)
    beta_pool, beta_b, beta_w = covfeat.fit_maps(P, Pbar, Z[:, :R], is_train, blr)
    dPg = (Pg[pairs[:, 1]] - Pg[pairs[:, 0]])[ev]                       # x's covariate change, early -> modern
    D = gm(np.load(c("z_raw.npy"), mmap_mode="r"))
    T = {h: E[f"z_{h}"].astype("float64") for h in ("a", "b")}
    blocks = (cells[ev, 0] // 6) * 100000 + cells[ev, 1] // 6
    ub, binv = np.unique(blocks, return_inverse=True)
    Wb = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)
    res = {"cache": a.cache, "population": a.population, "n_cells": int(len(ev)), "k_analogs": a.k_analogs,
           "exclude_km": a.exclude_km, "analog_median_km": float(np.median(an_km)), "ranks": {}}
    for r in a.ranks:
        Tr = {h: v[:, :r] for h, v in T.items()}
        Dr = D[:, :r]
        moves = {"desk": Dr[e_ix] - Dr[m_ix], "between": -(dPg @ beta_b[:, :r]),
                 "within": -(dPg @ beta_w[:, :r]), "pooled": -(dPg @ beta_pool[:, :r])}
        tA, tB = Tr["a"][e_ix] - Tr["a"][m_ix], Tr["b"][e_ix] - Tr["b"][m_ix]
        out = {"centered": {}, "uncentered": {}, "neighbours": {}}
        for target, (S_e, S_m) in (("space_for_time", (an_e, an_m)), ("random", (rn_e, rn_m))):
            vA = Tr["a"][S_e].mean(1) - Tr["a"][S_m].mean(1)
            vB = Tr["b"][S_e].mean(1) - Tr["b"][S_m].mean(1)
            for mode in ("centered", "uncentered"):
                def stats(w):
                    if mode == "centered":
                        cen = lambda v: v - (w[:, None] * v).sum(0) / w.sum()
                    else:
                        cen = lambda v: v
                    va, vb, ta, tb = cen(vA), cen(vB), cen(tA), cen(tB)
                    vv = (w * (va * vb).sum(1)).sum()
                    o = {}
                    for name in PREDICTORS:
                        if name == "truth":
                            num = 0.5 * ((w * (ta * vb).sum(1)).sum() + (w * (tb * va).sum(1)).sum())
                            nrm = (w * (ta * tb).sum(1)).sum()
                        else:
                            mv = cen(moves[name])
                            num = (w * (mv * vb).sum(1)).sum()
                            nrm = (w * (mv * mv).sum(1)).sum()
                        o[name] = ((num / vv) if vv > 0 else np.nan,
                                   (num / np.sqrt(nrm * vv)) if vv > 0 and nrm > 0 else np.nan)
                    return o
                point = stats(np.ones(len(ev)))
                boots = [stats(w[binv].astype(float)) for w in Wb]
                o = {}
                for name in PREDICTORS:
                    al = np.array([b[name][1] for b in boots])
                    o[name] = {"pull": float(point[name][0]), "align": float(point[name][1]),
                               "align_ci": [float(np.nanpercentile(al, 2.5)), float(np.nanpercentile(al, 97.5))]}
                out[mode][target] = o
        backcasts = {"desk": Dr[e_ix], "no_change": Dr[m_ix],
                     "between": Dr[m_ix] - dPg @ beta_b[:, :r], "within": Dr[m_ix] - dPg @ beta_w[:, :r]}
        Dm = Dr[modern]
        for name, B in backcasts.items():
            tot = 0.0
            for i, x in enumerate(ev):
                d2 = ((Dm - B[i]) ** 2).sum(1)
                d2[np.all(E["cells"][modern] == cells[x], axis=1)] = np.inf
                nb = modern[np.argsort(d2)[: a.k_neighbours]]
                e = e_ix[i]
                tot += float((((Tr["a"][e] - Tr["a"][nb]) * (Tr["b"][e] - Tr["b"][nb])).sum(1)).mean())
            out["neighbours"][name] = tot / len(ev)
        res["ranks"][str(r)] = out
    with open(os.path.join(a.out, "analog_pull.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "analog_pull.json"), encoding="utf-8"))
    print(f"analog pull v2 [{r['population']}] {r['n_cells']} held-out cells; space-for-time direction = analogs of "
          f"early minus analogs of modern covariates ({r['k_analogs']} each, exclusion {r['exclude_km']:.0f} km, "
          f"median analog distance {r['analog_median_km']:.0f} km)")
    for rk, o in r["ranks"].items():
        print(f"  r{rk}  alignment with the space-for-time direction, centered [95% CI]  (random-direction null) "
              f"| uncentered | pull (centered)")
        for name in PREDICTORS:
            c_, rn, u = (o["centered"]["space_for_time"][name], o["centered"]["random"][name],
                         o["uncentered"]["space_for_time"][name])
            print(f"    {name:8s} {c_['align']:+.3f} [{c_['align_ci'][0]:+.3f}, {c_['align_ci'][1]:+.3f}]  "
                  f"({rn['align']:+.3f})  | {u['align']:+.3f} | {c_['pull']:+.3f}")
        print("    neighbour misfit (true sq. distance from x's real past to the readout's nearest modern points): "
              + "  ".join(f"{k} {v:.3f}" for k, v in o["neighbours"].items()))


if __name__ == "__main__":
    main()
