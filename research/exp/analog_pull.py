"""E017 (S2): where do DESK's backcast errors point? Toward space-for-time ANALOGS -- the modern states of places whose
modern covariates match the cell's past covariates?

    python research/exp/analog_pull.py --cache <dir> --out <dir> [--population withheld] [--ranks 12 24 64]
    python research/exp/analog_pull.py --summarize <out>

The worry (user, S2): in a readout crowded with modern cell-years, a backcast state that leaves its own modern state
lands near someone else's, and the readout then borrows that place's habitat -- confidently (S1). If DESK reads a
cell's past covariates through spatial associations, it lands near the cell's covariate analogs.

For each held-out cell x (early group e, modern group m), in ESK coordinates (first r):
    analogs   the k modern cell-epochs (other cells, optionally > --exclude-km away) whose modern covariates (64 PCs)
              are nearest x's EARLY covariates;  random = k random modern cell-epochs (the null)
    v         = analog mean state - x's true modern state           (the analog direction)
    pull(P)   = sum <dP, v_B> / sum <v_A, v_B>   how far predictor P moves x toward its analogs, as a fraction of the
              (noise-free: halves A, B) squared distance; for the TRUE change, <dT_A, v_B> (independent halves)
    align(P)  = the same as a correlation (direction only)
Predictors: DESK; the linear space-for-time map (between-cell coefficients; positive control -- it IS space-for-time);
the within-cell map; the truth. S2 predicts pull(DESK) > pull(truth), and toward covariate analogs more than toward
random cells. NEIGHBOURS: each predictor's backcast point's k nearest modern cell-epochs in DESK space (the readout's
training points) and the noise-free TRUE squared distance from x's real early state to them (lower = the readout
borrows from places that really resemble x's past); no-change and the true early state's own neighbours as references.
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
    modern = np.where((E["epoch"] == 1) & ~E["withheld"])[0]               # candidate analogs / neighbours

    P = covfeat.cov_pcs(np.load(c("F_cov.npy"), mmap_mode="r"), is_train)
    Pbar = covfeat.cell_means(keys, P, trained_year)
    gm = lambda A: np.stack([np.asarray(A[np.asarray(g)], "float64").mean(0) for g in full])
    Pg = gm(P)
    rng = np.random.default_rng(0)
    # analogs: modern cell-epochs nearest x's EARLY covariates (other cells, optional exclusion radius)
    an, rnd = [], []
    for x in ev:
        e = pairs[x, 0]
        d = ((Pg[modern] - Pg[e]) ** 2).sum(1)
        dist_km = np.hypot(*(E["cells"][modern] - cells[x]).T.astype(float)) * 27.0
        bad = (dist_km <= max(a.exclude_km, 1e-9)) | (np.all(E["cells"][modern] == cells[x], axis=1))
        d[bad] = np.inf
        an.append(modern[np.argsort(d)[: a.k_analogs]])
        ok = modern[~bad]
        rnd.append(rng.choice(ok, a.k_analogs, replace=False))
    an, rnd = np.array(an), np.array(rnd)

    Z = np.load(c("esk_annual.npy")).astype("float64")
    R = max(a.ranks)
    beta_pool, beta_b, beta_w = covfeat.fit_maps(P, Pbar, Z[:, :R], is_train, blr)
    dPg = (Pg[pairs[:, 1]] - Pg[pairs[:, 0]])[ev]
    D = gm(np.load(c("z_raw.npy"), mmap_mode="r"))
    T = {h: E[f"z_{h}"].astype("float64") for h in ("a", "b", "full")}
    blocks = (cells[ev, 0] // 6) * 100000 + cells[ev, 1] // 6
    ub, binv = np.unique(blocks, return_inverse=True)
    W = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)
    res = {"cache": a.cache, "population": a.population, "n_cells": int(len(ev)), "k_analogs": a.k_analogs,
           "exclude_km": a.exclude_km, "ranks": {}}
    e_ix, m_ix = pairs[ev, 0], pairs[ev, 1]
    for r in a.ranks:
        Tr = {h: v[:, :r] for h, v in T.items()}
        Dr = D[:, :r]
        out = {"pull": {}, "neighbours": {}}
        for target, A_ in (("analogs", an), ("random", rnd)):
            vA = Tr["a"][A_].mean(1) - Tr["a"][m_ix]
            vB = Tr["b"][A_].mean(1) - Tr["b"][m_ix]
            vv = (vA * vB).sum(1)
            preds = {"desk": Dr[e_ix] - Dr[m_ix], "between": -(dPg @ beta_b[:, :r]),
                     "within": -(dPg @ beta_w[:, :r]), "pooled": -(dPg @ beta_pool[:, :r])}
            # a predictor's backcast move is early - modern = -(predicted early->modern change)
            tA, tB = Tr["a"][e_ix] - Tr["a"][m_ix], Tr["b"][e_ix] - Tr["b"][m_ix]
            rows = {}
            for name, dp in preds.items():
                num = (dp * vB).sum(1)
                rows[name] = (num, (dp * dp).sum(1))
            rows["truth"] = (0.5 * ((tA * vB).sum(1) + (tB * vA).sum(1)), (tA * tB).sum(1))

            def stat(num, nrm, w):
                den_v = (w * vv).sum()
                return ((w * num).sum() / den_v if den_v > 0 else np.nan,
                        (w * num).sum() / np.sqrt((w * nrm).sum() * den_v) if den_v > 0 and (w * nrm).sum() > 0
                        else np.nan)
            o = {}
            ones = np.ones(len(ev))
            for name, (num, nrm) in rows.items():
                p, al = stat(num, nrm, ones)
                bp = np.array([stat(num, nrm, w[binv].astype(float))[0] for w in W])
                o[name] = {"pull": float(p), "align": float(al),
                           "pull_ci": [float(np.nanpercentile(bp, 2.5)), float(np.nanpercentile(bp, 97.5))]}
            out["pull"][target] = o
        # neighbours in DESK space among modern cell-epochs (not x itself)
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
        # reference: the true early state's own nearest modern cell-epochs (noise-free distance ranking)
        tot = 0.0
        for i, x in enumerate(ev):
            e = e_ix[i]
            d2 = ((Tr["a"][e] - Tr["a"][modern]) * (Tr["b"][e] - Tr["b"][modern])).sum(1)
            d2[np.all(E["cells"][modern] == cells[x], axis=1)] = np.inf
            nb = np.argsort(d2)[: a.k_neighbours]
            tot += float(d2[nb].mean())
        out["neighbours"]["true_early_own"] = tot / len(ev)
        res["ranks"][str(r)] = out
    with open(os.path.join(a.out, "analog_pull.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "analog_pull.json"), encoding="utf-8"))
    print(f"analog pull [{r['population']}] {r['n_cells']} held-out cells, {r['k_analogs']} covariate analogs "
          f"(exclusion {r['exclude_km']:.0f} km)")
    for rk, o in r["ranks"].items():
        print(f"  r{rk}: pull toward analogs (random cells) [95% CI] / alignment")
        for name in ("truth", "desk", "between", "within", "pooled"):
            pa, pr = o["pull"]["analogs"].get(name), o["pull"]["random"].get(name)
            if pa:
                print(f"    {name:8s} {pa['pull']:+.3f} [{pa['pull_ci'][0]:+.3f}, {pa['pull_ci'][1]:+.3f}] "
                      f"({pr['pull']:+.3f})   align {pa['align']:+.3f} ({pr['align']:+.3f})")
        nb = o["neighbours"]
        print("    neighbour misfit (true sq. distance from x's real past to the readout's nearest modern points): "
              + "  ".join(f"{k} {v:.3f}" for k, v in nb.items()))


if __name__ == "__main__":
    main()
