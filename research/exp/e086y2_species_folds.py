"""E086y2: reference communities defined by RESIDENCE, scored without leakage by cross-fitting over SPECIES.

    python research/exp/e086y2_species_folds.py --cache <tempho cache> --cbc cbc_circles_res.npz
        --similarity species_similarity.csv --out <dir> [--ranks 24 64] [--folds 5]

The reference / dev split came from similarity to the House Finch, so a residence-based kernel (e.g. ALL sedentary
species) must draw on reference AND dev species -- and then the scored species would sit inside the kernel. Here:
  U      = every non-sealed species in the cache (96 reference + 273 dev) with an AVONET migration code; never sealed
           species (not in the cache), never the House Finch.
  folds  = U split into --folds species folds (seeded, stratified by sedentary-common / other).
  pools  ref (the 96 reference species), sed (all sedentary in U), res (sedentary + partial migrants), rand (a random
         subset of U the size of sed, seeded).
For each pool and fold f: the kernel is built from (pool minus fold f); the TARGETS are fold f's species. So every
species is scored by a kernel that never contained it, and every pool is scored on the same species. Kernels: breed
(BBS) and sum (breed + winter on the same species where CBC carries them). Everything else as E086y: joint cells
(BBS and CBC in both epochs), Nystrom (3,000 landmarks), odd-year features / even-year targets, readouts change and level
cross-fitted over 6x6 spatial blocks, BLR one block per half, no intercept; scores pooled over the targets of a
stratum: noise-free change corr and capture (mean predicted / mean observed change).
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE)); sys.path.insert(0, _HERE)
from lib import blr  # noqa: E402
from e086y_kernel_screen import nystrom  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--cbc", required=True)
    ap.add_argument("--similarity", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ranks", type=int, nargs="+", default=[24, 64]); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-land", type=int, default=3000); ap.add_argument("--n-boot", type=int, default=300)
    a = ap.parse_args()
    import pandas as pd
    j = lambda f: os.path.join(a.cache, f)
    meta = json.load(open(j("meta.json"))); comm, dev = list(meta["community"]), list(meta["dev_species"])
    sealed = set(meta.get("sealed_species", []))
    keys = np.load(j("keys.npy")); r, c, y = keys[:, 0], keys[:, 1], keys[:, 2]
    Xall = np.concatenate([np.load(j("X_comm.npy")).astype(np.float64),
                           np.asarray(np.load(j("X_dev.npy"), mmap_mode="r"), np.float64)], 1)
    names = comm + dev
    sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    U = [k for k, s in enumerate(names) if mig.get(s) in (1.0, 2.0, 3.0)]
    assert "houfin" not in names and not sealed & set(names)
    prev = (Xall > 0).mean(0)
    sed = np.array([mig.get(s) == 1.0 for s in names]); sc = sed & (prev >= 0.08)
    rng = np.random.default_rng(86)
    # stratified species folds over U
    fold = np.full(len(names), -1)
    for grp in (np.array([k for k in U if sc[k]]), np.array([k for k in U if not sc[k]])):
        p = rng.permutation(grp); fold[p] = np.arange(len(p)) % a.folds
    pools = {"ref": set(range(len(comm))), "sed": {k for k in U if sed[k]},
             "res": {k for k in U if mig.get(names[k]) in (1.0, 2.0)}}
    pools["rand"] = set(rng.choice(U, len(pools["sed"]), replace=False).tolist())
    # winter: CBC counts of every U species the file carries, party-hour standardized, 1966+
    D = np.load(a.cbc, allow_pickle=False); cs = list(D["species"])
    ph, yr = D["party_hours"].astype(float), D["year_end"]
    okc = (D["row"] >= 0) & (yr >= 1966) & (yr <= 2025) & np.isfinite(ph) & (ph > 0)
    wcol = {k: cs.index(names[k]) for k in U if names[k] in cs}
    wk = sorted(wcol)
    Yw = D["Y"][okc][:, [wcol[k] for k in wk]].astype(float) * ((np.median(ph[okc]) / ph[okc])[:, None] ** 0.8)
    df = pd.DataFrame(Yw); df["r"], df["c"], df["y"] = D["row"][okc], D["col"][okc], yr[okc]
    g = df.groupby(["r", "c", "y"]).mean()
    wr, wc, wy = (np.array([q[i] for q in g.index]) for i in range(3)); Xw = np.log1p(g.to_numpy())
    wpos = {k: i for i, k in enumerate(wk)}
    Lb = np.log1p(Xall)
    ep = lambda yy: np.where(yy <= 1995, 0, 1)
    bcell = r.astype(np.int64) * 1000 + c; wcell = wr.astype(np.int64) * 1000 + wc
    hb = {(int(q), int(e)) for q, e in zip(bcell, ep(y))}; hw = {(int(q), int(e)) for q, e in zip(wcell, ep(wy))}
    cells = np.array(sorted({q for q, e in hb if (q, 0) in hb and (q, 1) in hb and (q, 0) in hw and (q, 1) in hw}))
    pos = {q: i for i, q in enumerate(cells)}
    def cell_means(V, cell, yy, parity):
        out = np.full((len(cells), 2, V.shape[1]), np.nan)
        m = np.array([q in pos for q in cell]) & ((yy % 2) == parity)
        idx = np.array([pos[q] for q in cell[m]]); e = ep(yy[m])
        for q in (0, 1):
            s = np.zeros((len(cells), V.shape[1])); n = np.zeros(len(cells))
            np.add.at(s, idx[e == q], V[m][e == q]); np.add.at(n, idx[e == q], 1)
            out[:, q] = np.where(n[:, None] > 0, s / np.maximum(n, 1)[:, None], np.nan)
        return out
    TA, TB = cell_means(Lb, bcell, y, 1), cell_means(Lb, bcell, y, 0)
    mb = np.array([q in pos for q in bcell]) & ((y % 2) == 1); mw = np.array([q in pos for q in wcell]) & ((wy % 2) == 1)
    lb = rng.choice(len(Lb), min(a.n_land, len(Lb)), replace=False); lw = rng.choice(len(Xw), min(a.n_land, len(Xw)), replace=False)
    blk = (cells // 1000 // 6) * 1000 + (cells % 1000) // 6
    ub = np.unique(blk); fb_ = {b: i % 5 for i, b in enumerate(rng.permutation(ub))}; sfold = np.array([fb_[b] for b in blk])
    W = np.vstack([np.ones(len(cells)), rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub)), size=a.n_boot)[
        :, np.searchsorted(ub, blk)]])
    strata = {"sed_common": sc, "sedentary": sed, "all": np.ones(len(names), bool)}
    print(f"[e086y2] {len(cells)} joint cells; U {len(U)} species (sedentary common {int(sc[U].sum())}); pools "
          + ", ".join(f"{k} {len(v)}" for k, v in pools.items()) + f"; winter carries {len(wk)}", flush=True)
    res = {"n_cells": int(len(cells)), "pools": {k: sorted(names[i] for i in v) for k, v in pools.items()},
           "folds": {names[k]: int(fold[k]) for k in U}, "results": {}}
    for rk in a.ranks:
        for pn, pool in pools.items():
            P = {ro: {kn: np.full((len(cells), len(names)), np.nan) for kn in ("breed", "sum")} for ro in ("level", "change")}
            for f in range(a.folds):
                ks = sorted(k for k in pool if fold[k] != f)                 # kernel species: pool minus fold f
                tg = [k for k in U if fold[k] == f]                          # targets: fold f
                kw = [wpos[k] for k in ks if k in wpos]
                fbn = nystrom(Lb[lb][:, ks], rk); fwn = nystrom(Xw[lw][:, kw], rk)
                Pb = np.full((len(Lb), rk), np.nan); Pb[mb] = fbn(Lb[mb][:, ks])
                Pw = np.full((len(Xw), rk), np.nan); Pw[mw] = fwn(Xw[mw][:, kw])
                Fb, Fw = cell_means(Pb, bcell, y, 1), cell_means(Pw, wcell, wy, 1)
                for kn, parts in (("breed", [Fb]), ("sum", [Fb, Fw])):
                    F = np.concatenate(parts, 2); bl = [(i * rk, (i + 1) * rk) for i in range(len(parts))]
                    ok = np.isfinite(F).all((1, 2)) & np.isfinite(TA[:, :, tg]).all((1, 2)) & np.isfinite(TB[:, :, tg]).all((1, 2))
                    dF = F[:, 1] - F[:, 0]
                    for sf in range(5):
                        tr, te = ok & (sfold != sf), ok & (sfold == sf)
                        ml = blr.fit(np.concatenate([F[tr, 0], F[tr, 1]]), np.concatenate([TB[tr, 0][:, tg], TB[tr, 1][:, tg]]), bl)
                        P["level"][kn][np.ix_(te, tg)] = dF[te] @ np.asarray(ml["coef"]).T
                        mc = blr.fit(dF[tr], TB[tr, 1][:, tg] - TB[tr, 0][:, tg], bl)
                        P["change"][kn][np.ix_(te, tg)] = dF[te] @ np.asarray(mc["coef"]).T
            dA, dB = TA[:, 1] - TA[:, 0], TB[:, 1] - TB[:, 0]
            for ro in ("level", "change"):
                for kn in ("breed", "sum"):
                    Pm = P[ro][kn]; ok = np.isfinite(Pm[:, U]).all(1)
                    out = {}
                    for st, smask in strata.items():
                        m = np.array([k for k in U if smask[k]])
                        Pc_, A_, B_ = Pm[ok][:, m], dA[ok][:, m], dB[ok][:, m]
                        cors = []
                        for w in W[:, ok]:
                            sw = w.sum(); mu = lambda X: (w @ X) / sw
                            p_, a_, b_ = Pc_ - mu(Pc_), A_ - mu(A_), B_ - mu(B_)
                            cors.append((w @ (p_ * b_)).sum() / np.sqrt((w @ (p_ * p_)).sum() * max((w @ (a_ * b_)).sum(), 1e-12)))
                        out[st] = {"corr": float(cors[0]), "corr_ci": [float(x) for x in np.percentile(cors[1:], [2.5, 97.5])],
                                   "capture": float(Pc_.mean() / B_.mean()), "mean_obs": float(B_.mean()), "n_species": int(len(m))}
                    res["results"][f"r{rk}|{pn}|{kn}|{ro}"] = out
                    o = out["sed_common"]
                    print(f"  r{rk:<3d} pool {pn:5s} {kn:5s} {ro:6s} sed_common corr {o['corr']:+.3f} "
                          f"[{o['corr_ci'][0]:+.3f},{o['corr_ci'][1]:+.3f}] capture {o['capture']:+.2f} (obs {o['mean_obs']:+.3f}, "
                          f"{o['n_species']} sp) | sedentary corr {out['sedentary']['corr']:+.3f} capture {out['sedentary']['capture']:+.2f}",
                          flush=True)
            os.makedirs(a.out, exist_ok=True)
            json.dump(res, open(os.path.join(a.out, "e086y2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
