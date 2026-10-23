"""E086y: which REFERENCE KERNEL carries the change of sedentary common species? A screen on OBSERVED communities.

    python research/exp/e086y_kernel_screen.py --cache <tempho cache> --cbc cbc_circles_comm.npz
        --similarity species_similarity.csv --out <dir> [--ranks 24 64]

Cells: every grid cell with BBS rows AND CBC counts in BOTH epochs (1966-1995, 1996-2025). Kernels (uncentred Ruzicka on
log1p counts; Nystrom features, 3,000 random landmark cell-years of that season, all cells and years):
  breed      BBS communities of the reference (community) species -- the current kernel
  winter     CBC communities (winters 1966-2025, counts x (median party-hours / party-hours)^0.8) of the reference
             species CBC carries
  sum        [breed, winter] -- the sum kernel K_breed + K_winter; two readout blocks with their own amplitudes
  breed_sc   breed on the SEDENTARY COMMON reference species only (AVONET migration 1, BBS prevalence >= 0.08)
  sum_sc     [breed_sc, winter on the same species]
Rank r per half (sum kernels get r per half; also reported with r/2 per half, equal total dims to breed). Noise kept
apart: features from ODD years (BBS rows / winters), species targets from EVEN years; the noise-free denominator uses
the odd-year half of the target. Species: dev species (never in the reference set), strata sedentary common /
sedentary / all dev. Readouts, cross-fitted over 6x6 spatial blocks (5 folds), iid-prior BLR per block, applied WITHOUT the intercept:
  level   cell-epoch levels (both epochs as rows) -> species level; applied to the cell's feature change
  change  feature change -> species change across cells
Scores per stratum: noise-free pooled corr of change, and CAPTURE = mean predicted change / mean observed change.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from lib import blr  # noqa: E402


def ruz(A, B):
    """Ruzicka between rows of A (n, S) and B (m, S), log1p already applied."""
    out = np.empty((A.shape[0], B.shape[0]))
    for i in range(0, A.shape[0], 200):
        a = A[i:i + 200, None, :]
        out[i:i + 200] = np.minimum(a, B[None]).sum(-1) / np.maximum(np.maximum(a, B[None]).sum(-1), 1e-12)
    return out


def nystrom(X_land, r):
    K = ruz(X_land, X_land)
    w, U = np.linalg.eigh(K)
    o = np.argsort(w)[::-1][:r]
    w, U = np.maximum(w[o], 1e-9), U[:, o]
    return lambda X: ruz(X, X_land) @ U / np.sqrt(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--cbc", required=True)
    ap.add_argument("--similarity", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ranks", type=int, nargs="+", default=[24, 64]); ap.add_argument("--n-land", type=int, default=3000)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--transform", choices=("log1p", "sqrt", "raw"), default="log1p",
                    help="E086y-b: transform of the REFERENCE counts inside the Ruzicka kernel (targets stay log1p)")
    a = ap.parse_args()
    import pandas as pd
    j = lambda f: os.path.join(a.cache, f)
    meta = json.load(open(j("meta.json"))); comm, dev = list(meta["community"]), list(meta["dev_species"])
    assert "houfin" not in comm and "houfin" not in dev
    keys = np.load(j("keys.npy")); r, c, y = keys[:, 0], keys[:, 1], keys[:, 2]
    TF = {"log1p": np.log1p, "sqrt": np.sqrt, "raw": lambda v: v}[a.transform]
    Xc = TF(np.load(j("X_comm.npy")).astype(np.float64))
    Xd = np.asarray(np.load(j("X_dev.npy"), mmap_mode="r")); Ld = np.log1p(Xd.astype(np.float64))
    sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    pc_ref = (np.load(j("X_comm.npy")) > 0).mean(0); pc_dev = (Xd > 0).mean(0)
    ref_sc = [s for k, s in enumerate(comm) if mig.get(s) == 1.0 and pc_ref[k] >= 0.08]
    D = np.load(a.cbc, allow_pickle=False); cs = list(D["species"])
    ph, yr = D["party_hours"].astype(float), D["year_end"]
    okc = (D["row"] >= 0) & (yr >= 1966) & (yr <= 2025) & np.isfinite(ph) & (ph > 0)
    wsp = [s for s in comm if s in cs]
    Yw = D["Y"][okc][:, [cs.index(s) for s in wsp]].astype(float) * ((np.median(ph[okc]) / ph[okc])[:, None] ** 0.8)
    df = pd.DataFrame(Yw); df["r"], df["c"], df["y"] = D["row"][okc], D["col"][okc], yr[okc]
    g = df.groupby(["r", "c", "y"]).mean()
    wr, wc, wy = (np.array([k[i] for k in g.index]) for i in range(3)); Xw = TF(g.to_numpy())
    # joint cells: BBS rows and CBC winters in both epochs
    ep = lambda yy: np.where(yy <= 1995, 0, 1)
    bcell = r.astype(np.int64) * 1000 + c; wcell = wr.astype(np.int64) * 1000 + wc
    has = lambda cell, yy: {(int(k), int(e)) for k, e in zip(cell, ep(yy))}
    hb, hw = has(bcell, y), has(wcell, wy)
    cells = sorted({k for k, e in hb if (k, 0) in hb and (k, 1) in hb and (k, 0) in hw and (k, 1) in hw})
    cells = np.array(cells)
    print(f"[e086y] {len(cells)} joint cells; reference {len(comm)} (sedentary common {len(ref_sc)}); CBC carries {len(wsp)}; "
          f"{len(Xw):,} CBC cell-years", flush=True)
    rng = np.random.default_rng(86)
    lb = rng.choice(len(Xc), min(a.n_land, len(Xc)), replace=False); lw = rng.choice(len(Xw), min(a.n_land, len(Xw)), replace=False)
    sc_b = [comm.index(s) for s in ref_sc]; sc_w = [wsp.index(s) for s in ref_sc if s in wsp]
    # per cell x epoch x parity means
    def cell_means(V, cell, yy, parity):
        out = np.full((len(cells), 2, V.shape[1]), np.nan)
        pos = {k: i for i, k in enumerate(cells)}
        m = np.array([k in pos for k in cell]) & ((yy % 2) == parity)
        idx = np.array([pos[k] for k in cell[m]]); e = ep(yy[m])
        for q in (0, 1):
            s = np.zeros((len(cells), V.shape[1])); n = np.zeros(len(cells))
            np.add.at(s, idx[e == q], V[m][e == q]); np.add.at(n, idx[e == q], 1)
            out[:, q] = np.where(n[:, None] > 0, s / np.maximum(n, 1)[:, None], np.nan)
        return out
    sed = np.array([mig.get(s) == 1.0 for s in dev])
    strata = {"sed_common": sed & (pc_dev >= 0.08), "sedentary": sed, "all_dev": np.ones(len(dev), bool)}
    TA, TB = cell_means(Ld, bcell, y, 1), cell_means(Ld, bcell, y, 0)       # odd / even target halves
    blk = (cells // 1000 // 6) * 1000 + (cells % 1000) // 6
    ub = np.unique(blk); fold = {b: i % 5 for i, b in enumerate(rng.permutation(ub))}
    fid = np.array([fold[b] for b in blk])
    W = np.vstack([np.ones(len(cells)), rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub)), size=a.n_boot)[
        :, np.searchsorted(ub, blk)]])
    res = {"n_cells": int(len(cells)), "ref_sedentary_common": ref_sc, "results": {}}
    for rk in a.ranks:
        fb = nystrom(Xc[lb], rk); fw = nystrom(Xw[lw], rk)
        fbs = nystrom(Xc[lb][:, sc_b], rk); fws = nystrom(Xw[lw][:, sc_w], rk)
        h = rk // 2
        fbh, fwh = nystrom(Xc[lb], h), nystrom(Xw[lw], h)
        feats = {}
        for nm, fn, V, cell, yy in (("breed", fb, Xc, bcell, y), ("winter", fw, Xw, wcell, wy),
                                    ("breed_sc", fbs, Xc[:, sc_b], bcell, y), ("winter_sc", fws, Xw[:, sc_w], wcell, wy),
                                    ("breed_half", fbh, Xc, bcell, y), ("winter_half", fwh, Xw, wcell, wy)):
            pos = set(cells.tolist()); m = np.array([k in pos for k in cell]) & ((yy % 2) == 1)
            Phi = np.full((len(cell), rk if "half" not in nm else h), np.nan); Phi[m] = fn(V[m])
            feats[nm] = cell_means(Phi, cell, yy, 1)
        cands = {"breed": [feats["breed"]], "winter": [feats["winter"]], "sum": [feats["breed"], feats["winter"]],
                 "sum_half": [feats["breed_half"], feats["winter_half"]], "breed_sc": [feats["breed_sc"]],
                 "sum_sc": [feats["breed_sc"], feats["winter_sc"]]}
        for cn, parts in cands.items():
            F = np.concatenate(parts, axis=2); bl, s0 = [], 0
            for p in parts:
                bl.append((s0, s0 + p.shape[2])); s0 += p.shape[2]
            ok = np.isfinite(F).all((1, 2)) & np.isfinite(TA).all((1, 2)) & np.isfinite(TB).all((1, 2))
            dF = F[:, 1] - F[:, 0]
            P_lvl, P_chg = np.full((len(cells), len(dev)), np.nan), np.full((len(cells), len(dev)), np.nan)
            for f in range(5):
                tr, te = ok & (fid != f), ok & (fid == f)
                Xl = np.concatenate([F[tr, 0], F[tr, 1]]); Yl = np.concatenate([TB[tr, 0], TB[tr, 1]])
                ml = blr.fit(Xl, Yl, bl); P_lvl[te] = dF[te] @ np.asarray(ml["coef"]).T
                # coefficients only (no intercept): CAPTURE must be the part of the mean change the FEATURE change explains
                mc = blr.fit(dF[tr], TB[tr, 1] - TB[tr, 0], bl); P_chg[te] = dF[te] @ np.asarray(mc["coef"]).T
            dA, dB = (TA[:, 1] - TA[:, 0])[ok], (TB[:, 1] - TB[:, 0])[ok]
            out = {}
            for ro, P in (("level", P_lvl[ok]), ("change", P_chg[ok])):
                for st, m in strata.items():
                    cors = []
                    for w in W[:, ok]:
                        sw = w.sum(); mu = lambda X: (w @ X) / sw
                        Pc, Ac, Bc = P[:, m] - mu(P[:, m]), dA[:, m] - mu(dA[:, m]), dB[:, m] - mu(dB[:, m])
                        cors.append((w @ (Pc * Bc)).sum() / np.sqrt((w @ (Pc * Pc)).sum() * max((w @ (Ac * Bc)).sum(), 1e-12)))
                    out[f"{ro}|{st}"] = {"corr": float(cors[0]), "corr_ci": [float(x) for x in np.percentile(cors[1:], [2.5, 97.5])],
                                         "capture": float(P[:, m].mean() / dB[:, m].mean()),
                                         "mean_pred": float(P[:, m].mean()), "mean_obs": float(dB[:, m].mean())}
            res["results"][f"r{rk}|{cn}"] = out
            o = out
            print(f"  r{rk:<3d} {cn:9s} sed_common: level corr {o['level|sed_common']['corr']:+.3f} capture "
                  f"{o['level|sed_common']['capture']:+.2f} | change corr {o['change|sed_common']['corr']:+.3f} capture "
                  f"{o['change|sed_common']['capture']:+.2f} | sedentary change corr {o['change|sedentary']['corr']:+.3f} "
                  f"(obs mean {o['level|sed_common']['mean_obs']:+.3f}, n {int(ok.sum())})", flush=True)
    os.makedirs(a.out, exist_ok=True)
    res["transform"] = a.transform
    json.dump(res, open(os.path.join(a.out, "e086y.json" if a.transform == "log1p" else f"e086y_{a.transform}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
