"""E083: does a WINTER (CBC) community block add information about sedentary species' breeding-season abundance and
change, beyond the breeding (BBS) community coordinates? (concatenation vs breeding-only, no DESK training)

    python research/exp/e083_concat_readout.py --cache <research cache> --cbc cbc_circles_comm.npz
        --similarity species_similarity.csv --out DIR

Points: cell-years with both BBS (cache keys) and CBC (party-hour standardized, exponent 0.8, mean over the cell's
circles, winters ending 1966-2024), community species. Blocks:
  B   breeding: the cache's ESK coordinates of the observed BBS community (esk_annual), first 24 dims (the age model's r24)
  C   winter: an uncentred Ruzicka kernel-PCA basis of CBC communities (log1p), Nystrom with 2,000 landmarks drawn from
      TRAINING-cell CBC points, 24 dims (the same contract as ESK, fitted on winter data only)
  Cs  C shuffled across cells within year (a dimension control: same values, association broken)
Arms: B, B+C, B+Cs, C. Targets: log1p BBS counts of DEV species, SEDENTARY primary (AVONET 1), residents, migratory.
Readout: ridge per arm, lambda by 5-fold CV over spatial blocks on TRAINING cells; scored on DESK's held-out + buffer
cells (never in ESK fitting of the readout here).
  LEVEL: per species Spearman across test cell-years of predicted vs observed, median over species (>= 30 non-zero).
  CHANGE: per cell, window means 1966-1985 and 2005-2024 (>= 2 years each in both surveys); a ridge on feature change
          -> species change fitted on training cells, scored on test cells; per species Spearman, median.
Paired differences (B+C - B, B+C - B+Cs) with a 6x6-block bootstrap over test cells (species median recomputed).
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e069_presence_score import ridge_cv  # noqa: E402


def ruz_kernel(A, B):
    A, B = np.log1p(A), np.log1p(B)
    out = np.empty((len(A), len(B)))
    for i in range(0, len(A), 200):
        a = A[i:i + 200, None, :]
        mn = np.minimum(a, B[None]).sum(-1); mx = np.maximum(a, B[None]).sum(-1)
        out[i:i + 200] = np.where(mx > 0, mn / np.maximum(mx, 1e-12), 0.0)
    return out


def med_spearman(P, O, nzmask):
    vals = [spearmanr(P[:, s], O[:, s]).correlation for s in np.where(nzmask)[0]]
    return float(np.nanmedian(vals)), np.array(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--cbc", required=True)
    ap.add_argument("--similarity", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dims", type=int, default=24); ap.add_argument("--landmarks", type=int, default=2000)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--cv-change", action="store_true", help="E083b: CV over all co-located cells for CHANGE")
    ap.add_argument("--min-nz", type=int, default=10)
    a = ap.parse_args()
    import pandas as pd
    C = a.cache
    meta = json.load(open(os.path.join(C, "meta.json")))
    keys = np.load(os.path.join(C, "keys.npy")); esk = np.load(os.path.join(C, "esk_annual.npy"))[:, :a.dims].astype(float)
    Xd = np.log1p(np.load(os.path.join(C, "X_dev.npy")).astype(float))
    spl = np.load(os.path.join(C, "split.npz"))
    dev = list(meta["dev_species"]); assert "houfin" not in dev
    sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    mv = np.array([mig.get(s, np.nan) for s in dev])
    D = np.load(a.cbc, allow_pickle=False)
    comm, cs = list(meta["community"]), list(D["species"])
    sp = [s for s in comm if s in cs]
    yr, ph = D["year_end"], D["party_hours"]
    ok = (D["row"] >= 0) & (yr >= 1966) & (yr <= 2024) & np.isfinite(ph) & (ph > 0)
    Y = D["Y"][:, [cs.index(s) for s in sp]].astype(float)[ok] * ((np.median(ph[ok]) / ph[ok])[:, None] ** 0.8)
    df = pd.DataFrame(Y); df["r"], df["c"], df["y"] = D["row"][ok], D["col"][ok], yr[ok]
    g = df.groupby(["r", "c", "y"]).mean()
    cidx = {k: i for i, k in enumerate(g.index)}; CX = g.to_numpy()
    kk = np.array([cidx.get((int(r), int(c), int(y)), -1) for r, c, y in keys])
    both = kk >= 0
    train_cell = ~spl["holdout"] & ~spl["buffer"]
    tr_key = train_cell[keys[:, 0], keys[:, 1]]
    te_key = (spl["holdout"] | spl["buffer"])[keys[:, 0], keys[:, 1]]
    # winter basis: Nystrom on training-cell CBC points
    rng = np.random.default_rng(83)
    cbc_train_pts = np.unique(kk[both & tr_key])
    L = CX[rng.choice(cbc_train_pts, min(a.landmarks, len(cbc_train_pts)), replace=False)]
    Kll = ruz_kernel(L, L)
    w, V = np.linalg.eigh(Kll); o = np.argsort(-w)[:a.dims]
    proj = V[:, o] / np.sqrt(np.maximum(w[o], 1e-9))
    Zc_pts = ruz_kernel(CX, L) @ proj                     # every CBC cell-year in the winter basis
    rows = np.where(both)[0]
    Zb, Zc = esk[rows], Zc_pts[kk[rows]]
    T, tr, te = Xd[rows], tr_key[rows], te_key[rows]
    yy = keys[rows, 2]
    Zs = Zc.copy()
    for y in np.unique(yy):                               # shuffle the winter block across cells within year
        m = np.where(yy == y)[0]; Zs[m] = Zc[rng.permutation(m)]
    arms = {"B": Zb, "B+C": np.hstack([Zb, Zc]), "B+Cs": np.hstack([Zb, Zs]), "C": Zc}
    blocks_all = spl["block_id"][rows]
    res = {"points": int(len(rows)), "train_points": int(tr.sum()), "test_points": int(te.sum()), "dims": a.dims}
    classes = {"sedentary": mv == 1, "residents": np.isin(mv, (1, 2)), "migratory": mv == 3}
    # LEVEL
    preds = {}
    for nm, F in arms.items():
        ro, lam = ridge_cv(F[tr], T[tr], blocks_all[tr], [1e-3, 1e-2, 1e-1, 1, 10])
        preds[nm] = ro(F[te])
    Ot = T[te]; nz = (Ot > 0).sum(0) >= 30
    cellblock = (keys[rows][te, 0] // 6) * 1000 + keys[rows][te, 1] // 6
    ub = np.unique(cellblock)
    bidx = [np.concatenate([np.where(cellblock == b)[0] for b in rng.choice(ub, len(ub))]) for _ in range(a.boot)]
    level = {}
    for cname, cm in classes.items():
        sm = nz & cm
        r_ = {nm: med_spearman(P, Ot, sm)[0] for nm, P in preds.items()}
        for d1, d0 in (("B+C", "B"), ("B+C", "B+Cs")):
            bs = []
            for bi in bidx:
                bs.append(med_spearman(preds[d1][bi], Ot[bi], sm)[0] - med_spearman(preds[d0][bi], Ot[bi], sm)[0])
            r_[f"{d1} - {d0}"] = [float(r_[d1] - r_[d0])] + [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]
        r_["species"] = int(sm.sum())
        level[cname] = r_
    res["level"] = level
    # CHANGE (cell windows)
    cell_of = keys[rows, 0] * 1000 + keys[rows, 1]
    w0, w1 = (yy >= 1966) & (yy <= 1985), (yy >= 2005) & (yy <= 2024)
    cells_u = np.unique(cell_of)
    feat_d, tgt_d, is_tr, is_te, cb = {nm: [] for nm in arms}, [], [], [], []
    for c in cells_u:
        m0, m1 = (cell_of == c) & w0, (cell_of == c) & w1
        if m0.sum() >= 2 and m1.sum() >= 2:
            for nm, F in arms.items():
                feat_d[nm].append(F[m1].mean(0) - F[m0].mean(0))
            tgt_d.append(T[m1].mean(0) - T[m0].mean(0))
            i0 = np.where(m0)[0][0]; is_tr.append(tr[i0]); is_te.append(te[i0])
            cb.append(blocks_all[i0])
    tgt_d, is_tr, is_te, cb = np.array(tgt_d), np.array(is_tr), np.array(is_te), np.array(cb)
    change = {"cells_train": int(is_tr.sum()), "cells_test": int(is_te.sum())}
    if is_te.sum() >= 30 and is_tr.sum() >= 60:
        pd_ = {}
        for nm in arms:
            F = np.array(feat_d[nm])
            ro, lam = ridge_cv(F[is_tr], tgt_d[is_tr], cb[is_tr], [1e-3, 1e-2, 1e-1, 1, 10])
            pd_[nm] = ro(F[is_te])
        Od = tgt_d[is_te]; nzd = (np.abs(Od) > 0).sum(0) >= 20
        tcells = np.where(is_te)[0]
        tb = cb[is_te]; ubt = np.unique(tb)
        bidx2 = [np.concatenate([np.where(tb == b)[0] for b in rng.choice(ubt, len(ubt))]) for _ in range(a.boot)]
        for cname, cm in classes.items():
            sm = nzd & cm
            r_ = {nm: med_spearman(P, Od, sm)[0] for nm, P in pd_.items()}
            for d1, d0 in (("B+C", "B"), ("B+C", "B+Cs")):
                bs = [med_spearman(pd_[d1][bi], Od[bi], sm)[0] - med_spearman(pd_[d0][bi], Od[bi], sm)[0] for bi in bidx2]
                r_[f"{d1} - {d0}"] = [float(r_[d1] - r_[d0])] + [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]
            r_["species"] = int(sm.sum())
            change[cname] = r_
    res["change"] = change
    if a.cv_change:
        # E083b (after E083's result): 5-fold CV over spatial blocks across ALL co-located cells with both windows
        # (every cell is a test cell once), species with >= --min-nz non-zero changes; species named
        F_all = {nm: np.array(feat_d[nm]) for nm in arms}
        folds = np.random.default_rng(831).integers(5, size=len(np.unique(cb)))
        fmap = dict(zip(np.unique(cb), folds)); fold = np.array([fmap[b] for b in cb])
        pcv = {nm: np.zeros_like(tgt_d) for nm in arms}
        for k_ in range(5):
            trk, tek = fold != k_, fold == k_
            for nm in arms:
                ro, lam = ridge_cv(F_all[nm][trk], tgt_d[trk], cb[trk], [1e-3, 1e-2, 1e-1, 1, 10])
                pcv[nm][tek] = ro(F_all[nm][tek])
        nzc = (np.abs(tgt_d) > 0).sum(0) >= a.min_nz
        ubc = np.unique(cb)
        bidx3 = [np.concatenate([np.where(cb == b)[0] for b in rng.choice(ubc, len(ubc))]) for _ in range(a.boot)]
        cv = {"cells": int(len(tgt_d))}
        for cname, cm in classes.items():
            sm = nzc & cm
            r_ = {nm: med_spearman(P, tgt_d, sm)[0] for nm, P in pcv.items()}
            for d1, d0 in (("B+C", "B"), ("B+C", "B+Cs")):
                bs = [med_spearman(pcv[d1][bi], tgt_d[bi], sm)[0] - med_spearman(pcv[d0][bi], tgt_d[bi], sm)[0]
                      for bi in bidx3]
                r_[f"{d1} - {d0}"] = [float(r_[d1] - r_[d0])] + [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]
            r_["species"] = [dev[i] for i in np.where(sm)[0]]
            per = {nm: med_spearman(P, tgt_d, sm)[1] for nm, P in pcv.items()}
            r_["per_species_B_vs_BC"] = {dev[i]: [float(per["B"][j]), float(per["B+C"][j])]
                                         for j, i in enumerate(np.where(sm)[0])}
            cv[cname] = r_
        res["change_cv"] = cv
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e083b.json" if a.cv_change else "e083.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
