"""E069a: does DESK's BACKCAST z carry the community composition that CBC lists record in years DESK never trained on?

    python research/exp/e069_presence_score.py --cache <research cache> --cbc cbc_circles_comm.npz
        --similarity species_similarity.csv --out DIR

Effort-immune score (presence-only, conditioned within list). For a CBC list at cell q in winter t, each species ON the
list is one record; given a record, P(species s) = softmax_s(eta_s(z) + delta_s) over the species set. A list's own
effort and completeness are shared by every species on it and cancel; only WHICH species it recorded, relative to each
other, is scored. Unlisted species enter only through the normalizer (no absences are assumed).
  eta(z): a ridge readout z -> log1p BBS abundance of the CBC-matchable reference species, fitted on the cache's
          TRAINING cell-years from trained years (>= 1996), lambda by 5-fold CV over the cache's spatial blocks
          (BBS only; no CBC information);
  delta:  species offsets (winter detectability, season, reporting), fitted by maximum likelihood on CBC lists of
          2005-2024 with z at the list's year -- shared by every predictor below.
Eras: 1940-1952 (the output EMA starts in 1940, half-life ~10.5 y: burn-in, reported only), 1953-1965 and 1966-1995.
Predictors of the same early list: BACKCAST z (the cell's z_ema in the list's year), MODERN placebo (the cell's mean
z_ema over 2005-2024: "the community is what it is now"), REGIONAL placebo (modern + the mean backcast shift of the
lists in the same stratum and year, in eta space: change shared regionally, not local). 1940-1965 is before BBS,
never supervised; 1966-1995 are BBS years WITHHELD from this DESK's training. Lists in the cache's 3,902 cells only.
Reported per era and species set (all; residents = AVONET migration 1-2): mean log-likelihood per listed species for
each predictor; paired differences backcast - modern and backcast - regional, 95% bootstrap over circles (1,000);
mean within-list AUC (listed vs unlisted species) per predictor.
"""
import argparse
import json
import os

import numpy as np


def ridge_cv(Z, T, groups, lams, k=5, seed=69):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups); fold = dict(zip(ug, rng.integers(k, size=len(ug))))
    f = np.array([fold[g] for g in groups])
    best = None
    for lam in lams:
        err = 0.0
        for i in range(k):
            tr, te = f != i, f == i
            mu_z, mu_t = Z[tr].mean(0), T[tr].mean(0)
            A = (Z[tr] - mu_z).T @ (Z[tr] - mu_z)
            B = np.linalg.solve(A + lam * np.trace(A) / A.shape[0] * np.eye(A.shape[0]), (Z[tr] - mu_z).T @ (T[tr] - mu_t))
            err += (((Z[te] - mu_z) @ B + mu_t - T[te]) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, lam)
    lam = best[1]
    mu_z, mu_t = Z.mean(0), T.mean(0)
    A = (Z - mu_z).T @ (Z - mu_z)
    B = np.linalg.solve(A + lam * np.trace(A) / A.shape[0] * np.eye(A.shape[0]), (Z - mu_z).T @ (T - mu_t))
    return (lambda z: (z - mu_z) @ B + mu_t), lam


def split_ridge_cv(Zb, Zw, T, groups, lams, k=5, seed=69):
    """E069a2 (peer E059e): ridge on [cell-mean z, within-cell deviation dz] with SEPARATE penalties, chosen jointly by
    k-fold CV over spatial blocks. Returns f(zbar, z) = a + zbar Bb + (z - zbar) Bw."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups); fold = dict(zip(ug, rng.integers(k, size=len(ug))))
    f = np.array([fold[g] for g in groups])
    F = np.concatenate([Zb, Zw], 1); p = Zb.shape[1]

    def fit(rows, lb, lw):
        mu_f, mu_t = F[rows].mean(0), T[rows].mean(0)
        A = (F[rows] - mu_f).T @ (F[rows] - mu_f)
        tb, tw = np.trace(A[:p, :p]) / p, np.trace(A[p:, p:]) / p
        pen = np.concatenate([np.full(p, lb * tb), np.full(p, lw * tw)])
        return mu_f, mu_t, np.linalg.solve(A + np.diag(pen), (F[rows] - mu_f).T @ (T[rows] - mu_t))
    best = None
    for lb in lams:
        for lw in lams:
            err = 0.0
            for i in range(k):
                mu_f, mu_t, B = fit(f != i, lb, lw)
                err += (((F[f == i] - mu_f) @ B + mu_t - T[f == i]) ** 2).sum()
            if best is None or err < best[0]:
                best = (err, lb, lw)
    mu_f, mu_t, B = fit(np.ones(len(F), bool), best[1], best[2])
    ratio = float(np.linalg.norm(B[:p]) / max(np.linalg.norm(B[p:]), 1e-12))
    return (lambda zbar, z: (np.concatenate([zbar, z - zbar], 1) - mu_f) @ B + mu_t), (best[1], best[2]), ratio


def interact_ridge_cv(Zb, Zw, T, groups, lams, n_kind=4, k=5, seed=69):
    """E074 (peer E073): place-dependent slopes. Ridge on [zbar, dz, dz x s], s = the first n_kind dims of zbar
    standardized on the training rows ("kind of place"), with a separate penalty per block chosen jointly by k-fold CV
    over spatial blocks. Returns f(zbar, z)."""
    mu_s, sd_s = Zb[:, :n_kind].mean(0), Zb[:, :n_kind].std(0) + 1e-9

    def feats(zbar, z):
        dz = z - zbar
        s = (zbar[:, :n_kind] - mu_s) / sd_s
        return np.concatenate([zbar, dz, (dz[:, :, None] * s[:, None, :]).reshape(len(z), -1)], 1)
    F = feats(Zb, Zw); p = Zb.shape[1]; blocks = [(0, p), (p, 2 * p), (2 * p, F.shape[1])]
    rng = np.random.default_rng(seed)
    ug = np.unique(groups); fold = dict(zip(ug, rng.integers(k, size=len(ug))))
    f = np.array([fold[g] for g in groups])

    def stats(rows):
        mu_f, mu_t = F[rows].mean(0), T[rows].mean(0)
        Fc = F[rows] - mu_f
        A = Fc.T @ Fc
        return mu_f, mu_t, A, Fc.T @ (T[rows] - mu_t), [np.trace(A[a:b, a:b]) / (b - a) for a, b in blocks]

    def solve(st, lb):
        mu_f, mu_t, A, rhs, tr = st
        pen = np.concatenate([np.full(b - a, l * t) for (a, b), l, t in zip(blocks, lb, tr)])
        return np.linalg.solve(A + np.diag(pen), rhs)
    folds = [(stats(f != i), f == i) for i in range(k)]
    best = None
    for l1 in lams:
        for l2 in lams:
            for l3 in lams:
                err = 0.0
                for st, te in folds:
                    B = solve(st, (l1, l2, l3))
                    err += (((F[te] - st[0]) @ B + st[1] - T[te]) ** 2).sum()
                if best is None or err < best[0]:
                    best = (err, (l1, l2, l3))
    st = stats(np.ones(len(F), bool)); B = solve(st, best[1])
    return (lambda zbar, z: (feats(zbar, z) - st[0]) @ B + st[1]), best[1]


def list_ll(eta, delta, P):
    """mean log P(species | record) over listed species, per list. eta (n, S), P (n, S) bool presence."""
    s = eta + delta
    lse = np.log(np.exp(s - s.max(1, keepdims=True)).sum(1)) + s.max(1)
    return ((s - lse[:, None]) * P).sum(1) / np.maximum(P.sum(1), 1)


def list_auc(score, P):
    from scipy.stats import rankdata
    out = np.full(len(P), np.nan)
    for i in range(len(P)):
        n1 = P[i].sum(); n0 = len(P[i]) - n1
        if n1 >= 1 and n0 >= 1:
            r = rankdata(score[i]); out[i] = (r[P[i]].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def fit_delta(eta, P, steps=500, lr=0.1):
    import torch
    e, p = torch.as_tensor(eta, dtype=torch.float64), torch.as_tensor(P, dtype=torch.float64)
    d = torch.zeros(eta.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([d], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        s = e + d
        loss = -((s - torch.logsumexp(s, 1, keepdim=True)) * p).sum() / p.sum() + 1e-4 * (d ** 2).sum()
        loss.backward(); opt.step()
    return d.detach().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--cbc", required=True)
    ap.add_argument("--similarity", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--readout", choices=["level", "split", "interact"], default="level",
                    help="level: ridge on z (E069a); split: ridge on [cell-mean z, z - cell mean] with separate "
                         "penalties (E069a2, after the peer's E059e)")
    a = ap.parse_args()
    import pandas as pd
    C = a.cache
    meta = json.load(open(os.path.join(C, "meta.json")))
    keys = np.load(os.path.join(C, "keys.npy")); Xc = np.load(os.path.join(C, "X_comm.npy"))
    z_keys = np.load(os.path.join(C, "z_ema.npy")); Zall = np.load(os.path.join(C, "z_ema_all.npy"), mmap_mode="r")
    cells = np.load(os.path.join(C, "cells.npy")); years = np.load(os.path.join(C, "years.npy"))
    sp_ = np.load(os.path.join(C, "split.npz"))
    D = np.load(a.cbc, allow_pickle=False)
    comm, cs = list(meta["community"]), list(D["species"])
    sp = [s for s in comm if s in cs]
    assert "houfin" not in sp
    sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    resident = np.array([mig.get(s) in (1.0, 2.0) for s in sp])
    # readout on BBS training cell-years from trained years
    tr = sp_["is_train"] & (keys[:, 2] >= 1996)
    T = np.log1p(Xc[:, [comm.index(s) for s in sp]])
    trained = (years >= 1996) & (years <= 2025)
    zbar_cell = np.asarray(Zall[:, trained].mean(1), float)          # each cell's mean z over trained years
    kci = np.load(os.path.join(C, "key_cell_index.npy"))
    if a.readout == "level":
        _ro, lam = ridge_cv(z_keys[tr].astype(float), T[tr], sp_["block_id"][tr], [1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
        readout = lambda cell, z: _ro(z)
        between_within = None
    elif a.readout == "interact":
        _ro, lam = interact_ridge_cv(zbar_cell[kci[tr]], z_keys[tr].astype(float), T[tr], sp_["block_id"][tr],
                                     [1e-3, 1e-2, 1e-1, 1, 10])
        readout = lambda cell, z: _ro(zbar_cell[cell], z)
        between_within = None
    else:
        _ro, lam, between_within = split_ridge_cv(zbar_cell[kci[tr]], z_keys[tr].astype(float), T[tr],
                                                  sp_["block_id"][tr], [1e-3, 1e-2, 1e-1, 1, 10, 100])
        readout = lambda cell, z: _ro(zbar_cell[cell], z)
    # CBC lists in cached cells
    cix = {(int(r), int(c)): i for i, (r, c) in enumerate(cells)}
    ci = np.array([cix.get((int(r), int(c)), -1) for r, c in zip(D["row"], D["col"])])
    yr = D["year_end"]
    P = D["Y"][:, [cs.index(s) for s in sp]] > 0
    ok = (ci >= 0) & (yr >= years[0]) & (yr <= 2024) & (P.sum(1) >= 2)
    y0 = int(years[0])
    zt = lambda rows: np.asarray(Zall[ci[rows], yr[rows] - y0], float)
    mod_years = (years >= 2005) & (years <= 2024)
    zmod_cell = np.asarray(Zall[:, mod_years].mean(1), float)
    res = {"readout": a.readout, "species": len(sp), "residents": int(resident.sum()), "ridge_lambda": lam,
           "between_over_within_coef_norm": between_within}
    m_mod = ok & (yr >= 2005)
    rows_mod = np.where(m_mod)[0]
    eta_mod = readout(ci[rows_mod], zt(rows_mod))
    for subset, cols in (("all", np.ones(len(sp), bool)), ("residents", resident)):
        delta = fit_delta(eta_mod[:, cols], P[rows_mod][:, cols])
        out = {"modern_lists_for_delta": int(len(rows_mod))}
        for era, (e0, e1) in (("1940_1952", (1940, 1952)), ("1953_1965", (1953, 1965)), ("1966_1995", (1966, 1995))):
            rows = np.where(ok & (yr >= e0) & (yr <= e1))[0]
            Pm = P[rows][:, cols]
            keep = Pm.sum(1) >= 2
            rows, Pm = rows[keep], Pm[keep]
            eb = readout(ci[rows], zt(rows))[:, cols]
            em = readout(ci[rows], zmod_cell[ci[rows]])[:, cols]
            sh = pd.DataFrame(eb - em); sh["k"] = list(zip(D["stratum_idx"][rows], yr[rows]))
            er = em + sh.groupby("k").transform("mean").to_numpy()
            llb, llm, llr = list_ll(eb, delta, Pm), list_ll(em, delta, Pm), list_ll(er, delta, Pm)
            circ = D["circle_idx"][rows]; uc = np.unique(circ)
            idx = {c: np.where(circ == c)[0] for c in uc}
            rng = np.random.default_rng(69)
            bd = {"modern": [], "regional": []}
            for _ in range(a.boot):
                pick = np.concatenate([idx[c] for c in rng.choice(uc, len(uc))])
                bd["modern"].append((llb[pick] - llm[pick]).mean()); bd["regional"].append((llb[pick] - llr[pick]).mean())
            out[era] = {"lists": int(len(rows)), "circles": int(len(uc)),
                        "ll_backcast": float(llb.mean()), "ll_modern": float(llm.mean()), "ll_regional": float(llr.mean()),
                        "diff_vs_modern": float((llb - llm).mean()),
                        "diff_vs_modern_ci95": [float(x) for x in np.percentile(bd["modern"], [2.5, 97.5])],
                        "diff_vs_regional": float((llb - llr).mean()),
                        "diff_vs_regional_ci95": [float(x) for x in np.percentile(bd["regional"], [2.5, 97.5])],
                        "auc_backcast": float(np.nanmean(list_auc(eb + delta, Pm))),
                        "auc_modern": float(np.nanmean(list_auc(em + delta, Pm))),
                        # secondary (pre-registered in E069a2): shrinking the backcast shift by alpha
                        "ll_by_alpha": {str(al): float(list_ll(em + al * (eb - em), delta, Pm).mean())
                                        for al in (0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)}}
        res[subset] = out
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"e069a_{a.readout}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
