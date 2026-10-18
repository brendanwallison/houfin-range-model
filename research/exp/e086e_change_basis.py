"""E086e stage 1: is community CHANGE compact in its own basis, and is that basis what species change follows?

    python research/exp/e086e_change_basis.py --cache <tempho research cache> --out <dir> [--ks 4 8 16 24]

No model training. Basis U (from OBSERVED communities, TRAINING cells, TRAINED years only):
  each training cell's ESK (esk_annual, 64 dims) mean over two trained-era windows (first / second half of the trained
  years), separately for odd-year and even-year rows (halves A / B); within-cell change dA = A(w2) - A(w1), dB likewise;
  noise-free change covariance C = (dA'dB + dB'dA) / 2n; U = eigenvectors of C, ordered by noise-free change variance.
Evaluated on HELD-OUT cells, long change (withheld early epoch -> 2005-25; esk_epochs ABBA halves z_a / z_b):
  (a) noise-free change variance captured by the top-k U directions vs the top-k ESK dims (tr(P' C_ho P) / tr(C_ho));
  (b) ORACLE species relevance: per dev species, ridge of species change on observed community change projected to k
      dims (U_k or ESK_k; and ESK 64), fitted on training cells' trained-era window change (half A community, half B
      species), applied to held-out cells' long change (community half A); scored against species half B: pooled
      noise-free corr over the stratum (sum cov(pred, B) / sqrt(sum var(pred) * sum cov(A_sp, B_sp))) -- SEDENTARY
      primary, all dev species;
  (c) DESK in the change basis: DESK's predicted long change (z_raw epoch means, the cache's model) projected on U_j vs
      the observed change (noise-free corr per direction, j = 1..16), against the same on ESK dims 1..16.
Alignment: share of each U_j's mass in ESK dims 1-24 vs 25-64.
"""
import argparse
import json
import os

import numpy as np


def nf_corr(p, a, b):
    """Noise-free pooled corr of prediction p with truth halves a, b (columns = species / dims), centred."""
    p = p - p.mean(0); a = a - a.mean(0); b = b - b.mean(0)
    num = (p * b).sum()
    den = np.sqrt((p * p).sum() * max((a * b).sum(), 1e-12))
    return float(num / den)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 24, 32, 48, 64])
    ap.add_argument("--similarity", default=None)
    a = ap.parse_args()
    import pandas as pd
    j = lambda f: os.path.join(a.cache, f)
    keys = np.load(j("keys.npy")); Eann = np.load(j("esk_annual.npy")).astype("float64")
    Xdev = np.load(j("X_dev.npy"), mmap_mode="r")
    meta = json.load(open(j("meta.json"))); dev = list(meta["dev_species"])
    sp = np.load(j("split.npz")); ho, bf = sp["holdout"].astype(bool), sp["buffer"].astype(bool)
    withheld = set(int(y) for y in sp["withheld"])
    E = dict(np.load(j("esk_epochs.npz")))
    r, c, y = keys[:, 0], keys[:, 1], keys[:, 2]
    trained_years = sorted(set(int(v) for v in np.unique(y)) - withheld)
    mid = trained_years[len(trained_years) // 2]
    is_tr = ~ho[r, c] & ~bf[r, c] & ~np.isin(y, list(withheld))
    cell_id = r.astype(np.int64) * 10000 + c
    L = np.log1p(np.asarray(Xdev, "float64"))

    # trained-era window change on training cells, halves by year parity
    def win_change(V):
        out = {}
        for half in (0, 1):
            m = is_tr & (y % 2 == half)
            df = pd.DataFrame(V[m]); df["cell"] = cell_id[m]; df["w"] = (y[m] >= mid).astype(int)
            g = df.groupby(["cell", "w"]).mean()
            w0 = g.xs(0, level="w"); w1 = g.xs(1, level="w")
            common = w0.index.intersection(w1.index)
            out[half] = (w1.loc[common] - w0.loc[common])
        common = out[0].index.intersection(out[1].index)
        return out[0].loc[common].to_numpy(), out[1].loc[common].to_numpy(), common
    dA, dB, cells_tr = win_change(Eann)
    C = (dA.T @ dB + dB.T @ dA) / (2 * len(dA))
    ev, U = np.linalg.eigh(C); o = np.argsort(ev)[::-1]; ev, U = ev[o], U[:, o]
    # held-out long change (esk_epochs halves)
    idx = {}
    for i, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh):
            idx.setdefault(cell, [None, None])[0] = i
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = i
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    pc = E["cells"][pairs[:, 0]]
    held = ho[pc[:, 0], pc[:, 1]]
    hp = pairs[held]
    DA = E["z_a"][hp[:, 1]] - E["z_a"][hp[:, 0]]; DB = E["z_b"][hp[:, 1]] - E["z_b"][hp[:, 0]]
    Cho = (DA.T @ DB + DB.T @ DA) / (2 * len(DA))
    tot = np.trace(Cho)
    res = {"cache": a.cache, "n_train_cells": int(len(dA)), "n_heldout_cells": int(len(hp)), "mid_year": mid,
           "trained_era_change_eig_share": (ev / ev.sum())[:24].tolist()}
    I = np.eye(64)
    res["captured_heldout_change"] = {
        f"k{k}": {"U": float(np.trace(U[:, :k].T @ Cho @ U[:, :k]) / tot),
                  "ESK": float(np.trace(I[:, :k].T @ Cho @ I[:, :k]) / tot)} for k in a.ks}
    res["U_mass_in_esk_1_24"] = [float((U[:24, q] ** 2).sum()) for q in range(16)]
    # (b) oracle species relevance
    rows_ptr, rows_flat = E["rows_ptr"], E["rows_flat"]
    half_rows = lambda i, h: rows_flat[rows_ptr[i]:rows_ptr[i + 1]][(y[rows_flat[rows_ptr[i]:rows_ptr[i + 1]]] % 2) == h]
    def sp_epoch(i, h):
        rr = half_rows(i, h)
        return L[rr].mean(0) if len(rr) else np.full(L.shape[1], np.nan)
    SA = np.stack([sp_epoch(m, 0) - sp_epoch(e, 0) for e, m in hp]); SB = np.stack([sp_epoch(m, 1) - sp_epoch(e, 1) for e, m in hp])
    sA, sB, _ = win_change(L)               # training-cell species window change (same cells as dA)
    okh = np.isfinite(SA).all(1) & np.isfinite(SB).all(1)
    mig = {}
    if a.similarity:
        sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    strata = {"all": np.ones(len(dev), bool), "sedentary": np.array([mig.get(s) == 1.0 for s in dev])}
    res["oracle_species"] = {}

    def fit_pred(Ftr, Ytr, Fte, lam):
        mu = Ftr.mean(0); Fc = Ftr - mu
        W = np.linalg.solve(Fc.T @ Fc + lam * np.eye(Fc.shape[1]), Fc.T @ (Ytr - Ytr.mean(0)))
        return (Fte - mu) @ W + Ytr.mean(0)
    bases = {f"U{k}": U[:, :k] for k in a.ks}
    bases.update({f"ESK{k}": I[:, :k] for k in a.ks}); bases.setdefault("ESK64", I)
    for name, P in bases.items():
        Ftr, Fte = dA @ P, DA[okh] @ P
        best = None
        for lam in (1e-3, 1e-2, 1e-1, 1.0):
            # lambda by 2-fold split of training cells on the trained-era target (no held-out data)
            n = len(Ftr); h = np.arange(n) % 2 == 0
            p = fit_pred(Ftr[h], sB[h], Ftr[~h], lam)
            sc = nf_corr(p, sA[~h], sB[~h])
            if best is None or sc > best[0]:
                best = (sc, lam)
        pred = fit_pred(Ftr, sB, Fte, best[1])
        res["oracle_species"][name] = {s: nf_corr(pred[:, m], SA[okh][:, m], SB[okh][:, m])
                                       for s, m in strata.items() if m.any()}
        res["oracle_species"][name]["lambda"] = best[1]
    # (c) DESK in the change basis
    zr = np.load(j("z_raw_all.npy"), mmap_mode="r"); cells = np.load(j("cells.npy")); yrs = np.load(j("years.npy"))
    cix = {(int(a_), int(b_)): q for q, (a_, b_) in enumerate(cells)}
    def desk_epoch(i):
        rr = rows_flat[rows_ptr[i]:rows_ptr[i + 1]]
        return np.asarray(zr[[cix[(int(r[k]), int(c[k]))] for k in rr], y[rr] - yrs[0]], "float64").mean(0)
    Dd = np.stack([desk_epoch(m) - desk_epoch(e) for e, m in hp])
    res["desk_per_direction"] = {"U": [nf_corr((Dd @ U[:, [q]]), DA @ U[:, [q]], DB @ U[:, [q]]) for q in range(16)],
                                 "ESK": [nf_corr(Dd[:, [q]], DA[:, [q]], DB[:, [q]]) for q in range(16)]}
    # (d) EXTRAPOLATION COVERAGE (the user: the design must extrapolate in time without early supervision). DESK's own
    # covariate-driven change BEFORE the BBS era (mean z_raw 1940-1955 minus 1966-1986, every cached cell): the share of
    # its across-cell variance inside U_k vs ESK_k. A change block truncated to BBS-era change directions loses the rest.
    pre = (yrs >= 1940) & (yrs <= 1955); bas = (yrs >= 1966) & (yrs <= 1986)
    Dpre = np.asarray(zr[:, pre].mean(1) - zr[:, bas].mean(1), "float64")
    Dpre = Dpre[np.isfinite(Dpre).all(1)]
    Cpre = Dpre.T @ Dpre / len(Dpre)                    # uncentred: a uniform shift is change too
    res["pre_bbs_change_captured"] = {f"k{k}": {"U": float(np.trace(U[:, :k].T @ Cpre @ U[:, :k]) / np.trace(Cpre)),
                                                "ESK": float(np.trace(Cpre[:k, :k]) / np.trace(Cpre))} for k in a.ks}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e086e_change_basis.json"), "w"), indent=1)
    print(f"[e086e] {os.path.basename(a.cache.rstrip('/'))}: {len(dA)} training cells (window split at {mid}), "
          f"{len(hp)} held-out cells")
    for k in a.ks:
        cc = res["captured_heldout_change"][f"k{k}"]
        print(f"  k={k:2d} held-out change captured: U {cc['U']:.3f}  ESK {cc['ESK']:.3f}")
    for name, v in res["oracle_species"].items():
        print(f"  oracle species {name:6s} " + "  ".join(f"{s} {v[s]:+.3f}" for s in strata if s in v))
    d = res["desk_per_direction"]
    print("  DESK per direction (U 1-8):  " + " ".join(f"{x:+.2f}" for x in d["U"][:8]))
    print("  DESK per direction (ESK 1-8): " + " ".join(f"{x:+.2f}" for x in d["ESK"][:8]))
    for k in a.ks:
        pc_ = res["pre_bbs_change_captured"][f"k{k}"]
        print(f"  k={k:2d} PRE-BBS DESK change (1940-55 vs 1966-86) captured: U {pc_['U']:.3f}  ESK {pc_['ESK']:.3f}")
    print("  U_j mass in ESK 1-24 (j=1..8): " + " ".join(f"{x:.2f}" for x in res["U_mass_in_esk_1_24"][:8]))


if __name__ == "__main__":
    main()
