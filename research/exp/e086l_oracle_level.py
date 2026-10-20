"""E086l: the PRODUCTION-FORM readout on the OBSERVED community's ESK coordinates at rank k -- does the tail beyond 64
help the downstream readout even with perfect community knowledge (an upper bound for a rank-128 DESK)?

    python research/exp/e086l_oracle_level.py --cache <tempho cache> --esk <esk128_annual.npy from E086k>
        --similarity <species_similarity.csv> --out <dir> [--ks 24 64 96 128] [--n-boot 200]

Per dev species, the age model's form: one iid-prior BLR (research/lib/blr, ML amplitude) of the species' cell-level
log1p abundance on the community's ESK coordinates (first k dims), fitted on LEVELS at training cells (cell means over
trained years). Noise kept apart: community features from ODD surveyed years, species targets from EVEN years.
Scored on held-out cells:
  change   the level-fitted coefficients applied to the community's change (withheld early -> modern epoch, odd years),
           noise-free pooled corr with species change (even-year half; odd-year half for the noise-free denominator);
  level    modern-epoch species level (even years) vs the prediction from the odd-year community, pooled corr.
Strata: sedentary common (prevalence >= 0.08), sedentary, all dev. 6x6-block bootstrap over held-out cells for k vs 64.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from lib import blr  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--esk", required=True)
    ap.add_argument("--similarity", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[24, 64, 96, 128]); ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()
    import pandas as pd
    j = lambda f: os.path.join(a.cache, f)
    keys = np.load(j("keys.npy")); Z = np.load(a.esk).astype(np.float64)
    Ld = np.log1p(np.asarray(np.load(j("X_dev.npy"), mmap_mode="r"), np.float64))
    sp = np.load(j("split.npz")); ho, bf = sp["holdout"].astype(bool), sp["buffer"].astype(bool)
    withheld = set(int(y) for y in sp["withheld"])
    E = dict(np.load(j("esk_epochs.npz"))); ptr, flat = E["rows_ptr"], E["rows_flat"]
    r, c, y = keys[:, 0], keys[:, 1], keys[:, 2]
    meta = json.load(open(j("meta.json"))); dev = list(meta["dev_species"])
    sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    is_tr = ~ho[r, c] & ~bf[r, c] & ~np.isin(y, list(withheld))
    prev = (np.asarray(np.load(j("X_dev.npy"), mmap_mode="r"))[is_tr] > 0).mean(0)
    sed = np.array([mig.get(s) == 1.0 for s in dev])
    strata = {"sed_common": sed & (prev >= 0.08), "sedentary": sed, "all_dev": np.ones(len(dev), bool)}
    # training: cell means over trained years, odd-year community -> even-year species
    cid = r.astype(np.int64) * 100000 + c
    def cell_means(V, mask):
        u, inv = np.unique(cid[mask], return_inverse=True)
        s = np.zeros((len(u), V.shape[1])); np.add.at(s, inv, V[mask]); n = np.bincount(inv)
        return u, s / n[:, None]
    uz, Xtr = cell_means(Z, is_tr & (y % 2 == 1)); uy, Ytr = cell_means(Ld, is_tr & (y % 2 == 0))
    common = np.intersect1d(uz, uy)
    Xtr = Xtr[np.searchsorted(uz, common)]; Ytr = Ytr[np.searchsorted(uy, common)]
    # held-out cell-epochs
    idx = {}
    for i, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh):
            idx.setdefault(cell, [None, None])[0] = i
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = i
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    pc = E["cells"][pairs[:, 0]]; hp = pairs[ho[pc[:, 0], pc[:, 1]]]
    def hm(V, i, h):
        rr = flat[ptr[i]:ptr[i + 1]]; rr = rr[(y[rr] % 2) == h]
        return V[rr].mean(0) if len(rr) else np.full(V.shape[1], np.nan)
    ZA_e = np.stack([hm(Z, e, 1) for e, m in hp]); ZA_m = np.stack([hm(Z, m, 1) for e, m in hp])
    SB_e = np.stack([hm(Ld, e, 0) for e, m in hp]); SB_m = np.stack([hm(Ld, m, 0) for e, m in hp])
    SA_e = np.stack([hm(Ld, e, 1) for e, m in hp]); SA_m = np.stack([hm(Ld, m, 1) for e, m in hp])
    ok = np.isfinite(ZA_e).all(1) & np.isfinite(ZA_m).all(1) & np.isfinite(SB_e).all(1) & np.isfinite(SB_m).all(1) \
        & np.isfinite(SA_e).all(1) & np.isfinite(SA_m).all(1)
    ZA_e, ZA_m, SB_e, SB_m, SA_e, SA_m = (v[ok] for v in (ZA_e, ZA_m, SB_e, SB_m, SA_e, SA_m))
    hcells = E["cells"][hp[ok, 1]]
    blk = (hcells[:, 0] // 6) * 100000 + hcells[:, 1] // 6
    ub, binv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(86)
    W = np.vstack([np.ones(len(blk)), rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub)), size=a.n_boot)[:, binv]])
    dSA, dSB = SA_m - SA_e, SB_m - SB_e
    def wcorr_nf(P, A, B, m):          # pooled over species in m, per bootstrap weight row
        out = []
        for w in W:
            sw = w.sum(); mean = lambda X: (w @ X) / sw
            Pc, Ac, Bc = P[:, m] - mean(P[:, m]), A[:, m] - mean(A[:, m]), B[:, m] - mean(B[:, m])
            out.append((w @ (Pc * Bc)).sum() / np.sqrt((w @ (Pc * Pc)).sum() * max((w @ (Ac * Bc)).sum(), 1e-12)))
        return np.array(out)
    res = {"n_train_cells": int(len(common)), "n_heldout_cells": int(len(hcells)), "change": {}, "level": {}}
    draws = {}
    for k in a.ks:
        model = blr.fit(Xtr[:, :k], Ytr, [(0, k)])
        P_chg = (ZA_m[:, :k] - ZA_e[:, :k]) @ model["coef"].T
        P_lvl = blr.predict(model, ZA_m[:, :k])[0]
        P_lvl = np.asarray(P_lvl)
        draws[k] = {s: wcorr_nf(P_chg, dSA, dSB, m) for s, m in strata.items()}
        res["change"][f"k{k}"] = {s: float(v[0]) for s, v in draws[k].items()}
        res["level"][f"k{k}"] = {s: float(wcorr_nf(P_lvl, SA_m, SB_m, m)[0]) for s, m in strata.items()}
    res["change_vs_k64"] = {f"k{k}": {s: [float((draws[k][s] - draws[64][s])[0])] +
                                      [float(np.percentile((draws[k][s] - draws[64][s])[1:], q)) for q in (2.5, 97.5)]
                                      for s in strata} for k in a.ks if k != 64 and 64 in draws}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e086l.json"), "w"), indent=1)
    print(f"[e086l] {os.path.basename(a.cache.rstrip('/'))}: {res['n_train_cells']} training cells, {res['n_heldout_cells']} held-out")
    for k in a.ks:
        ch, lv = res["change"][f"k{k}"], res["level"][f"k{k}"]
        extra = res["change_vs_k64"].get(f"k{k}", {})
        print(f"  k={k:3d} change " + " ".join(f"{s} {v:+.3f}" for s, v in ch.items()) + " | level " +
              " ".join(f"{s} {v:+.3f}" for s, v in lv.items()) +
              ("" if not extra else " | vs k64 " + " ".join(f"{s} {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]" for s, d in extra.items())))


if __name__ == "__main__":
    main()
