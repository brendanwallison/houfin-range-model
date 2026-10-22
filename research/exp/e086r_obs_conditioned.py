"""E086r: an OBSERVATION-CONDITIONED readout -- does a held-out cell's own observed CBC change, added to DESK's
prediction, improve the species' predicted breeding-season (BBS) change, especially under sparse target data?

    python research/exp/e086r_obs_conditioned.py --cache <tempho research cache> --cbc cbc_circles_res.npz
        --similarity species_similarity.csv --out <dir> [--rank 24] [--thin 0.03] [--thin-seeds 1 2] [--cbc-keep 1 0.2]

Species: dev species that are also in the CBC file (asserts no houfin). Per species s, at held-out cells that have
CBC lists in both epochs (withheld early span; modern = after it), three predictions of the cell's BBS change:
  desk   DESK z (first --rank dims) -> species, the cache's z change between epochs through a readout
  cbc    b_s * the cell's observed CBC change of species s itself (effort-adjusted presence; see below)
  combo  a_s * desk + b_s * cbc, the two weights fitted jointly per species
Readouts (both use target data only at TRAINING cells x TRAINED years, rows thinned to --thin with each seed):
  within  deviations on deviations: DESK coefficients by the iid-prior BLR (research/lib/blr) on per-cell demeaned z
          -> demeaned log1p counts; a_s, b_s by per-species least squares of the species' deviation on [DESK's
          predicted deviation, the CBC deviation] over cell-years that have both a kept BBS row and a CBC value
  level   cell means: DESK by the BLR on cell-mean z -> cell-mean log1p counts (the production form, applied to
          change); a_s, b_s by least squares (with intercept) across CBC-covered training cells of the cell-mean
          species level on [DESK's predicted level, the cell-mean CBC value]
CBC value: per list, presence (count > 0), residualized per species on [log party-hours, log field counters] by
least squares over all lists 1966+ (missing effort -> that year's median); averaged over a cell's lists per winter
(year_end), then over the epoch's winters. --cbc-keep < 1 keeps each list with that probability (sparse winter data;
the evaluation cell set is fixed from the full list set, cells left without a kept list in an epoch get no CBC change).
Scored on held-out cells with >= 3 CBC lists in each epoch: noise-free pooled corr (odd / even-year halves of the BBS
change), 6x6-block bootstrap over cells, paired differences combo - desk and cbc - desk.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from lib import blr  # noqa: E402


def lsq(X, Y, ridge=1e-6):
    """Per-species least squares: X (S, n, p), Y (S, n) -> (S, p)."""
    XtX = np.einsum("snp,snq->spq", X, X) + ridge * np.eye(X.shape[2])
    return np.linalg.solve(XtX, np.einsum("snp,sn->sp", X, Y)[..., None])[..., 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--cbc", required=True)
    ap.add_argument("--similarity", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=24); ap.add_argument("--thin", type=float, default=0.03)
    ap.add_argument("--thin-seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--cbc-keep", type=float, nargs="+", default=[1.0, 0.2])
    ap.add_argument("--min-lists", type=int, default=3); ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    import pandas as pd
    j = lambda f: os.path.join(a.cache, f)
    meta = json.load(open(j("meta.json"))); dev = list(meta["dev_species"])
    keys = np.load(j("keys.npy")); r, c, y = keys[:, 0], keys[:, 1], keys[:, 2]
    sp = np.load(j("split.npz")); ho, bf = sp["holdout"].astype(bool), sp["buffer"].astype(bool)
    wh = np.array(sorted(int(v) for v in sp["withheld"])); w0, w1 = int(wh[0]), int(wh[-1])
    D = np.load(a.cbc, allow_pickle=False); cs = list(D["species"])
    species = [s for s in dev if s in cs]
    assert "houfin" not in species and "houfin" not in cs
    di = [dev.index(s) for s in species]; ci_ = [cs.index(s) for s in species]
    Ld = np.log1p(np.asarray(np.load(j("X_dev.npy"), mmap_mode="r")[:, di], np.float64))
    Z = np.load(j("z_ema.npy")).astype(np.float64)[:, :a.rank]
    sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    is_tr = ~ho[r, c] & ~bf[r, c] & ~np.isin(y, wh)
    prev = (np.asarray(np.load(j("X_dev.npy"), mmap_mode="r")[:, di])[is_tr] > 0).mean(0)
    sed = np.array([mig.get(s) == 1.0 for s in species])
    strata = {"sed_common": sed & (prev >= 0.08), "sedentary": sed, "sed_rare_mid": sed & (prev < 0.08),
              "migrants": ~sed, "all": np.ones(len(species), bool)}
    S = len(species)
    # ---- CBC lists: effort-adjusted presence of each species
    nr, nc = ho.shape
    lr, lc, ly = D["row"], D["col"], D["year_end"]
    on = (lr >= 0) & (lr < nr) & (lc >= 0) & (lc < nc) & (ly >= 1966) & (ly <= 2025)
    lr, lc, ly = lr[on], lc[on], ly[on]
    Pr = (np.asarray(D["Y"][on][:, ci_]) > 0).astype(np.float64)
    def logfill(v):
        v = np.asarray(v, np.float64).copy(); bad = ~np.isfinite(v) | (v <= 0)
        for yy in np.unique(ly):
            m = ly == yy; good = m & ~bad
            v[m & bad] = np.median(v[good]) if good.any() else np.median(v[~bad])
        return np.log(v)
    Eff = np.stack([np.ones(len(ly)), logfill(D["party_hours"][on]), logfill(D["field_counters"][on])], 1)
    beta = np.linalg.lstsq(Eff, Pr, rcond=None)[0]
    Padj = Pr - Eff[:, 1:] @ beta[1:]
    lcid = lr.astype(np.int64) * 1000 + lc
    early_l, mod_l = (ly >= w0) & (ly <= w1), ly > w1
    # held-out cell-epoch pairs (BBS), as e086l
    E = dict(np.load(j("esk_epochs.npz"))); ptr, flat = E["rows_ptr"], E["rows_flat"]
    idx = {}
    for i, (cell, ep, w) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(w):
            idx.setdefault(cell, [None, None])[0] = i
        elif ep == 1 and not w:
            idx.setdefault(cell, [None, None])[1] = i
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    pc = E["cells"][pairs[:, 0]]; hp = pairs[ho[pc[:, 0], pc[:, 1]]]
    hcells = E["cells"][hp[:, 1]]; hcid = hcells[:, 0].astype(np.int64) * 1000 + hcells[:, 1]
    ne = np.array([(lcid[early_l] == k).sum() for k in hcid]); nm = np.array([(lcid[mod_l] == k).sum() for k in hcid])
    def hm(V, i, h):
        rr = flat[ptr[i]:ptr[i + 1]]; rr = rr[(y[rr] % 2) == h] if h is not None else rr
        return V[rr].mean(0) if len(rr) else np.full(V.shape[1], np.nan)
    SA = np.stack([hm(Ld, m, 1) - hm(Ld, e, 1) for e, m in hp]); SB = np.stack([hm(Ld, m, 0) - hm(Ld, e, 0) for e, m in hp])
    dZ = np.stack([hm(Z, m, None) - hm(Z, e, None) for e, m in hp])
    ok = (ne >= a.min_lists) & (nm >= a.min_lists) & np.isfinite(SA).all(1) & np.isfinite(SB).all(1) & np.isfinite(dZ).all(1)
    hp, hcells, hcid, SA, SB, dZ = hp[ok], hcells[ok], hcid[ok], SA[ok], SB[ok], dZ[ok]
    blk = (hcells[:, 0] // 6) * 100000 + hcells[:, 1] // 6
    ub, binv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(86)
    W = np.vstack([np.ones(len(blk)), rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub)), size=a.n_boot)[:, binv]])
    def nfcorr(P, m):
        out = []
        for w in W:
            sw = w.sum(); mean = lambda X: (w @ X) / sw
            Pc, Ac, Bc = P[:, m] - mean(P[:, m]), SA[:, m] - mean(SA[:, m]), SB[:, m] - mean(SB[:, m])
            out.append((w @ (Pc * Bc)).sum() / np.sqrt((w @ (Pc * Pc)).sum() * max((w @ (Ac * Bc)).sum(), 1e-12)))
        return np.array(out)
    res = {"cache": a.cache, "rank": a.rank, "withheld": [w0, w1], "n_species": S, "n_eval_cells": int(len(hcid)),
           "n_eval_blocks": int(len(ub)), "strata_n": {k: int(v.sum()) for k, v in strata.items()},
           "effort_beta_median": [float(np.median(beta[k])) for k in (1, 2)], "regimes": {}}
    print(f"[e086r] {os.path.basename(a.cache.rstrip('/'))}: {S} species, {len(hcid)} held-out CBC cells "
          f"({len(ub)} blocks), withheld {w0}-{w1}; strata " + str(res["strata_n"]), flush=True)
    regimes = [("full", 1.0, 0)] + [(f"row{int(round(a.thin * 100)):02d}t{s}", a.thin, s) for s in a.thin_seeds]
    for keep in a.cbc_keep:
        lk = np.random.default_rng(1000 + int(keep * 100)).random(len(ly)) < keep if keep < 1 else np.ones(len(ly), bool)
        # cell-year CBC means over kept lists
        ky = lcid[lk] * 10000 + ly[lk]
        uk, kinv = np.unique(ky, return_inverse=True)
        cy = np.zeros((len(uk), S)); np.add.at(cy, kinv, Padj[lk]); cy /= np.bincount(kinv)[:, None]
        cy_cell, cy_year = uk // 10000, uk % 10000
        def epoch_cbc(cid, early):
            m = (cy_cell == cid) & ((cy_year >= w0) & (cy_year <= w1) if early else (cy_year > w1))
            return cy[m].mean(0) if m.any() else None
        dC = np.zeros((len(hcid), S)); has = np.zeros(len(hcid), bool)
        for i, k in enumerate(hcid):
            ee, mm = epoch_cbc(k, True), epoch_cbc(k, False)
            if ee is not None and mm is not None:
                dC[i] = mm - ee; has[i] = True
        rowkey = (r.astype(np.int64) * 1000 + c) * 10000 + y
        pos = np.searchsorted(uk, rowkey); pos = np.clip(pos, 0, len(uk) - 1)
        row_has = uk[pos] == rowkey; Crow = np.where(row_has[:, None], cy[pos], np.nan)
        for rname, frac, seed in regimes:
            keep_row = is_tr & (np.random.default_rng(seed).random(len(y)) < frac if frac < 1 else True)
            out = {"cbc_keep": keep, "eval_cells_with_cbc_change": int(has.sum())}
            cid = r.astype(np.int64) * 1000 + c
            # ---- within readout
            u, inv = np.unique(cid[keep_row], return_inverse=True); n = np.bincount(inv)
            rows = np.where(keep_row)[0]
            g2 = n[inv] >= 2
            def demean(V, rr, inv_, n_):
                s = np.zeros((n_.size, V.shape[1])); np.add.at(s, inv_, V[rr]); return V[rr] - (s / n_[:, None])[inv_]
            Zd, Yd = demean(Z, rows, inv, n)[g2], demean(Ld, rows, inv, n)[g2]
            mw = blr.fit(Zd, Yd, [(0, a.rank)]); Bw = np.asarray(mw["coef"])
            pr = rows[row_has[rows]]
            u2, inv2 = np.unique(cid[pr], return_inverse=True); n2 = np.bincount(inv2); g = n2[inv2] >= 2
            Zp, Yp, Cp = demean(Z, pr, inv2, n2)[g], demean(Ld, pr, inv2, n2)[g], demean(Crow, pr, inv2, n2)[g]
            Dp = Zp @ Bw.T
            ab = lsq(np.stack([Dp.T, Cp.T], 2), Yp.T); b1 = lsq(Cp.T[..., None], Yp.T)[:, 0]
            P_w = {"desk": dZ @ Bw.T}
            P_w["cbc"] = dC * b1; P_w["combo"] = P_w["desk"] * ab[:, 0] + dC * ab[:, 1]
            # ---- level readout
            Zm = np.zeros((len(u), a.rank)); np.add.at(Zm, inv, Z[rows]); Zm /= n[:, None]
            Ym = np.zeros((len(u), S)); np.add.at(Ym, inv, Ld[rows]); Ym /= n[:, None]
            ml = blr.fit(Zm, Ym, [(0, a.rank)]); Bl = np.asarray(ml["coef"])
            Cm = np.zeros((len(u2), S)); np.add.at(Cm, inv2, Crow[pr]); Cm /= n2[:, None]
            Zm2 = np.zeros((len(u2), a.rank)); np.add.at(Zm2, inv2, Z[pr]); Zm2 /= n2[:, None]
            Ym2 = np.zeros((len(u2), S)); np.add.at(Ym2, inv2, Ld[pr]); Ym2 /= n2[:, None]
            Dm = Zm2 @ Bl.T; one = np.ones_like(Dm)
            abl = lsq(np.stack([one.T, Dm.T, Cm.T], 2), Ym2.T); bl1 = lsq(np.stack([one.T, Cm.T], 2), Ym2.T)[:, 1]
            P_l = {"desk": dZ @ Bl.T, "cbc": dC * bl1, "combo": (dZ @ Bl.T) * abl[:, 1] + dC * abl[:, 2]}
            out.update({"train_rows": int(keep_row.sum()), "within_pairs": int(g.sum()), "level_cbc_cells": int(len(u2))})
            for ro_name, P, wts in (("within", P_w, (ab[:, 0], ab[:, 1])), ("level", P_l, (abl[:, 1], abl[:, 2]))):
                blockout = {}
                for st, m in strata.items():
                    if m.sum() == 0:
                        continue
                    dr = {k: nfcorr(v, m) for k, v in P.items()}
                    blockout[st] = {"skill": {k: float(v[0]) for k, v in dr.items()},
                                    "combo_minus_desk": [float((dr["combo"] - dr["desk"])[0])] +
                                    [float(x) for x in np.percentile((dr["combo"] - dr["desk"])[1:], [2.5, 97.5])],
                                    "cbc_minus_desk": [float((dr["cbc"] - dr["desk"])[0])] +
                                    [float(x) for x in np.percentile((dr["cbc"] - dr["desk"])[1:], [2.5, 97.5])],
                                    "median_weight_desk": float(np.median(wts[0][m])), "median_weight_cbc": float(np.median(wts[1][m]))}
                out[ro_name] = blockout
                sc = blockout["sed_common"]
                print(f"  cbc_keep {keep:.2f} {rname:9s} {ro_name:6s} sed_common desk {sc['skill']['desk']:+.3f} cbc "
                      f"{sc['skill']['cbc']:+.3f} combo {sc['skill']['combo']:+.3f} | combo-desk {sc['combo_minus_desk'][0]:+.3f} "
                      f"[{sc['combo_minus_desk'][1]:+.3f},{sc['combo_minus_desk'][2]:+.3f}] | weights a {sc['median_weight_desk']:+.2f} "
                      f"b {sc['median_weight_cbc']:+.2f}", flush=True)
            res["regimes"][f"keep{keep:.2f}|{rname}"] = out
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e086r.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
