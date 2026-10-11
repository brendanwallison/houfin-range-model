"""E047: does county-census-based cropland history (CropAT-US) carry local change signal that LUH lacks?

    python research/exp/cropat_test.py --cache <tempho research cache> --cropat <cropat_27km.npz> --out <dir>
    python research/exp/cropat_test.py --summarize <dir>

Per cell with both epochs (E039h/E044d pairs): epoch-mean cropland fraction over the cell's surveyed years in the
early withheld epoch (1966-86) and the modern epoch (2005-25; CropAT ends 2021, later years carry 2021 forward), from
CropAT-US (aggregated to the 27 km grid by cropat_aggregate.py) and from LUH (c3ann + c3nfx + c3per + c4ann + c4per,
raw fractions from the yearly states). Change dC per source; observed ESK change dZ (r24).
TYPE = 32 PCs of the cell's mean covariate levels; REGION = 96 RFF position fields at 300 km. Both residualizations
are fitted on TRAINING cells (not held out, not buffer) and applied to all. Maps (OLS on the 1-2 crop features, fitted
on training cells) graded on HELD-OUT cells by noise-free centred corr against the truth residualized the same way;
paired 6x6-block bootstrap of CropAT - LUH. Also: concordance of the two sources' levels and changes.
NOTE: the training cells' fitted change uses their observed early-epoch communities (a cross-sectional diagnostic of a
relationship, as E044d, not a backcast).
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
from type_dose import ridge_cv, wc  # noqa: E402

LUH_CROP = slice(0, 5)        # c3ann, c3nfx, c3per, c4ann, c4per within the landuse stream


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--cropat")
    ap.add_argument("--out")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--n-type", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--summarize", default=None)
    ap.add_argument("--types", action="store_true", help="E047b: multivariate crop-type (CropAT) vs LUH land-use arms")
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from scipy import sparse
    from change_oracle import group_rows
    from src.config_utils import load_config
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    N = len(keys)
    split = np.load(c("split.npz"))
    withheld = [int(y) for y in split["withheld"]]
    r = a.rank
    E, full, _, _ = group_rows(a.cache, keys, withheld)
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh):
            idx.setdefault(cell, [None, None])[0] = j
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = j
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    cells = E["cells"][pairs[:, 0]]
    held = split["holdout"][cells[:, 0], cells[:, 1]].astype(bool)
    buf = split["buffer"][cells[:, 0], cells[:, 1]].astype(bool)
    # CropAT covers CONUS only (zeros elsewhere): keep cells where BUI (CONUS-only) is available in 2015
    cfg = load_config()
    states_dir = os.path.join(cfg["paths"]["hist_dir"], "yearly_states")
    bui_avail = np.load(os.path.join(states_dir, "state_2015.npz"))["bui"][..., 0]
    conus = np.nan_to_num(bui_avail[cells[:, 0], cells[:, 1]]) > 0.5
    if os.environ.get("CROPAT_ALL_CELLS"):          # diagnostic: keep non-CONUS cells (CropAT arms then invalid)
        conus[:] = True
    ev, tc = np.where(held & conus)[0], np.where(~held & ~buf & conus)[0]
    n_non_conus = int((~conus & (held | (~held & ~buf))).sum())
    dZ = (E["z_full"][pairs[:, 1], :r] - E["z_full"][pairs[:, 0], :r]).astype("float64")
    Ta = (E["z_a"][pairs[:, 1], :r] - E["z_a"][pairs[:, 0], :r]).astype("float64")
    Tb = (E["z_b"][pairs[:, 1], :r] - E["z_b"][pairs[:, 0], :r]).astype("float64")

    def gmat(groups):
        rows = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
        cols = np.concatenate([np.asarray(g, int) for g in groups])
        vals = np.concatenate([np.full(len(g), 1.0 / len(g)) for g in groups])
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(groups), N))
    Ge = gmat([np.asarray(full[p]) for p in pairs[:, 0]])
    Gm = gmat([np.asarray(full[p]) for p in pairs[:, 1]])
    yr = keys[:, 2].astype(int)
    # CropAT crop fraction per key
    ca = np.load(a.cropat)
    cyears, crop = ca["years"].astype(int), ca["crop"].astype("float64")
    ymap = {int(y): i for i, y in enumerate(cyears)}
    yi = np.array([ymap[min(int(y), int(cyears.max()))] for y in yr])     # years after CropAT's last carry it
    C_cat = crop[yi, keys[:, 0], keys[:, 1]]
    # LUH crop fraction per key (raw landuse stream)
    C_luh = np.full(N, np.nan)
    L_all = np.full((N, 33), np.nan) if a.types else None
    for y in np.unique(yr):
        z = np.load(os.path.join(states_dir, f"state_{int(y)}.npz"))
        lu = z["landuse"][..., LUH_CROP].sum(-1)
        sel = yr == y
        C_luh[sel] = lu[keys[sel, 0], keys[sel, 1]]
        if a.types:
            L_all[sel] = z["landuse"][keys[sel, 0], keys[sel, 1]]
    ok_rows = np.isfinite(C_cat) & np.isfinite(C_luh)
    C_cat, C_luh = np.where(ok_rows, C_cat, 0.0), np.where(ok_rows, C_luh, 0.0)
    ep = lambda G, v: np.asarray(G @ v[:, None])[:, 0]
    lev_cat, lev_luh = ep(Gm, C_cat), ep(Gm, C_luh)
    d_cat, d_luh = lev_cat - ep(Ge, C_cat), lev_luh - ep(Ge, C_luh)
    # type and region bases (as E044d)
    F = np.asarray(np.load(c("F_cov_raw.npy"), mmap_mode="r"), "float64")
    ho = split["holdout"][keys[:, 0], keys[:, 1]]
    bfk = split["buffer"][keys[:, 0], keys[:, 1]]
    ty = ~np.isin(keys[:, 2], withheld)
    trn = np.where(~ho & ~bfk & ty & np.isfinite(F).all(1))[0]
    mu, sd = F[trn].mean(0), F[trn].std(0)
    keep = sd > 1e-6
    X = np.where(np.isfinite(F[:, keep]), (F[:, keep] - mu[keep]) / sd[keep], 0.0)
    _, _, Vt = np.linalg.svd(X[trn][:: max(1, len(trn) // 40000)] - X[trn].mean(0), full_matrices=False)
    Tm = np.hstack([np.ones((len(pairs), 1)), np.asarray(Gm @ X) @ Vt[:a.n_type].T])
    rgp = np.random.default_rng(1)
    om, ph = rgp.normal(size=(2, 96)) / 300.0, rgp.uniform(0, 2 * np.pi, 96)
    Tm2 = np.hstack([Tm, np.sqrt(2.0 / 96) * np.cos(cells.astype(float) * 27.0 @ om + ph)])

    def resid(V, basis):
        g, *_ = np.linalg.lstsq(basis[tc], V[tc], rcond=None)
        return V - basis @ g, basis @ g
    blk = (cells[ev, 0] // 6) * 100000 + cells[ev, 1] // 6
    ub, binv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(11)
    W = np.vstack([np.ones(len(ev)), rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)[:, binv]])

    def corr(D, A_, B_, w):
        vp, vt = wc(D, D, w).sum(), wc(A_, B_, w).sum()
        cv = 0.5 * (wc(D, A_, w).sum() + wc(D, B_, w).sum())
        return cv / np.sqrt(vp * vt) if vp > 0 and vt > 0 else np.nan

    def pcorr(x, y):
        x, y = x - x.mean(), y - y.mean()
        return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
    res = {"cache": a.cache, "cropat": a.cropat, "n_heldout": int(len(ev)), "n_train_cells": int(len(tc)),
           "n_cells_dropped_outside_conus": n_non_conus,
           "cropat_years": [int(cyears.min()), int(cyears.max())], "rows_missing_crop": int((~ok_rows).sum()),
           "concordance": {}, "levels": {}, "modes": {}}
    allc = np.r_[tc, ev]
    res["concordance"]["levels_modern"] = pcorr(lev_cat[allc], lev_luh[allc])
    res["concordance"]["change"] = pcorr(d_cat[allc], d_luh[allc])
    res["levels"] = {"cropat_mean_change": float(d_cat[allc].mean()), "luh_mean_change": float(d_luh[allc].mean()),
                     "cropat_sd_change": float(d_cat[allc].std()), "luh_sd_change": float(d_luh[allc].std())}
    for mode, basis in (("none", np.ones((len(pairs), 1))), ("type", Tm), ("type+region", Tm2)):
        dcat_r, _ = resid(d_cat[:, None], basis)
        dluh_r, _ = resid(d_luh[:, None], basis)
        dZ_r, zfit = resid(dZ, basis)
        Ta_r, Tb_r = Ta[ev] - zfit[ev], Tb[ev] - zfit[ev]
        res["concordance"][f"change_{mode}"] = pcorr(dcat_r[allc, 0], dluh_r[allc, 0])
        preds = {}
        for nm, Xf in (("cropat", dcat_r), ("luh", dluh_r), ("both", np.hstack([dcat_r, dluh_r]))):
            A_ = np.hstack([np.ones((len(tc), 1)), Xf[tc]])
            B, *_ = np.linalg.lstsq(A_, dZ_r[tc], rcond=None)
            preds[nm] = np.hstack([np.ones((len(ev), 1)), Xf[ev]]) @ B
        boots = {k: np.array([corr(D, Ta_r, Tb_r, w) for w in W]) for k, D in preds.items()}
        o = {k: {"corr": float(v[0]), "ci": [float(np.nanpercentile(v[1:], q)) for q in (2.5, 97.5)]}
             for k, v in boots.items()}
        d = boots["cropat"] - boots["luh"]
        o["cropat_minus_luh"] = {"diff": float(d[0]), "ci": [float(np.nanpercentile(d[1:], q)) for q in (2.5, 97.5)]}
        d2 = boots["both"] - boots["luh"]
        o["both_minus_luh"] = {"diff": float(d2[0]), "ci": [float(np.nanpercentile(d2[1:], q)) for q in (2.5, 97.5)]}
        res["modes"][mode] = o
    if a.types:
        Tk = ca["types"].astype("float64")[yi, :, keys[:, 0], keys[:, 1]]             # (N, 10)
        Tk = np.where(np.isfinite(Tk) & ok_rows[:, None], Tk, 0.0)
        L_all = np.where(np.isfinite(L_all) & ok_rows[:, None], L_all, 0.0)
        epm = lambda G, M: np.asarray(G @ M)
        dT = epm(Gm, Tk) - epm(Ge, Tk)
        dLa = epm(Gm, L_all) - epm(Ge, L_all)
        feats = {"A_cropat_types": dT, "B_luh_crop5": dLa[:, 0:5], "C_luh_all33": dLa,
                 "D_luh_all33+cropat_types": np.hstack([dLa, dT])}
        blk_tc = (cells[tc, 0] // 6) * 100000 + cells[tc, 1] // 6
        lams = [10.0 ** e for e in np.arange(-3, 3.01, 0.5)]
        res["types"] = {"names": [str(n) for n in ca["type_names"]], "modes": {}}
        for mode, basis in (("type", Tm), ("type+region", Tm2)):
            dZ_r, zfit = resid(dZ, basis)
            Ta_r, Tb_r = Ta[ev] - zfit[ev], Tb[ev] - zfit[ev]
            preds = {}
            for nm, Fx in feats.items():
                Fr, _ = resid(Fx, basis)
                sdv = Fr[tc].std(0)
                k_ = sdv > 1e-9
                Fz = Fr[:, k_] / sdv[k_]
                B, xm, ym, lam = ridge_cv(Fz[tc], dZ_r[tc], blk_tc, lams)
                preds[nm] = (Fz[ev] - xm) @ B + ym
            boots = {k: np.array([corr(D, Ta_r, Tb_r, w) for w in W]) for k, D in preds.items()}
            o = {k: {"corr": float(v[0]), "ci": [float(np.nanpercentile(v[1:], q)) for q in (2.5, 97.5)]}
                 for k, v in boots.items()}
            for x_, y_ in (("A_cropat_types", "B_luh_crop5"), ("D_luh_all33+cropat_types", "C_luh_all33")):
                d = boots[x_] - boots[y_]
                o[f"{x_} - {y_}"] = {"diff": float(d[0]), "ci": [float(np.nanpercentile(d[1:], q)) for q in (2.5, 97.5)]}
            res["types"]["modes"][mode] = o
    fn = "cropat_types.json" if a.types else "cropat_test.json"
    with open(os.path.join(a.out, fn), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    if os.path.exists(os.path.join(out, "cropat_types.json")):
        r = json.load(open(os.path.join(out, "cropat_types.json"), encoding="utf-8"))
        print(f"{os.path.basename(os.path.normpath(r['cache']))}: E047b crop-type arms ({r['n_heldout']} held-out cells)")
        for mode, o in r["types"]["modes"].items():
            arms = " | ".join(f"{k} {v['corr']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]" for k, v in o.items()
                              if "corr" in v)
            diffs = " | ".join(f"{k}: {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]" for k, v in o.items()
                               if "diff" in v)
            print(f"  [{mode:11s}] {arms}")
            print(f"                {diffs}")
        if not os.path.exists(os.path.join(out, "cropat_test.json")):
            return
    r = json.load(open(os.path.join(out, "cropat_test.json"), encoding="utf-8"))
    cc = r["concordance"]
    print(f"{os.path.basename(os.path.normpath(r['cache']))}: {r['n_heldout']} held-out / {r['n_train_cells']} training "
          f"cells (dropped outside CONUS: {r.get('n_cells_dropped_outside_conus')}); CropAT {r['cropat_years']}; rows missing crop "
          f"{r['rows_missing_crop']}")
    print(f"  concordance CropAT vs LUH: modern levels {cc['levels_modern']:+.3f}; change {cc['change']:+.3f}; "
          f"change within type {cc['change_type']:+.3f}; within type+region {cc['change_type+region']:+.3f}")
    lv = r["levels"]
    print(f"  crop-fraction change mean / sd: CropAT {lv['cropat_mean_change']:+.4f} / {lv['cropat_sd_change']:.4f}; "
          f"LUH {lv['luh_mean_change']:+.4f} / {lv['luh_sd_change']:.4f}")
    for mode, o in r["modes"].items():
        print(f"  [{mode:11s}] corr with community change: CropAT {o['cropat']['corr']:+.3f} "
              f"[{o['cropat']['ci'][0]:+.3f},{o['cropat']['ci'][1]:+.3f}]  LUH {o['luh']['corr']:+.3f} "
              f"[{o['luh']['ci'][0]:+.3f},{o['luh']['ci'][1]:+.3f}]  both {o['both']['corr']:+.3f} | CropAT - LUH "
              f"{o['cropat_minus_luh']['diff']:+.3f} [{o['cropat_minus_luh']['ci'][0]:+.3f},"
              f"{o['cropat_minus_luh']['ci'][1]:+.3f}]; both - LUH {o['both_minus_luh']['diff']:+.3f} "
              f"[{o['both_minus_luh']['ci'][0]:+.3f},{o['both_minus_luh']['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
