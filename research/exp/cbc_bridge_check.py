"""E063: does CBC regional change cover the withheld era better than extrapolating BBS's own later trend backward?

    python research/exp/cbc_bridge_check.py --cache <cache> --cbc-csv <indices csv> --strata-shp A [--strata-shp2 B] --out <dir>
    python research/exp/cbc_bridge_check.py --summarize <out>

Truth: BBS change of each species at each 27 km cell between the withheld early epoch (default 1966-1986) and the modern
epoch (2005-2025), in cross ABBA halves (noise-free variance = cov of halves; M2; observers not removed, M3).
Predictors, all assigned to cells through the state/province x BCR stratum polygons (BBS USGS/CWS stratification):
- CBC_withheld: the CBC stratum's change over the SAME era -- log1p(mean index, winters ending 1967-1986) to
  log1p(mean index, 2005-2024) -- information from a different survey that covers the years BBS is withheld from;
- BBS_trend: what is available WITHOUT any survey of the withheld era -- the species' own BBS change over the trained
  era (1996-2005 -> 2016-2025), as a leave-one-out 300 km Gaussian field of OTHER cells, extrapolated back (a linear
  trend: same direction and per-year rate; corr is scale-free);
- CBC_trained: the CBC stratum's change over the trained era (1996-2005 -> 2016-2024) -- does CBC help only because
  it covers the withheld years, or as any regional trend does?
Statistics: noise-free corr with the truth, pooled over species (sum cov / sqrt(sum var(pred) * sum cov(halves))),
centred (place-specific, M4) and uncentred; CBC_withheld beyond BBS_trend (residualized per species); paired
bootstrap differences over 6x6-cell blocks; prevalence tiers (M14); species-level corr of mean change. Species: the
cache's 96 reference + 273 dev species with a CBC product (medium/high suitability), never sealed, never House Finch.
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
TIERS = [(0.0, 0.02), (0.02, 0.1), (0.1, 0.3), (0.3, 1.01)]
PREDS = ("CBC_withheld", "BBS_trend", "CBC_trained")


def _corr(parts):
    sxy = sxx = syy = 0.0
    for x, a, b in parts:
        sxy += float(np.sum(x * 0.5 * (a + b))); sxx += float(np.sum(x * x)); syy += float(np.sum(a * b))
    return sxy / np.sqrt(sxx * syy) if sxx > 0 and syy > 0 else np.nan


def stratum_map(shps, H, W, tr, crs):
    import geopandas as gpd
    from rasterio.features import rasterize
    out = np.full((H, W), None, dtype=object)
    seen = set()
    for shp in shps:
        g = gpd.read_file(shp).to_crs(crs)
        codes = [c for c in g["ST_12"].astype(str).unique() if c not in seen]
        seen |= set(codes)
        lut = {c: i + 1 for i, c in enumerate(codes)}
        gs = g[g["ST_12"].astype(str).isin(lut)]
        ras = rasterize(((geom, lut[str(c)]) for geom, c in zip(gs.geometry, gs["ST_12"])), out_shape=(H, W),
                        transform=tr, fill=0, dtype="int32")
        inv = {v: k for k, v in lut.items()}
        take = (ras > 0) & np.equal(out, None)
        out[take] = [inv[v] for v in ras[take]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache"); ap.add_argument("--cbc-csv"); ap.add_argument("--out")
    ap.add_argument("--strata-shp"); ap.add_argument("--strata-shp2", default=None)
    ap.add_argument("--early", type=int, nargs=2, default=[1966, 1986])
    ap.add_argument("--modern", type=int, nargs=2, default=[2005, 2025])
    ap.add_argument("--trained", type=int, nargs=4, default=[1996, 2005, 2016, 2025])
    ap.add_argument("--min-years", type=int, default=2)
    ap.add_argument("--scale-km", type=float, default=300.0); ap.add_argument("--cell-km", type=float, default=27.0)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    import pandas as pd
    import rasterio
    from src.config_utils import load_data_config
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
    from src.community_encoder.train_DESK.validation_core import abba_halves
    os.makedirs(a.out, exist_ok=True)
    dcfg = load_data_config()
    c = lambda f: os.path.join(a.cache, f)
    meta = json.load(open(c("meta.json")))
    species = list(meta["community"]) + list(meta["dev_species"])
    assert "houfin" not in species and not set(species) & set(meta.get("sealed_species", []))
    X = np.concatenate([np.load(c("X_comm.npy")), np.load(c("X_dev.npy"))], axis=1).astype("float64")
    keys = np.load(c("keys.npy")); yr = keys[:, 2]
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]

    def epoch_pairs(w0, w1, v0, v1):
        by = {}
        for i in np.where(((yr >= w0) & (yr <= w1)) | ((yr >= v0) & (yr <= v1)))[0]:
            by.setdefault(cid[i], []).append(i)
        g = {"eA": [], "eB": [], "lA": [], "lB": [], "e": [], "l": []}
        cells = []
        for cc, rows in by.items():
            rows = np.array(rows)
            re_, rl = rows[yr[rows] <= w1], rows[yr[rows] >= v0]
            if len(re_) < 2 * a.min_years or len(rl) < 2 * a.min_years:
                continue
            he, hl = abba_halves(re_, yr[re_]), abba_halves(rl, yr[rl])
            if min(len(he[0]), len(he[1]), len(hl[0]), len(hl[1])) < 1:
                continue
            for k, v in (("eA", he[0]), ("eB", he[1]), ("lA", hl[0]), ("lB", hl[1]), ("e", re_), ("l", rl)):
                g[k].append(v)
            cells.append(cc)
        Y = {k: epoch_mean_observed(X, v) for k, v in g.items()}
        return np.array(cells), Y
    cells, Y = epoch_pairs(a.early[0], a.early[1], a.modern[0], a.modern[1])
    cA, cB = Y["lA"] - Y["eB"], Y["lB"] - Y["eA"]
    # BBS trained-era change at ALL cells with data, as a leave-one-out regional field evaluated at the truth cells
    t0, t1, t2, t3 = a.trained
    tcells, TY = epoch_pairs(t0, t1, t2, t3)
    tchg = TY["l"] - TY["e"]
    rc = lambda cc: np.column_stack([cc // 100000, cc % 100000]).astype("float64") * a.cell_km
    P, Q = rc(cells), rc(tcells)
    Wt = np.exp(-0.5 * ((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1) / a.scale_km ** 2)
    Wt[cells[:, None] == tcells[None, :]] = 0.0          # leave the truth cell itself out
    Wt /= np.maximum(Wt.sum(1, keepdims=True), 1e-12)
    bbs_trend = Wt @ tchg                                # (n_cells, S)

    with rasterio.open(dcfg["grid"]["ref_raster"]) as r:
        crs, tr, H, W = r.crs, r.transform, r.height, r.width
    smap = stratum_map([s for s in (a.strata_shp, a.strata_shp2) if s], H, W, tr, crs)
    cstr = np.array([smap[cc // 100000, cc % 100000] for cc in cells], dtype=object)
    d = pd.read_csv(a.cbc_csv, encoding="latin-1")
    d = d[(d["region_type"] == "stratum") & d["survey_suitability"].isin(["medium", "high"])]
    d = d[~d["scientific_name"].str.contains("/|,| and ", regex=True, na=True)]
    tax = pd.read_csv(os.path.join(dcfg["datasets_root"], "avonet", "eBird_taxonomy.csv"),
                      usecols=["SCIENTIFIC_NAME", "SPECIES_CODE", "CATEGORY"])
    tax = tax[tax["CATEGORY"] == "species"]
    d["code"] = d["scientific_name"].map(dict(zip(tax["SCIENTIFIC_NAME"], tax["SPECIES_CODE"])))
    d = d[d["code"].isin(set(species))]

    def cbc_change(y0, y1, z0, z1):
        m0 = d[(d.year_end >= y0) & (d.year_end <= y1)].groupby(["code", "region"])["annual_index"].mean()
        m1 = d[(d.year_end >= z0) & (d.year_end <= z1)].groupby(["code", "region"])["annual_index"].mean()
        return (np.log1p(m1) - np.log1p(m0)).dropna()
    cw = cbc_change(a.early[0] + 1, a.early[1], a.modern[0], min(a.modern[1], 2024))
    ct = cbc_change(t0, t1, t2, min(t3, 2024))
    prev = (X > 0).mean(axis=0)
    per = []
    for j, code in enumerate(species):
        if code not in cw.index.get_level_values(0):
            continue
        xw = np.array([cw.get((code, s), np.nan) for s in cstr])
        xt = np.array([ct.get((code, s), np.nan) for s in cstr])
        ok = np.isfinite(xw) & np.isfinite(xt) & np.isfinite(bbs_trend[:, j])
        if ok.sum() < 30:
            continue
        per.append(dict(code=code, prev=float(prev[j]), ok=ok, CBC_withheld=xw, CBC_trained=xt,
                        BBS_trend=bbs_trend[:, j], cA=cA[:, j], cB=cB[:, j]))
    print(f"[E063] {len(cells)} cells with BBS in both epochs; {len(per)} species with CBC strata covering >= 30 of them",
          flush=True)
    blocks = ((cells // 100000) // 6) * 10000 + (cells % 100000) // 6
    ub = np.unique(blocks)
    rng = np.random.default_rng(63)
    draws = [None] + [rng.choice(ub, size=len(ub), replace=True) for _ in range(a.n_boot)]

    def stats(w, tier=None):
        res = {}
        for centred in (True, False):
            sfx = "c" if centred else "u"
            acc = {p: [] for p in PREDS + ("CBCw_beyond_BBS",)}
            for s in per:
                if tier is not None and not (tier[0] <= s["prev"] < tier[1]):
                    continue
                m = s["ok"]; ww = w[m]
                if ww.sum() < 2:
                    continue
                rep = lambda v: np.repeat(v[m], ww)
                v = {k: rep(s[k]) for k in PREDS + ("cA", "cB")}
                if centred:
                    v = {k: z - z.mean() for k, z in v.items()}
                for p in PREDS:
                    acc[p].append((v[p], v["cA"], v["cB"]))
                b = v["BBS_trend"]; bb = float(np.sum(b * b))
                beta = float(np.sum(v["CBC_withheld"] * b) / bb) if bb > 0 else 0.0
                acc["CBCw_beyond_BBS"].append((v["CBC_withheld"] - beta * b, v["cA"], v["cB"]))
            for p, parts in acc.items():
                res[f"{p}_{sfx}"] = _corr(parts)
            res[f"diff_CBCw_minus_BBS_{sfx}"] = res[f"CBC_withheld_{sfx}"] - res[f"BBS_trend_{sfx}"]
            res[f"diff_CBCw_minus_CBCt_{sfx}"] = res[f"CBC_withheld_{sfx}"] - res[f"CBC_trained_{sfx}"]
            res[f"n_{sfx}"] = len(acc["BBS_trend"])
        return res

    def boot(tier=None):
        outs = []
        for dr in draws:
            if dr is None:
                w = np.ones(len(cells), dtype=int)
            else:
                cnt = dict(zip(*np.unique(dr, return_counts=True)))
                w = np.array([cnt.get(b, 0) for b in blocks], dtype=int)
            outs.append(stats(w, tier))
        pt, bs = outs[0], outs[1:]
        summ = {}
        for k, v in pt.items():
            if k.startswith("n_"):
                summ[k] = v; continue
            b = np.array([o[k] for o in bs], dtype="float64"); b = b[np.isfinite(b)]
            summ[k] = {"v": float(v), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
        return summ

    rep = {"cells": int(len(cells)), "species": len(per), "early": a.early, "modern": a.modern, "trained": a.trained,
           "all": boot()}
    for lo, hi in TIERS:
        rep[f"prev_{lo:g}-{min(hi, 1):g}"] = boot((lo, hi))
    sl = {}
    for p in PREDS:
        mx = np.array([np.nanmean(s[p][s["ok"]]) for s in per])
        mA = np.array([np.mean(s["cA"][s["ok"]]) for s in per]); mB = np.array([np.mean(s["cB"][s["ok"]]) for s in per])
        xc, Ac, Bc = mx - mx.mean(), mA - mA.mean(), mB - mB.mean()
        sl[p] = float(np.sum(xc * 0.5 * (Ac + Bc)) / np.sqrt(np.sum(xc ** 2) * max(np.sum(Ac * Bc), 1e-12)))
    rep["species_level"] = sl
    json.dump(rep, open(os.path.join(a.out, "cbc_bridge_check.json"), "w"), indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "cbc_bridge_check.json")))
    print(f"[E063] cells {r['cells']}, species {r['species']}, truth {r['early']} -> {r['modern']}, BBS trend from {r['trained']}")
    f = lambda d: f"{d['v']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]"
    for key in [k for k in r if k == "all" or k.startswith("prev_")]:
        s = r[key]
        for sfx, lab in (("c", "centred"), ("u", "uncentred")):
            print(f"  {key:13s} {lab:9s} n={s[f'n_{sfx}']:3d} | CBC_withheld {f(s[f'CBC_withheld_{sfx}'])} | BBS_trend "
                  f"{f(s[f'BBS_trend_{sfx}'])} | CBC_trained {f(s[f'CBC_trained_{sfx}'])} | CBCw-BBS "
                  f"{f(s[f'diff_CBCw_minus_BBS_{sfx}'])} | CBCw beyond BBS {f(s[f'CBCw_beyond_BBS_{sfx}'])} | CBCw-CBCt "
                  f"{f(s[f'diff_CBCw_minus_CBCt_{sfx}'])}")
    print("  species level:", {k: round(v, 3) for k, v in r["species_level"].items()})


if __name__ == "__main__":
    main()
