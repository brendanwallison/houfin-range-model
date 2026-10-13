"""E062: does RAW eBird (our own aggregation of the EBD) agree with BBS -- on level and on change -- cell by cell?

    python research/exp/ebird_raw_agree.py --cache <research cache> --grid <build_ebird_raw_grid npz> --out <dir>
    python research/exp/ebird_raw_agree.py --summarize <out>

The same BBS side as E058 (research/exp/ebird_change_agree.py): per 27 km cell with >= 2 surveyed years per ABBA half
in both windows (default 2010-2014 and 2019-2024), BBS change in cross halves, so cov(change_A, change_B) is the
noise-free change variance (M2; observers are NOT removed, M3). eBird side: per cell, window and year-parity half
(even / odd years), the smoothed log reporting rate log((detections + 0.5) / (checklists + 1)) per species, with
>= --min-chk checklists in every window-half. eBird change x = late - early (all years), and its halves xA, xB from
the parity halves -- which raw data allows and the S&T product did not, so agreement can be FULLY noise-corrected:
  E_att = sum cov(x, BBS change) / sqrt(sum var(x) * sum cov(cA, cB))       -- comparable with E058's 0.10
  E_dis = sum cov(x, BBS change) / sqrt(sum cov(xA, xB) * sum cov(cA, cB))   -- both surveys' noise removed
pooled over species, centred (place-specific, M4) and uncentred. Also, as E058: the regional BBS field R (leave-one-
out 300 km Gaussian of other cells' BBS change), eBird beyond R, a moved-eBird null (the eBird field shifted /
mirrored on the grid, 8 moves, M19), observer-continuity strata (M3), prevalence tiers (M14), species-level agreement;
and LEVEL agreement in the late window (eBird log reporting rate vs BBS log1p mean count, centred, disattenuated with
both surveys' halves). Species: the cache's 96 reference + 273 dev species present in the grid (no sealed species,
never House Finch). CIs: bootstrap over 6x6-cell blocks.
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
MOVES = [("shift_e40", 0, 40), ("shift_w40", 0, -40), ("shift_n20", -20, 0), ("shift_s20", 20, 0),
         ("flip_ew", None, "ew"), ("flip_ns", None, "ns"), ("shift_e70", 0, 70), ("shift_s35", 35, 0)]


def _pooled(parts, dis=None):
    """parts: (x, cA, cB) per species. Attenuated corr (var x) or, with dis=(xA, xB) per species, fully noise-free."""
    sxy = sxx = syy = 0.0
    for i, (x, a, b) in enumerate(parts):
        sxy += float(np.sum(x * 0.5 * (a + b)))
        syy += float(np.sum(a * b))
        sxx += float(np.sum(dis[i][0] * dis[i][1])) if dis is not None else float(np.sum(x * x))
    return sxy / np.sqrt(sxx * syy) if sxx > 0 and syy > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--grid")
    ap.add_argument("--out")
    ap.add_argument("--early", type=int, nargs=2, default=[2010, 2014])
    ap.add_argument("--late", type=int, nargs=2, default=[2019, 2024])
    ap.add_argument("--min-years", type=int, default=2)
    ap.add_argument("--min-chk", type=int, default=10)
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--scale-km", type=float, default=300.0)
    ap.add_argument("--cell-km", type=float, default=27.0)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    import pandas as pd
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
    from src.community_encoder.train_DESK.validation_core import abba_halves
    from src.data.preprocess import bbs
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    meta = json.load(open(c("meta.json")))
    species = list(meta["community"]) + list(meta["dev_species"])
    assert "houfin" not in species and not set(species) & set(meta.get("sealed_species", []))
    X = np.concatenate([np.load(c("X_comm.npy")), np.load(c("X_dev.npy"))], axis=1).astype("float64")
    keys = np.load(c("keys.npy"))
    yr = keys[:, 2]
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    runs = bbs.load_run_metadata()
    ry = pd.read_csv(c("route_years.csv"))
    mm = ry.merge(runs, left_on=["country", "state", "route", "year"], right_on=["CountryNum", "StateNum", "Route", "Year"])
    obs = mm.groupby(["row", "col", "year"])["ObsN"].apply(lambda s: frozenset(s.astype(int))).to_dict()
    obs_of = lambda rows: frozenset().union(*[obs.get((int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])),
                                                      frozenset()) for i in rows])
    e0, e1 = a.early
    l0, l1 = a.late
    by = {}
    for i in np.where(((yr >= e0) & (yr <= e1)) | ((yr >= l0) & (yr <= l1)))[0]:
        by.setdefault(cid[i], []).append(i)

    # eBird per cell: window x parity-half sums
    G = np.load(a.grid, allow_pickle=False)
    gk, gn, gd, gsp = G["keys"], G["n_chk"].astype("float64"), G["det"].astype("float64"), list(G["species"])
    gcol = {s: i for i, s in enumerate(gsp)}
    sp_idx = [(j, gcol[s]) for j, s in enumerate(species) if s in gcol]
    assert "houfin" not in gcol
    gcell = gk[:, 0].astype(np.int64) * 100000 + gk[:, 1]
    gyr = gk[:, 2]
    eb = {}
    for w, (w0, w1) in (("e", a.early), ("l", a.late)):
        for h in (0, 1):
            m = (gyr >= w0) & (gyr <= w1) & (gyr % 2 == h)
            dfn = pd.DataFrame({"c": gcell[m], "n": gn[m]})
            n_by = dfn.groupby("c")["n"].sum()
            d_by = pd.DataFrame(gd[m][:, [g for _, g in sp_idx]]).groupby(gcell[m]).sum()
            eb[(w, h)] = (n_by, d_by)

    g = {"eA": [], "eB": [], "lA": [], "lB": []}
    cells, kind, ebn, ebd = [], [], {}, {}
    for cc, rows in by.items():
        rows = np.array(rows)
        re_, rl = rows[yr[rows] <= e1], rows[yr[rows] >= l0]
        if len(re_) < 2 * a.min_years or len(rl) < 2 * a.min_years:
            continue
        if any(cc not in eb[k][0].index or eb[k][0][cc] < a.min_chk for k in eb):
            continue
        he, hl = abba_halves(re_, yr[re_]), abba_halves(rl, yr[rl])
        if min(len(he[0]), len(he[1]), len(hl[0]), len(hl[1])) < 1:
            continue
        g["eA"].append(he[0]), g["eB"].append(he[1]), g["lA"].append(hl[0]), g["lB"].append(hl[1])
        o1, o2 = obs_of(re_), obs_of(rl)
        kind.append("unknown" if not (o1 and o2) else
                    ("unchanged" if o1 == o2 else ("replaced" if not (o1 & o2) else "partial")))
        cells.append((cc // 100000, cc % 100000))
    cells, kind = np.array(cells), np.array(kind)
    Y = {k: epoch_mean_observed(X, v) for k, v in g.items()}
    cA, cB = Y["lA"] - Y["eB"], Y["lB"] - Y["eA"]
    lvA, lvB = Y["lA"], Y["lB"]
    ccode = cells[:, 0].astype(np.int64) * 100000 + cells[:, 1]

    def eb_index(cc_arr, w, h=None):
        hs = (0, 1) if h is None else (h,)
        n = sum(eb[(w, hh)][0].reindex(cc_arr).to_numpy(dtype="float64") for hh in hs)
        d = sum(eb[(w, hh)][1].reindex(cc_arr).to_numpy(dtype="float64") for hh in hs)
        return np.log((d + 0.5) / (n[:, None] + 1.0))
    xe, xl = eb_index(ccode, "e"), eb_index(ccode, "l")
    x = xl - xe
    xA = eb_index(ccode, "l", 0) - eb_index(ccode, "e", 0)
    xB = eb_index(ccode, "l", 1) - eb_index(ccode, "e", 1)
    LA, LB = eb_index(ccode, "l", 0), eb_index(ccode, "l", 1)
    # moved eBird change fields (M19): look up the change at a moved position among cells with eBird data
    allc = np.array(sorted(set(eb[("e", 0)][0].index) & set(eb[("e", 1)][0].index) &
                           set(eb[("l", 0)][0].index) & set(eb[("l", 1)][0].index)))
    rr, cc_ = allc // 100000, allc % 100000
    H, Wd = int(rr.max()) + 1, int(cc_.max()) + 1
    xall = eb_index(allc, "l") - eb_index(allc, "e")
    pos = {int(k): i for i, k in enumerate(allc)}
    moved = {}
    for nm, dr, dc in MOVES:
        if dr is None:
            tr_ = cells[:, 0] if dc == "ew" else H - 1 - cells[:, 0]
            tc_ = Wd - 1 - cells[:, 1] if dc == "ew" else cells[:, 1]
        else:
            tr_, tc_ = (cells[:, 0] + dr) % H, (cells[:, 1] + dc) % Wd
        idx = np.array([pos.get(int(r_) * 100000 + int(c2), -1) for r_, c2 in zip(tr_, tc_)])
        mv = np.full(x.shape, np.nan)
        mv[idx >= 0] = xall[idx[idx >= 0]]
        moved[nm] = mv
    xy = cells.astype("float64") * a.cell_km
    Wt = np.exp(-0.5 * ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1) / a.scale_km ** 2)
    np.fill_diagonal(Wt, 0.0)
    Wt /= np.maximum(Wt.sum(1, keepdims=True), 1e-12)
    prev = (X > 0).mean(axis=0)
    per = []
    for k_, (j, gi) in enumerate(sp_idx):
        if ((X[:, j] > 0).sum() < 10):
            continue
        per.append(dict(code=species[j], prev=float(prev[j]), x=x[:, k_], xA=xA[:, k_], xB=xB[:, k_],
                        cA=cA[:, j], cB=cB[:, j], reg=0.5 * (Wt @ cA[:, j] + Wt @ cB[:, j]),
                        LA=LA[:, k_], LB=LB[:, k_], L=xl[:, k_], lA=lvA[:, j], lB=lvB[:, j],
                        moved={nm: v[:, k_] for nm, v in moved.items()}))
    print(f"[E062] {len(cells)} cells with BBS in both windows and >= {a.min_chk} eBird checklists per window-half; "
          f"{len(per)} species; observers {dict(zip(*np.unique(kind, return_counts=True)))}", flush=True)
    blocks = (cells[:, 0] // 6) * 10000 + cells[:, 1] // 6
    ub = np.unique(blocks)
    rng = np.random.default_rng(62)
    draws = [None] + [rng.choice(ub, size=len(ub), replace=True) for _ in range(a.n_boot)]

    def stats(w, mask=None, tier=None):
        res = {}
        for centred in (True, False):
            sfx = "c" if centred else "u"
            acc = {k: [] for k in ("att", "reg", "beyond", "lvl")}
            dis, dlv, accm = [], [], {nm: [] for nm, _, _ in MOVES}
            for s in per:
                if tier is not None and not (tier[0] <= s["prev"] < tier[1]):
                    continue
                m = np.ones(len(cells), bool) if mask is None else mask.copy()
                ww = w[m]
                if ww.sum() < 2:
                    continue
                rep = lambda v: np.repeat(v[m], ww)
                v = {k: rep(s[k]) for k in ("x", "xA", "xB", "cA", "cB", "reg", "LA", "LB", "L", "lA", "lB")}
                if centred:
                    v = {k: z - z.mean() for k, z in v.items()}
                acc["att"].append((v["x"], v["cA"], v["cB"]))
                dis.append((v["xA"], v["xB"]))
                acc["reg"].append((v["reg"], v["cA"], v["cB"]))
                beta = float(np.sum(v["x"] * v["reg"]) / np.sum(v["reg"] ** 2)) if np.sum(v["reg"] ** 2) > 0 else 0.0
                acc["beyond"].append((v["x"] - beta * v["reg"], v["cA"], v["cB"]))
                acc["lvl"].append((v["L"], v["lA"], v["lB"]))
                dlv.append((v["LA"], v["LB"]))
                for nm in accm:
                    mv = rep(s["moved"][nm])
                    f = np.isfinite(mv)
                    if f.sum() < 2:
                        continue
                    t = [mv[f], v["cA"][f], v["cB"][f]]
                    if centred:
                        t = [z - z.mean() for z in t]
                    accm[nm].append(tuple(t))
            res[f"E_att_{sfx}"] = _pooled(acc["att"])
            res[f"E_dis_{sfx}"] = _pooled(acc["att"], dis)
            res[f"R_{sfx}"] = _pooled(acc["reg"])
            res[f"beyondR_{sfx}"] = _pooled(acc["beyond"])
            res[f"level_dis_{sfx}"] = _pooled(acc["lvl"], dlv)
            res[f"level_att_{sfx}"] = _pooled(acc["lvl"])
            res[f"null_{sfx}"] = [_pooled(v) for v in accm.values() if v]
            res[f"n_{sfx}"] = len(acc["att"])
        return res

    def boot(mask=None, tier=None):
        out = []
        for d in draws:
            if d is None:
                w = np.ones(len(cells), dtype=int)
            else:
                cnt = dict(zip(*np.unique(d, return_counts=True)))
                w = np.array([cnt.get(b, 0) for b in blocks], dtype=int)
            out.append(stats(w, mask, tier))
        pt, bs = out[0], out[1:]
        summ = {}
        for k, val in pt.items():
            if k.startswith("null_"):
                vv = [z for z in val if np.isfinite(z)]
                summ[k] = {"median": float(np.median(vv)) if vv else None, "max": float(np.max(vv)) if vv else None}
            elif k.startswith("n_"):
                summ[k] = val
            else:
                b = np.array([o[k] for o in bs], dtype="float64")
                b = b[np.isfinite(b)]
                summ[k] = {"v": float(val), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
                           if len(b) > 10 else None}
        return summ

    rep = {"cells": int(len(cells)), "species": len(per), "early": a.early, "late": a.late, "min_chk": a.min_chk,
           "grid": os.path.abspath(a.grid), "observer_classes": {k: int(v) for k, v in zip(*np.unique(kind, return_counts=True))},
           "all": boot()}
    for k in ("unchanged", "partial", "replaced"):
        rep[f"obs_{k}"] = boot(mask=(kind == k))
    for lo, hi in TIERS:
        rep[f"prev_{lo:g}-{min(hi, 1):g}"] = boot(tier=(lo, hi))
    mx = np.array([np.mean(s["x"]) for s in per]); mA = np.array([np.mean(s["cA"]) for s in per])
    mB = np.array([np.mean(s["cB"]) for s in per])
    xc, Ac, Bc = mx - mx.mean(), mA - mA.mean(), mB - mB.mean()
    rep["species_level_att"] = float(np.sum(xc * 0.5 * (Ac + Bc)) / np.sqrt(np.sum(xc ** 2) * max(np.sum(Ac * Bc), 1e-12)))
    json.dump(rep, open(os.path.join(a.out, "ebird_raw_agree.json"), "w"), indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "ebird_raw_agree.json")))
    print(f"[E062] cells {r['cells']}, species {r['species']}, BBS/eBird {r['early']} vs {r['late']}, min_chk "
          f"{r['min_chk']}, observers {r['observer_classes']}")
    f = lambda d: (f"{d['v']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]" if d.get("ci") else f"{d['v']:+.3f}")
    for key in [k for k in r if k == "all" or k.startswith("obs_") or k.startswith("prev_")]:
        s = r[key]
        for sfx, lab in (("c", "centred"), ("u", "uncentred")):
            print(f"  {key:15s} {lab:9s} n={s[f'n_{sfx}']:3d} | E_att {f(s[f'E_att_{sfx}'])} | E_dis "
                  f"{f(s[f'E_dis_{sfx}'])} | R {f(s[f'R_{sfx}'])} | beyond R {f(s[f'beyondR_{sfx}'])} | moved null "
                  f"med {s[f'null_{sfx}']['median']} max {s[f'null_{sfx}']['max']} | level att "
                  f"{f(s[f'level_att_{sfx}'])} dis {f(s[f'level_dis_{sfx}'])}")
    print(f"  species-level mean change corr (eBird attenuated) {r['species_level_att']:+.3f}")


if __name__ == "__main__":
    main()
