"""E058: does the eBird Status & Trends 2012-2022 trend agree with BBS CHANGE over the same decade, cell by cell?

    python research/exp/ebird_change_agree.py --cache <research cache> --trends-dir <dir of trends parquets> --out <dir>
    python research/exp/ebird_change_agree.py --summarize <out>

The August test (b6e451d) measured eBird vs BBS agreement on LEVELS (median per-species corr 0.405 over paired
cell-years). Nobody measured agreement on CHANGE, which is what any temporal use of the trend product rests on.

Per species (the cache's 96 reference + 273 dev species only; never the sealed half, never House Finch) and per 27 km
cell with >= --min-years surveyed years in each BBS window (default early 2010-2014, late 2019-2024; 2020 has no
surveys): BBS change = log1p(mean late counts) - log1p(mean early counts), in two ABBA halves (A, B) per window, so
cov(change_A, change_B) is the noise-free change variance (M2: survey noise removed; observers are NOT, M3). eBird
prediction = the trend parquet's abd_ppy, gridded to the model grid by preprocess/ebird_trend.rasterize_parquet, turned
into a log change over the windows' centre gap: gap * log(1 + ppy / 100). Only breeding and resident season products
are compared with breeding-season BBS counts.

Statistics, pooled over species (sum of covariances over the square roots of summed variances), centred (each species'
mean change over its cells removed: place-specific, M4) and uncentred:
- corr_nf(eBird, BBS change) and k = slope of BBS change on the eBird prediction (calibration, M1);
- the same for a REGIONAL BBS field: a leave-one-out Gaussian (--scale-km, default 300) average of OTHER cells' BBS
  change -- the observed-neighbours trend, as a yardstick for how much cell-level change is predictable at all;
- eBird beyond the regional field (partial: eBird residualized on the field, per species);
- a spatial-shift null for eBird (M19): the eBird field moved by whole-grid shifts and reflections, refit-free;
- the same by observer continuity (unchanged / partial / replaced between windows, BBS ObsN; M3) and by prevalence
  tier (M14); species-level agreement of the mean change across species; LEVEL agreement (eBird abd vs BBS late-window
  log1p mean, centred) for comparison with August's 0.405.
CIs: bootstrap over 6x6-cell blocks (species fixed).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
SEASONS_OK = ("breeding", "resident")
TIERS = [(0.0, 0.02), (0.02, 0.1), (0.1, 0.3), (0.3, 1.01)]
SHIFTS = [("shift_e40", 0, 40), ("shift_w40", 0, -40), ("shift_n20", -20, 0), ("shift_s20", 20, 0),
          ("flip_ew", None, "ew"), ("flip_ns", None, "ns"), ("shift_e70", 0, 70), ("shift_s35", 35, 0)]


def _trend_parquet(trends_dir, code):
    hits = sorted(p for p in glob.glob(os.path.join(trends_dir, f"{code}_*_ebird-trends_*.parquet"))
                  if "_folds_" not in os.path.basename(p))
    if not hits:
        return None, None
    season = os.path.basename(hits[0])[len(code) + 1:].split("_ebird-trends")[0]
    return hits[0], season


def _moved(grid, how):
    name, a, b = how
    if a is None:
        return grid[:, ::-1] if b == "ew" else grid[::-1, :]
    return np.roll(np.roll(grid, a, axis=0), b, axis=1)


def _pooled(parts):
    """parts: list of (x, cA, cB) per species (already centred or not). Pooled noise-free corr and k."""
    sxy = sxx = syy = 0.0
    for x, cA, cB in parts:
        y = 0.5 * (cA + cB)
        sxy += float(np.sum(x * y))
        sxx += float(np.sum(x * x))
        syy += float(np.sum(cA * cB))
    corr = sxy / np.sqrt(sxx * syy) if sxx > 0 and syy > 0 else np.nan
    k = sxy / sxx if sxx > 0 else np.nan
    return corr, k


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--trends-dir")
    ap.add_argument("--out")
    ap.add_argument("--early", type=int, nargs=2, default=[2010, 2014])
    ap.add_argument("--late", type=int, nargs=2, default=[2019, 2024])
    ap.add_argument("--min-years", type=int, default=2)
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--scale-km", type=float, default=300.0)
    ap.add_argument("--cell-km", type=float, default=27.0)
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
    from src.data.preprocess import bbs
    from src.data.preprocess.ebird_trend import rasterize_parquet
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    meta = json.load(open(c("meta.json")))
    comm, dev = list(meta["community"]), list(meta["dev_species"])
    species = comm + dev
    assert "houfin" not in species, "House Finch must never enter this analysis"
    assert not set(species) & set(meta.get("sealed_species", [])), "sealed species reached the analysis"
    X = np.concatenate([np.load(c("X_comm.npy")), np.load(c("X_dev.npy"))], axis=1).astype("float64")
    assert X.shape[1] == len(species), (X.shape, len(species))
    keys = np.load(c("keys.npy"))
    yr = keys[:, 2]
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]

    # observers per cell-year (as persistence_observer.py / E028)
    runs = bbs.load_run_metadata()
    ry = pd.read_csv(c("route_years.csv"))
    mm = ry.merge(runs, left_on=["country", "state", "route", "year"],
                  right_on=["CountryNum", "StateNum", "Route", "Year"])
    obs = mm.groupby(["row", "col", "year"])["ObsN"].apply(lambda s: frozenset(s.astype(int))).to_dict()
    obs_of = lambda rows: frozenset().union(*[obs.get((int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])),
                                                      frozenset()) for i in rows])

    e0, e1 = a.early
    l0, l1 = a.late
    gap = 0.5 * (l0 + l1) - 0.5 * (e0 + e1)
    by = {}
    for i in np.where(((yr >= e0) & (yr <= e1)) | ((yr >= l0) & (yr <= l1)))[0]:
        by.setdefault(cid[i], []).append(i)
    g = {"eA": [], "eB": [], "lA": [], "lB": [], "l": []}
    cells, kind = [], []
    for cc, rows in by.items():
        rows = np.array(rows)
        re_, rl = rows[yr[rows] <= e1], rows[yr[rows] >= l0]
        if len(re_) < 2 * a.min_years or len(rl) < 2 * a.min_years:
            continue
        he, hl = abba_halves(re_, yr[re_]), abba_halves(rl, yr[rl])
        if min(len(he[0]), len(he[1]), len(hl[0]), len(hl[1])) < 1:
            continue
        g["eA"].append(he[0]), g["eB"].append(he[1]), g["lA"].append(hl[0]), g["lB"].append(hl[1]), g["l"].append(rl)
        o1, o2 = obs_of(re_), obs_of(rl)
        kind.append("unknown" if not (o1 and o2) else
                    ("unchanged" if o1 == o2 else ("replaced" if not (o1 & o2) else "partial")))
        cells.append((cc // 100000, cc % 100000))
    cells, kind = np.array(cells), np.array(kind)
    Y = {k: epoch_mean_observed(X, v) for k, v in g.items()}
    cA, cB = Y["lA"] - Y["eB"], Y["lB"] - Y["eA"]      # cross halves: no shared survey year
    lvl = Y["l"]
    n_cells = len(cells)
    print(f"[E058] {n_cells} cells with >= {a.min_years} years per half in {e0}-{e1} and {l0}-{l1}; gap {gap:.1f} y; "
          f"observer classes {dict(zip(*np.unique(kind, return_counts=True)))}", flush=True)

    dcfg = load_data_config()
    with rasterio.open(dcfg["grid"]["ref_raster"]) as ref:
        ref_crs, transform, H, W = ref.crs, ref.transform, ref.height, ref.width
    prev = (X > 0).mean(axis=0)

    # regional BBS field: leave-one-out Gaussian average of OTHER cells' change (per half, so noise stays independent)
    xy = cells.astype("float64") * a.cell_km
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    Wt = np.exp(-0.5 * d2 / a.scale_km ** 2)
    np.fill_diagonal(Wt, 0.0)
    Wt /= np.maximum(Wt.sum(1, keepdims=True), 1e-12)

    per = []
    skipped = {}
    for j, code in enumerate(species):
        path, season = _trend_parquet(a.trends_dir, code)
        if path is None:
            skipped[code] = "no trends product"
            continue
        if season not in SEASONS_OK:
            skipped[code] = f"season {season}"
            continue
        grids, (sy, ey) = rasterize_parquet(path, ref_crs, transform, H, W)
        ppy, abd = grids["abd_ppy"], grids["abd"]
        x = ppy[cells[:, 0], cells[:, 1]].astype("float64")
        ab = np.log1p(np.clip(abd[cells[:, 0], cells[:, 1]].astype("float64"), 0, None))
        ok = np.isfinite(x) & np.isfinite(ab)
        if ok.sum() < a.min_cells:
            skipped[code] = f"{int(ok.sum())} cells"
            continue
        xe = gap * np.log1p(np.clip(x, -99.0, None) / 100.0)
        moved = {}
        for how in SHIFTS:
            mg = _moved(ppy, how)[cells[:, 0], cells[:, 1]].astype("float64")
            moved[how[0]] = np.where(np.isfinite(mg), gap * np.log1p(np.clip(mg, -99.0, None) / 100.0), np.nan)
        rA, rB = Wt @ cA[:, j], Wt @ cB[:, j]        # other cells only (diagonal zeroed): their noise is independent
        per.append(dict(code=code, j=j, season=season, years=(sy, ey), ok=ok, x=xe, abd=ab, prev=float(prev[j]),
                        cA=cA[:, j], cB=cB[:, j], lvl=lvl[:, j], reg=0.5 * (rA + rB), moved=moved))
    print(f"[E058] {len(per)} species compared; skipped {len(skipped)} "
          f"({sum('season' in v for v in skipped.values())} by season, "
          f"{sum(v == 'no trends product' for v in skipped.values())} without a product)", flush=True)

    blocks = (cells[:, 0] // 6) * 10000 + cells[:, 1] // 6
    ub = np.unique(blocks)
    rng = np.random.default_rng(58)
    draws = [None] + [rng.choice(ub, size=len(ub), replace=True) for _ in range(a.n_boot)]

    def stats(weights, cell_mask=None, tier=None):
        res = {}
        for centred in (True, False):
            sfx = "c" if centred else "u"
            acc = {"ebird": [], "regional": [], "ebird_beyond_regional": [], "level": []}
            accm = {nm: [] for nm, _, _ in SHIFTS}
            for s in per:
                if tier is not None and not (tier[0] <= s["prev"] < tier[1]):
                    continue
                m = s["ok"] if cell_mask is None else (s["ok"] & cell_mask)
                w = weights[m]
                if w.sum() < 2:
                    continue
                rep = lambda v: np.repeat(v[m], w)
                x, a_, b_, r, lv, ab = (rep(s["x"]), rep(s["cA"]), rep(s["cB"]), rep(s["reg"]), rep(s["lvl"]),
                                        rep(s["abd"]))
                if centred:
                    x, a_, b_, r, lv, ab = (v - v.mean() for v in (x, a_, b_, r, lv, ab))
                acc["ebird"].append((x, a_, b_))
                acc["regional"].append((r, a_, b_))
                beta = float(np.sum(x * r) / np.sum(r * r)) if np.sum(r * r) > 0 else 0.0
                acc["ebird_beyond_regional"].append((x - beta * r, a_, b_))
                # level: eBird abd vs BBS late level; one measurement each, so a plain corr (not noise-free)
                acc["level"].append((ab, lv, lv))
                for nm in accm:
                    mv = rep(s["moved"][nm])
                    f = np.isfinite(mv)
                    if f.sum() < 2:
                        continue
                    mv2, a2, b2 = mv[f], a_[f], b_[f]
                    if centred:
                        mv2, a2, b2 = mv2 - mv2.mean(), a2 - a2.mean(), b2 - b2.mean()
                    accm[nm].append((mv2, a2, b2))
            for k2, v in acc.items():
                if k2 == "level":
                    num = sum(float(np.sum(x * y)) for x, y, _ in v)
                    den = np.sqrt(sum(float(np.sum(x * x)) for x, _, _ in v) *
                                  sum(float(np.sum(y * y)) for _, y, _ in v)) if v else np.nan
                    res[f"{k2}_{sfx}"] = (num / den if v and den > 0 else np.nan, np.nan)
                else:
                    res[f"{k2}_{sfx}"] = _pooled(v)
            res[f"null_{sfx}"] = {nm: _pooled(v)[0] for nm, v in accm.items()}
            res[f"n_species_{sfx}"] = len(acc["ebird"])
        return res

    def boot(cell_mask=None, tier=None):
        out = []
        for d in draws:
            if d is None:
                w = np.ones(n_cells, dtype=int)
            else:
                cnt = dict(zip(*np.unique(d, return_counts=True)))
                w = np.array([cnt.get(b, 0) for b in blocks], dtype=int)
            out.append(stats(w, cell_mask, tier))
        pt, bs = out[0], out[1:]
        summ = {}
        for k2, v in pt.items():
            if isinstance(v, tuple):
                vals = np.array([b[k2][0] for b in bs], dtype="float64")
                vals = vals[np.isfinite(vals)]
                ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) > 10 else None
                summ[k2] = {"corr": float(v[0]), "k": (None if not np.isfinite(v[1]) else float(v[1])), "ci": ci}
            else:
                summ[k2] = v
        for sfx in ("c", "u"):
            nulls = [x for x in summ[f"null_{sfx}"].values() if np.isfinite(x)]
            summ[f"null_{sfx}_max"] = float(np.max(nulls)) if nulls else None
            summ[f"null_{sfx}_median"] = float(np.median(nulls)) if nulls else None
        return summ

    report = {"cells": int(n_cells), "gap_years": gap, "early": a.early, "late": a.late,
              "species_compared": [s["code"] for s in per], "skipped": skipped,
              "trend_windows": sorted({f"{s['years'][0]}-{s['years'][1]}" for s in per}),
              "observer_classes": {k: int(v) for k, v in zip(*np.unique(kind, return_counts=True))}}
    report["all"] = boot()
    for k in ("unchanged", "partial", "replaced"):
        report[f"obs_{k}"] = boot(cell_mask=(kind == k))
    for lo, hi in TIERS:
        report[f"prev_{lo:g}-{min(hi, 1):g}"] = boot(tier=(lo, hi))
    # species level: each species' mean BBS change (noise-free cross product needs halves) vs mean eBird change
    mx = np.array([np.mean(s["x"][s["ok"]]) for s in per])
    mA = np.array([np.mean(s["cA"][s["ok"]]) for s in per])
    mB = np.array([np.mean(s["cB"][s["ok"]]) for s in per])
    xc, Ac, Bc = mx - mx.mean(), mA - mA.mean(), mB - mB.mean()
    report["species_level"] = {"corr_nf": float(np.sum(xc * 0.5 * (Ac + Bc)) / np.sqrt(np.sum(xc ** 2) *
                                                                                     max(np.sum(Ac * Bc), 1e-12))),
                               "n_species": len(per)}
    json.dump(report, open(os.path.join(a.out, "ebird_change_agree.json"), "w"), indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "ebird_change_agree.json")))
    print(f"[E058] cells {r['cells']}, gap {r['gap_years']:.1f} y, BBS {r['early']} vs {r['late']}, eBird windows "
          f"{r['trend_windows']}, species {len(r['species_compared'])}; observers {r['observer_classes']}")
    fmt = lambda d: (f"{d['corr']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]" if d.get("ci") else f"{d['corr']:+.3f}")
    for key in [k for k in r if k == "all" or k.startswith("obs_") or k.startswith("prev_")]:
        s = r[key]
        for sfx, lab in (("c", "centred"), ("u", "uncentred")):
            print(f"  {key:16s} {lab:9s} n={s[f'n_species_{sfx}']:3d} | eBird {fmt(s[f'ebird_{sfx}'])} "
                  f"k {s[f'ebird_{sfx}']['k'] if s[f'ebird_{sfx}']['k'] is None else round(s[f'ebird_{sfx}']['k'], 3)} "
                  f"| regional BBS {fmt(s[f'regional_{sfx}'])} | eBird beyond regional "
                  f"{fmt(s[f'ebird_beyond_regional_{sfx}'])} | moved-eBird null median "
                  f"{s[f'null_{sfx}_median']} max {s[f'null_{sfx}_max']} | level {s[f'level_{sfx}']['corr']:+.3f}")
    print(f"  species-level mean change corr_nf {r['species_level']['corr_nf']:+.3f} "
          f"(n {r['species_level']['n_species']})")


if __name__ == "__main__":
    main()
