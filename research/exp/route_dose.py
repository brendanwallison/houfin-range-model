"""E053: does land-cover change measured ALONG THE SURVEY ROUTE predict route-level bird change better than the same
land cover averaged over the route's 27 km cell, beyond all of DESK's covariates?

    python research/exp/route_dose.py --cache <tempho research cache (split, type basis)> --routes <bbsrtsl020.shp>
        --bbs <bbs release dir> --nlcd 1990=<tif> --nlcd 2019=<tif> --nlcd-cells <nlcd_27km.npz>
        --avonet <AVONET1_BirdLife.csv> --taxonomy <eBird_taxonomy.csv> --out <dir>
    python research/exp/route_dose.py --summarize <dir>

Routes (US, RPID 101, RunType 1) with >= 2 valid runs in each epoch (1986-95, 2014-23). Per route and species (the
369 non-sealed community + dev species; AOU -> eBird code by scientific name, src.data.identify.bbs_crosswalk; lumps
summed): log1p SpeciesTotal, epoch means and ABBA halves (alternating years). Covariates per route:
    R  NLCD class fractions in a 400 m buffer around the route line, 1990 and 2019 (change)
    C  the same classes for the route's 27 km cell (E052's aggregation)
    A  all DESK covariate channels at the route's cell (yearly states, desk_meta normalization), epoch means (change)
TYPE = 32 PCs of the cell's modern-epoch mean covariates (fitted on training routes). Training / held-out routes from
the cache's split at the route's cell (buffer cells dropped). Guild-pooled ridge (as guild_dose --pooled): arms A,
A+R, A+C and A + moved R (8 draws; each route takes the R-change of the route nearest its shifted position). Within
type and kernel-local (LOO 150 km within each set); paired 6x6-cell-block bootstrap. Secondary: held-out routes whose
main observer (most frequent ObsN) is the same in both epochs. Downloaded files are read only.
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
from nlcd_aggregate import CLASSES  # noqa: E402
from type_dose import loo_local_op, ridge_cv  # noqa: E402

GUILDS = ("Grassland", "Forest", "Woodland", "Shrubland", "Wetland", "Human Modified")
EARLY, MODERN = (1986, 1995), (2014, 2023)


def route_nlcd(routes_nlcd_crs, path, buffer_m=400.0):
    """(n_routes, K) class fractions of each route's buffer."""
    import rasterio
    from rasterio.mask import mask as rmask
    lut = np.full(256, len(CLASSES), np.int64)
    for k, (_, codes) in enumerate(CLASSES):
        lut[list(codes)] = k
    out = np.full((len(routes_nlcd_crs), len(CLASSES)), np.nan)
    with rasterio.open(path) as s:
        for i, geom in enumerate(routes_nlcd_crs.geometry.buffer(buffer_m)):
            try:
                arr, _ = rmask(s, [geom], crop=True, filled=True, nodata=250)
            except ValueError:
                continue
            c = np.bincount(lut[arr[0].ravel()], minlength=len(CLASSES) + 1)
            valid = c[:-1].sum()
            if valid > 0:
                out[i] = c[:-1] / valid
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--routes")
    ap.add_argument("--bbs")
    ap.add_argument("--nlcd", action="append")
    ap.add_argument("--nlcd-cells")
    ap.add_argument("--avonet")
    ap.add_argument("--taxonomy")
    ap.add_argument("--out")
    ap.add_argument("--n-type", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--summarize", default=None)
    ap.add_argument("--cv", type=int, default=0, help="E053b: K-fold spatial-block CV over all routes")
    ap.add_argument("--rival-pcs", type=int, default=0,
                    help="E053c: rival block A = this many leading PCs of covariate change (unsupervised, all routes)")
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    import geopandas as gpd
    import pandas as pd
    import rasterio
    from src.config_utils import load_config
    from src.community_encoder.train_DESK import covariate_io as cio
    from src.data.identify.bbs_crosswalk import _read_species_table, crosswalk, load_ebird_taxonomy, normalize_bbs_species
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    meta = json.load(open(c("meta.json"), encoding="utf-8"))
    split = np.load(c("split.npz"))
    codes = list(meta["community"]) + list(meta["dev_species"])
    # species crosswalk (AOU -> code) and habitats
    xw, diag = crosswalk(normalize_bbs_species(_read_species_table(os.path.join(a.bbs, "SpeciesList.csv"))),
                         load_ebird_taxonomy(a.taxonomy), codes)
    aou2s = {}
    code_ix = {cd: i for i, cd in enumerate(codes)}
    for aou, cd in zip(xw["aou"], xw["species_code"]):
        aou2s.setdefault(int(aou), code_ix[cd])
    tax = pd.read_csv(a.taxonomy).set_index("SPECIES_CODE")["SCIENTIFIC_NAME"]
    av = pd.read_csv(a.avonet, encoding="latin-1").set_index("Species1")["Habitat"]
    hab = np.array([av.get(tax.get(cd), None) for cd in codes], dtype=object)
    # valid runs
    w = pd.read_csv(os.path.join(a.bbs, "Weather.csv"), dtype={"StateNum": str, "Route": str})
    w = w[(w["CountryNum"] == 840) & (w["RPID"] == 101) & (w["RunType"] == 1)]
    w["RTENO"] = w["StateNum"].astype(int) * 1000 + w["Route"].astype(int)
    inep = lambda y, e: (y >= e[0]) & (y <= e[1])
    w = w[inep(w["Year"], EARLY) | inep(w["Year"], MODERN)]
    ne = w[inep(w["Year"], EARLY)].groupby("RTENO")["Year"].nunique()
    nm = w[inep(w["Year"], MODERN)].groupby("RTENO")["Year"].nunique()
    ok_rt = sorted(set(ne[ne >= 2].index) & set(nm[nm >= 2].index))
    # route lines
    g = gpd.read_file(a.routes)
    g["RTENO"] = g["RTENO"].astype(int)
    g = g[g["RTENO"].isin(ok_rt)].dissolve(by="RTENO").reset_index()
    grid = rasterio.open(os.path.join(os.environ["HOUFIN_DATA"], "ref_grid_27km.tif"))
    gg = g.to_crs(grid.crs)
    cx, cy = gg.geometry.centroid.x.values, gg.geometry.centroid.y.values
    col, row = ~grid.transform * (cx, cy)
    row, col = np.floor(row).astype(int), np.floor(col).astype(int)
    rts = g["RTENO"].astype(int).values
    held = split["holdout"][row, col].astype(bool)
    buf = split["buffer"][row, col].astype(bool)
    ev, tc = np.where(held)[0], np.where(~held & ~buf)[0]
    n = len(rts)
    # route x year species counts
    wv = w[w["RTENO"].isin(rts)]
    rid2 = dict(zip(wv["RouteDataID"], zip(wv["RTENO"], wv["Year"])))
    obs = wv.groupby(["RTENO", "Year"])["ObsN"].first()
    years_of = {r: sorted(set(wv[wv["RTENO"] == r]["Year"])) for r in rts}
    ry = [(r, y) for r in rts for y in years_of[r]]
    ry_ix = {k: i for i, k in enumerate(ry)}
    X = np.zeros((len(ry), len(codes)), "float64")
    for f in sorted(glob.glob(os.path.join(a.bbs, "States", "*.csv"))):
        d = pd.read_csv(f, usecols=["RouteDataID", "AOU", "SpeciesTotal"])
        d = d[d["RouteDataID"].isin(rid2.keys())]
        for rid, aou, tot in zip(d["RouteDataID"], d["AOU"], d["SpeciesTotal"]):
            s_ = aou2s.get(int(aou))
            if s_ is not None:
                X[ry_ix[rid2[rid]], s_] += float(tot)
    L = np.log1p(X)
    rix = {r: i for i, r in enumerate(rts)}
    ry_r = np.array([rix[r] for r, _ in ry])
    ry_y = np.array([y for _, y in ry])

    def epoch_stats(M):
        full, ha, hb = np.zeros((n, M.shape[1])), np.zeros((n, M.shape[1])), np.zeros((n, M.shape[1]))
        for e_ix, ep in ((0, EARLY), (1, MODERN)):
            sgn = -1.0 if e_ix == 0 else 1.0
            for i in range(n):
                rows = np.where((ry_r == i) & inep(ry_y, ep))[0]
                rows = rows[np.argsort(ry_y[rows])]
                full[i] += sgn * M[rows].mean(0)
                ha[i] += sgn * M[rows[0::2]].mean(0)
                hb[i] += sgn * M[rows[1::2]].mean(0)
        return full, ha, hb
    dS, dA_, dB_ = epoch_stats(L)
    pres = np.zeros((n, len(codes)), bool)
    for i in range(n):
        pres[i] = (X[ry_r == i] > 0).any(0)
    main_obs = {}
    for i, r in enumerate(rts):
        oe = [obs.get((r, y)) for y in years_of[r] if inep(y, EARLY)]
        om = [obs.get((r, y)) for y in years_of[r] if inep(y, MODERN)]
        mo = lambda v: max(set(v), key=v.count) if v else None
        main_obs[i] = (mo(oe), mo(om))
    same_obs = np.array([main_obs[i][0] is not None and main_obs[i][0] == main_obs[i][1] for i in range(n)])
    # covariates
    nl = {}
    for t in a.nlcd:
        y, p = t.split("=", 1)
        with rasterio.open(p) as s_:
            crs = s_.crs
        nl[int(y)] = route_nlcd(g.to_crs(crs), p)
    yrs = sorted(nl)
    dR = np.nan_to_num(nl[yrs[1]] - nl[yrs[0]])
    zc = np.load(a.nlcd_cells)
    cy_ = list(zc["years"])
    dC = (zc["nlcd"][cy_.index(yrs[1])][:, row, col] - zc["nlcd"][cy_.index(yrs[0])][:, row, col]).T.astype("float64")
    cfg = load_config()
    states_dir = os.path.join(cfg["paths"]["hist_dir"], "yearly_states")
    dm = np.load(os.path.join(meta["run_dir"], "desk_meta.npz"), allow_pickle=True)
    schema = json.loads(str(dm["schema"]))
    mu, sd = dm["mu"].astype("float32"), dm["sd"].astype("float32")
    acc = {}
    for ep in (EARLY, MODERN):
        tot = None
        for y in range(ep[0], ep[1] + 1):
            covn, _ = cio.norm_grid(cio.load_state_stack(y, states_dir, schema), mu, sd)
            v = covn[row, col].astype("float64")
            tot = v if tot is None else tot + v
        acc[ep] = tot / (ep[1] - ep[0] + 1)
    dA = acc[MODERN] - acc[EARLY]
    keep = dA[tc].std(0) > 1e-9
    dA = dA[:, keep]
    if a.rival_pcs:
        # E053c: ~300 raw channels made the pooled baseline unstable (wetland A = -0.50 in E053b); its leading PCs
        # carry the same change at far fewer degrees of freedom. Unsupervised (no bird data), so fitting on all
        # routes leaks nothing about the outcome.
        dAs = (dA - dA.mean(0)) / dA.std(0)
        _, _, Va = np.linalg.svd(dAs, full_matrices=False)
        dA = dAs @ Va[:a.rival_pcs].T
    levels = acc[MODERN]
    _, _, Vt = np.linalg.svd(levels[tc] - levels[tc].mean(0), full_matrices=False)
    Tm = np.hstack([np.ones((n, 1)), levels @ Vt[:a.n_type].T])
    xy = np.column_stack([row, col]).astype(float)
    lo, hi = xy.min(0), xy.max(0)

    def moved(V, dr, dc, flip):
        t = xy.copy()
        if flip:
            t[:, 1] = xy[:, 1].max() + xy[:, 1].min() - t[:, 1]
        t = lo + np.mod(t + np.array([dr, dc]) - lo, hi - lo + 1)
        return V[((t[:, None, :] - xy[None, :, :]) ** 2).sum(-1).argmin(1)]
    feats = {"A": dA, "A+R": np.hstack([dA, dR]), "A+C": np.hstack([dA, dC])}
    for k, (dr, dc, fl) in enumerate([(0, 40, False), (0, -40, False), (20, 0, False), (-20, 0, False),
                                      (0, 0, True), (15, 30, False), (-15, -30, False), (10, 0, True)]):
        feats[f"A+moved{k}"] = np.hstack([dA, moved(dR, dr, dc, fl)])
    arms = list(feats)
    blk = (row // 6) * 100000 + col // 6
    lams = [10.0 ** e for e in np.arange(-3, 3.01, 0.5)]
    if a.cv:
        # E053b: every route out-of-fold; folds = random assignment of 6x6-cell blocks
        ublk = np.unique(blk)
        fold_of = dict(zip(ublk.tolist(), np.random.default_rng(5).integers(a.cv, size=len(ublk)).tolist()))
        fold = np.array([fold_of[b] for b in blk])
        folds = [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(a.cv)]
        eval_all, min_ev_fold = np.arange(n), 3
    else:
        folds, eval_all, min_ev_fold = [(tc, ev)], ev, 30
    posmap = np.full(n, -1)
    posmap[eval_all] = np.arange(len(eval_all))
    ub, binv = np.unique(blk[eval_all], return_inverse=True)
    rng = np.random.default_rng(11)
    Wall = np.vstack([np.ones(len(eval_all)), rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)),
                                                              size=a.n_boot)[:, binv]]).astype("float64")
    res = {"n_routes": n, "n_train": int(len(tc)), "n_heldout": int(len(ev)), "cv": a.cv, "rival_pcs": a.rival_pcs,
           "n_evaluated": int(len(eval_all)), "n_evaluated_same_observer": int(same_obs[eval_all].sum()),
           "n_heldout_same_observer": int(same_obs[ev].sum()),
           "crosswalk": {k: v for k, v in diag.items() if k != "unmatched_codes"},
           "epochs": [EARLY, MODERN], "nlcd_years": yrs, "modes": {}}
    for mode in ("type", "type+kloc"):
        out = {}
        for gname in GUILDS:
            spp = [s for s in range(len(codes)) if hab[s] == gname]
            per_s = {}
            for tcf, evf in folds:
                store = []
                for s in spp:
                    tcs, evs = tcf[pres[tcf, s]], evf[pres[evf, s]]
                    if len(tcs) < 60 or len(evs) < min_ev_fold:
                        continue
                    sc = dS[tcs, s].std()
                    if sc <= 0:
                        continue
                    gT, *_ = np.linalg.lstsq(Tm[tcs], dS[tcs, s][:, None], rcond=None)
                    ytr = (dS[tcs, s] - (Tm[tcs] @ gT)[:, 0]) / sc
                    ya = (dA_[evs, s] - (Tm[evs] @ gT)[:, 0]) / sc
                    yb = (dB_[evs, s] - (Tm[evs] @ gT)[:, 0]) / sc
                    if mode == "type+kloc":
                        ktr, kev = loo_local_op(xy[tcs], 150.0), loo_local_op(xy[evs], 150.0)
                        ytr, ya, yb = ktr(ytr[:, None])[:, 0], kev(ya[:, None])[:, 0], kev(yb[:, None])[:, 0]
                    rec = {"s": s, "ytr": ytr, "blk": blk[tcs], "evs": evs, "ya": ya, "yb": yb, "F": {}}
                    for arm in arms:
                        Fx = feats[arm]
                        gF, *_ = np.linalg.lstsq(Tm[tcs], Fx[tcs], rcond=None)
                        Ftr, Fev = Fx[tcs] - Tm[tcs] @ gF, Fx[evs] - Tm[evs] @ gF
                        if mode == "type+kloc":
                            Ftr, Fev = ktr(Ftr), kev(Fev)
                        rec["F"][arm] = (Ftr, Fev)
                    store.append(rec)
                if not store:
                    continue
                for r_ in store:
                    d = per_s.setdefault(r_["s"], {"evs": [], "ya": [], "yb": [], "P": {arm: [] for arm in arms}})
                    d["evs"].append(r_["evs"])
                    d["ya"].append(r_["ya"])
                    d["yb"].append(r_["yb"])
                for arm in arms:
                    Ftr_all = np.vstack([r_["F"][arm][0] for r_ in store])
                    sdv = Ftr_all.std(0)
                    k_ = sdv > 1e-9
                    B, xm, ym, _ = ridge_cv(Ftr_all[:, k_] / sdv[k_], np.concatenate([r_["ytr"] for r_ in store])[:, None],
                                            np.concatenate([r_["blk"] for r_ in store]), lams)
                    for r_ in store:
                        per_s[r_["s"]]["P"][arm].append(((r_["F"][arm][1][:, k_] / sdv[k_] - xm) @ B + ym)[:, 0])
            tot = {arm: np.zeros((3, Wall.shape[0])) for arm in arms}
            tot_s = {arm: np.zeros((3, Wall.shape[0])) for arm in arms}
            n_used = 0
            for s_, d in per_s.items():
                evs = np.concatenate(d["evs"])
                if len(evs) < 30:
                    continue
                ww = Wall[:, posmap[evs]]
                ya, yb = np.concatenate(d["ya"]), np.concatenate(d["yb"])
                same = same_obs[evs]
                ok_s = False
                for msk, accd in ((np.ones(len(evs), bool), tot), (same, tot_s)):
                    if msk.sum() < 5:
                        continue
                    wm = ww * msk[None, :]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        ctr = lambda M: M - ((wm * M).sum(1) / wm.sum(1))[:, None]
                        cah = ctr(np.broadcast_to(ya[None, :], wm.shape))
                        cbh = ctr(np.broadcast_to(yb[None, :], wm.shape))
                        vth = (wm * cah * cbh).sum(1)
                        if accd is tot:
                            if vth[0] <= 0:
                                break
                            ok_s = True
                        for arm in arms:
                            p = np.concatenate(d["P"][arm])
                            cp = ctr(np.broadcast_to(p[None, :], wm.shape))
                            accd[arm][0] += 0.5 * ((wm * cp * cah).sum(1) + (wm * cp * cbh).sum(1))
                            accd[arm][1] += (wm * cp * cp).sum(1)
                            accd[arm][2] += vth
                n_used += int(ok_s)
            if n_used == 0:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                boots = {arm: tot[arm][0] / np.sqrt(tot[arm][1] * tot[arm][2]) for arm in arms}
                boots_same = {arm: tot_s[arm][0] / np.sqrt(np.maximum(tot_s[arm][1] * tot_s[arm][2], 1e-300))
                              for arm in arms}
            ci = lambda v: [float(np.nanpercentile(v[1:], q)) for q in (2.5, 97.5)]
            o = {"n_species": n_used}
            for arm in ("A", "A+R", "A+C"):
                o[arm] = {"corr": float(boots[arm][0]), "ci": ci(boots[arm])}
            gR, gC = boots["A+R"] - boots["A"], boots["A+C"] - boots["A"]
            nulls = np.array([boots[k] - boots["A"] for k in arms if k.startswith("A+moved")])
            o["gain_route"] = {"diff": float(gR[0]), "ci": ci(gR)}
            o["gain_cell"] = {"diff": float(gC[0]), "ci": ci(gC)}
            o["route_minus_cell"] = {"diff": float((gR - gC)[0]), "ci": ci(gR - gC)}
            o["null_gains"] = [float(v) for v in nulls[:, 0]]
            o["route_net_of_null"] = {"diff": float((gR - np.median(nulls, 0))[0]), "ci": ci(gR - np.median(nulls, 0))}
            gRs, gCs = boots_same["A+R"] - boots_same["A"], boots_same["A+C"] - boots_same["A"]
            o["same_observer"] = {"gain_route": {"diff": float(gRs[0]), "ci": ci(gRs)},
                                  "gain_cell": {"diff": float(gCs[0]), "ci": ci(gCs)},
                                  "route_minus_cell": {"diff": float((gRs - gCs)[0]), "ci": ci(gRs - gCs)}}
            out[gname] = o
        res["modes"][mode] = out
    with open(os.path.join(a.out, "route_dose.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "route_dose.json"), encoding="utf-8"))
    print(f"routes {r['n_routes']} (train {r['n_train']}, held-out {r['n_heldout']}); cv {r.get('cv')}: evaluated "
          f"{r.get('n_evaluated', r['n_heldout'])}, rival PCs {r.get('rival_pcs', 0)} (same observer {r.get('n_evaluated_same_observer', r['n_heldout_same_observer'])}); "
          f"crosswalk {r['crosswalk']}; NLCD {r['nlcd_years']}")
    f_ = lambda d: f"{d['diff']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]"
    for mode, o in r["modes"].items():
        print(f"  [{mode}]")
        for gname, v in o.items():
            nul = v["null_gains"]
            print(f"    {gname:14s} ({v['n_species']:3d} spp) A {v['A']['corr']:+.3f} | +route {f_(v['gain_route'])} "
                  f"(nulls median {np.median(nul):+.3f} max {max(nul):+.3f}) | +cell {f_(v['gain_cell'])} | "
                  f"route-cell {f_(v['route_minus_cell'])} || same-observer route-cell "
                  f"{f_(v['same_observer']['route_minus_cell'])}")


if __name__ == "__main__":
    main()
