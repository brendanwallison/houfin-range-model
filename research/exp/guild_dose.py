"""E049: does crop-type change act WHERE IT SHOULD -- on grassland/open-country birds rather than forest birds?

    python research/exp/guild_dose.py --cache <tempho research cache> --cropat <cropat_27km.npz with types>
        --avonet <AVONET1_BirdLife.csv> --taxonomy <eBird_taxonomy.csv> --out <dir>
    python research/exp/guild_dose.py --summarize <dir>

Species: the 96 community + 273 dev species (never the sealed half), each assigned its AVONET habitat class (BirdLife
taxonomy, matched on the eBird scientific name). Per species and cell: abundance change = epoch mean of log1p(count),
modern (2005-25) minus early withheld epoch (1966-86), over the cell's surveyed rows (ABBA halves for the truth).
Only cells where the species was recorded in either epoch count (its range). Each species' change is scaled by its
training-cell sd.
Predictors (cell-level change, CONUS cells): CropAT density + 10 crop-type fractions (11), LUH's 5 crop classes, all 33
LUH land-use channels, LUH 33 + CropAT 11. All residualized WITHIN TYPE (32 PCs of the cell's mean covariate levels,
fitted on training cells), optionally then KERNEL-LOCAL (LOO 150 km within each set). Per species, a ridge map
(lambda by 6x6-block CV) fitted on its training cells, graded on its held-out cells. Guild score = pooled noise-free
corr over the guild's species (sum of cross-half covariances / sqrt(sum pred var x sum truth noise-free var)), species
with resolvable change only; paired 6x6-block bootstrap (one draw shared by all species).
E049b (--pooled): ONE map per guild and arm, shared by the guild's species (rows = training cell x species pairs,
features standardized over the pooled rows, lambda by 6x6-block CV), applied to each species' held-out cells -- the
"the guild responds alike to what is grown" estimator, far more data per coefficient than per-species maps.
E049c (--pooled --rival-all, the E047b skeptic's correction): the rival is ALL covariate change DESK sees (every
standardized F_cov_raw channel), and the null is CropAT's own change field moved in space (shifted / mirrored, each
cell taking the field of the CONUS cell nearest its moved position -- same smoothness, misaligned), 8 draws. Gain =
(all + CropAT) - all; reported against the null gains, and in the west / east halves of the held-out cells.
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
from type_dose import loo_local_op, ridge_cv  # noqa: E402

GUILDS = ("Grassland", "Forest", "Woodland", "Shrubland", "Wetland", "Human Modified")
ARMS = ("cropat11", "luh_crop5", "luh33", "luh33+cropat11")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--cropat")
    ap.add_argument("--avonet")
    ap.add_argument("--taxonomy")
    ap.add_argument("--out")
    ap.add_argument("--n-type", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--summarize", default=None)
    ap.add_argument("--pooled", action="store_true", help="E049b: one shared map per guild and arm")
    ap.add_argument("--rival-all", action="store_true", help="E049c: all-covariate rival + moved-CropAT null")
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    import pandas as pd
    from scipy import sparse
    from change_oracle import group_rows
    from src.config_utils import load_config
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    N = len(keys)
    split = np.load(c("split.npz"))
    withheld = [int(y) for y in split["withheld"]]
    meta = json.load(open(c("meta.json"), encoding="utf-8"))
    E, full, half_a, half_b = group_rows(a.cache, keys, withheld)
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
    cfg = load_config()
    states_dir = os.path.join(cfg["paths"]["hist_dir"], "yearly_states")
    bui_avail = np.load(os.path.join(states_dir, "state_2015.npz"))["bui"][..., 0]
    conus = np.nan_to_num(bui_avail[cells[:, 0], cells[:, 1]]) > 0.5
    ev, tc = np.where(held & conus)[0], np.where(~held & ~buf & conus)[0]

    def gmat(groups):
        rows = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
        cols = np.concatenate([np.asarray(g, int) for g in groups])
        vals = np.concatenate([np.full(len(g), 1.0 / len(g)) for g in groups])
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(groups), N))
    G = {h: (gmat([np.asarray(gs[p]) for p in pairs[:, 0]]), gmat([np.asarray(gs[p]) for p in pairs[:, 1]]))
         for h, gs in (("full", full), ("a", half_a), ("b", half_b))}
    chg = lambda M, h: np.asarray(G[h][1] @ M) - np.asarray(G[h][0] @ M)
    # species and habitats
    codes = list(meta["community"]) + list(meta["dev_species"])
    Xs = np.hstack([np.load(c("X_comm.npy"), mmap_mode="r"), np.load(c("X_dev.npy"), mmap_mode="r")]).astype("float32")
    assert Xs.shape[1] == len(codes)
    tax = pd.read_csv(a.taxonomy).set_index("SPECIES_CODE")["SCIENTIFIC_NAME"]
    av = pd.read_csv(a.avonet, encoding="latin-1").set_index("Species1")["Habitat"]
    hab = np.array([av.get(tax.get(cd), None) for cd in codes], dtype=object)
    L = np.log1p(np.nan_to_num(Xs, nan=0.0))
    dS = {h: chg(L, h) for h in ("full", "a", "b")}
    P0 = (Xs > 0).astype("float32")
    pres = (np.asarray(G["full"][0] @ P0) + np.asarray(G["full"][1] @ P0)) > 0          # in range either epoch
    # predictors
    yr = keys[:, 2].astype(int)
    ca = np.load(a.cropat)
    cy = ca["years"].astype(int)
    ymap = {int(y): i for i, y in enumerate(cy)}
    yi = np.array([ymap[min(int(y), int(cy.max()))] for y in yr])
    Ck = np.concatenate([ca["crop"][yi, keys[:, 0], keys[:, 1]][:, None],
                         ca["types"][yi, :, keys[:, 0], keys[:, 1]]], axis=1).astype("float64")
    Ck = np.nan_to_num(Ck, nan=0.0)
    Lall = np.zeros((N, 33))
    for y in np.unique(yr):
        z = np.load(os.path.join(states_dir, f"state_{int(y)}.npz"))
        s_ = yr == y
        Lall[s_] = z["landuse"][keys[s_, 0], keys[s_, 1], :33]
    Lall = np.nan_to_num(Lall, nan=0.0)
    dC, dL = chg(Ck, "full"), chg(Lall, "full")
    feats = {"cropat11": dC, "luh_crop5": dL[:, :5], "luh33": dL, "luh33+cropat11": np.hstack([dL, dC])}
    # type basis
    F = np.asarray(np.load(c("F_cov_raw.npy"), mmap_mode="r"), "float64")
    ho = split["holdout"][keys[:, 0], keys[:, 1]]
    bfk = split["buffer"][keys[:, 0], keys[:, 1]]
    ty = ~np.isin(keys[:, 2], withheld)
    trn = np.where(~ho & ~bfk & ty & np.isfinite(F).all(1))[0]
    mu, sd = F[trn].mean(0), F[trn].std(0)
    keep = sd > 1e-6
    X = np.where(np.isfinite(F[:, keep]), (F[:, keep] - mu[keep]) / sd[keep], 0.0)
    _, _, Vt = np.linalg.svd(X[trn][:: max(1, len(trn) // 40000)] - X[trn].mean(0), full_matrices=False)
    Tm = np.hstack([np.ones((len(pairs), 1)), np.asarray(G["full"][1] @ X) @ Vt[:a.n_type].T])
    if a.rival_all:
        dXall = chg(X, "full")
        cidx = np.where(conus)[0]
        xy = cells[cidx].astype(float)
        lo, hi = xy.min(0), xy.max(0)

        def moved(V, dr, dc, flip):
            t = xy.copy()
            if flip:
                t[:, 1] = xy[:, 1].max() + xy[:, 1].min() - t[:, 1]
            t = lo + np.mod(t + np.array([dr, dc]) - lo, hi - lo + 1)
            src = cidx[((t[:, None, :] - xy[None, :, :]) ** 2).sum(-1).argmin(1)]
            out = V.copy()
            out[cidx] = V[src]
            return out
        feats = {"all": dXall, "all+cropat11": np.hstack([dXall, dC])}
        for k, (dr, dc, fl) in enumerate([(0, 40, False), (0, -40, False), (20, 0, False), (-20, 0, False),
                                          (0, 0, True), (15, 30, False), (-15, -30, False), (10, 0, True)]):
            feats[f"all+moved{k}"] = np.hstack([dXall, moved(dC, dr, dc, fl)])
    arm_list = list(feats)
    west = cells[:, 1] < np.median(cells[ev, 1])
    blk_all = (cells[:, 0] // 6) * 100000 + cells[:, 1] // 6
    lams = [10.0 ** e for e in np.arange(-3, 3.01, 0.5)]
    ub, binv = np.unique(blk_all[ev], return_inverse=True)
    rng = np.random.default_rng(11)
    Wev = np.vstack([np.ones(len(ev)), rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)),
                                                       size=a.n_boot)[:, binv]]).astype("float64")
    res = {"cache": a.cache, "n_species": len(codes), "habitat_counts": {}, "modes": {}}
    for g in GUILDS:
        res["habitat_counts"][g] = int((hab == g).sum())
    for mode in ("type", "type+kloc"):
        acc = {g: {arm: [] for arm in arm_list} for g in GUILDS}       # per species: (cv, vp, vt) arrays over draws
        accw = {h: {g: {arm: [] for arm in arm_list} for g in GUILDS} for h in ("west", "east")}
        n_used = {g: 0 for g in GUILDS}
        store = {g: [] for g in GUILDS}
        for s in range(len(codes)):
            g = hab[s]
            if g not in GUILDS:
                continue
            tcs, evs = tc[pres[tc, s]], ev[pres[ev, s]]
            if len(tcs) < 60 or len(evs) < 30:
                continue
            sc = dS["full"][tcs, s].std()
            if sc <= 0:
                continue
            Tg = Tm[np.r_[tcs, evs]]
            gT, *_ = np.linalg.lstsq(Tm[tcs], np.c_[dS["full"][tcs, s]], rcond=None)
            ytr = (dS["full"][tcs, s] - (Tm[tcs] @ gT)[:, 0]) / sc
            ya = (dS["a"][evs, s] - (Tm[evs] @ gT)[:, 0]) / sc
            yb = (dS["b"][evs, s] - (Tm[evs] @ gT)[:, 0]) / sc
            if mode == "type+kloc":
                ktr, kev = loo_local_op(cells[tcs], 150.0), loo_local_op(cells[evs], 150.0)
                ytr, ya, yb = ktr(ytr[:, None])[:, 0], kev(ya[:, None])[:, 0], kev(yb[:, None])[:, 0]
            w = Wev[:, np.searchsorted(ev, evs)]
            wa = lambda v: (w * v[None, :]).sum(1) / w.sum(1)
            ca_, cb_ = ya - wa(ya)[:, None], yb - wa(yb)[:, None]
            vt = (w * ca_ * cb_).sum(1)
            if vt[0] <= 0:
                continue
            n_used[g] += 1
            if a.pooled:
                rec = {"ytr": ytr, "blk": blk_all[tcs], "w": w, "ca": ca_, "cb": cb_, "vt": vt, "F": {},
                       "west": west[evs]}
                for arm in arm_list:
                    Fx = feats[arm]
                    gF, *_ = np.linalg.lstsq(Tm[tcs], Fx[tcs], rcond=None)
                    Ftr, Fev = Fx[tcs] - Tm[tcs] @ gF, Fx[evs] - Tm[evs] @ gF
                    if mode == "type+kloc":
                        Ftr, Fev = ktr(Ftr), kev(Fev)
                    rec["F"][arm] = (Ftr, Fev)
                store[g].append(rec)
                continue
            for arm in arm_list:
                Fx = feats[arm]
                gF, *_ = np.linalg.lstsq(Tm[tcs], Fx[tcs], rcond=None)
                Ftr, Fev = Fx[tcs] - Tm[tcs] @ gF, Fx[evs] - Tm[evs] @ gF
                if mode == "type+kloc":
                    Ftr, Fev = ktr(Ftr), kev(Fev)
                sdv = Ftr.std(0)
                k_ = sdv > 1e-9
                B, xm, ym, _ = ridge_cv(Ftr[:, k_] / sdv[k_], ytr[:, None], blk_all[tcs], lams)
                p = ((Fev[:, k_] / sdv[k_] - xm) @ B + ym)[:, 0]
                cp = p[None, :] - wa(p)[:, None]
                acc[g][arm].append(((0.5 * ((w * cp * ca_).sum(1) + (w * cp * cb_).sum(1))), (w * cp * cp).sum(1), vt))
        if a.pooled:
            for g in GUILDS:
                if not store[g]:
                    continue
                for arm in arm_list:
                    Ftr_all = np.vstack([r_["F"][arm][0] for r_ in store[g]])
                    y_all = np.concatenate([r_["ytr"] for r_ in store[g]])[:, None]
                    b_all = np.concatenate([r_["blk"] for r_ in store[g]])
                    sdv = Ftr_all.std(0)
                    k_ = sdv > 1e-9
                    B, xm, ym, _ = ridge_cv(Ftr_all[:, k_] / sdv[k_], y_all, b_all, lams)
                    for r_ in store[g]:
                        w = r_["w"]
                        p = ((r_["F"][arm][1][:, k_] / sdv[k_] - xm) @ B + ym)[:, 0]
                        cp = p[None, :] - ((w * p[None, :]).sum(1) / w.sum(1))[:, None]
                        acc[g][arm].append((0.5 * ((w * cp * r_["ca"]).sum(1) + (w * cp * r_["cb"]).sum(1)),
                                            (w * cp * cp).sum(1), r_["vt"]))
                        if a.rival_all:
                            for h, msk in (("west", r_["west"]), ("east", ~r_["west"])):
                                if msk.sum() < 10:
                                    continue
                                wm = w * msk[None, :]
                                ctr = lambda M: M - ((wm * M).sum(1) / wm.sum(1))[:, None]
                                cph = ctr(np.broadcast_to(p[None, :], wm.shape))
                                cah = ctr(r_["ca"])
                                cbh = ctr(r_["cb"])
                                vth = (wm * cah * cbh).sum(1)
                                if vth[0] <= 0:
                                    continue
                                accw[h][g][arm].append((0.5 * ((wm * cph * cah).sum(1) + (wm * cph * cbh).sum(1)),
                                                        (wm * cph * cph).sum(1), vth))
        out = {}
        for g in GUILDS:
            if n_used[g] == 0:
                continue
            o = {"n_species": n_used[g]}
            boots = {}
            for arm in arm_list:
                cv = np.sum([t[0] for t in acc[g][arm]], 0)
                vp = np.sum([t[1] for t in acc[g][arm]], 0)
                vt = np.sum([t[2] for t in acc[g][arm]], 0)
                r = cv / np.sqrt(vp * vt)
                boots[arm] = r
                o[arm] = {"corr": float(r[0]), "ci": [float(np.nanpercentile(r[1:], q)) for q in (2.5, 97.5)]}
            if a.rival_all:
                gain = boots["all+cropat11"] - boots["all"]
                nulls = np.array([boots[k] - boots["all"] for k in arm_list if k.startswith("all+moved")])
                net = gain - np.median(nulls, 0)
                o["gain_over_all"] = {"diff": float(gain[0]),
                                      "ci": [float(np.nanpercentile(gain[1:], q)) for q in (2.5, 97.5)]}
                o["null_gains"] = [float(v) for v in nulls[:, 0]]
                o["gain_net_of_null_median"] = {"diff": float(net[0]),
                                                "ci": [float(np.nanpercentile(net[1:], q)) for q in (2.5, 97.5)]}
                o["share_of_nulls_below_real"] = float((nulls[:, 0] < gain[0]).mean())
                for h in ("west", "east"):
                    if all(accw[h][g][k] for k in ("all", "all+cropat11")):
                        rr = {}
                        for k in ("all", "all+cropat11"):
                            cv = np.sum([t[0] for t in accw[h][g][k]], 0)
                            vp = np.sum([t[1] for t in accw[h][g][k]], 0)
                            vt = np.sum([t[2] for t in accw[h][g][k]], 0)
                            rr[k] = cv / np.sqrt(vp * vt)
                        dh = rr["all+cropat11"] - rr["all"]
                        o[f"gain_over_all_{h}"] = {"diff": float(dh[0]),
                                                   "ci": [float(np.nanpercentile(dh[1:], q)) for q in (2.5, 97.5)]}
                o["_boot_gain"] = gain.tolist()
            else:
                for x_, y_ in (("cropat11", "luh_crop5"), ("luh33+cropat11", "luh33")):
                    d = boots[x_] - boots[y_]
                    o[f"{x_} - {y_}"] = {"diff": float(d[0]),
                                         "ci": [float(np.nanpercentile(d[1:], q)) for q in (2.5, 97.5)]}
                o["_boot_gain"] = (boots["luh33+cropat11"] - boots["luh33"]).tolist()
            out[g] = o
        if "Grassland" in out and "Forest" in out:
            dg = np.array(out["Grassland"]["_boot_gain"]) - np.array(out["Forest"]["_boot_gain"])
            out["grassland_minus_forest_gain"] = {"diff": float(dg[0]),
                                                  "ci": [float(np.nanpercentile(dg[1:], q)) for q in (2.5, 97.5)]}
        for g in GUILDS:
            if g in out:
                out[g].pop("_boot_gain", None)
        res["modes"][mode] = out
    res["pooled"], res["rival_all"], res["arms"] = bool(a.pooled), bool(a.rival_all), arm_list
    fn = "guild_dose_rival.json" if a.rival_all else ("guild_dose_pooled.json" if a.pooled else "guild_dose.json")
    with open(os.path.join(a.out, fn), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    for fn in ("guild_dose.json", "guild_dose_pooled.json", "guild_dose_rival.json"):
        if os.path.exists(os.path.join(out, fn)):
            print(f"== {fn}")
            _summ(json.load(open(os.path.join(out, fn), encoding="utf-8")))


def _summ(r):
    print(f"{os.path.basename(os.path.normpath(r['cache']))}: {r['n_species']} species; habitats {r['habitat_counts']}")
    for mode, o in r["modes"].items():
        print(f"  [{mode}]")
        for g in GUILDS:
            if g not in o:
                continue
            v = o[g]
            if r.get("rival_all"):
                f_ = lambda d: f"{d['diff']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]"
                extra = " ".join(f"{h} {f_(v[f'gain_over_all_{h}'])}" for h in ("west", "east")
                                 if f"gain_over_all_{h}" in v)
                print(f"    {g:14s} ({v['n_species']:3d} spp) all {v['all']['corr']:+.3f} all+CropAT "
                      f"{v['all+cropat11']['corr']:+.3f} | gain {f_(v['gain_over_all'])}; moved-CropAT null gains "
                      f"median {np.median(v['null_gains']):+.3f} max {max(v['null_gains']):+.3f} (real beats "
                      f"{v['share_of_nulls_below_real']:.0%}); net of null {f_(v['gain_net_of_null_median'])} | {extra}")
                continue
            arms = " | ".join(f"{k} {v[k]['corr']:+.3f}" for k in ARMS)
            dd = v["luh33+cropat11 - luh33"]
            da = v["cropat11 - luh_crop5"]
            print(f"    {g:14s} ({v['n_species']:3d} spp) {arms} || +CropAT over LUH33 {dd['diff']:+.3f} "
                  f"[{dd['ci'][0]:+.3f},{dd['ci'][1]:+.3f}]; CropAT - LUHcrop5 {da['diff']:+.3f} "
                  f"[{da['ci'][0]:+.3f},{da['ci'][1]:+.3f}]")
        if "grassland_minus_forest_gain" in o:
            v = o["grassland_minus_forest_gain"]
            print(f"    grassland gain - forest gain {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
