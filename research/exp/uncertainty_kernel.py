"""E072: how much extra dissimilarity (uncertainty) does the readout need, and with what shape?

    python research/exp/uncertainty_kernel.py --cache <dir> --out <dir> [--seed-caches <dir> ...] [--drain]
    python research/exp/uncertainty_kernel.py --combine <run dir> ... --similarity <csv> --out <dir>
    python research/exp/uncertainty_kernel.py --selftest

The age model reads habitat as H = z.beta (iid prior: kernel s^2 z.z'). DESK's trajectory stays the ONLY source of
predicted change; the question is the size and shape of the uncertainty to add on top of it so the backcast is honest
(considerations G1, G2, G4; the user 2026-10-14: excess dissimilarity is less harmful than excess similarity, and the
added term must not stop DESK carrying temporal signal).

Per cache (one DESK model), on HELD-OUT cells, for the 273 out-of-community species:
  readouts   production form (one amplitude on raw z r24, levels) and split form (cell mean z / within deviation, own
             amplitudes), fitted on training cells x trained years -- species_readout_probe --level, unchanged
  windows    6-year windows 1966-2001, each against the modern epoch 2005-2025: span = 2015 - window centre,
             reach = first trained year - window centre (<= 0: inside the trained years)
  truth      log1p of ABBA half means (independent halves), so (Ta - P)(Tb - P) is a noise-free squared error
  saved      per window and cell x species: Ta, Tb, predictions, posterior variance V of the predicted change,
             the across-seed variance of the predicted change (--seed-caches: each seed its own split readout),
             plus each cell's position, block and modern mean z. Everything else is computed by --combine.
--drain: the "B" arm -- the split readout plus a free separable space x time block (joint RFF, 300 km x ell_t) fitted
jointly by marginal likelihood. Does the block take the change away from DESK's within term (amplitude, spread of the
DESK part of the predicted change, its corr and k)?

--combine: the error variogram eps(span) = sum (Ta-P)(Tb-P) / sum S_s (S_s: the species' noise-free spatial variance
of the modern level on the same cells; dimensionless: error variance per unit of spatial habitat contrast), by
stratum and readout, with block-bootstrap CIs; its linear and saturating fits and their values at 80 and 100 years;
the reach effect (tempho minus base at equal spans); the spatial correlation of errors by distance and by DESK
similarity; the seed-spread multiplier and rank correlations; and cross-fitted calibration (fit on one species half,
R = error / predicted variance on the other).
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

WINDOWS = [(1966, 1971), (1972, 1977), (1978, 1983), (1984, 1989), (1990, 1995), (1996, 2001)]
MODERN = (2005, 2025)
MOD_MID = 0.5 * (MODERN[0] + MODERN[1])


def cell_mean(keys, X, mask):
    """Each row's cell mean of X over that cell's rows in mask (NaN-safe: rows outside mask contribute nothing)."""
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    uc, inv = np.unique(cid, return_inverse=True)
    w = np.asarray(mask, float)
    Xz = np.where(mask[:, None], X, 0.0)
    sums = np.zeros((len(uc), X.shape[1]))
    np.add.at(sums, inv, Xz)
    cnt = np.bincount(inv, weights=w, minlength=len(uc))
    return (sums / np.maximum(cnt, 1)[:, None])[inv]


def abba(rows, years):
    from src.community_encoder.train_DESK.validation_core import abba_halves
    return abba_halves(rows, years)


def lmean(X, rows):
    return np.log1p(np.asarray(X[np.asarray(rows, int)], "float64").mean(0))


def split_fit(z, zbar, Y, trn, r):
    from lib import blr
    m = blr.fit(np.hstack([zbar, z - zbar])[trn], Y[trn], [(0, r), (r, 2 * r)])
    return m


def run(a):
    from lib import blr
    from src.data.preprocess.bbs_community import log1p_community
    c = lambda d, f: os.path.join(d, f)
    r = a.rank
    keys = np.load(c(a.cache, "keys.npy"))
    split = np.load(c(a.cache, "split.npz"))
    withheld = [int(y) for y in split["withheld"]]
    ho = split["holdout"][keys[:, 0], keys[:, 1]]
    bf = split["buffer"][keys[:, 0], keys[:, 1]]
    ty = ~np.isin(keys[:, 2], withheld)
    meta = json.load(open(c(a.cache, "meta.json"), encoding="utf-8"))
    species = list(meta["dev_species"])
    Xdev = np.asarray(np.load(c(a.cache, "X_dev.npy")), "float64")
    Y = np.asarray(log1p_community(Xdev), "float64")
    t0 = int(keys[ty, 2].min())

    def load_z(d):
        k2 = np.load(c(d, "keys.npy"))
        assert k2.shape == keys.shape and (k2 == keys).all(), f"keys differ: {d}"
        z = np.asarray(np.load(c(d, "z_raw.npy"), mmap_mode="r")[:, :r], "float64")
        fin = np.isfinite(z).all(1)
        return np.where(fin[:, None], z, 0.0), fin

    z, fin = load_z(a.cache)
    trn = np.where(~ho & ~bf & ty & fin)[0]
    zbar = cell_mean(keys, z, ty & fin)
    m_iid = blr.fit(z[trn], Y[trn], [(0, r)])
    m_sp = split_fit(z, zbar, Y, trn, r)
    print(f"[E072] {os.path.basename(os.path.normpath(a.cache))}: T0 {t0}, {len(trn)} training rows, "
          f"split median log amp between/within {np.median(np.log(m_sp['a']), 0).round(2).tolist()}", flush=True)

    # held-out cells and their rows
    hrows = np.where(ho & fin)[0]
    cid = keys[hrows, 0].astype(np.int64) * 100000 + keys[hrows, 1]
    order = np.argsort(cid, kind="stable")
    hrows, cid = hrows[order], cid[order]
    ucell, start = np.unique(cid, return_index=True)
    bounds = list(start) + [len(hrows)]
    cell_rows = [hrows[bounds[i]:bounds[i + 1]] for i in range(len(ucell))]
    yr = keys[:, 2]

    def groups(y0, y1):
        out = {}
        for i, rows in enumerate(cell_rows):
            g = rows[(yr[rows] >= y0) & (yr[rows] <= y1)]
            if len(g) >= 2:
                ha, hb = abba(g, yr[g])
                if len(ha) and len(hb):
                    out[i] = (g, ha, hb)
        return out

    gm = groups(*MODERN)
    seeds = [a.cache] + [d for d in (a.seed_caches or []) if os.path.normpath(d) != os.path.normpath(a.cache)]
    seed_models = []
    for d in seeds[1:]:
        zk, fk = load_z(d)
        trk = np.where(~ho & ~bf & ty & fk)[0]
        zbk = cell_mean(keys, zk, ty & fk)
        seed_models.append((zk, split_fit(zk, zbk, Y, trk, r)["coef"][:, r:]))
        print(f"[E072]   seed readout fitted: {os.path.basename(os.path.normpath(d))}", flush=True)
    W_sp, W_iid = m_sp["coef"][:, r:], m_iid["coef"]
    Ai_sp, Ai_iid = m_sp["Ainv"][:, r:, r:], m_iid["Ainv"]
    cells_xy = np.stack([keys[cr[0], :2] for cr in cell_rows]).astype(float) * 27.0
    blk = (keys[[cr[0] for cr in cell_rows], 0] // 6).astype(np.int64) * 100000 + keys[[cr[0] for cr in cell_rows], 1] // 6
    mod_cells = np.array(sorted(gm))
    la_mod_a = np.stack([lmean(Xdev, gm[i][1]) for i in mod_cells])
    la_mod_b = np.stack([lmean(Xdev, gm[i][2]) for i in mod_cells])
    zmod = np.stack([z[gm[i][0]].mean(0) for i in mod_cells])
    save = {"species": np.array(species), "t0": t0, "cells_xy": cells_xy[mod_cells], "block": blk[mod_cells],
            "mod_cells": mod_cells, "la_mod_a": la_mod_a.astype("float32"), "la_mod_b": la_mod_b.astype("float32"),
            "zmod": zmod.astype("float32"), "windows": np.array(WINDOWS), "n_seeds": len(seeds)}
    pos = {int(i): j for j, i in enumerate(mod_cells)}
    for wi, (y0, y1) in enumerate(WINDOWS):
        gw = groups(y0, y1)
        cl = np.array([i for i in sorted(gw) if i in gm])
        if len(cl) < 20:
            continue
        Ta = np.stack([lmean(Xdev, gm[i][1]) - lmean(Xdev, gw[i][1]) for i in cl])
        Tb = np.stack([lmean(Xdev, gm[i][2]) - lmean(Xdev, gw[i][2]) for i in cl])
        dz = np.stack([z[gm[i][0]].mean(0) - z[gw[i][0]].mean(0) for i in cl])
        Psp, Piid = dz @ W_sp.T, dz @ W_iid.T
        Vsp = np.einsum("cp,spq,cq->cs", dz, Ai_sp, dz)
        Viid = np.einsum("cp,spq,cq->cs", dz, Ai_iid, dz)
        sv = np.zeros_like(Psp)
        if seed_models:
            Pk = [Psp] + [np.stack([zk[gm[i][0]].mean(0) - zk[gw[i][0]].mean(0) for i in cl]) @ Wk.T
                          for zk, Wk in seed_models]
            sv = np.var(np.stack(Pk), axis=0, ddof=1)
        save.update({f"w{wi}_cells": np.array([pos[int(i)] for i in cl]), f"w{wi}_Ta": Ta.astype("float32"),
                     f"w{wi}_Tb": Tb.astype("float32"), f"w{wi}_Psp": Psp.astype("float32"),
                     f"w{wi}_Piid": Piid.astype("float32"), f"w{wi}_Vsp": Vsp.astype("float32"),
                     f"w{wi}_Viid": Viid.astype("float32"), f"w{wi}_seedvar": sv.astype("float32"),
                     f"w{wi}_dz2": (dz ** 2).sum(1).astype("float32")})
        print(f"[E072]   window {y0}-{y1}: {len(cl)} held-out cells, span {MOD_MID - 0.5 * (y0 + y1):.1f}, "
              f"reach {t0 - 0.5 * (y0 + y1):+.1f}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "e072_arrays.npz"), **save)
    if a.drain:
        drain(a, keys, z, zbar, Y, trn, m_sp, cell_rows, gm, groups, r)


def drain(a, keys, z, zbar, Y, trn, m_sp, cell_rows, gm, groups, r):
    """B arm: split readout + a free separable space x time block, all amplitudes by marginal likelihood."""
    from lib import blr
    from rrr_backcast import wc
    xy = keys[:, :2].astype(float) * 27.0
    t = keys[:, 2].astype(float)
    res = {"split_only": {}, "with_block": {}}

    def score(P, Ta, Tb):
        Pc, Ac, Bc = (X - X.mean(0) for X in (P, Ta, Tb))
        w = np.ones(len(P))
        vp, vt = wc(Pc, Pc, w), wc(Ac, Bc, w)
        cv = 0.5 * (wc(Pc, Ac, w) + wc(Pc, Bc, w))
        return {"corr": float(cv / np.sqrt(vp * vt)) if vp > 0 else None, "k": float(cv / vp) if vp > 0 else None,
                "pred_var": float(vp)}
    win = []
    for (y0, y1) in WINDOWS:
        gw = groups(y0, y1)
        cl = [i for i in sorted(gw) if i in gm]
        if len(cl) >= 20:
            win.append((y0, y1, gw, cl))
    Ta = np.vstack([np.stack([lmean_Y(Y, gm[i][1]) - lmean_Y(Y, gw[i][1]) for i in cl]) for _, _, gw, cl in win])

    def preds(feat_fn, coefW, coefG):
        out_d, out_g = [], []
        for _, _, gw, cl in win:
            dz = np.stack([z[gm[i][0]].mean(0) - z[gw[i][0]].mean(0) for i in cl])
            out_d.append(dz @ coefW.T)
            if coefG is not None:
                dg = np.stack([feat_fn(gm[i][0]).mean(0) - feat_fn(gw[i][0]).mean(0) for i in cl])
                out_g.append(dg @ coefG.T)
        return np.vstack(out_d), (np.vstack(out_g) if coefG is not None else None)
    # truth halves on the raw count scale, as in run()
    Xdev = np.asarray(np.load(os.path.join(a.cache, "X_dev.npy")), "float64")
    TA = np.vstack([np.stack([lmean(Xdev, gm[i][1]) - lmean(Xdev, gw[i][1]) for i in cl]) for _, _, gw, cl in win])
    TB = np.vstack([np.stack([lmean(Xdev, gm[i][2]) - lmean(Xdev, gw[i][2]) for i in cl]) for _, _, gw, cl in win])
    del Ta
    Pd0, _ = preds(None, m_sp["coef"][:, r:], None)
    res["split_only"] = {"median_log_amp_between_within": np.median(np.log(m_sp["a"]), 0).tolist(),
                         "desk_part": score(Pd0, TA, TB)}
    print(f"[E072 drain] split only: {res['split_only']}", flush=True)
    rng = np.random.default_rng(5)
    ng = a.n_block
    om_x = rng.normal(size=(2, ng)) / 300.0
    ph = rng.uniform(0, 2 * np.pi, ng)
    zw = z - zbar
    for ell_t in a.ell_t:
        om_t = rng.normal(size=ng) / float(ell_t)
        feat = lambda rows: np.sqrt(2.0 / ng) * np.cos(xy[rows] @ om_x + t[rows, None] * om_t + ph)
        G = feat(np.arange(len(keys)))
        X = np.hstack([zbar, zw, G])
        m = blr.fit(X[trn], Y[trn], [(0, r), (r, 2 * r), (2 * r, 2 * r + ng)])
        Pd, Pg = preds(feat, m["coef"][:, r:2 * r], m["coef"][:, 2 * r:])
        o = {"median_log_amp_between_within_block": np.median(np.log(m["a"]), 0).tolist(),
             "desk_part": score(Pd, TA, TB), "block_part": score(Pg, TA, TB), "total": score(Pd + Pg, TA, TB)}
        o["desk_part_spread_ratio_vs_split_only"] = o["desk_part"]["pred_var"] / res["split_only"]["desk_part"]["pred_var"]
        res["with_block"][f"ell_t{ell_t:g}"] = o
        print(f"[E072 drain] ell_t {ell_t:g}: {o}", flush=True)
    json.dump(res, open(os.path.join(a.out, "e072_drain.json"), "w"), indent=1)


def lmean_Y(Y, rows):
    return np.asarray(Y[np.asarray(rows, int)], "float64").mean(0)


# ---------------------------------------------------------------------------------------------------- combine
def half_of(code):
    return zlib.crc32(("e072:" + str(code)).encode()) % 2


def load_run(d):
    z = dict(np.load(os.path.join(d, "e072_arrays.npz"), allow_pickle=False))
    la, lb = z["la_mod_a"].astype("float64"), z["la_mod_b"].astype("float64")
    S = ((la - la.mean(0)) * (lb - lb.mean(0))).mean(0)          # noise-free spatial variance of the modern level
    prev = ((la + lb) > 0).sum(0)
    z["S"], z["prev"] = S, prev
    return z


def window_terms(z, wi, readout="sp", place=False):
    Ta, Tb = z[f"w{wi}_Ta"].astype("float64"), z[f"w{wi}_Tb"].astype("float64")
    P = z[f"w{wi}_P{readout}"].astype("float64")
    V = z[f"w{wi}_V{readout}"].astype("float64")
    if place:
        Ta, Tb, P = Ta - Ta.mean(0), Tb - Tb.mean(0), P - P.mean(0)
    e = (Ta - P) * (Tb - P)
    return e, V, Ta * Tb, P


def combine(a):
    import pandas as pd
    sim = pd.read_csv(a.similarity).set_index("species_code")
    out = {"runs": {}}
    runs = {os.path.basename(os.path.normpath(d)): load_run(d) for d in a.combine}
    rng = np.random.default_rng(17)
    lines = []

    def strata(z):
        sp = [str(s) for s in z["species"]]
        mig = np.array([sim["migration"].get(s, np.nan) for s in sp])
        ok = (z["S"] > 0) & (z["prev"] >= a.min_cells)
        half = np.array([half_of(s) for s in sp])
        return {"all": ok, "sedentary": ok & (mig == 1), "resident": ok & (mig <= 2), "migratory": ok & (mig == 3)}, half

    def spans(z):
        W = z["windows"]
        return {wi: (MOD_MID - 0.5 * (W[wi][0] + W[wi][1]), int(z["t0"]) - 0.5 * (W[wi][0] + W[wi][1]))
                for wi in range(len(W)) if f"w{wi}_Ta" in z}

    for name, z in runs.items():
        St, half = strata(z)
        sp_ = spans(z)
        blocks = z["block"]
        ub, binv_all = np.unique(blocks, return_inverse=True)
        draws = [np.ones(len(ub))] + [rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub))).astype(float)
                                      for _ in range(a.n_boot)]
        R = {}
        for readout in ("sp", "iid"):
            for place in (False, True):
                tag = f"{readout}_{'place' if place else 'total'}"
                for sname, m in St.items():
                    rows = []
                    for wi, (span, reach) in sp_.items():
                        e, V, tt, P = window_terms(z, wi, readout, place)
                        cells = z[f"w{wi}_cells"]
                        bw = np.array([d[binv_all[cells]] for d in draws])          # (B+1, n_cells)
                        num = bw @ e[:, m].sum(1)
                        den = bw.sum(1) * z["S"][m].sum()
                        sv = z[f"w{wi}_seedvar"].astype("float64")
                        rows.append({"window": wi, "span": span, "reach": reach, "n_cells": int(len(cells)),
                                     "eps": num / den, "V": (bw @ V[:, m].sum(1)) / den,
                                     "tau": (bw @ tt[:, m].sum(1)) / den,
                                     "pred": (bw @ (P[:, m] ** 2).sum(1)) / den,
                                     "seedvar": (bw @ sv[:, m].sum(1)) / den})
                    R[(tag, sname)] = rows
        res = {}
        for (tag, sname), rows in R.items():
            sx = np.array([r_["span"] for r_ in rows])
            E = np.stack([r_["eps"] for r_ in rows], 1)                          # (B+1, n_windows)
            ex = E - np.stack([r_["V"] for r_ in rows], 1)                      # excess over the posterior
            fits = [np.polyfit(sx, ex[b], 1) for b in range(len(ex))]
            slope = np.array([f[0] for f in fits])
            icpt = np.array([f[1] for f in fits])
            at = lambda T: icpt + slope * T
            ci = lambda v: [float(np.nanpercentile(v[1:], 2.5)), float(np.nanpercentile(v[1:], 97.5))]
            # saturating alternative: c (1 - exp(-span / ell)), grid over ell, best ell on the point estimate
            best = None
            for ell in (5, 10, 20, 40, 80, 160, 320):
                g = 1 - np.exp(-sx / ell)
                cc = (g @ ex[0]) / (g @ g)
                sse = ((ex[0] - cc * g) ** 2).sum()
                if best is None or sse < best[0]:
                    best = (sse, ell, cc)
            sse_lin = ((ex[0] - at(0)[0] - slope[0] * sx) ** 2).sum()
            res.setdefault(tag, {})[sname] = {
                "windows": [{"span": r_["span"], "reach": r_["reach"], "n_cells": r_["n_cells"],
                             "eps": float(r_["eps"][0]), "eps_ci": ci(r_["eps"]), "no_change": float(r_["tau"][0]),
                             "posterior_V": float(r_["V"][0]), "pred_energy": float(r_["pred"][0]),
                             "R_posterior": float(r_["eps"][0] / r_["V"][0]) if r_["V"][0] > 0 else None,
                             "seed_multiplier": float((r_["eps"][0] - r_["V"][0]) / r_["seedvar"][0])
                             if r_["seedvar"][0] > 0 else None,
                             "seed_multiplier_ci": ci((r_["eps"] - r_["V"]) / r_["seedvar"])
                             if r_["seedvar"][0] > 0 else None} for r_ in rows],
                "linear": {"intercept": float(icpt[0]), "intercept_ci": ci(icpt), "slope_per_decade": float(10 * slope[0]),
                           "slope_ci": [10 * x for x in ci(slope)], "sse": float(sse_lin),
                           "at_80y": float(at(80)[0]), "at_80y_ci": ci(at(80)), "at_100y": float(at(100)[0]),
                           "at_100y_ci": ci(at(100))},
                "saturating": {"ell": best[1], "c": float(best[2]), "sse": float(best[0]),
                               "at_80y": float(best[2] * (1 - np.exp(-80 / best[1]))),
                               "at_100y": float(best[2] * (1 - np.exp(-100 / best[1])))}}
        out["runs"][name] = {"t0": int(z["t0"]), "n_species": {k: int(v.sum()) for k, v in St.items()},
                             "variogram": res, "shape": shape_checks(z, St, a), "calibration": crossfit(z, St, half, sp_)}
        for tag in ("sp_total", "sp_place", "iid_total"):
            for sname in ("sedentary", "all"):
                o = res[tag][sname]
                ws = " ".join(f"{w['span']:.0f}y:{w['eps']:.3f}" for w in o["windows"])
                lines.append(f"{name:28s} {tag:9s} {sname:9s} n{out['runs'][name]['n_species'][sname]:3d} | eps by span "
                             f"{ws} | lin slope/dec {o['linear']['slope_per_decade']:+.3f} "
                             f"[{o['linear']['slope_ci'][0]:+.3f},{o['linear']['slope_ci'][1]:+.3f}] "
                             f"@80y {o['linear']['at_80y']:.3f} | sat ell {o['saturating']['ell']} @80y "
                             f"{o['saturating']['at_80y']:.3f}")
    out["reach"] = reach_effect(runs, a)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "e072_combined.json"), "w"), indent=1, default=float)
    print("\n".join(lines))
    for name, o in out["runs"].items():
        print(f"== {name}: shape {json.dumps(o['shape'], default=float)[:900]}")
        print(f"   calibration {json.dumps(o['calibration'], default=float)[:900]}")
    print(f"== reach {json.dumps(out['reach'], default=float)[:1500]}")


def shape_checks(z, St, a):
    """Spatial correlation of errors (distance vs DESK similarity) and per-cell predictors of error size."""
    from scipy.stats import spearmanr
    xy = z["cells_xy"]
    zm = z["zmod"].astype("float64")
    zn = zm / np.maximum(np.linalg.norm(zm, axis=1, keepdims=True), 1e-12)
    res = {}
    for sname in ("all", "sedentary"):
        m = St[sname]
        S = z["S"][m]
        dbins = [0, 100, 200, 400, 800, 1600, 1e9]
        acc = {"dist": np.zeros((len(dbins) - 1, 2)), "sim": np.zeros((5, 2)), "diag": [0.0, 0.0]}
        rho_rows = []
        sims_all = (zn @ zn.T)
        qs = np.quantile(sims_all[np.triu_indices(len(zn), 1)], [0.2, 0.4, 0.6, 0.8])
        for wi in range(len(WINDOWS)):
            if f"w{wi}_Ta" not in z:
                continue
            for place in (True,):
                e, V, tt, P = window_terms(z, wi, "sp", place)
                Ta, Tb = z[f"w{wi}_Ta"].astype("float64"), z[f"w{wi}_Tb"].astype("float64")
                Pp = z[f"w{wi}_Psp"].astype("float64")
                Ra = (Ta - Ta.mean(0)) - (Pp - Pp.mean(0))
                Rb = (Tb - Tb.mean(0)) - (Pp - Pp.mean(0))
                cells = z[f"w{wi}_cells"]
                A_, B_ = Ra[:, m] / np.sqrt(S), Rb[:, m] / np.sqrt(S)
                C = 0.5 * (A_ @ B_.T + B_ @ A_.T) / m.sum()                     # cross-cell: halves independent
                n = len(cells)
                iu = np.triu_indices(n, 1)
                d = np.sqrt(((xy[cells][:, None, :] - xy[cells][None, :, :]) ** 2).sum(-1))[iu]
                s = sims_all[np.ix_(cells, cells)][iu]
                cv = C[iu]
                acc["diag"][0] += np.trace(C)
                acc["diag"][1] += n
                for k in range(len(dbins) - 1):
                    mk = (d > dbins[k]) & (d <= dbins[k + 1])
                    acc["dist"][k] += [cv[mk].sum(), mk.sum()]
                # similarity quintiles WITHIN 200-800 km (distance held roughly fixed)
                band = (d > 200) & (d <= 800)
                qb = np.digitize(s[band], qs)
                for k in range(5):
                    acc["sim"][k] += [cv[band][qb == k].sum(), (qb == k).sum()]
                # per-cell predictors of error size (noise-free cell error energy)
                Ec = (e[:, m] / S).mean(1)
                sv = (z[f"w{wi}_seedvar"].astype("float64")[:, m] / S).mean(1)
                mv = (Pp[:, m] ** 2 / S).mean(1)
                rho_rows.append({"window": wi,
                                 "spearman_err_seedvar": float(spearmanr(Ec, sv)[0]) if sv.any() else None,
                                 "spearman_err_predicted_move": float(spearmanr(Ec, mv)[0]),
                                 "spearman_err_dz2": float(spearmanr(Ec, z[f"w{wi}_dz2"])[0])})
        c0 = acc["diag"][0] / acc["diag"][1]
        res[sname] = {"diag_var": float(c0),
                      "corr_by_distance_km": {f"{dbins[k]}-{dbins[k + 1]:g}": float(acc["dist"][k, 0] /
                                              max(acc["dist"][k, 1], 1) / c0) for k in range(len(dbins) - 1)},
                      "corr_by_desk_similarity_quintile_200_800km": [float(acc["sim"][k, 0] / max(acc["sim"][k, 1], 1)
                                                                          / c0) for k in range(5)],
                      "per_cell_spearman": rho_rows}
    return res


def crossfit(z, St, half, sp_):
    """Fit eps(span) = a + b span on one species half (split readout, total), predict the other: R = err / pred var."""
    out = {}
    for sname in ("all", "sedentary", "resident"):
        res = {}
        for fit_h in (0, 1):
            fm, tm = St[sname] & (half == fit_h), St[sname] & (half != fit_h)
            if fm.sum() < 5 or tm.sum() < 5:
                continue
            sx, ex = [], []
            for wi in sp_:
                e, V, _, _ = window_terms(z, wi, "sp", False)
                sx.append(sp_[wi][0])
                ex.append((e[:, fm].sum() - V[:, fm].sum()) / (len(e) * z["S"][fm].sum()))
            b, a0 = np.polyfit(sx, ex, 1)
            Rw = []
            for wi in sp_:
                e, V, _, _ = window_terms(z, wi, "sp", False)
                gam = max(a0 + b * sp_[wi][0], 0.0)
                pv = V[:, tm].sum() + gam * len(e) * z["S"][tm].sum()
                Rw.append({"span": sp_[wi][0], "R_after": float(e[:, tm].sum() / pv),
                           "R_before": float(e[:, tm].sum() / V[:, tm].sum())})
            res[f"fit_half{fit_h}"] = Rw
        out[sname] = res
    return out


def reach_effect(runs, a):
    """eps at equal spans: each tempho run minus the base run (reach 0), split readout, total, by stratum."""
    base = [n for n, z in runs.items() if int(z["t0"]) <= 1966]
    if not base:
        return {}
    zb = runs[base[0]]
    out = {}
    for name, z in runs.items():
        if name == base[0]:
            continue
        o = {}
        for sname in ("all", "sedentary"):
            rows = []
            for wi in range(len(WINDOWS)):
                if f"w{wi}_Ta" not in z or f"w{wi}_Ta" not in zb:
                    continue
                vals = []
                for zz in (z, zb):
                    import pandas as pd  # noqa: F401
                    mig_ok = species_mask(zz, sname, a)
                    e, V, _, _ = window_terms(zz, wi, "sp", False)
                    vals.append(e[:, mig_ok].sum() / (len(e) * zz["S"][mig_ok].sum()))
                W = WINDOWS[wi]
                rows.append({"window": f"{W[0]}-{W[1]}", "reach": int(z["t0"]) - 0.5 * (W[0] + W[1]),
                             "eps_run": float(vals[0]), "eps_base": float(vals[1]), "diff": float(vals[0] - vals[1])})
            o[sname] = rows
        out[name] = o
    return out


_SIM = {}


def species_mask(z, sname, a):
    import pandas as pd
    if "sim" not in _SIM:
        _SIM["sim"] = pd.read_csv(a.similarity).set_index("species_code")
    sim = _SIM["sim"]
    sp = [str(s) for s in z["species"]]
    mig = np.array([sim["migration"].get(s, np.nan) for s in sp])
    ok = (z["S"] > 0) & (z["prev"] >= a.min_cells)
    return {"all": ok, "sedentary": ok & (mig == 1)}[sname]


# ---------------------------------------------------------------------------------------------------- selftest
def selftest():
    """Planted check of the estimator: truth = prediction + error with a KNOWN variogram + independent half noise.
    eps must recover the planted error variance per unit spatial variance, and the linear fit its slope."""
    rng = np.random.default_rng(0)
    n, S_ = 400, 60
    S_true = rng.uniform(0.5, 2.0, S_)                                   # spatial variance of each species' level
    gam = lambda span: 0.02 + 0.004 * span                               # planted excess per unit S
    z = {"species": np.array([f"sp{i}" for i in range(S_)]), "t0": 1966, "windows": np.array(WINDOWS)}
    lev = rng.normal(size=(n, S_)) * np.sqrt(S_true)
    noise = 0.3
    la = lev + rng.normal(size=(n, S_)) * noise
    lb = lev + rng.normal(size=(n, S_)) * noise
    z["la_mod_a"], z["la_mod_b"] = la, lb
    for wi, (y0, y1) in enumerate(WINDOWS):
        span = MOD_MID - 0.5 * (y0 + y1)
        P = rng.normal(size=(n, S_)) * 0.3
        err = rng.normal(size=(n, S_)) * np.sqrt(gam(span) * S_true)
        T = P + err
        z[f"w{wi}_Ta"] = T + rng.normal(size=(n, S_)) * noise
        z[f"w{wi}_Tb"] = T + rng.normal(size=(n, S_)) * noise
        z[f"w{wi}_Psp"] = P
        z[f"w{wi}_Vsp"] = np.zeros((n, S_))
    Sx = ((la - la.mean(0)) * (lb - lb.mean(0))).mean(0)
    sx, ex = [], []
    for wi, (y0, y1) in enumerate(WINDOWS):
        e, V, _, _ = window_terms(z, wi, "sp", False)
        sx.append(MOD_MID - 0.5 * (y0 + y1))
        ex.append(e.sum() / (n * Sx.sum()))
    b, a0 = np.polyfit(sx, ex, 1)
    print(f"[selftest] planted slope 0.004 intercept 0.020 -> recovered {b:.4f} {a0:.4f}; "
          f"eps by span {np.round(ex, 4).tolist()} vs planted {[round(gam(s), 4) for s in sx]}")
    assert abs(b - 0.004) < 0.0015 and abs(a0 - 0.02) < 0.03, "estimator does not recover the planted variogram"
    print("[selftest] ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--seed-caches", nargs="*", default=None)
    ap.add_argument("--out")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--drain", action="store_true")
    ap.add_argument("--ell-t", type=float, nargs="+", default=[10.0, 30.0, 80.0])
    ap.add_argument("--n-block", type=int, default=256)
    ap.add_argument("--combine", nargs="*", default=None)
    ap.add_argument("--similarity", default=None)
    ap.add_argument("--min-cells", type=int, default=20, help="species seen (either half) in >= this many modern cells")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.combine:
        return combine(a)
    run(a)


if __name__ == "__main__":
    main()
