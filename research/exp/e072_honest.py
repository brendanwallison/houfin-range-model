"""E072 (v2): honest uncertainty for predicted habitat change -- representation-shaped (errors-in-variables, the user's
"option 2") and scaled by covariate novelty, calibrated where change is novel.

    python research/exp/e072_honest.py --cache <tempho cache> --similarity <csv> --out <dir>
        [--seed-caches <dir> ...] [--n-map-boot 30]

Representations of within-place change (E073):
  delta   the clock-free community-learned covariate delta: W = within-PCs of covariate change (climate, physical
          cover, human; EMA half-life 50 y), B = ridge from W to the community's ESK64 within deviations; delta = W B
  desk    DESK's own raw z within deviation
Each is read out per species by a SINGLE-slope within BLR (the production-like form; E074 rules the place-dependent
readout out for backcasts). delta's readout is also shrunk (prior x0.1, E073 B4).
Designs (training rows):
  full    training cells x trained years
  cover   the same minus training cells whose cover trend is beyond the p70 range (E073 C3) -- held-out cells beyond
          that range are then novel in cover, the 1920-like case
Uncertainty terms per held-out cell c and species s (change = early epoch -> 2005-25):
  V   the readout's posterior variance (E018: ~100x too small)
  U   errors-in-variables of the REPRESENTATION: delta -- the variance of P over block-bootstrap refits of the
      community map B (n-map-boot); desk -- the variance of P over seeds (each seed its own readout)
  nov_g  covariate novelty of the cell's change per group g (climate, cover, human): mean squared whitened component
      against the covariance of IN-ERA changes at training cells (late third - early third of the trained years)
Error: noise-free squared error e2 = (Ta - P)(Tb - P), in units of S_s (the species' noise-free spatial variance of the
modern level on held-out cells). Variance models var = V + S_s * (g0 + sum g_g nov_g) + gU * U, coefficients >= 0 by
NNLS on cell means; fitted on one species half, evaluated on the other (R = sum e2 / sum var; R > 1 overconfident) by
cover-novelty tercile and by beyond / inside the cover range; and fitted on INSIDE cells only, evaluated on BEYOND cells
(the extrapolation test). Finally the fitted law is applied to 1920-39 vs 2005-25 at every cache cell (by Great Plains
zone): predicted change-error variance in S units and the implied corr(H 1920, H 2000) = 1 - var / 2.
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

from anchored_delta import (GROUPS, Y0, ZONES, ZONES_TIF, Cov, Estimand, cell_mean, cover_trend_groups,  # noqa: E402
                            ema, fit_blocks, scaled_coef)

NC = ("climate", "cover", "human")          # the clock-free groups
HL = 50


def nnls(A, y):
    from scipy.optimize import nnls as _n
    return _n(A, y)[0]


def block_ids(keys):
    return (keys[:, 0] // 6).astype(np.int64) * 100000 + keys[:, 1] // 6


def comm_maps(W, Zw, rows, keys, n_boot, seed=0):
    """Ridge W -> Zw on rows, plus block-bootstrap refits (whole 6x6 blocks resampled with replacement)."""
    blk = block_ids(keys[rows])
    ub, inv = np.unique(blk, return_inverse=True)
    k = W.shape[1]
    XtX = np.zeros((len(ub), k, k))
    XtY = np.zeros((len(ub), k, Zw.shape[1]))
    for i in range(len(ub)):
        r = rows[inv == i]
        XtX[i] = W[r].T @ W[r]
        XtY[i] = W[r].T @ Zw[r]

    def solve(w):
        G = np.tensordot(w, XtX, 1)
        lam = 1e-1 * np.trace(G) / k
        return np.linalg.solve(G + lam * np.eye(k), np.tensordot(w, XtY, 1))
    B = solve(np.ones(len(ub)))
    rng = np.random.default_rng(seed)
    Bs = [solve(rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub))).astype(float)) for _ in range(n_boot)]
    return B, Bs


def in_era_change(keys, F, rows, ty):
    """Per training cell among ``rows``: late-third mean minus early-third mean of F over trained years."""
    yrs = keys[:, 2]
    ty_years = np.unique(yrs[ty])
    y0, y1 = int(ty_years.min()), int(ty_years.max())
    span = (y1 - y0 + 1) // 3
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    sel = np.zeros(len(keys), bool)
    sel[rows] = True
    out = []
    for lo, hi in ((y0, y0 + span - 1), (y1 - span + 1, y1)):
        m = sel & (yrs >= lo) & (yrs <= hi)
        uc, inv = np.unique(cid[m], return_inverse=True)
        s = np.zeros((len(uc), F.shape[1]))
        np.add.at(s, inv, F[m])
        out.append((uc, s / np.bincount(inv)[:, None]))
    both = np.intersect1d(out[0][0], out[1][0])
    return out[1][1][np.searchsorted(out[1][0], both)] - out[0][1][np.searchsorted(out[0][0], both)]


def novelty(dW_eval, dW_ref):
    """Mean squared whitened component of each row of dW_eval against the covariance of dW_ref."""
    C = np.cov(dW_ref, rowvar=False) + 1e-9 * np.eye(dW_ref.shape[1])
    Ci = np.linalg.inv(C)
    return np.einsum("ij,jk,ik->i", dW_eval, Ci, dW_eval) / dW_ref.shape[1]


def dense_dW(cov, groups, hl, early=(1920, 1939), modern=(2005, 2025)):
    """Per cache cell: within-feature change (modern-epoch mean minus early-window mean) for each group, from the
    yearly states (EMA from 1902), with the same standardization and loadings as the key features."""
    E = ema(cov.st, hl)
    Xs = np.where(np.isfinite(E), (E - cov.mu[hl]) / cov.sd[hl], 0.0)
    yi = lambda a, b: slice(a - Y0, b - Y0 + 1)
    D = Xs[yi(*modern)].mean(0) - Xs[yi(*early)].mean(0)                           # (cells, C)
    out = {}
    for g in groups:
        cols, V = cov.V[(hl, g)]
        out[g] = D[:, cols] @ V
    return out


def modern_level_S(est, Y):
    """Noise-free spatial variance per species of the modern-epoch level over held-out cells (alternate-year halves)."""
    la, lb = [], []
    yrs = est.keys[:, 2]
    for rows in est.modern:
        r = rows[np.argsort(yrs[rows])]
        if len(r) < 2:
            la.append(np.full(Y.shape[1], np.nan)); lb.append(np.full(Y.shape[1], np.nan))
            continue
        la.append(Y[r[0::2]].mean(0)); lb.append(Y[r[1::2]].mean(0))
    la, lb = np.array(la), np.array(lb)
    ok = np.isfinite(la).all(1)
    la, lb = la[ok], lb[ok]
    return ((la - la.mean(0)) * (lb - lb.mean(0))).mean(0)


def run(a):
    est = Estimand(a.cache, a.n_boot, a.similarity)
    keys, Y = est.keys, est.Y
    cov = Cov(keys, ~est.ho & ~est.bf & est.ty, est.ty)
    fin = cov.fin
    trn = np.where(~est.ho & ~est.bf & est.ty & fin)[0]
    uc, slope, thr, trn_cells, inv = cover_trend_groups(est, cov)
    keep = ~(trn_cells & (np.abs(slope) > thr))
    designs = {"full": trn, "cover": trn[keep[inv[trn]]]}
    ev_cid = est.cells_ev[:, 0].astype(np.int64) * 100000 + est.cells_ev[:, 1]
    beyond = np.abs(slope[np.searchsorted(uc, ev_cid)]) > thr
    Yw = Y - cell_mean(keys, Y, est.ty & fin)
    Zf = np.asarray(np.load(os.path.join(a.cache, "esk_annual.npy"), mmap_mode="r"), "float64")
    Zw = Zf - cell_mean(keys, Zf, est.ty & fin)
    Wg = {g: cov.W[(HL, g)] for g in NC}
    W = np.hstack([Wg[g] for g in NC])
    dW_ev = {g: est.delta(Wg[g]) for g in NC}
    dWall_ev = np.hstack([dW_ev[g] for g in NC])
    S = modern_level_S(est, Y)
    Ta, Tb = est.T["dev"]
    zs = {}
    for d in [a.cache] + list(a.seed_caches or []):
        z = np.asarray(np.load(os.path.join(d, "z_raw.npy"), mmap_mode="r")[:, :24], "float64")
        z = np.where(np.isfinite(z), z, 0.0)
        zs[d] = z - cell_mean(keys, z, est.ty & fin)
    try:
        import rasterio
        with rasterio.open(ZONES_TIF) as src:
            zg = src.read(1)
        zone_ev = np.array([ZONES.get(int(zg[r, c]), "other") for r, c in est.cells_ev])
        zone_all = np.array([ZONES.get(int(zg[r, c]), "other") for r, c in cov.cells])
    except Exception:
        zone_ev = zone_all = None
    res = {"cache": a.cache, "threshold": thr, "n_beyond": int(beyond.sum()), "designs": {}}
    save = {}
    for dname, rows in designs.items():
        ref = {g: in_era_change(keys, Wg[g], rows, est.ty) for g in NC}
        nov = {g: novelty(dW_ev[g], ref[g]) for g in NC}
        # E072b: isolation from supervision (skeptic 2: regional loss costs ~0.06), as a pseudo-novelty "iso" whose
        # log1p is the distance in cells to the nearest training cell of this design
        from scipy.spatial import cKDTree
        tc = np.unique(keys[rows][:, :2], axis=0).astype(float)
        dist, _ = cKDTree(tc).query(est.cells_ev.astype(float))
        nov["iso"] = np.expm1(np.maximum(dist - 1.0, 0.0))     # lnov(iso) = max(dist - 1, 0): 0 next to training
        B, Bs = comm_maps(W, Zw, rows, keys, a.n_map_boot)
        dl = W @ B
        m = fit_blocks([dl], Yw, rows)
        reps = {}
        dd = dWall_ev @ B
        coef_ml = m["coef"]
        coef_sh = scaled_coef(dl[rows], Yw[rows], m, 0.1)
        for tag, coef in (("delta", coef_ml), ("delta_shr", coef_sh)):
            P = dd @ coef.T
            Pb = np.stack([(dWall_ev @ Bb) @ coef.T for Bb in Bs])
            V = np.einsum("cp,spq,cq->cs", dd, m["Ainv"], dd) * (1.0 if tag == "delta" else 0.0)
            reps[tag] = (P, Pb.var(0, ddof=1), V)
        # DESK: main cache's readout + seed spread (each seed its own readout on the same rows)
        Pk = []
        for d, zw in zs.items():
            mk = fit_blocks([zw], Yw, rows)
            Pk.append(est.delta(zw) @ mk["coef"].T)
            if d == a.cache:
                Vd = np.einsum("cp,spq,cq->cs", est.delta(zw), mk["Ainv"], est.delta(zw))
        reps["desk"] = (Pk[0], np.var(np.stack(Pk), 0, ddof=1) if len(Pk) > 1 else np.zeros_like(Pk[0]), Vd)
        res["designs"][dname] = {"n_rows": int(len(rows)), "reps": {}}
        for rname, (P, U, V) in reps.items():
            e2 = (Ta - P) * (Tb - P)
            save[f"{dname}|{rname}|P"], save[f"{dname}|{rname}|U"] = P.astype("float32"), U.astype("float32")
            res["designs"][dname]["reps"][rname] = evaluate(est, e2, U, V, S, nov, beyond)
        # 1920 projection (delta only; the representation that can be encoded from 1902 covariates here)
        if dname == "full":
            dW1920 = dense_dW(cov, NC, HL)
            nov1920 = {g: novelty(dW1920[g], ref[g]) for g in NC}
            nov1920["iso"] = np.zeros(len(cov.cells))
            X1920 = np.hstack([dW1920[g] for g in NC])
            res["designs"][dname]["projection_1920"] = project(est, X1920, B, Bs, coef_ml, coef_sh, nov1920, nov, S,
                                                               dWall_ev, zone_all, zone_ev,
                                                               res["designs"][dname]["reps"])
    np.savez_compressed(os.path.join(a.out, "e072_preds.npz"), **{k.replace("|", "__"): v for k, v in save.items()})
    json.dump(res, open(os.path.join(a.out, "e072_honest.json"), "w"), indent=1, default=float)
    summarize(res)


MODELS = {"M0_posterior": [], "M1_const": ["1"], "M2_cover": ["1", "nov_cover"],
          "M3_novelty": ["1", "nov_climate", "nov_cover", "nov_human"], "M4_eiv": ["1", "U"],
          "M5_eiv+cover": ["1", "U", "nov_cover"], "M6_eiv_only": ["U"],
          "M7_eiv+iso": ["1", "U", "nov_iso"], "M8_iso": ["1", "nov_iso"]}


def lnov(v):
    """Novelty enters as log(1 + nov): it is heavy-tailed (up to ~400x the reference beyond the cover range)."""
    return np.log1p(np.maximum(v, 0.0))


def design_matrix(terms, e2, U, V, S, nov, cells, sp):
    """Cell-level regression on RATIOS OF SUMS over species (in sp): sum_s (e2 - V) / sum_s S, so the fit matches the
    pooled calibration statistic (a mean of per-species ratios is dominated by rare, small-S species)."""
    SS = S[sp].sum()
    y = (e2[np.ix_(cells, sp)] - V[np.ix_(cells, sp)]).sum(1) / SS
    cols = []
    for t in terms:
        if t == "1":
            cols.append(np.ones(len(cells)))
        elif t == "U":
            cols.append(U[np.ix_(cells, sp)].sum(1) / SS)
        else:
            cols.append(lnov(nov[t[4:]][cells]))
    return (np.stack(cols, 1) if cols else np.zeros((len(cells), 0))), y


def predicted_var(terms, gam, U, V, S, nov):
    out = V.copy()
    for t, g in zip(terms, gam):
        if t == "1":
            out = out + g * S[None, :]
        elif t == "U":
            out = out + g * U
        else:
            out = out + g * lnov(nov[t[4:]])[:, None] * S[None, :]
    return out


def evaluate(est, e2, U, V, S, nov, beyond):
    ok_sp = S > 0
    rng = np.random.default_rng(5)
    half = rng.random(len(S)) < 0.5
    terc = np.quantile(nov["cover"], [1 / 3, 2 / 3])
    tbin = np.digitize(nov["cover"], terc)
    allc = np.arange(e2.shape[0])
    out = {}
    for sname in ("sedentary", "all"):
        msp = est.strata[sname] & ok_sp
        o = {}
        for mname, terms in MODELS.items():
            rec = {}
            # (i) species cross-fit on all cells
            Rs = []
            for fit_h in (True, False):
                fsp, tsp = np.where(msp & (half == fit_h))[0], np.where(msp & (half != fit_h))[0]
                if not terms:
                    gam = []
                else:
                    X, y = design_matrix(terms, e2, U, V, S, nov, allc, fsp)
                    gam = nnls(X, y)
                pv = predicted_var(terms, gam, U, V, S, nov)
                Rs.append({"gamma": [float(g) for g in gam],
                           "R_all": float(e2[:, tsp].sum() / pv[:, tsp].sum()),
                           "R_by_cover_tercile": [float(e2[np.ix_(tbin == b, tsp)].sum() / pv[np.ix_(tbin == b, tsp)].sum())
                                                  for b in range(3)],
                           "R_beyond": float(e2[np.ix_(beyond, tsp)].sum() / pv[np.ix_(beyond, tsp)].sum()),
                           "R_inside": float(e2[np.ix_(~beyond, tsp)].sum() / pv[np.ix_(~beyond, tsp)].sum())})
            rec["crossfit"] = Rs
            # (ii) fit on INSIDE cells (all species of the stratum), evaluate on BEYOND cells
            sp = np.where(msp)[0]
            if terms:
                X, y = design_matrix(terms, e2, U, V, S, nov, np.where(~beyond)[0], sp)
                gam = nnls(X, y)
            else:
                gam = []
            pv = predicted_var(terms, gam, U, V, S, nov)
            rec["inside_to_beyond"] = {"gamma": [float(g) for g in gam],
                                       "R_beyond": float(e2[np.ix_(beyond, sp)].sum() / pv[np.ix_(beyond, sp)].sum()),
                                       "R_inside": float(e2[np.ix_(~beyond, sp)].sum() / pv[np.ix_(~beyond, sp)].sum())}
            o[mname] = rec
        o["mean_nov"] = {g: [float(nov[g][~beyond].mean()), float(nov[g][beyond].mean())] for g in nov}
        spi = np.where(msp)[0]
        o["mean_U_over_S"] = [float(U[np.ix_(~beyond, spi)].sum(1).mean() / S[spi].sum()),
                              float(U[np.ix_(beyond, spi)].sum(1).mean() / S[spi].sum())]
        o["mean_e2_over_S"] = [float(e2[np.ix_(~beyond, spi)].sum(1).mean() / S[spi].sum()),
                               float(e2[np.ix_(beyond, spi)].sum(1).mean() / S[spi].sum())]
        out[sname] = o
    return out


def project(est, X1920, B, Bs, coef_ml, coef_sh, nov1920, nov_ev, S, dWall_ev, zone_all, zone_ev, reps_eval):
    """Apply the inside-fitted laws (sedentary) to 1920-39 vs 2005-25 at every cache cell; compare with the held-out
    1966-86 vs 2005-25 change. Units: S (species spatial variance)."""
    out = {}
    msp = est.strata["sedentary"] & (S > 0)
    for tag, coef in (("delta", coef_ml), ("delta_shr", coef_sh)):
        SS = S[msp].sum()
        Ub = (np.stack([(X1920 @ Bb) @ coef[msp].T for Bb in Bs]).var(0, ddof=1).sum(1) / SS)[:, None]  # pooled
        Ue = (np.stack([(dWall_ev @ Bb) @ coef[msp].T for Bb in Bs]).var(0, ddof=1).sum(1) / SS)[:, None]
        pred_change = (((X1920 @ B) @ coef[msp].T) ** 2).sum(1) / SS
        rec = {"median_U_over_S_1920": float(np.median(Ub)), "median_U_over_S_eval": float(np.median(Ue)),
               "median_pred_change2_over_S_1920": float(np.median(pred_change))}
        for mname in ("M2_cover", "M5_eiv+cover", "M3_novelty", "M4_eiv", "M7_eiv+iso"):
            gam = reps_eval[tag]["sedentary"][mname]["inside_to_beyond"]["gamma"]
            terms = MODELS[mname]
            v = np.zeros_like(Ub)
            for t, g in zip(terms, gam):
                v += g * (1.0 if t == "1" else (Ub if t == "U" else lnov(nov1920[t[4:]])[:, None]))
            ve = np.zeros_like(Ue)
            for t, g in zip(terms, gam):
                ve += g * (1.0 if t == "1" else (Ue if t == "U" else lnov(nov_ev[t[4:]])[:, None]))
            zr = {}
            if zone_all is not None:
                for zn in ("west", "great_plains", "east"):
                    zm, ze = zone_all == zn, zone_ev == zn
                    if zm.sum() and ze.sum():
                        zr[zn] = {"var_1920": float(np.median(v[zm])), "var_eval": float(np.median(ve[ze])),
                                  "corr_H1920_H2000": float(1 - np.median(v[zm]) / 2)}
            rec[mname] = {"median_var_1920": float(np.median(v)), "median_var_eval": float(np.median(ve)),
                          "by_zone": zr}
        rec["novelty_1920_median"] = {g: float(np.median(nov1920[g])) for g in nov1920}
        rec["novelty_eval_median"] = {g: float(np.median(nov_ev[g])) for g in nov_ev}
        out[tag] = rec
    return out


def summarize(res):
    for dname, d in res["designs"].items():
        print(f"===== design {dname} (training rows {d['n_rows']})")
        for rname, r in d["reps"].items():
            for sname in ("sedentary", "all"):
                o = r[sname]
                print(f"-- {rname} / {sname}: mean e2/S inside {o['mean_e2_over_S'][0]:.3f} beyond {o['mean_e2_over_S'][1]:.3f};"
                      f" U/S {o['mean_U_over_S'][0]:.4f} / {o['mean_U_over_S'][1]:.4f}; nov(in/beyond) "
                      + " ".join(f"{g} {v[0]:.2f}/{v[1]:.2f}" for g, v in o["mean_nov"].items()))
                for mname in MODELS:
                    cf = o[mname]["crossfit"]
                    ib = o[mname]["inside_to_beyond"]
                    print(f"   {mname:14s} crossfit R {cf[0]['R_all']:.2f}/{cf[1]['R_all']:.2f} terciles "
                          f"{[round(x, 2) for x in cf[0]['R_by_cover_tercile']]} beyond {cf[0]['R_beyond']:.2f} | "
                          f"inside->beyond R_in {ib['R_inside']:.2f} R_beyond {ib['R_beyond']:.2f} gamma "
                          f"{[round(g, 4) for g in ib['gamma']]}")
        if "projection_1920" in d:
            print(f"   1920 projection: {json.dumps(d['projection_1920'], default=float)[:1500]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--similarity", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-caches", nargs="*", default=None)
    ap.add_argument("--n-map-boot", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    run(a)


if __name__ == "__main__":
    main()
