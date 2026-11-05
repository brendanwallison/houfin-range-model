"""SDM baseline B2: dispersal-limited SDM (MigClim class, expected-occupancy automaton) on B1-static and B1-dyn
suitability. Pre-registered in research/experiments/SDM_BASELINE_B2_2026-10-30.md. CPU, NumPy/SciPy.

    python sdm_b2_migsdm.py --design <design_step3> --bbs-npz <grid54 bbs npz> --metadata <model input metadata.pkl> --out <json>
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdm_b0_score import REGIONS, load_design, nb_lp, paired_bootstrap, per_obs_lp, step3_lp, summarise  # noqa: E402
from sdm_b1_sdm import Features, fit_nb, front, predict, region_of  # noqa: E402

SIGMA_U = 0.97
A_GRID = (30.0, 60.0, 120.0)
L_KM = 400.0


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class Automaton:
    def __init__(self, Dkm, years, native, release, release_year=1940):
        self.Kl = np.exp(-Dkm / L_KM).astype(np.float32); np.fill_diagonal(self.Kl, 0.0)
        self.Ks = {}
        for a in A_GRID:
            k = np.exp(-Dkm / a).astype(np.float32); np.fill_diagonal(k, 0.0); self.Ks[a] = k
        self.years, self.native, self.release = years, native, release
        self.t_rel = int(np.searchsorted(years, release_year))

    def run(self, S, a, r, q, S0, e0):
        nT, nL = S.shape
        P = np.zeros((nT, nL), np.float32)
        p = np.zeros(nL, np.float32); p[self.native] = 1.0
        Ks = self.Ks[a]
        for t in range(nT):
            if t == self.t_rel:
                p[self.release] = 1.0
            P[t] = p
            Est = 1.0 - np.exp(-S[t] / S0)
            press = r * ((1 - q) * (Ks @ p) + q * (self.Kl @ p))
            col = (1.0 - np.exp(-press)) * Est
            ext = e0 * np.exp(-S[t] / S0)
            p = (p * (1.0 - ext) + (1.0 - p) * col).astype(np.float32)
        return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--bbs-npz", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--seed-mode", default="native+release", choices=["native+release", "release"],
                    help="B6: release-only seeding to match a release-only synthetic truth")
    a_ = ap.parse_args()
    D = load_design(a_.design)
    Z = np.load(os.path.join(a_.design, "z_cells.npy")).astype(np.float64)
    years = D["years"]; nT, nL, M = Z.shape
    lr, lc = D["land_rows"], D["land_cols"]
    lut = -np.ones((lr.max() + 1, lc.max() + 1), int); lut[lr, lc] = np.arange(nL)
    cell = lut[D["row"], D["col"]]
    b = np.load(a_.bbs_npz)
    ish = b["initpop_shape"]
    native = np.where(ish[lr, lc] > 0)[0] if a_.seed_mode == "native+release" else np.array([], int)
    meta = pickle.load(open(a_.metadata, "rb"))
    inv = np.asarray(meta["inv_locations"])
    release = np.array([lut[r_, c_] for r_, c_ in inv if lut[r_, c_] >= 0])
    lat, lon = np.radians(D["land_lat"]), np.radians(D["land_lon"])
    Dkm = 6371.0 * np.arccos(np.clip(np.sin(lat)[:, None] * np.sin(lat)[None] +
                                     np.cos(lat)[:, None] * np.cos(lat)[None] * np.cos(lon[:, None] - lon[None]), -1, 1))
    auto = Automaton(Dkm, years, native, release)
    print(f"native seed cells {native.size}, release cells {release.tolist()}", flush=True)

    real_train = ~D["holdout"] & ~D["pseudo"] & (D["weight"] > 0)
    g = np.where(D["obsroute"] >= 0, D["obsroute"], -1)
    n_g = int(D["obsroute"].max()) + 1
    T, S_ = D["temporal"], D["spatial"]; Sr = S_ & ~D["pseudo"]
    region = region_of(D["lon"])
    pr = np.load(os.path.join(a_.design, "step3_pred.npz")); pred3 = {k: pr[k] for k in pr.files}
    lp3 = step3_lp(D, pred3)
    y, w = D["y"], D["weight"]
    out = {"native_cells": int(native.size), "release_cells": release.tolist(), "arms": {}}
    east95 = D["land_lon"] >= -95

    for arm, lam in (("static", 100.0), ("dyn", 1000.0)):
        t0 = time.time()
        zrows = Z[:, :, :][D["t"], cell] if arm == "dyn" else Z[(years >= 1966) & (years <= 2025)].mean(0)[cell]
        feat = Features(zrows, real_train)
        X = feat(zrows)
        fit = fit_nb(X[real_train], y[real_train], w[real_train], g[real_train], n_g, lam)
        mu_b1, mu0_b1 = predict(D, fit, X)
        lp_b1 = per_obs_lp(D, mu_b1, fit["phi"], mu_marg=mu0_b1, sigma=SIGMA_U)
        uexp = mu_b1 / mu0_b1                                   # fitted observer effects (1 for unseen / pseudo)
        if arm == "static":
            Sc = np.exp(np.clip(feat(Z[(years >= 1966) & (years <= 2025)].mean(0)) @ fit["b"], -30, 15))
            Sgrid = np.broadcast_to(Sc, (nT, nL)).astype(np.float32).copy()
        else:
            Sgrid = np.stack([np.exp(np.clip(feat(Z[t]) @ fit["b"], -30, 15)) for t in range(nT)]).astype(np.float32)
        tr = real_train
        tt, tc = D["t"][tr], cell[tr]
        ytr, wtr, utr = y[tr], w[tr], uexp[tr]
        Smed = float(np.median(Sgrid[-20:]))

        def unpack(th):
            return dict(r=float(np.exp(th[0])), q=float(sig(th[1])), S0=float(np.exp(th[2])), e0=float(sig(th[3])),
                        m=float(np.exp(th[4])), phi=float(np.exp(th[5])))

        def nll(th, a):
            pp = unpack(th)
            P = auto.run(Sgrid, a, pp["r"], pp["q"], pp["S0"], pp["e0"])
            mu = np.maximum(pp["m"] * P[tt, tc] * Sgrid[tt, tc] * utr, 1e-6)
            return -float((wtr * nb_lp(ytr, mu, pp["phi"])).sum()) / wtr.sum()

        # REVISION (after a degenerate first fit: from r = 1 every cell saturates by ~1908 and the likelihood is flat
        # in the spread parameters): coarse grid over (a, r, q) with (m, phi) profiled, then Nelder-Mead from the
        # best 3 grid points. Rules unchanged.
        def prof_mp(P):
            base = np.maximum(P[tt, tc] * Sgrid[tt, tc] * utr, 1e-9)
            f = lambda v: -float((wtr * nb_lp(ytr, np.maximum(np.exp(v[0]) * base, 1e-6), np.exp(v[1]))).sum()) / wtr.sum()
            rr = minimize(f, np.array([0.0, np.log(1.5)]), method="Nelder-Mead", options={"maxiter": 200})
            return rr.x, -float(rr.fun)
        grid = []
        S0_0, e0_0 = Smed / 3, .01
        for a in A_GRID:
            for r in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
                for q in (0.0, 0.02, 0.1):
                    P = auto.run(Sgrid, a, r, q, S0_0, e0_0)
                    (lm, lphi), lpd = prof_mp(P)
                    grid.append({"a": a, "r": r, "q": q, "lm": float(lm), "lphi": float(lphi), "lpd": lpd})
        grid.sort(key=lambda g_: -g_["lpd"])
        print(arm, "grid top 5", [(g_["a"], g_["r"], g_["q"], round(g_["lpd"], 4)) for g_ in grid[:5]],
              "grid worst", round(grid[-1]["lpd"], 4), f"{time.time() - t0:.0f}s", flush=True)
        best = None
        for g0 in grid[:3]:
            a = g0["a"]
            th0 = np.array([np.log(g0["r"]), np.log(max(g0["q"], 1e-3) / (1 - max(g0["q"], 1e-3))), np.log(S0_0),
                            np.log(e0_0 / (1 - e0_0)), g0["lm"], g0["lphi"]])
            r_ = minimize(nll, th0, args=(a,), method="Nelder-Mead",
                          options={"maxiter": a_.maxiter, "xatol": 1e-3, "fatol": 1e-5})
            rec = {"a_km": a, "start": {k: g0[k] for k in ("a", "r", "q")}, "train_mean_lpd": -float(r_.fun),
                   "nit": int(r_.nit), "params": unpack(r_.x), "msg": str(r_.message)}
            print(arm, rec, f"{time.time() - t0:.0f}s", flush=True)
            if best is None or rec["train_mean_lpd"] > best["train_mean_lpd"]:
                best = dict(rec, th=r_.x)
        best["grid_top10"] = grid[:10]
        pp = best["params"]; a = best["a_km"]
        P = auto.run(Sgrid, a, pp["r"], pp["q"], pp["S0"], pp["e0"])
        mu0 = np.maximum(pp["m"] * P[D["t"], cell] * Sgrid[D["t"], cell], 1e-6)
        mu = mu0 * uexp
        lp = per_obs_lp(D, mu, pp["phi"], mu_marg=mu0, sigma=SIGMA_U)
        summ = summarise(D, lp, mu)
        paired = {}
        for name, m_, k3 in (("T_marg", T, "marg"), ("T_cond", T, "cond"), ("S_real_marg", Sr, "marg"), ("S_all_marg", S_, "marg")):
            paired[f"age_minus_b2|{name}"] = paired_bootstrap(D, lp3[k3] - lp[k3], m_)
            paired[f"b2_minus_b1|{name}"] = paired_bootstrap(D, lp[k3] - lp_b1[k3], m_)
        for i, (rn, _, _) in enumerate(REGIONS):
            mm = Sr & (region == i)
            paired[f"age_minus_b2|S_real_marg|{rn}"] = paired_bootstrap(D, lp3["marg"] - lp["marg"], mm)
            paired[f"b2_minus_b1|S_real_marg|{rn}"] = paired_bootstrap(D, lp["marg"] - lp_b1["marg"], mm)
        dens = pp["m"] * P.astype(np.float64) * Sgrid
        fr = front(D, dens, years, lr, lc)
        fr_occ = front(D, P.astype(np.float64), years, lr, lc)        # thresholds relative to modern median p
        occ_first = np.where((P >= .5).any(0), years[np.argmax(P >= .5, 0)], np.nan)
        # occupancy front with an absolute .5 threshold, same eastern cell set
        real = D["obsroute"] >= 0
        first_obs = np.full(nL, np.inf); surveyed = np.full(nL, np.inf)
        np.minimum.at(surveyed, cell[real], D["year"][real])
        pos = real & (y > 0); np.minimum.at(first_obs, cell[pos], D["year"][pos])
        first_obs[np.isinf(first_obs)] = np.nan; surveyed[np.isinf(surveyed)] = np.nan
        ok = (D["land_lon"] >= -95) & (D["land_lat"] >= 25) & (D["land_lat"] < 49) & np.isfinite(first_obs) & (first_obs >= surveyed + 3)
        lag = occ_first[ok] - first_obs[ok]
        bins = [1966, 1975, 1985, 1995, 2010]
        occ_front = {"median": float(np.nanmedian(lag)), "never": float(np.mean(np.isnan(occ_first[ok]))),
                     "by_arrival": {f"{x}-{z - 1}": float(np.nanmedian(lag[(first_obs[ok] >= x) & (first_obs[ok] < z)])) for x, z in zip(bins[:-1], bins[1:])}}
        area = P[:, east95].sum(1)
        mil = {str(k): (int(years[np.argmax(area >= k)]) if (area >= k).any() else None) for k in (10, 100, 500)}
        out["arms"][arm] = {"b1_lambda": lam, "fit": {k: v for k, v in best.items() if k != "th"}, "summary": summ,
                            "paired": paired, "front_density": fr, "front_occupancy_p50": occ_front,
                            "east95_occupied_cells_reach": mil}
        print(f"ARM B2-{arm}: a {a} params {pp} | T marg {summ['T']['marg']:.4f} cond {summ['T']['cond']:.4f} | "
              f"S_real {summ['S_real']['marg']:.4f} | front10 {fr['10pct']['median']} {fr['10pct']['by_arrival']} | "
              f"front1 {fr['1pct']['median']} | occ50 {occ_front['median']} {occ_front['by_arrival']} | area {mil}", flush=True)
        json.dump(out, open(a_.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
