"""B2-Allee: dispersal-limited SDM with an establishment Allee term and a maturation lag, on STATIC suitability
(pre-registered in research/experiments/SDM_BASELINE_KISSMIG_B2ALLEE_2026-10-30.md sec. 3). CPU only.

    python sdm_b2_allee.py --design <design_step3> --bbs-npz <npz> --metadata <pkl> --out <json>
"""
import argparse
import itertools
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
from sdm_b2_migsdm import A_GRID, L_KM, sig  # noqa: E402

SIGMA_U = 0.97


class AlleeAutomaton:
    def __init__(self, Dkm, years, native, release, release_year=1940):
        self.Kl = np.exp(-Dkm / L_KM).astype(np.float32); np.fill_diagonal(self.Kl, 0.0)
        self.Ks = {}
        for a in A_GRID:
            k = np.exp(-Dkm / a).astype(np.float32); np.fill_diagonal(k, 0.0); self.Ks[a] = k
        self.years, self.native, self.release = years, native, release
        self.t_rel = int(np.searchsorted(years, release_year))

    def run(self, S, a, r, q, S0, e0, c50, h, L):
        nT, nL = S.shape
        P = np.zeros((nT, nL), np.float32)
        p = np.zeros(nL, np.float32); p[self.native] = 1.0
        Ks = self.Ks[a]
        for t in range(nT):
            if t == self.t_rel:
                p[self.release] = 1.0
            P[t] = p
            src = P[max(t - L, 0)] if L > 0 else p              # only cells occupied >= L years ago send propagules
            press = r * ((1 - q) * (Ks @ src) + q * (self.Kl @ src))
            allee = press ** h / (press ** h + c50 ** h)
            col = (1.0 - np.exp(-press)) * allee * (1.0 - np.exp(-S[t] / S0))
            ext = e0 * np.exp(-S[t] / S0)
            p = (p * (1.0 - ext) + (1.0 - p) * col).astype(np.float32)
        return P


def main():
    ap = argparse.ArgumentParser()
    for k in ("--design", "--bbs-npz", "--metadata", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--maxiter", type=int, default=300)
    a_ = ap.parse_args()
    t0 = time.time()
    D = load_design(a_.design)
    Z = np.load(os.path.join(a_.design, "z_cells.npy")).astype(np.float64)
    years = D["years"]; nT, nL, M = Z.shape
    lr, lc = D["land_rows"], D["land_cols"]
    lut = -np.ones((lr.max() + 1, lc.max() + 1), int); lut[lr, lc] = np.arange(nL)
    cell = lut[D["row"], D["col"]]
    b = np.load(a_.bbs_npz)
    native = np.where(b["initpop_shape"][lr, lc] > 0)[0]
    inv = np.asarray(pickle.load(open(a_.metadata, "rb"))["inv_locations"])
    release = np.array([lut[r_, c_] for r_, c_ in inv if lut[r_, c_] >= 0])
    lat, lon = np.radians(D["land_lat"]), np.radians(D["land_lon"])
    Dkm = 6371.0 * np.arccos(np.clip(np.sin(lat)[:, None] * np.sin(lat)[None] +
                                     np.cos(lat)[:, None] * np.cos(lat)[None] * np.cos(lon[:, None] - lon[None]), -1, 1))
    auto = AlleeAutomaton(Dkm, years, native, release)
    real_train = ~D["holdout"] & ~D["pseudo"] & (D["weight"] > 0)
    g = np.where(D["obsroute"] >= 0, D["obsroute"], -1); n_g = int(D["obsroute"].max()) + 1
    T, S_ = D["temporal"], D["spatial"]; Sr = S_ & ~D["pseudo"]
    region = region_of(D["lon"])
    y, w = D["y"], D["weight"]
    win = (years >= 1966) & (years <= 2025)
    zrows = Z[win].mean(0)[cell]
    feat = Features(zrows, real_train)
    X = feat(zrows)
    fit = fit_nb(X[real_train], y[real_train], w[real_train], g[real_train], n_g, 100.0)
    mu_b1, mu0_b1 = predict(D, fit, X)
    lp_b1 = per_obs_lp(D, mu_b1, fit["phi"], mu_marg=mu0_b1, sigma=SIGMA_U)
    uexp = mu_b1 / mu0_b1
    Sc = np.exp(np.clip(feat(Z[win].mean(0)) @ fit["b"], -30, 15))
    Sgrid = np.broadcast_to(Sc, (nT, nL)).astype(np.float32).copy()
    tr = real_train; tt, tc = D["t"][tr], cell[tr]; ytr, wtr, utr = y[tr], w[tr], uexp[tr]
    Smed = float(np.median(Sc))
    print(f"B1-static refit phi {fit['phi']:.3f}; {time.time() - t0:.0f}s", flush=True)

    def prof_mp(P):
        base = np.maximum(P[tt, tc] * Sgrid[tt, tc] * utr, 1e-9)
        f = lambda v: -float((wtr * nb_lp(ytr, np.maximum(np.exp(v[0]) * base, 1e-6), np.exp(v[1]))).sum()) / wtr.sum()
        rr = minimize(f, np.array([0.0, np.log(1.5)]), method="Nelder-Mead", options={"maxiter": 200})
        return rr.x, -float(rr.fun)

    S0_0, e0_0 = Smed / 3, .01
    grid = []
    for a, r, q, c50, h, L in itertools.product(A_GRID, (0.01, 0.03, 0.1, 0.3, 1.0), (0.0, 0.05), (0.03, 0.3), (2.0, 4.0), (0, 3, 6)):
        P = auto.run(Sgrid, a, r, q, S0_0, e0_0, c50, h, L)
        (lm, lphi), lpd = prof_mp(P)
        grid.append(dict(a=a, r=r, q=q, c50=c50, h=h, L=L, lm=float(lm), lphi=float(lphi), lpd=lpd))
    grid.sort(key=lambda g_: -g_["lpd"])
    print("grid top 5", [{k: g_[k] for k in ("a", "r", "q", "c50", "h", "L", "lpd")} for g_ in grid[:5]],
          f"{time.time() - t0:.0f}s", flush=True)

    def unpack(th):
        return dict(r=float(np.exp(th[0])), q=float(sig(th[1])), c50=float(np.exp(th[2])), h=float(1 + 5 * sig(th[3])),
                    S0=float(np.exp(th[4])), e0=float(sig(th[5])), m=float(np.exp(th[6])), phi=float(np.exp(th[7])))

    def nll(th, a, L):
        pp = unpack(th)
        P = auto.run(Sgrid, a, pp["r"], pp["q"], pp["S0"], pp["e0"], pp["c50"], pp["h"], L)
        mu = np.maximum(pp["m"] * P[tt, tc] * Sgrid[tt, tc] * utr, 1e-6)
        return -float((wtr * nb_lp(ytr, mu, pp["phi"])).sum()) / wtr.sum()

    best = None
    for g0 in grid[:3]:
        qq = min(max(g0["q"], 1e-3), .999); hh = (g0["h"] - 1) / 5
        th0 = np.array([np.log(g0["r"]), np.log(qq / (1 - qq)), np.log(g0["c50"]), np.log(hh / (1 - hh)),
                        np.log(S0_0), np.log(e0_0 / (1 - e0_0)), g0["lm"], g0["lphi"]])
        r_ = minimize(nll, th0, args=(g0["a"], g0["L"]), method="Nelder-Mead",
                      options={"maxiter": a_.maxiter, "xatol": 1e-3, "fatol": 1e-5})
        rec = {"a_km": g0["a"], "L": g0["L"], "start": {k: g0[k] for k in ("r", "q", "c50", "h")},
               "train_mean_lpd": -float(r_.fun), "nit": int(r_.nit), "params": unpack(r_.x), "msg": str(r_.message)}
        print(rec, f"{time.time() - t0:.0f}s", flush=True)
        if best is None or rec["train_mean_lpd"] > best["train_mean_lpd"]:
            best = rec
    pp, a, L = best["params"], best["a_km"], best["L"]
    P = auto.run(Sgrid, a, pp["r"], pp["q"], pp["S0"], pp["e0"], pp["c50"], pp["h"], L)
    mu0 = np.maximum(pp["m"] * P[D["t"], cell] * Sgrid[D["t"], cell], 1e-6)
    lp = per_obs_lp(D, mu0 * uexp, pp["phi"], mu_marg=mu0, sigma=SIGMA_U)
    pr = np.load(os.path.join(a_.design, "step3_pred.npz")); pred3 = {k: pr[k] for k in pr.files}
    lp3 = step3_lp(D, pred3)
    summ = summarise(D, lp, mu0 * uexp)
    paired = {}
    for name, m_, k in (("T_marg", T, "marg"), ("T_cond", T, "cond"), ("S_real_marg", Sr, "marg")):
        paired[f"age_minus|{name}"] = paired_bootstrap(D, lp3[k] - lp[k], m_)
        paired[f"minus_b1static|{name}"] = paired_bootstrap(D, lp[k] - lp_b1[k], m_)
    for i, (rn, _, _) in enumerate(REGIONS):
        mm = Sr & (region == i)
        paired[f"age_minus|S_real_marg|{rn}"] = paired_bootstrap(D, lp3["marg"] - lp["marg"], mm)
    fr = front(D, pp["m"] * P.astype(np.float64) * Sgrid, years, lr, lc)
    real = D["obsroute"] >= 0
    first_obs = np.full(nL, np.inf); surveyed = np.full(nL, np.inf)
    np.minimum.at(surveyed, cell[real], D["year"][real])
    pos = real & (y > 0); np.minimum.at(first_obs, cell[pos], D["year"][pos])
    first_obs[np.isinf(first_obs)] = np.nan; surveyed[np.isinf(surveyed)] = np.nan
    ok = (D["land_lon"] >= -95) & (D["land_lat"] >= 25) & (D["land_lat"] < 49) & np.isfinite(first_obs) & (first_obs >= surveyed + 3)
    occ_first = np.where((P >= .5).any(0), years[np.argmax(P >= .5, 0)], np.nan)
    lag = occ_first[ok] - first_obs[ok]
    bins = [1966, 1975, 1985, 1995, 2010]
    occ_front = {"median": float(np.nanmedian(lag)), "never": float(np.mean(np.isnan(occ_first[ok]))),
                 "by_arrival": {f"{x}-{z - 1}": float(np.nanmedian(lag[(first_obs[ok] >= x) & (first_obs[ok] < z)]))
                                for x, z in zip(bins[:-1], bins[1:])}}
    area = P[:, D["land_lon"] >= -95].sum(1)
    out = {"fit": best, "grid_top10": grid[:10], "summary": summ, "paired": paired, "front_density": fr,
           "front_occupancy_p50": occ_front,
           "east95_area_reach": {str(k): (int(years[np.argmax(area >= k)]) if (area >= k).any() else None) for k in (10, 100, 500)}}
    json.dump(out, open(a_.out, "w"), indent=1)
    print(f"B2-Allee: a {a} L {L} params {pp} | T marg {summ['T']['marg']:.4f} | S_real {summ['S_real']['marg']:.4f} | "
          f"front10 {fr['10pct']['median']} {fr['10pct']['by_arrival']} | front1 {fr['1pct']['median']} | "
          f"occ50 {occ_front['median']} {occ_front['by_arrival']} | area {out['east95_area_reach']}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
