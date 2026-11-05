"""SDM baseline B1: correlative NB2 count SDM on the age model's own Z (pre-registered in
research/experiments/SDM_BASELINE_B1_2026-10-30.md). CPU, NumPy/SciPy only.

    python sdm_b1_sdm.py --design <design_step3 dir> --out <dir>/b1_results.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdm_b0_score import REGIONS, load_design, paired_bootstrap, per_obs_lp, step3_lp, summarise  # noqa: E402

SIGMA_U = 0.97
LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
KNOTS = (1980, 1995, 2010)


def region_of(lon):
    r = np.zeros(lon.shape, int)
    for i, (_, lo, hi) in enumerate(REGIONS):
        r[(lon >= lo) & (lon < hi)] = i
    return r


class Features:
    """Standardised [Z, Z^2] (+ optional region x piecewise-linear year), fitted on training rows."""

    def __init__(self, Zrows, train, year=None, region=None):
        F = np.hstack([Zrows, Zrows ** 2])
        self.mu, self.sd = F[train].mean(0), F[train].std(0) + 1e-9
        self.trend = year is not None

    def __call__(self, Zrows, year=None, region=None):
        F = (np.hstack([Zrows, Zrows ** 2]) - self.mu) / self.sd
        cols = [np.ones((F.shape[0], 1)), F]
        if self.trend:
            for r in range(len(REGIONS)):
                ind = (region == r).astype(float)
                if r > 0:
                    cols.append(ind[:, None])
                tt = [(year - 1966) / 10.0] + [np.maximum(year - k, 0) / 10.0 for k in KNOTS]
                cols.append(np.stack([ind * x for x in tt], 1))
        return np.hstack(cols)


def fit_nb(X, y, w, g, n_g, lam, sigma=SIGMA_U, x0=None):
    """Penalised NB2: -sum w ll + lam/2 |b[1:]|^2 + |u|^2 / (2 sigma^2). Returns b, u, phi."""
    p = X.shape[1]
    has = g >= 0
    gi = np.where(has, g, 0)

    def unpack(th):
        return th[:p], th[p:p + n_g], th[-1]

    def f(th):
        b, u, lphi = unpack(th)
        phi = np.exp(lphi)
        eta = X @ b + np.where(has, u[gi], 0.0)
        eta = np.clip(eta, -30, 15)
        mu = np.exp(eta)
        ll = (gammaln(y + phi) - gammaln(phi) - gammaln(y + 1) + phi * (lphi - np.log(phi + mu))
              + y * (eta - np.log(phi + mu)))
        obj = -(w * ll).sum() + 0.5 * lam * (b[1:] ** 2).sum() + 0.5 * (u ** 2).sum() / sigma ** 2
        de = -w * (y - mu) * phi / (phi + mu)
        gb = X.T @ de
        gb[1:] += lam * b[1:]
        gu = np.bincount(gi[has], de[has], minlength=n_g) + u / sigma ** 2
        gl = -(w * phi * (digamma(y + phi) - digamma(phi) + lphi - np.log(phi + mu) + 1 - (y + phi) / (phi + mu))).sum()
        return obj, np.concatenate([gb, gu, [gl]])

    if x0 is None:
        x0 = np.zeros(p + n_g + 1)
        x0[0] = np.log(max(np.average(y, weights=w), 1e-3))
    r = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 3000, "maxcor": 30})
    b, u, lphi = unpack(r.x)
    return {"b": b, "u": u, "phi": float(np.exp(lphi)), "x": r.x, "converged": bool(r.success),
            "nit": int(r.nit), "msg": str(r.message)}


def predict(design, fit, X):
    g = design["obsroute"]
    has = g >= 0
    mu0 = np.exp(np.clip(X @ fit["b"], -30, 15))
    u = np.zeros(g.shape)
    seen_u = fit["u"]
    u[has] = seen_u[g[has]]
    return mu0 * np.exp(u), mu0


def p0_marginal(mu0, phi, sigma, nodes=24):
    t, w = np.polynomial.hermite.hermgauss(nodes)
    x, wn = np.sqrt(2.0) * t, w / np.sqrt(np.pi)
    return sum(wk * (phi / (phi + mu0 * np.exp(sigma * xk))) ** phi for xk, wk in zip(x, wn))


def front(design, dens, years, lr, lc):
    """front_speed.py re-implemented on the design table: first model year >= thr x median(2010-25, ok cells)."""
    lut = -np.ones((lr.max() + 1, lc.max() + 1), int); lut[lr, lc] = np.arange(len(lr))
    cell = lut[design["row"], design["col"]]
    real = design["obsroute"] >= 0
    nl = len(lr)
    first_obs = np.full(nl, np.inf); surveyed = np.full(nl, np.inf)
    np.minimum.at(surveyed, cell[real], design["year"][real])
    pos = real & (design["y"] > 0)
    np.minimum.at(first_obs, cell[pos], design["year"][pos])
    first_obs[np.isinf(first_obs)] = np.nan; surveyed[np.isinf(surveyed)] = np.nan
    east = (design["land_lon"] >= -95) & (design["land_lat"] >= 25) & (design["land_lat"] < 49)
    ok = east & np.isfinite(first_obs) & (first_obs >= surveyed + 3)
    res = {"n_cells": int(ok.sum())}
    for name, thr in (("1pct", .01), ("10pct", .10)):
        occ = dens >= thr * np.nanmedian(dens[years >= 2010][:, ok])
        fm = np.where(occ.any(0), years[np.argmax(occ, 0)], np.nan)
        lag = fm[ok] - first_obs[ok]
        bins = [1966, 1975, 1985, 1995, 2010]
        res[name] = {"median": float(np.nanmedian(lag)), "never": float(np.mean(np.isnan(fm[ok]))),
                     "by_arrival": {f"{a}-{b - 1}": float(np.nanmedian(lag[(first_obs[ok] >= a) & (first_obs[ok] < b)]))
                                    for a, b in zip(bins[:-1], bins[1:])}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="*", default=["dyn", "static", "native", "dyn+yr", "dyn+pz"])
    a = ap.parse_args()
    D = load_design(a.design)
    Z = np.load(os.path.join(a.design, "z_cells.npy")).astype(np.float64)       # (time, land, M)
    years = D["years"]
    lr, lc = D["land_rows"], D["land_cols"]
    lut = -np.ones((lr.max() + 1, lc.max() + 1), int); lut[lr, lc] = np.arange(len(lr))
    cell = lut[D["row"], D["col"]]
    ti = D["t"]
    y, w = D["y"], D["weight"]
    region = region_of(D["lon"])
    land_region = region_of(D["land_lon"])
    real_train = ~D["holdout"] & ~D["pseudo"] & (w > 0)
    blocks = (D["row"] // 4) * 1000 + D["col"] // 4
    win = (years >= 1966) & (years <= 2025)
    Zstatic = Z[win].mean(0)                                                          # (land, M)
    pr = np.load(os.path.join(a.design, "step3_pred.npz")); pred3 = {k: pr[k] for k in pr.files}
    lp3 = step3_lp(D, pred3)
    T, S = D["temporal"], D["spatial"]
    Sr = S & ~D["pseudo"]
    n_g = int(D["obsroute"].max()) + 1
    out = {"age_step3": {"summary": summarise(D, lp3, pred3["mu"])}, "arms": {}}

    # age model front (validation of the front code against front_speed.json) and early-east lambda analogue
    dens3 = np.load(os.path.join(a.design, "step3_density.npy")).astype(np.float64)
    out["age_step3"]["front"] = front(D, dens3, years, lr, lc)
    print("age step3 front (should match front_speed.json fitted -8 / -4):", out["age_step3"]["front"], flush=True)
    obs_real_ho = {"T": T, "S_real": Sr}
    p0_3 = p0_marginal(pred3["mu"] * np.exp(-pred3["eps_part"]), pred3["phi"], float(pred3["sigma_u"]))
    out["age_step3"]["p0"] = {s: {rn: {"obs": float((y[m & (region == i)] == 0).mean()), "pred": float(p0_3[m & (region == i)].mean())}
                                  for i, (rn, _, _) in enumerate(REGIONS) if (m & (region == i)).sum() > 50}
                              for s, m in obs_real_ho.items()}

    for arm in a.arms:
        zrows = Zstatic[cell] if arm == "static" else Z[ti, cell]
        train = real_train.copy()
        if arm == "native":
            train &= D["lon"] < -105
        if arm == "dyn+pz":
            train = ~D["holdout"] & (w > 0)
        trend = arm == "dyn+yr"
        feat = Features(zrows, train, year=D["year"] if trend else None, region=region if trend else None)
        X = feat(zrows, D["year"].astype(float), region) if trend else feat(zrows)
        # inner split: 20% of training 216 km blocks
        ub = np.unique(blocks[train])
        rng = np.random.default_rng(0)
        vb = rng.choice(ub, size=max(1, int(.2 * ub.size)), replace=False)
        iv = train & np.isin(blocks, vb); it = train & ~iv
        tune = []
        x0 = None
        for lam in LAMBDAS:
            ft = fit_nb(X[it], y[it], w[it], np.where(D["obsroute"][it] >= 0, D["obsroute"][it], -1), n_g, lam)
            mu0 = np.exp(np.clip(X[iv] @ ft["b"], -30, 15))
            from sdm_b0_score import integrated_lp
            lpv = integrated_lp(y[iv], mu0, ft["phi"], SIGMA_U)
            tune.append({"lam": lam, "inner_marg_lpd": float(lpv.mean()), "phi": ft["phi"], "converged": ft["converged"], "nit": ft["nit"]})
            print(arm, tune[-1], flush=True)
        best = max(tune, key=lambda r: r["inner_marg_lpd"])
        fit = fit_nb(X[train], y[train], w[train], np.where(D["obsroute"][train] >= 0, D["obsroute"][train], -1), n_g, best["lam"])
        mu_plug, mu0 = predict(D, fit, X)
        lp = per_obs_lp(D, mu_plug, fit["phi"], mu_marg=mu0, sigma=SIGMA_U)
        summ = summarise(D, lp, mu_plug)
        paired = {}
        for name, m, k3, kb in (("T_marg", T, "marg", "marg"), ("T_cond", T, "cond", "cond"),
                                ("S_all_marg", S, "marg", "marg"), ("S_real_marg", Sr, "marg", "marg")):
            paired[name] = paired_bootstrap(D, lp3[k3] - lp[kb], m)
        for i, (rn, _, _) in enumerate(REGIONS):
            for name, m, k in (("S_real_marg", Sr, "marg"), ("T_marg", T, "marg")):
                mm = m & (region == i)
                if mm.sum() > 100:
                    paired[f"{name}|{rn}"] = paired_bootstrap(D, lp3[k] - lp[k], mm)
        for rn, e0, e1 in (("east_1966-93", 1966, 1993), ("east_1994-2020", 1994, 2020)):
            mm = Sr & (region == 3) & (D["year"] >= e0) & (D["year"] <= e1)
            if mm.sum() > 100:
                paired[f"S_real_marg|{rn}"] = paired_bootstrap(D, lp3["marg"] - lp["marg"], mm)
        # structural readouts: expected count at u = 0 for every land cell and year (Z-driven)
        if arm == "static":
            Xc = feat(Zstatic)
            dens = np.broadcast_to(np.exp(np.clip(Xc @ fit["b"], -30, 15)), (len(years), len(lr)))
        else:
            dens = np.empty((len(years), len(lr)))
            for t_ in range(len(years)):
                if trend:
                    Xc = feat(Z[t_], np.full(len(lr), float(years[t_])), land_region)
                else:
                    Xc = feat(Z[t_])
                dens[t_] = np.exp(np.clip(Xc @ fit["b"], -30, 15))
        band = (D["land_lon"] >= -80) & (D["land_lat"] >= 25) & (D["land_lat"] < 49)
        ce = (D["land_lon"] >= -95) & (D["land_lon"] < -80) & (D["land_lat"] >= 25) & (D["land_lat"] < 49)
        def era(sel, a0, a1):
            return float(dens[(years >= a0) & (years <= a1)][:, sel].mean())
        struct = {"east80_1940-55": era(band, 1940, 1955), "east80_1966-75": era(band, 1966, 1975),
                  "east80_2010-20": era(band, 2010, 2020), "central_east_1966-75": era(ce, 1966, 1975),
                  "central_east_2010-20": era(ce, 2010, 2020), "all_1966-75": era(np.ones(len(lr), bool), 1966, 1975),
                  "all_2010-20": era(np.ones(len(lr), bool), 2010, 2020)}
        struct["EARLY_EAST_RATIO_1966-75_over_2010-20"] = struct["east80_1966-75"] / struct["east80_2010-20"]
        struct["east80_1940-55_over_2010-20"] = struct["east80_1940-55"] / struct["east80_2010-20"]
        struct["central_east_ratio"] = struct["central_east_1966-75"] / struct["central_east_2010-20"]
        west = D["land_lon"] < -105
        struct["west_1966-75"] = era(west, 1966, 1975); struct["west_2010-20"] = era(west, 2010, 2020)
        struct["west_ratio"] = struct["west_1966-75"] / struct["west_2010-20"]
        p0 = p0_marginal(mu0, fit["phi"], SIGMA_U)
        out["arms"][arm] = {"tuning": tune, "lambda": best["lam"], "phi": fit["phi"], "converged": fit["converged"],
                            "nit": fit["nit"], "msg": fit["msg"], "n_train": int(train.sum()), "summary": summ,
                            "paired_age_minus_b1": paired, "structure": struct, "front": front(D, dens, years, lr, lc),
                            "p0": {s: {rn: {"obs": float((y[m & (region == i)] == 0).mean()), "pred": float(p0[m & (region == i)].mean())}
                                       for i, (rn, _, _) in enumerate(REGIONS) if (m & (region == i)).sum() > 50}
                                   for s, m in obs_real_ho.items()}}
        s_ = summ
        print(f"ARM {arm}: lam {best['lam']} phi {fit['phi']:.3f} conv {fit['converged']} | T marg {s_['T']['marg']:.4f} "
              f"cond {s_['T']['cond']:.4f} | S_all {s_['S_all']['marg']:.4f} S_real {s_['S_real']['marg']:.4f} | "
              f"early-east ratio {struct['EARLY_EAST_RATIO_1966-75_over_2010-20']:.3f} | front {out['arms'][arm]['front']['1pct']['median']}",
              flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
