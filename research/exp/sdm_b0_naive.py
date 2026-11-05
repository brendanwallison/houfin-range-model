"""SDM baseline plan B0: naive forecasts on the age model's held-out sets (pre-registered in
research/experiments/SDM_BASELINE_B0_2026-10-30.md). CPU, NumPy/pandas/SciPy only.

    python sdm_b0_naive.py --design <dir> --out <dir>/b0_results.json

Temporal: persistence keyed by observer-route / route / cell (shrinkage chain), optional 324 km tile trend; tuned on
an inner forecast from 2015 to training rows 2016-19. Spatial: Gaussian-kernel interpolation (S-IDW) and tile mean
(S-tile) of training cell-year counts; tuned by leave-one-block-out on training cells. Every NB concentration is fitted
by maximum likelihood on the inner (training-only) predictions, then frozen.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdm_b0_score import load_design, nb_lp, paired_bootstrap, step3_lp, summarise  # noqa: E402

TILE = 6            # 6 x 54 km = 324 km trend / tile regions
BLOCK = 4           # 4 x 54 km = 216 km held-out blocks
MU_FLOOR = 1e-4
_KCACHE, _TCACHE = {}, {}


def fit_phi(y, mu):
    f = lambda lp: -nb_lp(y, mu, np.exp(lp)).sum()
    r = minimize_scalar(f, bounds=(-6, 6), method="bounded")
    return float(np.exp(r.x)), float(-r.fun / y.size)


# ---------------------------------------------------------------------------------------------- temporal
def level_table(df, key, k):
    """Per key: mean count, n rows and mean year over the key's last k distinct surveyed years (df = history)."""
    d = df[df[key] >= 0][[key, "year", "y"]]
    yrs = d[[key, "year"]].drop_duplicates().sort_values([key, "year"], ascending=[True, False])
    yrs["rk"] = yrs.groupby(key).cumcount()
    keep = yrs[yrs["rk"] < k][[key, "year"]]
    d = d.merge(keep, on=[key, "year"])
    g = d.groupby(key).agg(m=("y", "mean"), n=("y", "size"), ybar=("year", "mean"))
    return g


def kernel_window_mean(design, hist_mask, years_win, h_km, cell_index, Dkm):
    """Per land cell: Gaussian-kernel mean count of history rows in years_win (cell aggregates)."""
    m = hist_mask & np.isin(design["year"], years_win)
    nl = Dkm.shape[0]
    S = np.bincount(cell_index[m], design["y"][m], minlength=nl)
    N = np.bincount(cell_index[m], minlength=nl).astype(float)
    if h_km not in _KCACHE:
        K = np.exp(-0.5 * (Dkm / h_km) ** 2)
        K[Dkm > 3 * h_km] = 0.0
        _KCACHE[h_km] = K
    K = _KCACHE[h_km]
    KS, KN = K @ S, K @ N
    glob = S.sum() / max(N.sum(), 1)
    return np.where(KN > 1e-3, (KS + 0.01 * glob) / (KN + 0.01), glob)


def tile_trends(design, hist_mask, H, tiles_of_obs):
    """Per 324 km tile: Poisson log-linear slope over H-9..H with route fixed effects (profile likelihood grid)."""
    m = hist_mask & (design["year"] >= H - 9) & (design["year"] <= H) & (design["route"] >= 0)
    df = pd.DataFrame({"tile": tiles_of_obs[m], "route": design["route"][m], "t": design["year"][m] - H,
                       "y": design["y"][m]})
    grid = np.linspace(-0.15, 0.15, 61)
    out = {}
    for tile, d in df.groupby("tile"):
        if len(d) < 20 or d["route"].nunique() < 3 or d["y"].sum() < 10:
            continue
        t, y = d["t"].to_numpy(float), d["y"].to_numpy(float)
        codes, uniq = pd.factorize(d["route"])
        Yg = np.bincount(codes, y)
        ll = []
        for b in grid:
            Eg = np.bincount(codes, np.exp(b * t))
            ll.append(b * (y * t).sum() - (Yg[Yg > 0] * np.log(Eg[Yg > 0])).sum())
        out[tile] = float(grid[int(np.argmax(ll))])
    return out


def predict_temporal(design, ctx, H, target, k, a, trend, finest):
    """mu for target rows from training history <= H. finest in {cell, route, obsroute} (shrinkage chain)."""
    hist = ctx["train"] & (design["year"] <= H)
    df = pd.DataFrame({"y": design["y"][hist], "year": design["year"][hist], "cell": ctx["cell"][hist],
                       "route": design["route"][hist], "obsroute": design["obsroute"][hist]})
    yrs_win = np.arange(H - k + 1, H + 1)
    kern = kernel_window_mean(design, hist, yrs_win, 100.0, ctx["cell"], ctx["Dkm"])     # per land cell
    tc = ctx["cell"][target]
    mu = kern[tc].copy()
    ybar = np.full(mu.shape, H - (k - 1) / 2.0)
    chain = ["cell"] + (["route"] if finest in ("route", "obsroute") else []) + (["obsroute"] if finest == "obsroute" else [])
    keys = {"cell": ctx["cell"], "route": design["route"], "obsroute": design["obsroute"]}
    for key in chain:
        g = level_table(df, key, k)
        kv = keys[key][target]
        hit = np.isin(kv, g.index.to_numpy())
        gg = g.reindex(kv[hit])
        n, m_, yb = gg["n"].to_numpy(float), gg["m"].to_numpy(float), gg["ybar"].to_numpy(float)
        mu[hit] = (n * m_ + a * mu[hit]) / (n + a)
        ybar[hit] = (n * yb + a * ybar[hit]) / (n + a)
    if trend:
        if H not in _TCACHE:
            _TCACHE[H] = tile_trends(design, ctx["train"] & (design["year"] <= H), H, ctx["tile"])
        b = _TCACHE[H]
        bt = np.array([b.get(t, 0.0) for t in ctx["tile"][target]])
        mu = mu * np.exp(bt * (design["year"][target] - ybar))
    return np.maximum(mu, MU_FLOOR)


# ---------------------------------------------------------------------------------------------- spatial
def predict_spatial(design, ctx, target, h_km, w, kind, lobo):
    """mu for target rows by kernel (kind 'idw') or tile ('tile') average of training cell-year counts within +-w yr.
    lobo: exclude cells in the target cell's own 216 km block (inner tuning on training cells)."""
    src = ctx["train_s"]
    years = design["years"]
    nl = ctx["Dkm"].shape[0]
    ti = np.searchsorted(years, design["year"])
    S = np.zeros((len(years), nl)); N = np.zeros((len(years), nl))
    np.add.at(S, (ti[src], ctx["cell"][src]), design["y"][src])
    np.add.at(N, (ti[src], ctx["cell"][src]), 1.0)
    cs = lambda A: np.vstack([np.zeros((1, nl)), np.cumsum(A, 0)])
    cS, cN = cs(S), cs(N)
    if kind == "idw":
        K = np.exp(-0.5 * (ctx["Dkm"] / h_km) ** 2); K[ctx["Dkm"] > 3 * h_km] = 0.0
    else:
        K = (ctx["land_tile"][:, None] == ctx["land_tile"][None, :]).astype(float)
    if lobo:
        K = K * (ctx["land_block"][:, None] != ctx["land_block"][None, :])
    mu = np.empty(target.sum())
    tcell, tti = ctx["cell"][target], ti[target]
    for t in np.unique(tti):
        lo, hi = max(t - w, 0), min(t + w, len(years) - 1)
        Sw, Nw = cS[hi + 1] - cS[lo], cN[hi + 1] - cN[lo]
        glob = Sw.sum() / max(Nw.sum(), 1)
        sel = tti == t
        c = tcell[sel]
        KS, KN = K[c] @ Sw, K[c] @ Nw
        mu[sel] = np.where(KN > 1e-3, (KS + 0.01 * glob) / (KN + 0.01), glob)
    return np.maximum(mu, MU_FLOOR)


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    design = load_design(a.design)
    lr, lc = design["land_rows"], design["land_cols"]
    lut = -np.ones((lr.max() + 1, lc.max() + 1), int); lut[lr, lc] = np.arange(len(lr))
    cell = lut[design["row"], design["col"]]
    assert (cell >= 0).all()
    lat, lon = np.radians(design["land_lat"]), np.radians(design["land_lon"])
    Dkm = 6371.0 * np.arccos(np.clip(np.sin(lat)[:, None] * np.sin(lat)[None] +
                                     np.cos(lat)[:, None] * np.cos(lat)[None] * np.cos(lon[:, None] - lon[None]), -1, 1))
    ctx = {"cell": cell, "Dkm": Dkm,
           "tile": (design["row"] // TILE) * 1000 + design["col"] // TILE,
           "land_tile": (lr // TILE) * 1000 + lc // TILE, "land_block": (lr // BLOCK) * 1000 + lc // BLOCK,
           "train": ~design["holdout"] & ~design["pseudo"] & (design["weight"] > 0),
           "train_s": ~design["holdout"] & (design["weight"] > 0)}
    y = design["y"]
    res = {"tuning": {}, "baselines": {}}

    # ---- temporal: inner forecast from 2015 onto training rows 2016-19
    inner_t = ctx["train"] & (design["year"] >= 2016) & (design["year"] <= 2019)
    T = design["temporal"]
    preds = {}
    for finest in ("obsroute", "route", "cell"):
        for trend in (False, True):
            best = None
            for k, aa in itertools.product((1, 2, 3, 5, 8), (0.5, 1, 2, 4, 8)):
                mu_i = predict_temporal(design, ctx, 2015, inner_t, k, aa, trend, finest)
                phi, mlp = fit_phi(y[inner_t], mu_i)
                if best is None or mlp > best["inner_lpd"]:
                    best = {"k": k, "a": aa, "phi": phi, "inner_lpd": mlp}
            name = f"P-{finest}" + ("+trend" if trend else "")
            res["tuning"][name] = best
            mu = np.full(y.shape, np.nan)
            mu[T] = predict_temporal(design, ctx, 2020, T, best["k"], best["a"], trend, finest)
            preds[name] = (mu, best["phi"], "T")
            print(name, best, flush=True)

    # ---- spatial: leave-one-block-out on training real rows
    inner_s = ctx["train"]
    S = design["spatial"]
    for kind in ("idw", "tile"):
        best = None
        hs = (50, 75, 100, 150, 200, 300) if kind == "idw" else (0,)
        for h, w in itertools.product(hs, (0, 1, 2, 4)):
            mu_i = predict_spatial(design, ctx, inner_s, h, w, kind, lobo=True)
            phi, mlp = fit_phi(y[inner_s], mu_i)
            if best is None or mlp > best["inner_lpd"]:
                best = {"h_km": h, "w": w, "phi": phi, "inner_lpd": mlp}
        name = "S-" + kind
        res["tuning"][name] = best
        mu = np.full(y.shape, np.nan)
        mu[S] = predict_spatial(design, ctx, S, best["h_km"], best["w"], kind, lobo=False)
        preds[name] = (mu, best["phi"], "S")
        print(name, best, flush=True)

    # ---- score: age model (step 3) and every baseline on the same rows
    pr = np.load(os.path.join(a.design, "step3_pred.npz"))
    pred = {k: pr[k] for k in pr.files}
    lp_age = step3_lp(design, pred)
    res["age_step3"] = summarise(design, lp_age, pred["mu"])
    for name, (mu, phi, which) in preds.items():
        m = T if which == "T" else S
        lp = np.full(y.shape, np.nan)
        lp[m] = nb_lp(y[m], mu[m], phi)
        full = {"plug": np.where(m, lp, 0.0)}
        mu_f = np.where(m, mu, 1.0)
        summ = summarise(design, full, mu_f)
        summ = {k: v for k, v in summ.items() if (k == "T") == (which == "T")}
        comp = {}
        sets_ = {"T": T} if which == "T" else {"S_all": S, "S_real": S & ~design["pseudo"]}
        for sname, sm in sets_.items():
            for akind in ("marg", "cond", "plug"):
                diff = lp_age[akind] - np.where(m, lp, 0.0)
                comp[f"{sname}|age_{akind}_minus_baseline"] = paired_bootstrap(design, diff, sm)
                if sname == "S_real":
                    reg = {}
                    for rn, lo, hi in (("west", -180, -105), ("plains", -105, -95), ("midwest", -95, -85), ("east", -85, 0)):
                        rm = sm & (design["lon"] >= lo) & (design["lon"] < hi)
                        if rm.sum() > 100:
                            reg[rn] = paired_bootstrap(design, diff, rm)
                    comp[f"{sname}|age_{akind}_minus_baseline|by_region"] = reg
            if sname == "T":
                for akind in ("marg", "cond"):
                    reg = {}
                    for rn, lo, hi in (("west", -180, -105), ("plains", -105, -95), ("midwest", -95, -85), ("east", -85, 0)):
                        rm = sm & (design["lon"] >= lo) & (design["lon"] < hi)
                        if rm.sum() > 100:
                            reg[rn] = paired_bootstrap(design, lp_age[akind] - np.where(m, lp, 0.0), rm)
                    comp[f"T|age_{akind}_minus_baseline|by_region"] = reg
        res["baselines"][name] = {"summary": summ, "paired": comp, "phi": phi}
    json.dump(res, open(a.out, "w"), indent=1)
    # compact table
    print("\nmodel | T | S_all | S_real")
    a3 = res["age_step3"]
    print(f"age step3 plug | {a3['T']['plug']:.4f} | {a3['S_all']['plug']:.4f} | {a3['S_real']['plug']:.4f}")
    print(f"age step3 marg | {a3['T']['marg']:.4f} | {a3['S_all']['marg']:.4f} | {a3['S_real']['marg']:.4f}")
    print(f"age step3 cond | {a3['T']['cond']:.4f} | {a3['S_all']['cond']:.4f} | {a3['S_real']['cond']:.4f}")
    for name, b in res["baselines"].items():
        s = b["summary"]
        print(name, {k: round(v["plug"], 4) for k, v in s.items()})


if __name__ == "__main__":
    main()
