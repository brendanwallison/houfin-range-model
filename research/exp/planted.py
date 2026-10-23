"""Planted calibration (Phase 1.4): a copy of a REAL cache whose dev species are simulated from known
generators, through the real sampling design, so every metric can be read against what it CAN show.

    python research/exp/planted.py --cache <real cache> --out <planted cache> [--n-per 12] [--seed 0]

Keys, splits, z, covariates and route coverage stay real (symlinked); only X_dev is replaced. Each
planted species' log mean lambda(cell, year) comes from one generator:

    z          mu + beta . z_raw[:r]  (a perfect DESK readout: what the fast arm can know at best)
    spacetime  mu + f(x, t), f a separable space x time field (what the spacetime GP should win)
    mix_<rho>  mu + g(x) + h(x, t), static field g plus a spatially varying temporal field h holding a
               share rho of the variance, rho in {0, 0.1, 0.3, 0.6} (the power curve for change metrics)

Counts: every QC route surveying the cell that year draws NB(mean lambda, dispersion k); the cell-year
value is their mean -- exactly the suite's route-mean aggregation, so a cell with one route is as noisy
as it really is. ``mu`` is drawn from the real dev species' mean log1p abundances, so prevalence (the
zero share that makes most real change unresolvable) is realistic. ``truth.npz`` keeps each species'
generator and its noise-free log lambda per row, for ceilings computed without split halves.
"""
import argparse
import json
import os
import shutil

import numpy as np

LINKED = ("keys.npy", "X_comm.npy", "cells.npy", "years.npy", "z_ema_all.npy", "z_raw_all.npy",
          "z_ema.npy", "z_raw.npy", "key_cell_index.npy", "F_cov.npy", "F_cov_raw.npy",
          "split.npz", "route_years.csv", "esk_annual.npy", "esk_epochs.npz")


def rff_field(xy, n_feat, ls, rng):
    """A smooth spatial Gaussian field by random Fourier features (Matern-ish via RBF), unit variance."""
    W = rng.normal(size=(xy.shape[1], n_feat)) / ls
    b = rng.uniform(0, 2 * np.pi, n_feat)
    return np.sqrt(2.0 / n_feat) * np.cos(xy @ W + b)               # (n, n_feat), E[f f'] = k


def ar1_series(n_series, n_t, phi, rng):
    out = np.zeros((n_series, n_t))
    out[:, 0] = rng.normal(size=n_series)
    for t in range(1, n_t):
        out[:, t] = phi * out[:, t - 1] + np.sqrt(1 - phi ** 2) * rng.normal(size=n_series)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per", type=int, default=12, help="species per generator")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--dispersion", type=float, default=2.0)
    ap.add_argument("--cell-km", type=float, default=27.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generators", nargs="+", default=("z", "spacetime", "mix"),
                    help="which generators to plant (e.g. z alone for a readout floor with enough species)")
    a = ap.parse_args()
    import pandas as pd
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    for f in LINKED:
        src = os.path.join(a.cache, f)
        if os.path.exists(src) and not os.path.exists(os.path.join(a.out, f)):
            os.symlink(os.path.abspath(src), os.path.join(a.out, f))
    keys = np.load(os.path.join(a.cache, "keys.npy"))
    cells = np.load(os.path.join(a.cache, "cells.npy"))
    years = np.load(os.path.join(a.cache, "years.npy"))
    ci = np.load(os.path.join(a.cache, "key_cell_index.npy"))
    yi = keys[:, 2] - years[0]
    z_all = np.load(os.path.join(a.cache, "z_raw_all.npy"), mmap_mode="r")
    zk = np.asarray(z_all[ci, yi, : a.rank], "float64")
    zk = (zk - zk.mean(0)) / np.maximum(zk.std(0), 1e-9)
    real = np.log1p(np.load(os.path.join(a.cache, "X_dev.npy"), mmap_mode="r"))
    mu_pool = np.asarray(real.mean(0))
    mu_pool = mu_pool[mu_pool > 0]

    ry = pd.read_csv(os.path.join(a.cache, "route_years.csv"))
    n_routes = (ry.groupby(["row", "col", "year"]).size()
                .reindex(pd.MultiIndex.from_arrays([keys[:, 0], keys[:, 1], keys[:, 2]]))
                .fillna(1).to_numpy().astype(int))

    xy = cells.astype("float64") * a.cell_km
    gens, logl = [], []
    t_norm = (years - years.mean()) / years.std()

    def scaled(v, sd):
        return (v - v.mean()) / max(v.std(), 1e-12) * sd

    for s in range(a.n_per if "z" in a.generators else 0):         # z readouts
        logl.append(scaled(zk @ rng.normal(size=zk.shape[1]), 0.8))
        gens.append("z")
    for s in range(a.n_per if "spacetime" in a.generators else 0):  # separable space x time
        phi = rff_field(xy, 64, 400.0, rng)
        ser = ar1_series(64, len(years), 0.95, rng)
        logl.append(scaled((phi[ci] * ser.T[yi]).sum(1), 0.8))
        gens.append("spacetime")
    for rho in ((0.0, 0.1, 0.3, 0.6) if "mix" in a.generators else ()):   # static + temporal share rho
        for s in range(a.n_per):
            g = rff_field(xy, 64, 500.0, rng) @ rng.normal(size=64)
            phi = rff_field(xy, 32, 300.0, rng)
            ser = ar1_series(32, len(years), 0.97, rng) + t_norm[None, :] * rng.normal(size=(32, 1))
            h = (phi[ci] * ser.T[yi]).sum(1)
            v = np.sqrt(1 - rho) * scaled(g[ci], 1.0) + np.sqrt(rho) * scaled(h, 1.0)
            logl.append(scaled(v, 0.8))
            gens.append(f"mix_{rho:g}")
    L = np.stack(logl, 1)
    mu = rng.choice(np.log(np.expm1(mu_pool) + 1e-3), L.shape[1])
    lam = np.exp(np.clip(mu[None, :] + L, -12, 8))
    k = float(a.dispersion)
    X = np.zeros_like(lam, dtype="float32")
    maxr = int(n_routes.max())
    for r in range(1, maxr + 1):                                     # route-mean of NB draws
        rows = np.where(n_routes == r)[0]
        if not len(rows):
            continue
        p = k / (k + lam[rows])
        draws = rng.negative_binomial(k, p[None, :, :].repeat(r, 0))
        X[rows] = draws.mean(0)
    np.save(os.path.join(a.out, "X_dev.npy"), X)
    np.savez_compressed(os.path.join(a.out, "truth.npz"), generator=np.array(gens),
                        log_lambda=(mu[None, :] + L).astype("float32"), mu=mu)
    meta = json.load(open(os.path.join(a.cache, "meta.json"), encoding="utf-8"))
    meta.update({"planted_from": os.path.abspath(a.cache), "planted_seed": a.seed,
                 "dev_species": [f"{g}_{i:02d}" for i, g in enumerate(gens)],
                 "sealed_species": [], "planted_dispersion": k})
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    zero = float((X == 0).mean())
    print(f"[planted] {X.shape[1]} species x {X.shape[0]:,} cell-years; zero share {zero:.2f} "
          f"(real {float((np.asarray(real) == 0).mean()):.2f}); generators "
          + ", ".join(f"{g}:{gens.count(g)}" for g in dict.fromkeys(gens)))


if __name__ == "__main__":
    main()
