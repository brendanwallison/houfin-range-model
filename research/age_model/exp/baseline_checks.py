"""Baseline checks on one point's per-observation file (per_obs_*.npz from loglik_decompose.py).

    python research/age_model/exp/baseline_checks.py <per_obs.npz> [--out <dir>]

H8 (route heterogeneity): within (year, cell) groups holding >= 2 routes, compare the observed
   between-route variance of counts to the NB variance the model implies there (mu + mu^2/conc,
   conc recovered from the fitted means is not stored, so we report the observed variance-to-mean
   ratio against the fitted mean and the observed spread ratio max/min across routes); and the
   calibration: median fitted vs observed by observed-count decile, per region.
H9 (early west): observed vs fitted annual mean count per region, 1966-1985 -- does the fit lag
   the established western population as the survey expands west (1967-68)?
"""
import argparse
import os

import numpy as np

REGIONS = [("west", -180, -105), ("plains", -105, -95), ("midwest", -95, -85), ("east", -85, 0)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = np.load(a.npz)
    y, mu, yr, q = d["y"], d["mu"], d["year"], d["quality"] == 0
    rows, cols, lon = d["rows"], d["cols"], d["lon"]
    # --- H8: between-route spread within (year, cell)
    key = (yr.astype(np.int64) * 1_000_000 + rows.astype(np.int64) * 1000 + cols.astype(np.int64))
    order = np.argsort(key); k_sorted = key[order]
    starts = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    sizes = np.diff(np.r_[starts, k_sorted.size])
    multi = starts[sizes >= 2]
    vr, mus, spread = [], [], []
    for s, n in zip(starts, sizes):
        if n < 2:
            continue
        idx = order[s:s + n]
        if not q[idx].all():
            continue
        yy = y[idx]
        if yy.mean() > 0:
            vr.append(yy.var(ddof=1) / yy.mean()); mus.append(mu[idx].mean())
            spread.append((yy.max() + 1) / (yy.min() + 1))
    vr, mus, spread = map(np.asarray, (vr, mus, spread))
    print(f"H8: {len(multi)} (year, cell) groups with >=2 routes; {vr.size} with a positive mean")
    for lo, hi in ((0, 1), (1, 3), (3, 10), (10, 1e9)):
        m = (mus >= lo) & (mus < hi)
        if m.any():
            print(f"   fitted mean [{lo:g},{hi:g}): n={m.sum():5d}  observed var/mean median {np.median(vr[m]):6.1f}"
                  f"  (Poisson 1; NB conc 0.4 implies 1+mu/0.4 ~ {1 + np.median(mus[m]) / 0.4:5.1f})"
                  f"  (max+1)/(min+1) median {np.median(spread[m]):5.1f}")
    # --- calibration by region
    print("calibration: median fitted by observed-count band (positives), per region")
    bands = [(1, 3), (3, 10), (10, 30), (30, 100), (100, 1e9)]
    for name, lo, hi in REGIONS:
        m = q & (lon >= lo) & (lon < hi)
        cells = [f"{blo:g}-{bhi:g}: {np.median(mu[m & (y >= blo) & (y < bhi)]):5.1f}" if (m & (y >= blo) & (y < bhi)).any() else f"{blo:g}-{bhi:g}: -"
                 for blo, bhi in bands]
        z = m & (y == 0)
        print(f"   {name:8s} zeros->{np.median(mu[z]) if z.any() else float('nan'):5.2f} | " + " | ".join(cells))
    # --- H9: early annual means by region
    print("H9: observed / fitted mean count by region and year (1966-1985)")
    for name, lo, hi in REGIONS:
        m = q & (lon >= lo) & (lon < hi)
        line = []
        for t in range(1966, 1986, 2):
            mm = m & (yr == t)
            line.append(f"{t % 100:02d}:{y[mm].mean():.1f}/{mu[mm].mean():.1f}(n{mm.sum()})" if mm.any() else f"{t % 100:02d}:-")
        print(f"   {name:8s} " + " ".join(line))


if __name__ == "__main__":
    main()
