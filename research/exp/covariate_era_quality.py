"""Covariate era quality: does the covariate stack get visibly better at some year (e.g. 1940)?

Question (user, 2026-10-29): would starting the downstream House Finch simulation in 1940 instead of 1902 buy
better covariates? Measured directly on the stored yearly states.

The stored states are EMA-smoothed along the year axis at build time (streams.PerVariableYearStreamer:
s_t = a*x_t + (1-a)*s_{t-1}, a = 1 - exp(-1/tau)). The raw per-year grids are not on this machine, so the raw
series is recovered by exact inversion x_t = (s_t - (1-a)*s_{t-1}) / a. tau is not recorded in the schema (built
before ema_tau became provenance), so it is ESTIMATED: BUI and HYDE are linear interpolations between sparse
snapshots, so the correct tau is the one whose inverse makes their second differences vanish between snapshots.
The EMA started before 1902, so x_1902 is not recoverable; the raw series runs 1903-2025.

Measurements, per decade and per region (West = lon < -100, East = lon >= -100):
  (a) climate (Tave / Tmin / Tmax / PPT, q50, 12 monthly channels each):
      - var ratio: within-decade interannual variance / mean within-decade variance of 1990s, 2000s, 2010s
        (per cell, averaged over months; within-decade so century trends do not count as variance).
        A CRU-style relaxation to climatology mutes early variance.
      - near-normal share: share of cell-month-years whose anomaly vs the cell's 1961-1990 mean is < 0.05 sd
        (sd = cell's 1961-2020 interannual sd). Gaussian chance ~4%. Relaxed cells sit on the normal.
      - exact-normal share: share whose anomaly is < 1e-3 sd (chance ~0.1%): cells sitting exactly on the
        reference normal, the direct fingerprint of CRU "no station within the correlation decay distance".
      - coherence: correlation, across years within the decade, between a cell's anomaly and that of the cell
        k columns east (k = 4 ~ 108 km, k = 15 ~ 405 km), averaged over cells and months.
  (b) LUH-3 and HYDE: per centre year, share of changing cell-channels whose second difference is non-zero
      (> 1e-3 x their typical yearly change). Interior years of a linear interpolation segment give ~0, knots ~1.
  (c) BUI: same knot analysis, and share of CONUS cells with any built-up per year.
"""
import argparse
import json
import os

import numpy as np

S = "/home/breallis/houfin/work/houfin/processed/encoder/states/"
YEARS = np.arange(1902, 2026)          # stored state years
RYEARS = np.arange(1903, 2026)         # recoverable raw years
DECADES = [(1903, 1909)] + [(y, y + 9) for y in range(1910, 2020, 10)] + [(2020, 2025)]


def lon_grid():
    import rasterio
    from pyproj import Transformer
    from src.config_utils import load_data_config
    with rasterio.open(load_data_config()["grid"]["ref_raster"]) as ds:
        rows, cols = np.meshgrid(np.arange(ds.height), np.arange(ds.width), indexing="ij")
        x, y = rasterio.transform.xy(ds.transform, rows.ravel(), cols.ravel())
        lon, lat = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True).transform(np.array(x), np.array(y))
    return lon.reshape(rows.shape), lat.reshape(rows.shape)


def load(schema):
    v = schema["streams"][0]["variables"]
    cl_names = [n for n in v if n.split("_")[0] in ("Tave", "Tmin", "Tmax", "PPT") and n.endswith("_q50")]
    ci = [v.index(n) for n in cl_names]
    cl, lu, hy, bu = [], [], [], []
    for y in YEARS:
        z = np.load(os.path.join(S, "yearly_states", f"state_{y}.npz"))
        cl.append(z["climate"][..., ci]); lu.append(z["landuse"]); hy.append(z["hyde"]); bu.append(z["bui"])
    return cl_names, np.stack(cl), np.stack(lu), np.stack(hy), np.stack(bu)


def deema(s, tau):
    """Raw years 1903.. from stored states 1902.. (drops 1902: its EMA predecessor is not stored)."""
    a = 1.0 if tau == 0 else 1.0 - np.exp(-1.0 / tau)
    return (s[1:] - (1.0 - a) * s[:-1]) / a


def _rel_d2(x, mask):
    """|d2| relative to the cell-channel's mean |d1| (cell-channels that never change are dropped).
    Returns (centre years, (T-2, M) relative second differences)."""
    xm = x[:, mask].reshape(x.shape[0], -1)
    d1 = np.diff(xm, axis=0)
    d2 = np.diff(d1, axis=0)
    scale = np.nanmean(np.abs(d1), axis=0)
    mag = np.nanmean(np.abs(xm), axis=0)
    ok = np.isfinite(scale) & (scale > 1e-9 * (mag + 1e-12))
    return RYEARS[1:-1], np.abs(d2[:, ok]) / scale[ok]


def knot_profile(x, mask, tol=1e-3):
    """Per centre year: share of changing cell-channels whose second difference is non-zero."""
    yrs, r = _rel_d2(x, mask)
    return yrs, (r > tol).mean(1)


def sparsity_score(x, mask):
    """Median relative second difference over all years and cell-channels (lower = more piecewise-linear)."""
    return float(np.median(_rel_d2(x, mask)[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    schema = json.load(open(os.path.join(S, "state_schema.json")))
    cl_names, cl, lu, hy, bu = load(schema)
    lon, lat = lon_grid()
    land = np.isfinite(cl).all(axis=(0, 3)) & np.isfinite(lu).all(axis=(0, 3))
    bvars = schema["streams"][3]["variables"]
    conus = land & (bu[..., bvars.index("bui_avail")].max(0) > 0.5)
    regions = {"West": land & (lon < -100), "East": land & (lon >= -100)}
    out = {"n_cells": {k: int(m.sum()) for k, m in regions.items()}, "n_conus": int(conus.sum())}
    print(out)

    # ---- tau estimation -------------------------------------------------------------------------
    taus = [0, 0.5, 1, 1.5, 1.9, 2, 2.1, 2.5, 3, 4, 5, 10]
    tau_tab = {}
    for tau in taus:
        tau_tab[tau] = {"bui": sparsity_score(deema(bu[..., 1:], tau), conus),
                        "hyde": sparsity_score(deema(hy, tau), land),
                        "landuse": sparsity_score(deema(lu, tau), land)}
    print("tau sparsity (median |d2| / typical |d1|; lower = more piecewise-linear):")
    for t, r in tau_tab.items():
        print(f"  tau={t:>4}: " + "  ".join(f"{k}={v:.2e}" for k, v in r.items()))
    tau = min(taus, key=lambda t: tau_tab[t]["bui"] + tau_tab[t]["hyde"])
    print("estimated tau:", tau)
    out["tau_table"] = {str(k): v for k, v in tau_tab.items()}
    out["tau_est"] = tau
    clr, lur, hyr, bur = (deema(a, tau) for a in (cl, lu, hy, bu))
    del cl, lu, hy, bu

    # ---- (a) climate ---------------------------------------------------------------------------------
    yi = {y: i for i, y in enumerate(RYEARS)}
    base_idx = [i for i, y in enumerate(RYEARS) if 1961 <= y <= 1990]
    sd_idx = [i for i, y in enumerate(RYEARS) if 1961 <= y <= 2020]
    ref_dec = [(1990, 1999), (2000, 2009), (2010, 2019)]
    clim = {}
    for base in ("Tave", "Tmin", "Tmax", "PPT"):
        ch = [i for i, n in enumerate(cl_names) if n.startswith(base + "_")]
        x = clr[..., ch]                                                      # (T, H, W, 12)
        normal = x[base_idx].mean(0)
        sd = x[sd_idx].std(0)
        anom = x - normal

        def wvar(y0, y1):
            return x[yi[y0]:yi[y1] + 1].var(0, ddof=1)                        # (H, W, 12)
        ref = np.mean([wvar(*d) for d in ref_dec], axis=0)
        res = {}
        for reg, m in regions.items():
            rows = []
            for y0, y1 in DECADES:
                v = wvar(y0, y1)
                ratio = (v[m] / np.maximum(ref[m], 1e-12)).mean(1)            # per cell, mean over months
                a = anom[yi[y0]:yi[y1] + 1][:, m]                             # (n, N, 12)
                sdm = np.maximum(sd[m], 1e-12)[None]
                nn = (np.abs(a) < 0.05 * sdm).mean()
                ex = (np.abs(a) < 1e-3 * sdm).mean()
                coh = {}
                seg = x[yi[y0]:yi[y1] + 1]
                seg = seg - seg.mean(0)
                for k in (4, 15):
                    a1, a2 = seg[:, :, :-k], seg[:, :, k:]
                    mk = m[:, :-k] & land[:, k:]
                    num = (a1 * a2).sum(0)
                    den = np.sqrt((a1 ** 2).sum(0) * (a2 ** 2).sum(0))
                    r = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)[mk]
                    coh[k] = float(np.nanmean(r))
                rows.append({"decade": f"{y0}-{y1}", "var_ratio_median": float(np.median(ratio)),
                             "var_ratio_p10": float(np.percentile(ratio, 10)),
                             "near_normal_share": float(nn), "exact_normal_share": float(ex),
                             "coh_108km": coh[4], "coh_405km": coh[15]})
            res[reg] = rows
        clim[base] = res
    # muted cells: within-decade variance < 25% of the modern reference, split CONUS / outside CONUS
    muted = {}
    sub = {"West-CONUS": regions["West"] & conus, "West-nonUS": regions["West"] & ~conus,
           "East-CONUS": regions["East"] & conus, "East-nonUS": regions["East"] & ~conus}
    for base in ("Tave", "PPT"):
        ch = [i for i, n in enumerate(cl_names) if n.startswith(base + "_")]
        x = clr[..., ch]
        ref = np.mean([x[yi[a]:yi[b] + 1].var(0, ddof=1) for a, b in ref_dec], axis=0)
        muted[base] = {}
        for y0, y1 in DECADES:
            ratio = (x[yi[y0]:yi[y1] + 1].var(0, ddof=1) / np.maximum(ref, 1e-12)).mean(-1)
            row = {}
            for k, m in sub.items():
                mm = m & (ratio < 0.25)
                row[k] = {"share": float(mm.sum() / max(m.sum(), 1)),
                          "lat_med": float(np.median(lat[mm])) if mm.any() else None,
                          "lon_med": float(np.median(lon[mm])) if mm.any() else None}
            muted[base][f"{y0}-{y1}"] = row
    out["muted_cells"] = muted
    print("\nshare of cells with within-decade variance < 25% of modern (median lat/lon of those cells):")
    for base, rows in muted.items():
        for dec, row in rows.items():
            print(f"  {base} {dec}: " + "  ".join(
                f"{k}={v['share']:.3f}" + (f"({v['lat_med']:.0f},{v['lon_med']:.0f})" if v["share"] > 0.005 else "")
                for k, v in row.items()))
    out["climate"] = clim
    for base, res in clim.items():
        for reg, rows in res.items():
            print(f"\n{base} {reg}: decade  var_ratio(med/p10)  near_normal  exact_normal  coh108  coh405")
            for r in rows:
                print(f"  {r['decade']}  {r['var_ratio_median']:.2f}/{r['var_ratio_p10']:.2f}   "
                      f"{r['near_normal_share']:.3f}   {r['exact_normal_share']:.4f}   "
                      f"{r['coh_108km']:.2f}  {r['coh_405km']:.2f}")

    # ---- (b,c) knot structure ---------------------------------------------------------------------------
    knots = {}
    groups = {"landuse": (lur, land), "hyde": (hyr, land), "bui": (bur[..., 1:], conus)}
    lvars = schema["streams"][1]["variables"]
    for sub in ("c3ann", "pastr", "primf", "secdf", "urban"):
        groups[f"landuse:{sub}"] = (lur[..., [lvars.index(sub)]], land)
    for g, (x, m) in groups.items():
        for reg, rm in list(regions.items()) + [("All", land)]:
            yrs, r = knot_profile(x, m & rm)
            knots[f"{g}|{reg}"] = {int(y): float(v) for y, v in zip(yrs, r)}
    out["knots"] = knots
    for g in groups:
        for reg in ("All", "West", "East"):
            r = knots[f"{g}|{reg}"]
            big = [y for y, v in r.items() if v > 0.5]
            print(f"\n{g} [{reg}]: kink years (>50% of changing cell-channels bend): {big}")
            if reg == "All":
                print("   ", " ".join(f"{y}:{v:.2f}" for y, v in r.items()))

    # BUI coverage (q99 > 0 within CONUS)
    q99 = bur[..., bvars.index("bui_q99")]
    q50 = bur[..., bvars.index("bui_q50")]
    cov = {}
    for reg, rm in regions.items():
        mm = conus & rm
        cov[reg] = {int(y): {"any_built": float((q99[i][mm] > 0).mean()),
                             "median_cell_built": float((q50[i][mm] > 0).mean())}
                    for i, y in enumerate(RYEARS) if y % 10 == 0 or y in (1903, 1905, 1945, 2025)}
    out["bui_coverage"] = cov
    print("\nBUI share of CONUS cells with any built-up (q99>0) / median-pixel built (q50>0):")
    for reg in cov:
        print(" ", reg, " ".join(f"{y}:{v['any_built']:.2f}/{v['median_cell_built']:.2f}" for y, v in cov[reg].items()))

    with open(os.path.join(args.out, "covariate_era_quality.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
