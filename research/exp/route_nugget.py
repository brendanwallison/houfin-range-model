"""N1: measure the between-route NUGGET directly -- route-level communities, effort-matched, by true distance.

    ESK_DESK_CONFIG=<overlay> python research/exp/route_nugget.py --out <dir> [--k 3] [--seed 0]
    python research/exp/route_nugget.py --summarize <dir>

Why: the atlas compared a cell's 40-year change (mostly the SAME routes) with adjacent-cell differences
(DIFFERENT routes). If two routes a few km apart already differ by a lot -- sub-cell habitat, observers -- the
spatial curve carries a nugget the temporal change does not, ESK may discard it, and a "matched size"
comparison is then confounded (the skeptic's reading: R1's strong form). Here every unit is ONE route:

  * every route-epoch with >= 2k surveyed years contributes two disjoint, year-balanced (ABBA over 2k randomly
    chosen years) halves of exactly k years each: equal effort for every unit, so the cross-half signal
    S = K(1a,1b) - K(1a,2b) - K(2a,1b) + K(2a,2b) means the same thing for every pair type;
  * SPATIAL pairs: two routes in the modern epoch (2005-2025), binned by the distance between their start
    points (projected grid CRS); the smallest bins are the nugget;
  * TEMPORAL pairs: the same route, early (1966-1986) vs modern epoch;
  * each half is projected into the ESK basis, so every bin also gets retention at r = 6 / 24 / 64.
Communities are log1p of the k-year mean count of the reference community's species, per route (the cell
aggregation divides by QC routes; a route is its own unit here). Block-bootstrap CIs resample 6x6-cell blocks of
the first route's cell.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

EARLY, MODERN = (1966, 1986), (2005, 2025)
BINS_KM = (0, 10, 20, 30, 45, 60, 90, 135, 200, 300)
RANKS = (6, 24, 64)
KEYS = ["CountryNum", "StateNum", "Route"]


def ruzicka(X, Y):
    mx = np.maximum(X, Y).sum(1)
    return np.where(mx > 0, np.minimum(X, Y).sum(1) / np.maximum(mx, 1e-300), 0.0)


def cross(xa, xb, i, j):
    return (ruzicka(xa[i], xb[i]) - ruzicka(xa[i], xb[j]) - ruzicka(xa[j], xb[i])
            + ruzicka(xa[j], xb[j]))


def route_counts():
    """Route-year x reference-community counts, with the route's grid cell and projected start point."""
    import geopandas as gpd
    import pandas as pd
    from src.config_utils import load_config, load_data_config
    from src.community_encoder.train_DESK.bbs_community_points import species_order
    from src.community_encoder.train_DESK.validate_gp_species import crosswalk_all_species
    from src.data.preprocess import bbs
    from src.data.preprocess.bbs_community import route_grid_map
    cfg = load_config()
    dcfg = load_data_config()
    comm_csv = (cfg.get("trend", {}) or {}).get("community_trend_list") or dcfg["community_trend_list"]
    comm = species_order(comm_csv)
    xw, _ = crosswalk_all_species(os.path.join(bbs.BBS_PARENT_DIR, "SpeciesList.csv"),
                                  os.path.join(dcfg["datasets_root"], "avonet", "eBird_taxonomy.csv"))
    ix = {c: i for i, c in enumerate(comm)}
    xw = xw[xw["species_code"].str.lower().isin(ix)]
    obs, coverage = bbs.load_usca_observations(aou_filter=None, return_coverage=True)
    routes = bbs.load_routes()
    land_mask, _, transform, crs, nx, ny = bbs.load_grid_reference(bbs.MASK_PATH)
    rc = route_grid_map(routes, transform, crs, nx, ny, land_mask)
    cov = coverage[KEYS + ["Year"]].drop_duplicates().merge(rc, on=KEYS, how="inner").reset_index(drop=True)
    cov["k"] = np.arange(len(cov))
    o = obs.merge(xw, left_on="AOU", right_on="aou", how="inner")
    o = o.merge(cov[KEYS + ["Year", "k"]], on=KEYS + ["Year"], how="inner")
    R = np.zeros((len(cov), len(comm)), "float64")
    np.add.at(R, (o["k"].to_numpy(), o["species_code"].str.lower().map(ix).to_numpy()),
              o["SpeciesTotal"].to_numpy(float))
    pts = routes[KEYS + ["Latitude", "Longitude"]].drop_duplicates(KEYS)
    g = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["Longitude"], pts["Latitude"]),
                         crs="EPSG:4326").to_crs(crs)
    pts = pd.DataFrame({**{k: pts[k].astype(int).to_numpy() for k in KEYS},
                        "x": g.geometry.x.to_numpy(), "y": g.geometry.y.to_numpy()})
    cov = cov.merge(pts, on=KEYS, how="left")
    return R, cov, comm


def halves(cov, R, k, rng):
    """Per route-epoch with >= 2k years: x_a, x_b (log1p of k-year mean counts) and the unit table."""
    units, xa, xb = [], [], []
    for (c, s, r), g in cov.groupby(KEYS):
        for e, (lo, hi) in enumerate((EARLY, MODERN)):
            gg = g[(g["Year"] >= lo) & (g["Year"] <= hi)].sort_values("Year")
            if len(gg) < 2 * k:
                continue
            pick = np.sort(rng.choice(len(gg), 2 * k, replace=False))
            abba = np.array([0, 1, 1, 0] * k)[: 2 * k].astype(bool)      # year-balanced halves
            ka, kb = gg["k"].to_numpy()[pick][~abba], gg["k"].to_numpy()[pick][abba]
            xa.append(np.log1p(R[ka].mean(0)))
            xb.append(np.log1p(R[kb].mean(0)))
            units.append((c, s, r, e, int(gg["row"].iloc[0]), int(gg["col"].iloc[0]),
                          float(gg["x"].iloc[0]), float(gg["y"].iloc[0])))
    return np.array(units, dtype=object), np.array(xa), np.array(xb)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--k", type=int, default=3, help="years per half (every unit: exactly k)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-km", type=float, default=300.0)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from src.config_utils import load_config
    from src.community_encoder.train_DESK.esk_kernel import project_points_to_z
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    R, cov, comm = route_counts()
    U, xa, xb = halves(cov, R, a.k, rng)
    cfg = load_config()
    L = max(RANKS)
    proj = lambda v: np.asarray(project_points_to_z(np.asarray(v, "float32"), cfg["desk"]["z_dir"], L),
                                "float64")
    za, zb = proj(xa), proj(xb)
    ep = U[:, 3].astype(int)
    xy = U[:, 6:8].astype(float) / 1000.0
    cells = U[:, 4:6].astype(int)
    res = {"k": a.k, "n_units": int(len(U)), "n_species": len(comm), "bins_km": list(BINS_KM),
           "spatial": [], "temporal": {}}

    def summarize_pairs(i, j):
        sk = cross(xa, xb, i, j)
        out = {"n_pairs": int(len(i)), "signal_exact": float(sk.mean())}
        blk = (cells[i, 0] // 6) * 100000 + cells[i, 1] // 6
        ub, inv = np.unique(blk, return_inverse=True)
        W = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)
        nk = np.bincount(inv, weights=sk, minlength=len(ub))
        cnt = np.bincount(inv, minlength=len(ub)).astype(float)
        out["signal_exact_ci"] = list(np.percentile((W @ nk) / np.maximum(W @ cnt, 1), [2.5, 97.5]))
        for r in RANKS:
            sr = ((za[i, :r] - za[j, :r]) * (zb[i, :r] - zb[j, :r])).sum(1)
            nr = np.bincount(inv, weights=sr, minlength=len(ub))
            boot = (W @ nr) / np.where(W @ nk != 0, W @ nk, np.nan)
            out[f"r{r}"] = {"signal": float(sr.mean()), "retention": float(sr.sum() / sk.sum()),
                            "retention_ci": [float(np.nanpercentile(boot, 2.5)),
                                             float(np.nanpercentile(boot, 97.5))]}
        return out

    # spatial: all modern route pairs within max-km, binned by distance
    m = np.where(ep == 1)[0]
    from scipy.spatial import cKDTree
    tree = cKDTree(xy[m])
    pairs = np.array(sorted(tree.query_pairs(a.max_km)), dtype=int)
    i, j = m[pairs[:, 0]], m[pairs[:, 1]]
    dist = np.hypot(*(xy[i] - xy[j]).T)
    for lo, hi in zip(BINS_KM[:-1], BINS_KM[1:]):
        s = (dist >= lo) & (dist < hi)
        if s.sum() < 30:
            continue
        res["spatial"].append(dict(summarize_pairs(i[s], j[s]), lo_km=lo, hi_km=hi,
                                   mean_km=float(dist[s].mean()),
                                   same_cell_share=float((cells[i[s]] == cells[j[s]]).all(1).mean())))
    # temporal: the same route, early vs modern
    key = {tuple(u[:3]): n for n, u in enumerate(U) if u[3] == 0}
    te = [(key[tuple(u[:3])], n) for n, u in enumerate(U) if u[3] == 1 and tuple(u[:3]) in key]
    ti, tj = np.array([p[0] for p in te]), np.array([p[1] for p in te])
    res["temporal"] = summarize_pairs(ti, tj)

    # COHERENCE of temporal change: how much of a route's 40-y change does a route d km away share? The
    # cross-route covariance of the two change vectors, half A of one against half B of the other (independent
    # noise): C = K(1e_a,2e_b) - K(1e_a,2m_b) - K(1m_a,2e_b) + K(1m_a,2m_b); in ESK coordinates a dot product.
    # C(d) / S_temporal is the share of the change that is regional at scale d; C_r / C is how much of THAT part
    # ESK keeps -- set against the nugget's retention (smallest spatial bin) and the structured spatial part's.
    res["coherence"] = []
    T = np.column_stack([ti, tj])                                   # (early unit, modern unit) per route
    txy = xy[ti]
    ttree = cKDTree(txy)
    tp = np.array(sorted(ttree.query_pairs(a.max_km)), dtype=int)
    if len(tp):
        p1, p2 = T[tp[:, 0]], T[tp[:, 1]]
        d12 = np.hypot(*(txy[tp[:, 0]] - txy[tp[:, 1]]).T)
        ck = (ruzicka(xa[p1[:, 0]], xb[p2[:, 0]]) - ruzicka(xa[p1[:, 0]], xb[p2[:, 1]])
              - ruzicka(xa[p1[:, 1]], xb[p2[:, 0]]) + ruzicka(xa[p1[:, 1]], xb[p2[:, 1]]))
        s_t = res["temporal"]["signal_exact"]
        for lo, hi in zip(BINS_KM[:-1], BINS_KM[1:]):
            s = (d12 >= lo) & (d12 < hi)
            if s.sum() < 30:
                continue
            row = {"lo_km": lo, "hi_km": hi, "n_pairs": int(s.sum()), "cov_exact": float(ck[s].mean()),
                   "shared_share": float(ck[s].mean() / s_t)}
            for r in RANKS:
                dza = za[p1[s, 0], :r] - za[p1[s, 1], :r]
                dzb = zb[p2[s, 0], :r] - zb[p2[s, 1], :r]
                cr = (dza * dzb).sum(1)
                row[f"r{r}_retention"] = float(cr.sum() / ck[s].sum()) if ck[s].sum() != 0 else None
            res["coherence"].append(row)
    with open(os.path.join(a.out, "route_nugget.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "route_nugget.json"), encoding="utf-8"))
    ci = lambda v: f"[{v[0]:.3f},{v[1]:.3f}]"
    print(f"N1 route-level variogram: k={r['k']} years per half for every unit, {r['n_units']} route-epochs")
    print(f"  {'pairs':>22s} {'n':>7s}  {'exact signal':>22s}   retention r6 / r24 / r64")
    for b in r["spatial"]:
        print(f"  spatial {b['lo_km']:4.0f}-{b['hi_km']:<4.0f} km {b['n_pairs']:7d}  {b['signal_exact']:.3f} "
              f"{ci(b['signal_exact_ci'])}   {b['r6']['retention']:.3f} / {b['r24']['retention']:.3f} "
              f"{ci(b['r24']['retention_ci'])} / {b['r64']['retention']:.3f}  (same cell {b['same_cell_share']:.0%})")
    t = r["temporal"]
    print(f"  {'temporal same route':>22s} {t['n_pairs']:7d}  {t['signal_exact']:.3f} {ci(t['signal_exact_ci'])}   "
          f"{t['r6']['retention']:.3f} / {t['r24']['retention']:.3f} {ci(t['r24']['retention_ci'])} / "
          f"{t['r64']['retention']:.3f}")
    if r.get("coherence"):
        print("  coherence of the 40-y change between routes d km apart: shared share of a route's change, and "
              "ESK retention of the shared part (r6 / r24 / r64)")
        for c in r["coherence"]:
            f = lambda v: "  -  " if v is None else f"{v:.3f}"
            print(f"    {c['lo_km']:4.0f}-{c['hi_km']:<4.0f} km {c['n_pairs']:7d}  shared {c['shared_share']:.3f}   "
                  f"{f(c['r6_retention'])} / {f(c['r24_retention'])} / {f(c['r64_retention'])}")


if __name__ == "__main__":
    main()
