"""Summarise rangr truth candidates: first-arrival year (N >= 1) by band (SLOW-front truth selection, B6 record).
    python sdm_b6_arrival.py --design <design_step3> --b6 <b6 dir> --tags tag1 tag2 ...
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--design", required=True); ap.add_argument("--b6", required=True); ap.add_argument("--tags", nargs="+")
a = ap.parse_args()
D = np.load(os.path.join(a.design, "design.npz"))
lr, lc, lon, lat, years = D["land_rows"], D["land_cols"], D["land_lon"], D["land_lat"], D["years"]
g = json.load(open(os.path.join(a.b6, "grid.json")))
bands = {"east80": (lon >= -80) & (lat >= 25) & (lat < 49), "central_east": (lon >= -95) & (lon < -80) & (lat >= 25) & (lat < 49)}
out = {}
for tag in a.tags:
    p = os.path.join(a.b6, f"N_{tag}.f64")
    if not os.path.exists(p):
        print(tag, "missing"); continue
    N = np.fromfile(p, dtype="<f8").reshape(g["Nx"], g["Ny"], len(years)).transpose(2, 1, 0)[:, lr, lc]
    N = np.nan_to_num(N)
    first = np.where((N >= 1).any(0), years[np.argmax(N >= 1, 0)], np.nan)
    r = {"total_2025": float(N[-1].sum()), "alive": bool(N[-1].sum() > 0)}
    for bn, bm in bands.items():
        f = first[bm]
        r[bn] = {"median_first": float(np.nanmedian(f)) if np.isfinite(f).any() else None,
                 "p10": float(np.nanpercentile(f, 10)) if np.isfinite(f).any() else None,
                 "p90": float(np.nanpercentile(f, 90)) if np.isfinite(f).any() else None,
                 "reached_by_2010": float(np.mean(f <= 2010))}
    ok = (r["alive"] and r["east80"]["median_first"] is not None and r["central_east"]["median_first"] is not None
          and 1955 <= r["east80"]["median_first"] <= 1975 and 1975 <= r["central_east"]["median_first"] <= 1995
          and r["central_east"]["reached_by_2010"] >= .8)
    r["meets_targets"] = ok
    r["score"] = (abs(r["east80"]["median_first"] - 1965) + abs(r["central_east"]["median_first"] - 1985)
                  if r["east80"]["median_first"] and r["central_east"]["median_first"] else None)
    out[tag] = r
    print(tag, json.dumps(r))
json.dump(out, open(os.path.join(a.b6, "slow_candidates.json"), "w"), indent=1)
