"""FIA forest structure per 27 km cell, early (periodic inventories, 1968-1988) vs modern (annual, 2010-2019).

    python research/exp/fia_cells.py --fia <dir with ENTIRE_SURVEY/COND/PLOT.csv> --grid <ref_grid_27km.tif>
        --out <fia_27km.npz> [--min-cond 5]

Forest conditions (COND_STATUS_CD 1) restricted in BOTH epochs to unreserved productive forest (RESERVCD 0,
SITECLCD 1-6: the TIMBERLAND definition several periodic inventories were limited to), weighted by CONDPROP_UNADJ.
Early: for each state, the periodic inventory (SURVEY.ANN_INVENTORY N) whose INVYR is closest to 1976 within
1966-1988 (the BBS early epoch is 1966-86). Modern: every annual-inventory condition with INVYR 2010-2019.
Per cell and epoch:
  ss_share     share of forest area in seedling/sapling stands (STDSZCD 3) -- the early-successional share
  large_share  share in large-diameter stands (STDSZCD 1)
  mean_age     mean STDAGE (valid ages >= 0 only)
  young_share  share of aged area with STDAGE <= 20
Cells need >= --min-cond forest conditions with each field in BOTH epochs, else NaN (mask false). Writes, for
guild_dose.py --aux: years 1940-2025 with the early value for years < 1996 and the modern value after, so a modern-
minus-early epoch change is exactly modern - early; plus fia_mask (H, W) and a per-state audit (CSV next to --out).
Plot coordinates are FIA's public ones (fuzzed up to ~1.6 km, some swapped within county) -- immaterial at 27 km
except for swaps across a cell boundary.
"""
import argparse
import json
import os

import numpy as np

NAMES = ("ss_share", "large_share", "mean_age", "young_share")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fia", required=True); ap.add_argument("--grid", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-cond", type=int, default=5)
    ap.add_argument("--early", type=int, nargs=2, default=(1966, 1988)); ap.add_argument("--target", type=int, default=1976)
    ap.add_argument("--modern", type=int, nargs=2, default=(2010, 2019))
    a = ap.parse_args()
    import pandas as pd
    import rasterio
    from pyproj import Transformer
    j = lambda f: os.path.join(a.fia, f)
    srv = pd.read_csv(j("ENTIRE_SURVEY.csv"), usecols=["CN", "INVYR", "STATECD", "STATEAB", "ANN_INVENTORY"])
    plot = pd.read_csv(j("ENTIRE_PLOT.csv"), usecols=["CN", "SRV_CN", "INVYR", "STATECD", "LAT", "LON", "PLOT_STATUS_CD",
                                                       "DESIGNCD"], dtype={"CN": str, "SRV_CN": str})
    cond = pd.read_csv(j("ENTIRE_COND.csv"), usecols=["PLT_CN", "INVYR", "STATECD", "COND_STATUS_CD", "RESERVCD",
                                                       "SITECLCD", "STDSZCD", "STDAGE", "CONDPROP_UNADJ"],
                       dtype={"PLT_CN": str})
    srv["CN"] = srv["CN"].astype(str)
    plot = plot.merge(srv[["CN", "ANN_INVENTORY", "STATEAB"]].rename(columns={"CN": "SRV_CN"}), on="SRV_CN", how="left")
    # choose the early periodic inventory per state
    per = plot[(plot.ANN_INVENTORY == "N") & plot.INVYR.between(*a.early)]
    early_yr = {}
    for st, g in per.groupby("STATECD"):
        yrs = sorted(g.INVYR.unique())
        early_yr[int(st)] = int(min(yrs, key=lambda y: (abs(y - a.target), y)))
    plot["epoch"] = -1
    is_e = (plot.ANN_INVENTORY == "N") & (plot.STATECD.map(early_yr) == plot.INVYR)
    is_m = (plot.ANN_INVENTORY == "Y") & plot.INVYR.between(*a.modern)
    plot.loc[is_e, "epoch"] = 0
    plot.loc[is_m, "epoch"] = 1
    plot = plot[plot.epoch >= 0]
    c = cond.merge(plot[["CN", "LAT", "LON", "epoch", "STATEAB", "DESIGNCD"]].rename(columns={"CN": "PLT_CN"}),
                   on="PLT_CN", how="inner")
    c = c[(c.COND_STATUS_CD == 1) & (c.RESERVCD == 0) & c.SITECLCD.between(1, 6)
          & np.isfinite(c.CONDPROP_UNADJ) & (c.CONDPROP_UNADJ > 0)].copy()
    grid = rasterio.open(a.grid)
    H, W = grid.height, grid.width
    tf = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    x, y = tf.transform(c.LON.to_numpy(), c.LAT.to_numpy())
    rr, cc = rasterio.transform.rowcol(grid.transform, x, y)
    c["r"], c["c"] = np.asarray(rr), np.asarray(cc)
    c = c[(c.r >= 0) & (c.r < H) & (c.c >= 0) & (c.c < W)]
    w = c.CONDPROP_UNADJ.to_numpy()
    szok = c.STDSZCD.isin([1, 2, 3, 5]).to_numpy()
    agok = (np.isfinite(c.STDAGE) & (c.STDAGE >= 0)).to_numpy()
    out = np.full((2, 4, H, W), np.nan, "float64")
    nsz = np.zeros((2, H, W)); nag = np.zeros((2, H, W))
    for e in (0, 1):
        m = (c.epoch == e).to_numpy()
        r_, c_ = c.r.to_numpy()[m], c.c.to_numpy()[m]

        def acc(val, ok):
            num = np.zeros((H, W)); den = np.zeros((H, W)); cnt = np.zeros((H, W))
            np.add.at(num, (r_[ok[m]], c_[ok[m]]), (w * val)[m][ok[m]])
            np.add.at(den, (r_[ok[m]], c_[ok[m]]), w[m][ok[m]])
            np.add.at(cnt, (r_[ok[m]], c_[ok[m]]), 1.0)
            return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan), cnt
        sz = c.STDSZCD.to_numpy(); ag = c.STDAGE.to_numpy()
        out[e, 0], nsz[e] = acc((sz == 3).astype(float), szok)
        out[e, 1], _ = acc((sz == 1).astype(float), szok)
        out[e, 2], nag[e] = acc(np.where(agok, ag, 0.0), agok)
        out[e, 3], _ = acc((np.where(agok, ag, 999) <= 20).astype(float), agok)
    mask = (nsz.min(0) >= a.min_cond) & (nag.min(0) >= a.min_cond)
    out[:, :, ~mask] = np.nan
    years = np.arange(1940, 2026)
    arr = np.where((years < 1996)[:, None, None, None], out[0][None], out[1][None]).astype("float32")
    # per-state audit
    rows = []
    for st, g in c.groupby("STATEAB"):
        for e in (0, 1):
            h = g[g.epoch == e]
            if len(h) == 0:
                continue
            rows.append({"state": st, "epoch": ["early", "modern"][e],
                         "invyr": ",".join(str(v) for v in sorted(h.INVYR.unique())), "n_cond": len(h),
                         "stdszcd_valid": round(float(h.STDSZCD.isin([1, 2, 3, 5]).mean()), 3),
                         "stdage_valid": round(float((np.isfinite(h.STDAGE) & (h.STDAGE >= 0)).mean()), 3),
                         "ss_share": round(float(np.average((h.STDSZCD == 3), weights=h.CONDPROP_UNADJ)), 3),
                         "designcd": ",".join(str(int(v)) for v in sorted(h.DESIGNCD.dropna().unique())[:6])})
    aud = pd.DataFrame(rows)
    aud.to_csv(os.path.splitext(a.out)[0] + "_audit.csv", index=False)
    info = {"early_invyr_by_statecd": early_yr, "cells_masked_in": int(mask.sum()), "min_cond": a.min_cond,
            "n_cond_early": int((c.epoch == 0).sum()), "n_cond_modern": int((c.epoch == 1).sum()),
            "median_cond_per_cell": [float(np.median(nsz[e][mask])) for e in (0, 1)] if mask.any() else None}
    np.savez(a.out, years=years, fia=arr, fia_mask=mask, fia_names=np.array(NAMES), info=np.array(json.dumps(info)))
    print(f"[fia] -> {a.out}: {json.dumps(info)}")
    print(aud.to_string(index=False))


if __name__ == "__main__":
    main()
