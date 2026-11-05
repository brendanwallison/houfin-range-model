"""FIELD54: compare the 54 km states before (v1) and after (v2) the climate sub-point union fix.

Affected cells = 54 km land cells that gained sub-points (from the supplementary
climr run). Reports, per climate channel over affected cells x years: corr(v1, v2),
max |change| (raw and / sd of the channel over all land cell-years); how many cells
changed at all; and that unaffected cells and non-climate streams are bit-identical.

    python research/exp/field54_climate_v2_compare.py --s1 .../states --s2 .../states_v2 \
        --missing $HOUFIN_DATA54/climate_extra/missing_subpoints.csv --mask .../ocean_mask_54km.tif \
        --out .../field54_climate_v2_compare.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.masks import read_land_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--missing", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--width", type=int, default=112)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    land = np.asarray(read_land_mask(args.mask), bool)
    miss = pd.read_csv(args.missing, usecols=["row", "col"])
    aff = np.zeros_like(land)
    aff[miss["row"].to_numpy() // 2, miss["col"].to_numpy() // 2] = True
    aff &= land
    other = land & ~aff
    sc = json.load(open(os.path.join(args.s2, "state_schema.json")))
    sc1 = json.load(open(os.path.join(args.s1, "state_schema.json")))
    assert sc["streams"] == sc1["streams"], "schema differs between v1 and v2"
    names = [s["name"] for s in sc["streams"]]
    cvars = next(s["variables"] for s in sc["streams"] if s["name"] == "climate")
    C = len(cvars)

    files = sorted(glob.glob(os.path.join(args.s2, "yearly_states", "state_*.npz")))
    a1, a2, allv = [], [], []
    other_maxdiff = {n: 0.0 for n in names}
    aff_maxdiff_nonclim = {n: 0.0 for n in names if n != "climate"}
    for f in files:
        z1 = np.load(os.path.join(args.s1, "yearly_states", os.path.basename(f)))
        z2 = np.load(f)
        for n in names:
            x1, x2 = z1[n].astype("float64"), z2[n].astype("float64")
            d = np.abs(x1 - x2)
            other_maxdiff[n] = max(other_maxdiff[n], float(np.nanmax(d[other])))
            if n != "climate":
                aff_maxdiff_nonclim[n] = max(aff_maxdiff_nonclim[n], float(np.nanmax(d[aff])))
            else:
                a1.append(x1[aff]); a2.append(x2[aff]); allv.append(x2[land])
    a1 = np.concatenate(a1)        # (years*n_aff, C)
    a2 = np.concatenate(a2)
    allv = np.concatenate(allv)
    sd = allv.std(0)
    n_aff = int(aff.sum())
    n_years = len(files)
    d = np.abs(a2 - a1)
    changed_cell = (d.reshape(n_years, n_aff, C) > 0).any(axis=(0, 2))
    per = {}
    for j, v in enumerate(cvars):
        c = np.corrcoef(a1[:, j], a2[:, j])[0, 1] if a1[:, j].std() > 0 else float("nan")
        per[v] = {"corr": float(c), "max_abs": float(d[:, j].max()),
                  "max_abs_over_sd": float(d[:, j].max() / sd[j]) if sd[j] > 0 else None,
                  "rms_over_sd": float(np.sqrt((d[:, j] ** 2).mean()) / sd[j]) if sd[j] > 0 else None}
    corrs = np.array([p["corr"] for p in per.values()])
    mos = np.array([p["max_abs_over_sd"] for p in per.values()], dtype=float)
    worst = sorted(per.items(), key=lambda kv: kv[1]["corr"])[:8]
    res = {
        "n_land": int(land.sum()), "n_affected_cells": n_aff,
        "n_affected_cells_changed": int(changed_cell.sum()), "n_years": n_years,
        "climate_corr_v1_v2_affected": {"min": float(np.nanmin(corrs)),
                                        "median": float(np.nanmedian(corrs)),
                                        "n_below_0.99": int((corrs < 0.99).sum())},
        "climate_max_abs_over_sd": {"median": float(np.nanmedian(mos)), "max": float(np.nanmax(mos))},
        "worst_channels": worst,
        "unaffected_cells_max_abs_diff_by_stream": other_maxdiff,
        "affected_cells_max_abs_diff_nonclimate": aff_maxdiff_nonclim,
        "per_channel": per,
    }
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_channel"}, indent=1))


if __name__ == "__main__":
    main()
