"""E078: size of DESK's backcast z change per cell, arm vs control (no readout).

    python research/exp/e078_zchange.py --cache-root <research_cache> --pairs ARM:CONTROL ... --cover cover.npz --out DIR

Median |mean z_ema 1953-65 - mean z_ema 2005-24| over the cached cells, overall and in the cover-change top tercile vs
the rest, and the ratio arm / control.
"""
import argparse
import json
import os

import numpy as np


def change(root, cache):
    Z = np.load(os.path.join(root, cache, "z_ema_all.npy"), mmap_mode="r")
    y = np.load(os.path.join(root, cache, "years.npy")); cells = np.load(os.path.join(root, cache, "cells.npy"))
    a = np.asarray(Z[:, (y >= 1953) & (y <= 1965)].mean(1)); b = np.asarray(Z[:, (y >= 2005) & (y <= 2024)].mean(1))
    return np.linalg.norm(a - b, axis=1), cells


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-root", required=True); ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--cover", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    M = np.load(a.cover)
    res = {}
    for p in a.pairs:
        arm, ctl = p.split(":")
        da, cells = change(a.cache_root, arm); dc, _ = change(a.cache_root, ctl)
        top = M["change"][cells[:, 0], cells[:, 1]] > M["terciles"][1]
        r = lambda m: {"control": float(np.median(dc[m])), "arm": float(np.median(da[m])),
                       "ratio": float(np.median(da[m]) / np.median(dc[m]))}
        res[arm] = {"all": r(np.ones(len(da), bool)), "cover_top": r(top), "cover_rest": r(~top)}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e078_zchange.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
