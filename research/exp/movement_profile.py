"""E041 measure (3): does a model keep moving before 1950? Raw z (r24) of every surveyed cell, 1902-2025, and the
median over cells of |z(decade) - z(1946-55)| for each decade -- the shape of the backcast trajectory, and whether it
saturates before 1950 (production's does: E035, M22).

    ESK_DESK_CONFIG=<run overlay> RESEARCH_OUT=<run dir> python research/exp/movement_profile.py --keys-cache <dir> --out <dir> --tag <name>
    python research/exp/movement_profile.py --summarize <out>

Run under research/exp/static_channels.py for a model trained with static channels, so it is encoded as trained.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

DECADES = [(1902, 1911), (1912, 1921), (1922, 1931), (1932, 1941), (1946, 1955), (1956, 1965), (1966, 1975),
           (1976, 1985), (1986, 1995), (1996, 2005), (2006, 2015), (2016, 2025)]
REF = (1946, 1955)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keys-cache")
    ap.add_argument("--out")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from src.config_utils import load_config
    from src.community_encoder.train_DESK.validate_spacetime import encode_points
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config()
    keys = np.load(os.path.join(a.keys_cache, "keys.npy"))
    cells = np.unique(keys[:, :2], axis=0)
    years = np.arange(1902, 2026)
    grid = np.array([(r_, c_, y) for r_, c_ in cells for y in years], "int32")
    Z, ok = encode_points(cfg, grid)
    Z = Z.reshape(len(cells), len(years), -1)[:, :, : a.rank].astype("float64")
    Z[~ok.reshape(len(cells), len(years))] = np.nan
    wm = lambda lo, hi: np.nanmean(Z[:, (years >= lo) & (years <= hi)], axis=1)
    ref = wm(*REF)
    prof = {f"{lo}-{hi}": float(np.nanmedian(np.linalg.norm(wm(lo, hi) - ref, axis=1))) for lo, hi in DECADES}
    res = {"tag": a.tag, "n_cells": int(len(cells)), "ref": REF, "profile": prof,
           "pre1950_over_post1950": prof["1902-1911"] / max(prof["2016-2025"], 1e-12)}
    with open(os.path.join(a.out, f"movement_{a.tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    rows = [json.load(open(os.path.join(out, f), encoding="utf-8")) for f in sorted(os.listdir(out))
            if f.startswith("movement_") and f.endswith(".json")]
    if not rows:
        return
    ds = list(rows[0]["profile"])
    print(f"median |z(decade) - z(1946-55)| over {rows[0]['n_cells']} cells (raw z, r24):")
    print(f"  {'model':34s} " + " ".join(f"{d[:4]}s" for d in ds) + "  1902-11 / 2016-25")
    for r in rows:
        print(f"  {r['tag']:34s} " + " ".join(f"{r['profile'][d]:.3f}" for d in ds)
              + f"  {r['pre1950_over_post1950']:.2f}")


if __name__ == "__main__":
    main()
