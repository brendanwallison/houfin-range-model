"""E035f: is production DESK's movement after 1950 a clock read from land-use MANAGEMENT trends?

    ESK_DESK_CONFIG=config/overlays/production.json python research/exp/clock_freeze.py --keys-cache <dir> --out <dir>
    python research/exp/clock_freeze.py --summarize <out>

Production raw z (r24) for every surveyed cell, 1930-2025, encoded with one group of covariate channels FROZEN at its
1950 value in every later year (covariate_io.load_state_stack replaced for the process): LUH fertilizer + irrigation
(the steepest post-1950 trends, flat before), LUH class fractions, all LUH, temperature, all dynamic channels. For each
variant: the median over cells of |z(window) - z(1946-1955)| for later windows -- how much of DESK's movement since
1950 survives when that group stops moving -- and the same for 1930-39 (before the freeze: identical by construction,
a check). If DESK's post-1950 movement is mostly a management clock, freezing fertilizer + irrigation removes most of
it, freezing the class fractions removes little.
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

TEMP = ("Tave", "Tmax", "Tmin", "DD5", "DD18", "DDsub0", "DDsub18", "NFFD", "Eref")
CLASSES = ("c3ann", "c3nfx", "c3per", "c4ann", "c4per", "pastr", "primf", "primn", "range", "secdf", "secdn", "urban")
FREEZE_YEAR = 1950
WINDOWS = [(1930, 1939), (1956, 1965), (1966, 1975), (1976, 1985), (1986, 1995), (1996, 2005), (2006, 2015),
           (2016, 2025)]
REF = (1946, 1955)


def groups_from_schema(schema):
    g = {"fertl_irrig": [], "luh_classes": [], "luh_all": [], "temperature": [], "dynamic": []}
    for s in schema["streams"]:
        st = int(s["start"])
        cols = list(range(st, int(s["end"])))
        if s["name"] == "climate":
            g["dynamic"] += cols
            g["temperature"] += [st + j for j, v in enumerate(s.get("variables", [])) if v.split("_")[0] in TEMP]
        elif s["name"] == "landuse":
            g["dynamic"] += cols
            g["luh_all"] += cols
            names = s.get("variables") or []
            g["fertl_irrig"] += [st + j for j, v in enumerate(names) if v.startswith(("fertl", "irrig"))]
            g["luh_classes"] += [st + j for j, v in enumerate(names) if v in CLASSES]
        elif s["name"] in ("hyde", "bui"):
            g["dynamic"] += cols
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keys-cache")
    ap.add_argument("--out")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--max-cells", type=int, default=0, help="smoke runs only")
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from src.config_utils import load_config
    from src.community_encoder.train_DESK import covariate_io as cio
    from src.community_encoder.train_DESK.validate_spacetime import encode_points
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config()
    schema = cio.load_schema(os.path.join(cfg["paths"]["hist_dir"], "yearly_states"))
    G = groups_from_schema(schema)
    keys = np.load(os.path.join(a.keys_cache, "keys.npy"))
    cells = np.unique(keys[:, :2], axis=0)
    if a.max_cells:
        cells = cells[:: max(1, len(cells) // a.max_cells)][: a.max_cells]
    years = np.arange(1930, 2026)
    grid = np.array([(r_, c_, y) for r_, c_ in cells for y in years], "int32")
    orig = cio.load_state_stack
    frozen = {"cols": None}
    cache = {}

    def patched(year, states_dir, schema_):
        x = orig(year, states_dir, schema_)
        if frozen["cols"] is not None and int(year) > FREEZE_YEAR:
            if FREEZE_YEAR not in cache:
                cache[FREEZE_YEAR] = orig(FREEZE_YEAR, states_dir, schema_)
            x = x.copy()
            x[..., frozen["cols"]] = cache[FREEZE_YEAR][..., frozen["cols"]]
        return x
    cio.load_state_stack = patched
    res = {"n_cells": int(len(cells)), "freeze_year": FREEZE_YEAR, "ref_window": REF, "windows": WINDOWS,
           "group_sizes": {k: len(v) for k, v in G.items()}, "variants": {}}
    Zs = {}
    for name in ["actual"] + list(G):
        frozen["cols"] = None if name == "actual" else np.array(G[name], int)
        Z, ok = encode_points(cfg, grid)
        Z = Z.reshape(len(cells), len(years), -1)[:, :, : a.rank].astype("float64")
        Z[~ok.reshape(len(cells), len(years))] = np.nan
        Zs[name] = Z
        wm = lambda lo, hi: np.nanmean(Z[:, (years >= lo) & (years <= hi)], axis=1)
        ref = wm(*REF)
        res["variants"][name] = {f"{lo}-{hi}": float(np.nanmedian(np.linalg.norm(wm(lo, hi) - ref, axis=1)))
                                 for lo, hi in WINDOWS}
        print(f"[clock] {name}: done", flush=True)
    cio.load_state_stack = orig
    # share of the ACTUAL post-1950 movement that each freeze removes, per window: 1 - |frozen - ref| / |actual - ref|
    # is not additive across groups; also report the direct size of the removed part |z_actual - z_frozen|
    wm_ = lambda Z, lo, hi: np.nanmean(Z[:, (years >= lo) & (years <= hi)], axis=1)
    res["removed_part"] = {name: {f"{lo}-{hi}": float(np.nanmedian(np.linalg.norm(wm_(Zs["actual"], lo, hi) -
                                                                                 wm_(Zs[name], lo, hi), axis=1)))
                                  for lo, hi in WINDOWS} for name in G}
    with open(os.path.join(a.out, "clock_freeze.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "clock_freeze.json"), encoding="utf-8"))
    ws = [f"{lo}-{hi}" for lo, hi in r["windows"]]
    print(f"production DESK raw z (r24), {r['n_cells']} cells: median |z(window) - z({r['ref_window'][0]}-"
          f"{r['ref_window'][1]})| with a channel group frozen at its {r['freeze_year']} value after {r['freeze_year']}"
          f" (group sizes {r['group_sizes']})")
    print(f"  {'variant':14s} " + " ".join(f"{w:>10s}" for w in ws))
    for name, v in r["variants"].items():
        print(f"  {name:14s} " + " ".join(f"{v[w]:10.3f}" for w in ws))
    print("  size of the part each freeze removes, median |z_actual - z_frozen|:")
    for name, v in r["removed_part"].items():
        print(f"  {name:14s} " + " ".join(f"{v[w]:10.3f}" for w in ws))


if __name__ == "__main__":
    main()
