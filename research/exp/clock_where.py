"""E035g: clock or intensification? WHERE production DESK responds to the fertilizer + irrigation trend.

    ESK_DESK_CONFIG=config/overlays/production.json python research/exp/clock_where.py --keys-cache <dir> --out <dir>
    python research/exp/clock_where.py --summarize <out>

LUH's management channels are rates per hectare of each crop, defined even where a cell has almost no cropland. A
model that uses the fertilizer/irrigation trend as a CLOCK responds wherever the channel moves; one that reads it as
INTENSIFICATION responds where crops are. Per surveyed cell: the part of production raw z (r24) that freezing
fertilizer + irrigation at their 1950 values removes, |z_actual - z_frozen| averaged over 1996-2025 and over 1966-85,
against the cell's mean 1996-2025 crop fraction (c3ann + c4ann + c3per + c4per + c3nfx) and forest fraction (primf +
secdf). Reported: Spearman correlations and the median removed part by crop-fraction tercile (ratio top / bottom).
CLOCK predicts a flat relation (ratio ~1); INTENSIFICATION a strong one (ratio >> 1).
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

FREEZE_YEAR = 1950
CROPS = ("c3ann", "c4ann", "c3per", "c4per", "c3nfx")
FOREST = ("primf", "secdf")
WINDOWS = {"1996-2025": (1996, 2025), "1966-1985": (1966, 1985)}


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
    from scipy.stats import spearmanr
    from src.config_utils import load_config
    from src.community_encoder.train_DESK import covariate_io as cio
    from src.community_encoder.train_DESK.validate_spacetime import encode_points
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config()
    states_dir = os.path.join(cfg["paths"]["hist_dir"], "yearly_states")
    schema = cio.load_schema(states_dir)
    lu = [s for s in schema["streams"] if s["name"] == "landuse"][0]
    names, st = lu["variables"], int(lu["start"])
    mgmt = np.array([st + j for j, v in enumerate(names) if v.startswith(("fertl", "irrig"))], int)
    crop_c = np.array([st + names.index(v) for v in CROPS], int)
    forest_c = np.array([st + names.index(v) for v in FOREST], int)
    keys = np.load(os.path.join(a.keys_cache, "keys.npy"))
    cells = np.unique(keys[:, :2], axis=0)
    if a.max_cells:
        cells = cells[:: max(1, len(cells) // a.max_cells)][: a.max_cells]
    years = np.arange(1966, 2026)
    grid = np.array([(r_, c_, y) for r_, c_ in cells for y in years], "int32")
    # crop and forest fractions, 1996-2025 mean, raw state values (fractions of the cell)
    crop = np.zeros(len(cells))
    forest = np.zeros(len(cells))
    for y in range(1996, 2026):
        x = cio.load_state_stack(y, states_dir, schema)
        crop += np.nansum(x[cells[:, 0], cells[:, 1]][:, crop_c], 1) / 30.0
        forest += np.nansum(x[cells[:, 0], cells[:, 1]][:, forest_c], 1) / 30.0
    orig = cio.load_state_stack
    state = {"freeze": False}
    cache = {}

    def patched(year, sd_, sch_):
        x = orig(year, sd_, sch_)
        if state["freeze"] and int(year) > FREEZE_YEAR:
            if "ref" not in cache:
                cache["ref"] = orig(FREEZE_YEAR, sd_, sch_)
            x = x.copy()
            x[..., mgmt] = cache["ref"][..., mgmt]
        return x
    cio.load_state_stack = patched
    Z = {}
    for name, frz in (("actual", False), ("frozen", True)):
        state["freeze"] = frz
        z, ok = encode_points(cfg, grid)
        z = z.reshape(len(cells), len(years), -1)[:, :, : a.rank].astype("float64")
        z[~ok.reshape(len(cells), len(years))] = np.nan
        Z[name] = z
    cio.load_state_stack = orig
    res = {"n_cells": int(len(cells)), "freeze_year": FREEZE_YEAR, "n_mgmt_channels": int(len(mgmt)),
           "crop_fraction_quantiles": [float(np.quantile(crop, q)) for q in (0.1, 0.33, 0.5, 0.67, 0.9)],
           "windows": {}}
    t1, t2 = np.quantile(crop, [1 / 3, 2 / 3])
    terc = np.digitize(crop, [t1, t2])
    for w, (lo, hi) in WINDOWS.items():
        sel = (years >= lo) & (years <= hi)
        rem = np.linalg.norm(np.nanmean(Z["actual"][:, sel], 1) - np.nanmean(Z["frozen"][:, sel], 1), axis=1)
        ok = np.isfinite(rem)
        med = [float(np.median(rem[ok & (terc == k)])) for k in range(3)]
        res["windows"][w] = {"spearman_removed_vs_crop": float(spearmanr(rem[ok], crop[ok])[0]),
                             "spearman_removed_vs_forest": float(spearmanr(rem[ok], forest[ok])[0]),
                             "median_removed_by_crop_tercile": med,
                             "ratio_top_over_bottom": med[2] / med[0] if med[0] > 0 else None,
                             "median_removed_lowest_decile_crop": float(np.median(rem[ok & (crop <= np.quantile(crop, 0.1))])),
                             "median_removed_all": float(np.median(rem[ok]))}
    np.savez(os.path.join(a.out, "clock_where_cells.npz"), cells=cells, crop=crop, forest=forest)
    with open(os.path.join(a.out, "clock_where.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "clock_where.json"), encoding="utf-8"))
    q = r["crop_fraction_quantiles"]
    print(f"production DESK, {r['n_cells']} cells: part of raw z removed by freezing fertilizer + irrigation "
          f"({r['n_mgmt_channels']} channels) at {r['freeze_year']}, against the cell's crop fraction "
          f"(deciles 10/33/50/67/90%: {', '.join(f'{v:.3f}' for v in q)})")
    for w, v in r["windows"].items():
        m = v["median_removed_by_crop_tercile"]
        rt = v["ratio_top_over_bottom"]
        print(f"  {w}: Spearman(removed, crop) {v['spearman_removed_vs_crop']:+.3f}, (removed, forest) "
              f"{v['spearman_removed_vs_forest']:+.3f}; median removed by crop tercile low/mid/high "
              f"{m[0]:.3f} / {m[1]:.3f} / {m[2]:.3f} (ratio {rt:.2f}); lowest-decile-crop cells "
              f"{v['median_removed_lowest_decile_crop']:.3f}; all {v['median_removed_all']:.3f}")


if __name__ == "__main__":
    main()
