"""E048: a new states dir = the existing yearly states + 12 CropAT-US channels folded into the LANDUSE stream.

    python research/exp/build_cropat_states.py --src <states dir (has yearly_states/ + state_schema.json)>
        --cropat <cropat_27km.npz with types (cropat_aggregate.py --types)> --out <new states dir> [--tau 2]

New channels, appended to landuse (33 -> 45; every later stream shifts by 12):
    cropat_density              crop fraction of the cell (county-census-based CropAT-US, 1850-2021)
    cropat_<type> x 10          each crop type's fraction of the cell (corn, soybean, three wheats, cotton, sorghum,
                                barley, rice, others)
    cropat_avail                1 where CropAT exists (CONUS: BUI available in 2015), else 0 -- the stream's
                                indicator_variable, so it is exempt from transform/standardization and masked
                                atomically with the stream (covariate_io, augment), exactly as bui_avail
Each CropAT channel gets the landuse stream's light input EMA (tau 2 y, from 1850; years after 2021 hold 2021, the
same constant hold build_states uses past a source's last snapshot). Outside CONUS each channel is a time-CONSTANT
fill (its CONUS mean over 1940-2025), so no temporal signal is invented there; cells invalid in the source states stay
NaN, so the valid-cell mask is unchanged. Folding into landuse (rather than a new stream) keeps the architecture
identical apart from the landuse input width (a new stream costs ~500k parameters). Self-checks: original channels
bit-identical, valid masks identical, schema dims consistent. The source states are only read.
"""
import argparse
import json
import os
import shutil

import numpy as np

TYPE_NAMES = ["corn", "soybean", "winter_wheat", "spring_wheat", "durum_wheat", "cotton", "sorghum", "barley", "rice",
              "others"]
NEW_VARS = ["cropat_density"] + [f"cropat_{t}" for t in TYPE_NAMES] + ["cropat_avail"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src")
    ap.add_argument("--cropat")
    ap.add_argument("--out")
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--fill-years", type=int, nargs=2, default=[1940, 2025])
    a = ap.parse_args()
    src_ys = os.path.join(a.src, "yearly_states")
    schema = json.load(open(os.path.join(a.src, "state_schema.json")))
    streams = schema["streams"]
    li = [i for i, s in enumerate(streams) if s["name"] == "landuse"][0]
    assert not streams[li].get("indicator_variable"), "landuse already has an indicator"
    years = sorted(int(f[6:10]) for f in os.listdir(src_ys) if f.startswith("state_") and f.endswith(".npz"))
    z0 = np.load(os.path.join(src_ys, f"state_{years[0]}.npz"))
    H, W = z0["landuse"].shape[:2]
    # CropAT raw yearly stack (Y, H, W, 11)
    ca = np.load(a.cropat)
    cy = ca["years"].astype(int)
    raw = np.concatenate([ca["crop"][:, :, :, None], np.moveaxis(ca["types"], 1, -1)], axis=-1).astype("float64")
    assert raw.shape[1:3] == (H, W), f"CropAT grid {raw.shape[1:3]} != states grid {(H, W)}"
    raw = np.nan_to_num(raw, nan=0.0)                      # outside the raster: filled below anyway
    bui = np.load(os.path.join(src_ys, "state_2015.npz"))["bui"]
    bui_vars = streams[[i for i, s in enumerate(streams) if s["name"] == "bui"][0]]["variables"]
    avail = (np.nan_to_num(bui[..., bui_vars.index("bui_avail")]) > 0.5).astype("float64")
    # EMA from CropAT's first year through the last state year
    alpha = 1.0 - np.exp(-1.0 / a.tau) if a.tau > 0 else 1.0
    ema, state = {}, None
    for y in range(int(cy.min()), max(years) + 1):
        curr = raw[np.searchsorted(cy, min(y, int(cy.max())))]
        state = curr.copy() if state is None else alpha * curr + (1.0 - alpha) * state
        if y in years:
            ema[y] = state.copy()
    fy = [y for y in years if a.fill_years[0] <= y <= a.fill_years[1]]
    on = avail > 0.5
    fill = np.mean([ema[y][on] for y in fy], axis=(0, 1))   # (11,) CONUS mean per channel
    os.makedirs(os.path.join(a.out, "yearly_states"), exist_ok=True)
    for y in years:
        z = np.load(os.path.join(src_ys, f"state_{y}.npz"))
        arrays = {k: z[k] for k in z.files}
        lu = arrays["landuse"]
        new = np.where(on[..., None], ema[y], fill[None, None, :])
        new = np.concatenate([new, avail[..., None]], axis=-1)
        invalid = np.isnan(lu).any(-1)
        new[invalid] = np.nan
        arrays["landuse"] = np.concatenate([lu, new.astype(lu.dtype)], axis=-1)
        np.savez_compressed(os.path.join(a.out, "yearly_states", f"state_{y}.npz"), **arrays)
    # schema: widen landuse, shift later streams, declare the indicator
    ns = json.loads(json.dumps(schema))
    shift = len(NEW_VARS)
    ls = ns["streams"][li]
    ls["variables"] = list(ls["variables"]) + NEW_VARS
    ls["end"], ls["dim"] = int(ls["end"]) + shift, int(ls["dim"]) + shift
    ls["indicator_variable"] = "cropat_avail"
    for s in ns["streams"][li + 1:]:
        s["start"], s["end"] = int(s["start"]) + shift, int(s["end"]) + shift
    ns["cropat"] = {"source": os.path.abspath(a.cropat), "tau": a.tau, "fill": fill.tolist(), "variables": NEW_VARS,
                    "note": "E048: CropAT-US channels folded into landuse; see research/exp/build_cropat_states.py"}
    for k in ("dim", "total_dim", "n_channels"):
        if k in ns:
            ns[k] = int(ns[k]) + shift
    with open(os.path.join(a.out, "state_schema.json"), "w") as fh:
        json.dump(ns, fh, indent=1)
    for extra in os.listdir(a.src):                         # carry any other sidecars (not history_vectors)
        p = os.path.join(a.src, extra)
        if os.path.isfile(p) and extra not in ("state_schema.json", "history_vectors.npy"):
            shutil.copy2(p, os.path.join(a.out, extra))
    # self-checks on two years
    for y in (years[0], 1990 if 1990 in years else years[-1]):
        zs = np.load(os.path.join(src_ys, f"state_{y}.npz"))
        zn = np.load(os.path.join(a.out, "yearly_states", f"state_{y}.npz"))
        for k in zs.files:
            o, n_ = zs[k], zn[k]
            if k == "landuse":
                assert np.array_equal(np.nan_to_num(o, nan=-9e9), np.nan_to_num(n_[..., :o.shape[-1]], nan=-9e9))
                assert np.array_equal(np.isnan(o).any(-1), np.isnan(n_).any(-1)), "valid mask changed"
            else:
                assert np.array_equal(np.nan_to_num(o, nan=-9e9), np.nan_to_num(n_, nan=-9e9)), k
    tot = sum(int(s["dim"]) for s in ns["streams"])
    assert ns["streams"][-1]["end"] == tot, "schema ends inconsistent"
    print(f"[cropat-states] {len(years)} years {years[0]}-{years[-1]} -> {a.out}; landuse 33 -> {ls['dim']}; total "
          f"{tot} channels; CONUS cells {int(on.sum())}; fill {np.round(fill, 4).tolist()}", flush=True)


if __name__ == "__main__":
    main()
