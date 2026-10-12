"""E060: build DESK's auxiliary eBird targets from the Status & Trends trend product (reference species only).

    python research/exp/build_ebird_aux.py --cache <research cache> --trends-dir <dir> --out-dir <dir> [--years 2012 2021]

For each of the cache's 96 REFERENCE species (never dev or sealed species -- DESK already trains on the reference set
through ESK, so no new species information enters the representation; never House Finch) with a breeding or resident
trend product: the mid-window relative abundance ``abd`` and the per-year trend ``abd_ppy`` are gridded onto the model
grid by the production rasterizer, and two targets are written, each (n_years, H, W, S) of log1p abundance:

  ebird_aux_st.npz  SPATIOTEMPORAL: abd x (1 + ppy/100) ** (year - mid-window) -- the product's own annual path,
                    inside its window only (never extrapolated outside it, as the August design required);
  ebird_aux_sp.npz  SPATIAL ONLY: abd in every year -- the same places and species with the temporal part removed,
                    so the two arms separate "eBird's coverage regularizes" from "eBird's trend helps".

Years: the window common to every kept species (--years, default 2012-2021: the products' windows are 2011-2021 or
2012-2022). Footprint: cells where at least one kept species is estimated; inside it, a species without an estimate is
written as 0 abundance (outside its modelled range); outside it every column is NaN and DESK never sees the cell.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
SEASONS_OK = ("breeding", "resident")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--trends-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--years", type=int, nargs=2, default=[2012, 2021])
    a = ap.parse_args()
    import rasterio
    from src.config_utils import load_data_config
    from src.data.preprocess.ebird_trend import rasterize_parquet

    meta = json.load(open(os.path.join(a.cache, "meta.json")))
    ref = [c for c in meta["community"] if c != "houfin"]
    assert not set(ref) & set(meta.get("sealed_species", [])) and not set(ref) & set(meta["dev_species"])
    with rasterio.open(load_data_config()["grid"]["ref_raster"]) as r:
        crs, tr, H, W = r.crs, r.transform, r.height, r.width
    years = list(range(a.years[0], a.years[1] + 1))
    abd, ppy, mids, kept, skipped = [], [], [], [], {}
    for code in ref:
        hits = sorted(p for p in glob.glob(os.path.join(a.trends_dir, f"{code}_*_ebird-trends_*.parquet"))
                      if "_folds_" not in os.path.basename(p))
        if not hits:
            skipped[code] = "no product"
            continue
        season = os.path.basename(hits[0])[len(code) + 1:].split("_ebird-trends")[0]
        if season not in SEASONS_OK:
            skipped[code] = f"season {season}"
            continue
        g, (sy, ey) = rasterize_parquet(hits[0], crs, tr, H, W)
        if sy > years[0] or ey < years[-1]:
            skipped[code] = f"window {sy}-{ey}"
            continue
        abd.append(g["abd"]), ppy.append(g["abd_ppy"]), mids.append(0.5 * (sy + ey)), kept.append(code)
    A, P = np.stack(abd, -1).astype("float64"), np.stack(ppy, -1).astype("float64")   # (H, W, S)
    foot = np.isfinite(A).any(-1)
    A0 = np.where(np.isfinite(A), np.clip(A, 0, None), 0.0)
    P0 = np.where(np.isfinite(P), np.clip(P, -99.0, None), 0.0)
    mid = np.array(mids)
    st = np.stack([np.log1p(A0 * (1 + P0 / 100.0) ** (y - mid)) for y in years]).astype("float32")
    sp = np.repeat(np.log1p(A0)[None], len(years), axis=0).astype("float32")
    st[:, ~foot] = np.nan
    sp[:, ~foot] = np.nan
    os.makedirs(a.out_dir, exist_ok=True)
    for name, Y in (("ebird_aux_st.npz", st), ("ebird_aux_sp.npz", sp)):
        np.savez_compressed(os.path.join(a.out_dir, name), years=np.array(years), Y=Y, footprint=foot,
                            species=np.array(kept))
    within = st[:, foot] - st[:, foot].mean(0, keepdims=True)
    info = {"species_kept": kept, "skipped": skipped, "years": years, "footprint_cells": int(foot.sum()),
            "grid": [H, W], "zero_filled_share": float(1 - np.isfinite(A[foot]).mean()),
            "st_within_cell_sd": float(within.std()), "sp_between_cell_sd": float(sp[0][foot].std())}
    json.dump(info, open(os.path.join(a.out_dir, "ebird_aux_meta.json"), "w"), indent=1)
    print(f"[E060] {len(kept)} of {len(ref)} reference species kept ({len(skipped)} skipped: "
          f"{sorted(set(v.split()[0] for v in skipped.values()))}); years {years[0]}-{years[-1]}; footprint "
          f"{int(foot.sum())} cells; zero-filled {info['zero_filled_share']:.1%}; within-cell sd (temporal) "
          f"{info['st_within_cell_sd']:.3f} vs between-cell sd {info['sp_between_cell_sd']:.3f}")


if __name__ == "__main__":
    main()
