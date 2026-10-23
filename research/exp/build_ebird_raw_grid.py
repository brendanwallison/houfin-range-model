"""E062: aggregate the reduced raw eBird (EBD) checklists onto the model's 27 km grid, by cell and year.

    python research/exp/build_ebird_raw_grid.py --reduced DIR [DIR ...] --out grid.npz [--short]

Input: src/data/preprocess/ebd_reduce.py outputs (checklists.parquet: complete breeding-season checklists, late May -
mid July, 2002-2025, <= 5 h, <= 5 km, <= 10 observers, inside the study box; counts.parquet: detections of the kept
species -- the 96 reference + 273 dev species; sealed species and House Finch were refused at read time). A kept
checklist without a row for a species is a genuine non-detection of it.

Output (npz): keys (n, 3) int32 [row, col, year] for every grid cell-year with >= 1 checklist; n_chk (n,) checklists;
det (n, S) int32 checklists detecting each species; species (S,). --short keeps only checklists <= 60 min and <= 2 km
(an effort-restricted variant, since eBird effort composition changed with its ~25x growth over 2008-2024).
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reduced", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--short", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    import rasterio
    from rasterio.warp import transform as warp_transform
    from src.config_utils import load_data_config

    with rasterio.open(load_data_config()["grid"]["ref_raster"]) as r:
        crs, tr, H, W = r.crs, r.transform, r.height, r.width
    chks, dets = [], []
    for d in a.reduced:
        ch = pd.read_parquet(os.path.join(d, "checklists.parquet"),
                             columns=["checklist_id", "lat", "lon", "year", "duration_min", "distance_km"])
        if a.short:
            ch = ch[(ch["duration_min"] <= 60) & (ch["distance_km"] <= 2.0)]
        xs, ys = warp_transform("EPSG:4326", crs, ch["lon"].tolist(), ch["lat"].tolist())
        col = np.floor((np.asarray(xs) - tr.c) / tr.a).astype(np.int64)
        row = np.floor((np.asarray(ys) - tr.f) / tr.e).astype(np.int64)
        ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
        ch = ch.assign(row=row, col=col)[ok]
        chks.append(ch[["checklist_id", "row", "col", "year"]])
        ct = pd.read_parquet(os.path.join(d, "counts.parquet"), columns=["checklist_id", "species_code"])
        dets.append(ct[ct["checklist_id"].isin(set(ch["checklist_id"]))])
        print(f"[E062] {os.path.basename(d)}: {int(ok.sum()):,} checklists on the grid, {len(dets[-1]):,} detections",
              flush=True)
    ch = pd.concat(chks, ignore_index=True)
    ct = pd.concat(dets, ignore_index=True)
    assert "houfin" not in set(ct["species_code"])
    g = ch.groupby(["row", "col", "year"]).size().rename("n_chk").reset_index()
    g["k"] = np.arange(len(g))
    kidx = ch.merge(g[["row", "col", "year", "k"]], on=["row", "col", "year"])[["checklist_id", "k"]]
    ct = ct.merge(kidx, on="checklist_id")
    species = np.array(sorted(ct["species_code"].unique()))
    si = {s: i for i, s in enumerate(species)}
    det = np.zeros((len(g), len(species)), dtype=np.int32)
    agg = ct.groupby(["k", "species_code"]).size().reset_index(name="n")
    det[agg["k"].to_numpy(), agg["species_code"].map(si).to_numpy()] = agg["n"].to_numpy()
    np.savez_compressed(a.out, keys=g[["row", "col", "year"]].to_numpy(np.int32), n_chk=g["n_chk"].to_numpy(np.int32),
                        det=det, species=species)
    info = {"cell_years": int(len(g)), "cells": int(g[["row", "col"]].drop_duplicates().shape[0]),
            "checklists": int(len(ch)), "detections": int(len(ct)), "species": int(len(species)), "short": a.short,
            "checklists_by_year": {int(y): int(n) for y, n in ch.groupby("year").size().items()}}
    json.dump(info, open(os.path.splitext(a.out)[0] + "_meta.json", "w"), indent=1)
    print(f"[E062] grid: {info['cell_years']:,} cell-years over {info['cells']:,} cells, {info['checklists']:,} "
          f"checklists, {len(species)} species -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
