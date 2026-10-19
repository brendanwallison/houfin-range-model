"""Spatial audit of what a fitted run calls a western sink.

This deliberately separates three ecological statements that a single phrase
"the West is a sink" conflates: fundamentally unsuitable habitat
(Allee-free lambda <= 1), fundamentally suitable but Allee-dead habitat, and a
viable positive-equilibrium source.  It reports extent and connected regions,
then overlays the historical native seed and positive BBS observations.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as crs_transform
from scipy import ndimage

from src.config_utils import load_age_model_config, load_data_config


def _share(mask, values=None):
    if mask.size == 0:
        return None
    if values is None:
        return float(np.mean(mask))
    den = float(np.sum(values))
    return None if den <= 0 else float(np.sum(values[mask]) / den)


def _observation_overlap(meta, klass, west, first, last):
    years = np.asarray(meta["years"])[np.asarray(meta["obs_time_indices"])]
    rows, cols = np.asarray(meta["obs_rows"]), np.asarray(meta["obs_cols"])
    counts = np.asarray(meta["observed_results"], dtype=float)
    keep = (years >= first) & (years <= last) & (counts > 0) & west[rows, cols]
    k = klass[rows[keep], cols[keep]]
    y = counts[keep]
    cells = np.unique(np.stack([rows[keep], cols[keep]], axis=1), axis=0) if keep.any() else np.empty((0, 2))
    return {
        "years": [int(first), int(last)],
        "positive_records": int(keep.sum()),
        "positive_cells": int(cells.shape[0]),
        "record_fraction_fundamental_sink": _share(k == 0),
        "record_fraction_allee_dead": _share(k == 1),
        "count_fraction_fundamental_sink": _share(k == 0, y),
        "count_fraction_allee_dead": _share(k == 1, y),
    }


def _components(mask, positive_cells, transform, crs, cell_area_km2, label_name):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    rows = []
    for i in range(1, n + 1):
        rr, cc = np.where(lab == i)
        if rr.size == 0:
            continue
        x, y = rasterio.transform.xy(transform, rr, cc, offset="center")
        lon, lat = crs_transform(crs, "EPSG:4326", list(x), list(y))
        rows.append({
            "class": label_name,
            "component": int(i),
            "cells": int(rr.size),
            "area_km2": float(rr.size * cell_area_km2),
            "centroid_lon": float(np.mean(lon)),
            "centroid_lat": float(np.mean(lat)),
            "lon_min": float(np.min(lon)), "lon_max": float(np.max(lon)),
            "lat_min": float(np.min(lat)), "lat_max": float(np.max(lat)),
            "positive_bbs_cells": int(np.sum(positive_cells[rr, cc])),
        })
    return sorted(rows, key=lambda r: r["cells"], reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--components-csv", default=None)
    a = ap.parse_args()

    cfg, dcfg = load_age_model_config(), load_data_config()
    fields_path = Path(a.viz_dir) / "07_source_sink_fields.npz"
    with np.load(fields_path) as z:
        lam_fund = np.asarray(z["lam_fundamental_modern"])
        viable = np.asarray(z["equilibrium_exists"]).astype(bool)
        allee_dead = (np.asarray(z["allee_dead_mask"]).astype(bool)
                      if "allee_dead_mask" in z else (lam_fund > 1) & ~viable)
        window = np.asarray(z["window_years"]).astype(int).tolist()
    land = np.isfinite(lam_fund)
    fundamental_sink = land & (lam_fund <= 1.0)
    klass = np.full(lam_fund.shape, -1, dtype=np.int8)
    klass[fundamental_sink] = 0
    klass[allee_dead] = 1
    klass[land & viable] = 2

    zones_path = os.path.expandvars(dcfg["regions"]["great_plains_zones"])
    with rasterio.open(zones_path) as src:
        zones = src.read(1)
    west = land & (zones == 1)
    ref_path = os.path.expandvars(dcfg["grid"]["ref_raster"])
    with rasterio.open(ref_path) as src:
        transform, crs = src.transform, src.crs
        cell_area_km2 = abs(transform.a * transform.e) / 1e6
    with open(os.path.join(cfg["input_dir"], "metadata.pkl"), "rb") as fh:
        meta = pickle.load(fh)

    obs_rows, obs_cols = np.asarray(meta["obs_rows"]), np.asarray(meta["obs_cols"])
    obs_y = np.asarray(meta["observed_results"])
    positive_cells = np.zeros(land.shape, dtype=bool)
    positive_cells[obs_rows[obs_y > 0], obs_cols[obs_y > 0]] = True
    native = np.asarray(meta.get("initpop_latent", np.zeros(land.shape))) > 0

    west_n = int(west.sum())
    def west_stats(mask):
        m = west & mask
        return {"cells": int(m.sum()), "area_km2": float(m.sum() * cell_area_km2),
                "fraction_of_west": (float(m.sum() / west_n) if west_n else None),
                "native_seed_cells": int((m & native).sum()),
                "positive_bbs_cells": int((m & positive_cells).sum())}

    years = np.asarray(meta["years"])
    report = {
        "source_fields": str(fields_path),
        "modern_window": window,
        "definitions": {"fundamental_sink": "Allee-free lambda <= 1",
                        "allee_dead": "Allee-free lambda > 1 but no positive equilibrium",
                        "viable": "positive equilibrium exists"},
        "west": {"cells": west_n, "area_km2": float(west_n * cell_area_km2)},
        "classes": {"fundamental_sink": west_stats(fundamental_sink),
                    "allee_dead": west_stats(allee_dead),
                    "viable": west_stats(viable)},
        "native_seed": {"west_cells": int((west & native).sum()),
                        "fraction_in_fundamental_sink": _share(fundamental_sink[west & native]),
                        "fraction_allee_dead": _share(allee_dead[west & native])},
        "observations": {
            "early_bbs": _observation_overlap(meta, klass, west, 1966, 1979),
            "modern_bbs": _observation_overlap(meta, klass, west, 2010, int(years[-1])),
        },
    }
    comps = (_components(west & fundamental_sink, positive_cells, transform, crs,
                         cell_area_km2, "fundamental_sink") +
             _components(west & allee_dead, positive_cells, transform, crs,
                         cell_area_km2, "allee_dead"))
    report["largest_components"] = sorted(comps, key=lambda r: r["cells"], reverse=True)[:20]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    csv_path = Path(a.components_csv) if a.components_csv else out.with_suffix(".components.csv")
    if comps:
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(comps[0]))
            w.writeheader(); w.writerows(comps)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
