#!/usr/bin/env python3
"""Persist the west / Great-Plains-barrier / east zone raster on the model grid.

WHY THIS EXISTS AS A BUILD STEP. The source vector
(``data/ecoregions/NA_CEC_Eco_Level1.shp``, EPA/CEC Ecoregions of North America
Level I) is NOT in git -- ``/data/`` is gitignored -- and is not fetched by
``scripts/tacc/download_all.sh``. It exists only where it was downloaded by hand.
So the barrier-crossing diagnostic cannot read the shapefile on HPC; it reads this
raster instead, and skips itself cleanly if the raster is absent. Run this locally
where the shapefile lives, then ship the output:

    python scripts/build_great_plains_mask.py
    scp <processed>/regions/great_plains_zones_27km.tif \\
        ls6.tacc.utexas.edu:/work/07980/bwa386/ls6/houfin/processed/regions/

Zones are the GAP-FREE corridor partition (see
``src/data/preprocess/great_plains.corridor_zones``): the barrier is the closed
x-interval between the polygon's per-row west and east edges, so every cell is
assigned exactly one zone. A partition with holes would let a dispersal operator
projected onto it silently lose mass into unclassified cells.

Land masking is deliberately NOT applied -- the zones are pure geography and the
consumer intersects them with its own land mask, so this raster stays valid if the
land mask is rebuilt.

Follows the grid/IO/provenance-tag pattern of ``scripts/build_disease_arrival_map.py``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config_utils import load_age_model_config, load_data_config
from src.data.preprocess.great_plains import (
    ZONE_BARRIER, ZONE_EAST, ZONE_NAMES, ZONE_WEST,
    build_zone_array, load_great_plains_geom,
)

DEFAULT_SHP = _REPO / "data" / "ecoregions" / "NA_CEC_Eco_Level1.shp"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapefile", default=str(DEFAULT_SHP),
                    help="EPA/CEC Ecoregions Level I shapefile.")
    ap.add_argument("--grid", default=None,
                    help="Reference raster defining the model grid "
                         "(default: age-model ocean mask, else data/ref_grid_27km.tif).")
    ap.add_argument("--out", default=None,
                    help="Output GeoTIFF (default: <processed>/regions/great_plains_zones_27km.tif).")
    ap.add_argument("--clean-tol-cells", type=float, default=2.0,
                    help="Morphological smoothing scale, in grid cells (default 2).")
    args = ap.parse_args()

    acfg = load_age_model_config()
    dcfg = load_data_config()

    # Same grid-donor fallback chain as build_disease_arrival_map.py: the ocean mask
    # is the grid of record, but it is a built product and may be absent on a fresh
    # checkout, whereas ref_grid_27km.tif builds standalone from config alone.
    grid_path = args.grid or acfg["ocean_mask"]
    if not os.path.exists(grid_path):
        fallback = str(_REPO / "data" / "ref_grid_27km.tif")  # repo-relative, not cwd
        print(f"  {grid_path} unavailable; falling back to {fallback}")
        grid_path = fallback
    if not os.path.exists(grid_path):
        raise SystemExit(f"no grid donor raster found (tried {grid_path})")
    if not os.path.exists(args.shapefile):
        raise SystemExit(
            f"ecoregion shapefile not found: {args.shapefile}\n"
            "Download EPA/CEC Ecoregions of North America Level I from "
            "https://www.epa.gov/eco-research/ecoregions-north-america into "
            "data/ecoregions/ . It is not in git and not in download_all.sh.")

    with rasterio.open(grid_path) as src:
        ny, nx = src.height, src.width
        transform, crs = src.transform, src.crs
    res_m = abs(transform.a)
    print(f"grid: {grid_path}  {ny}x{nx} @ {res_m:.0f} m  {crs}")

    box_bounds = dcfg["grid"]["box_bounds"]
    project_crs = dcfg["grid"]["box_crs"]
    tol = args.clean_tol_cells * res_m
    print(f"reading {args.shapefile} -> {project_crs}, cleaning at {tol:.0f} m")
    geom = load_great_plains_geom(args.shapefile, project_crs, tol)

    zones = build_zone_array(geom, transform, ny, nx, box_bounds)
    for code in (ZONE_WEST, ZONE_BARRIER, ZONE_EAST):
        n = int((zones == code).sum())
        print(f"  {ZONE_NAMES[code]:>8}: {n:6d} cells ({n / zones.size:6.1%})")

    out = args.out or os.path.join(
        os.path.normpath(os.path.join(os.path.dirname(acfg["input_dir"].rstrip("/")), "..")),
        "regions", "great_plains_zones_27km.tif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with rasterio.open(tmp, "w", driver="GTiff", height=ny, width=nx, count=1,
                       dtype="uint8", crs=crs, transform=transform, nodata=0,
                       compress="deflate") as dst:
        dst.write(zones, 1)
        dst.update_tags(
            source=os.path.basename(args.shapefile),
            method=("per-row corridor partition between the cleaned GREAT PLAINS "
                    "Level I polygon's west and east edges; gap-free by construction"),
            codes=", ".join(f"{c}={n}" for c, n in sorted(ZONE_NAMES.items())),
            clean_tol_m=f"{tol:.0f}",
            land_masked="no (consumer intersects with its own land mask)",
        )
    os.replace(tmp, out)
    print(f"wrote {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(zones, cmap=mcolors.ListedColormap(["#4575b4", "#fdae61", "#d73027"]),
              norm=mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], 3))
    ax.set_title("Great Plains barrier zones (west | corridor | east)")
    ax.axis("off")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in
                       (("#4575b4", "west"), ("#fdae61", "barrier corridor"),
                        ("#d73027", "east"))], loc="lower left", fontsize=8)
    png = os.path.splitext(out)[0] + ".png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
