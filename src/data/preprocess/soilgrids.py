"""Reproject the aggregated SoilGrids 5000 m tiles onto the model grid (static).

SoilGrids COGs are in interrupted Goode Homolosine (ESRI:54052), so they must be
reprojected — not block-aggregated — to the model Albers grid; `reproject_match`
with ``average`` is the linear areal aggregate (5 km → ~25 km). Soil is
time-invariant, so this runs once, one output raster per property×depth.
"""
import glob
import os

import rioxarray  # noqa: F401  (registers .rio)

from src.config_utils import load_data_config
from src.processing import regrid


def preprocess(in_dir, out_dir, ref):
    """Reproject every ``*_mean_5000.tif`` in ``in_dir`` onto ``ref``. Returns count."""
    os.makedirs(out_dir, exist_ok=True)
    tiles = sorted(glob.glob(os.path.join(in_dir, "*_mean_5000.tif")))
    for tif in tiles:
        da = rioxarray.open_rasterio(tif, masked=True)
        out = regrid.reproject_to_ref(da, ref, resampling="average")
        out.rio.to_raster(os.path.join(out_dir, os.path.basename(tif).replace("_5000.tif", "_grid.tif")))
    print(f"SoilGrids: reprojected {len(tiles)} tiles -> {out_dir}")
    return len(tiles)


def main():
    cfg = load_data_config()
    dr = cfg["datasets_root"]
    scfg = cfg.get("soilgrids", {})
    in_dir = os.path.join(dr, scfg.get("out_subdir", "soilgrids_5000m"))
    out_dir = os.path.join(dr, "soilgrids_grid")
    preprocess(in_dir, out_dir, regrid.load_ref(cfg))


if __name__ == "__main__":
    main()
