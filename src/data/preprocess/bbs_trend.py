"""Align USGS BBS trend rasters (tr{AOU}.tif, %/yr) onto the model grid.

For each community species (``community_trend_list``: species_code + AOU), read
its BBS trend raster and reproject onto ``grid.ref_raster``. Because the ref grid
is built on the BBS 27 km lattice (same CRS ESRI:102003, resolution and origin
residue -- see build_ref_grid / data_config ``grid``), this is a **nearest
clip/pad with zero resampling** of the ground truth. Cells outside a species'
mapped range are NaN.

The cell value is the long-term (1966-2022) geometric-mean **percent-per-year**
population change; it is NOT winsorized here (that stays faithful) -- the
trend->abundance step (train_DESK/trend_community.py) clips the heavy tails that
inverse-distance interpolation produces at sparse-coverage range margins.

Output ``trends.bbs_trend_grid`` (.npz): ``rate`` (n_species, H, W) float32,
``species_code`` (n_species,), ``aou`` (n_species,), ``valid`` (H, W) bool.

    python -m src.data.preprocess.bbs_trend
"""
import argparse
import os

import numpy as np
import pandas as pd

from src.config_utils import load_data_config
from src.processing import regrid


def align_trend_raster(aou, trend_dir, ref):
    """Reproject one ``tr{AOU}.tif`` onto the ref grid; return (H, W) float32 or None.

    NaN outside the species' mapped range. ``nearest`` because the grids share
    CRS + lattice, so this copies source cells exactly (no interpolation).
    """
    import rioxarray  # noqa: F401 (registers .rio)

    path = os.path.join(trend_dir, f"tr{int(aou):05d}.tif")
    if not os.path.exists(path):
        return None
    da = rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True)
    out = regrid.reproject_to_ref(da, ref, resampling="nearest")
    return np.asarray(out.values, dtype="float32")


def build(community_csv, trend_dir, out_path):
    comm = pd.read_csv(community_csv)
    ref = regrid.load_ref()
    H, W = int(ref.sizes["y"]), int(ref.sizes["x"])

    rates, codes, aous, missing = [], [], [], []
    for _, r in comm.iterrows():
        code, aou = str(r["species_code"]), int(r["aou"])
        arr = align_trend_raster(aou, trend_dir, ref)
        if arr is None:
            missing.append((code, aou))
            continue
        rates.append(arr)
        codes.append(code)
        aous.append(aou)
    if not rates:
        raise SystemExit(f"no BBS trend rasters found in {trend_dir} for {len(comm)} community species")

    rate = np.stack(rates).astype("float32")          # (n_species, H, W)
    valid = np.isfinite(rate).any(axis=0)             # cells covered by >=1 species
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, rate=rate, species_code=np.array(codes),
             aou=np.array(aous, dtype=int), valid=valid)
    print(f"[bbs_trend] {len(codes)} species aligned to {H}x{W} grid "
          f"({int(valid.sum())} covered cells); {len(missing)} missing rasters"
          + (f": {missing}" if missing else "."))
    print(f"[bbs_trend] wrote -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default=None, help="community_trend.csv (default: config).")
    ap.add_argument("--trend-dir", default=None, help="Dir of BBS tr{AOU}.tif (default: config).")
    ap.add_argument("--out", default=None, help="Output .npz (default: trends.bbs_trend_grid).")
    args = ap.parse_args()

    dcfg = load_data_config()
    dr = dcfg["datasets_root"]
    community = args.community or dcfg["community_trend_list"]
    trend_dir = args.trend_dir or os.path.join(dr, dcfg["sciencebase"]["out_subdirs"]["bbs_trends"])
    out = args.out or dcfg["trends"]["bbs_trend_grid"]
    build(community, trend_dir, out)


if __name__ == "__main__":
    main()
