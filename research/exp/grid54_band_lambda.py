"""Regional fundamental lambda by era for one fitted point (54 km ladder readout).

Same lambda as map_diagnostics (local_growth_lambda of the reconstructed Sa/Sj/Fmax); regions by cell-centre lon/lat.
Usage: band_lambda.py RUN_DIR OUT_JSON   (AGE_MODEL_CONFIG / DATA_CONFIG select the fit, as for map_diagnostics)
"""
import json
import sys
from pathlib import Path

import jax
import numpy as np
import rasterio
from rasterio.warp import transform as crs_transform

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "scripts/viz")
from map_diagnostics import reconstruct_map  # noqa: E402
from src.config_utils import load_age_model_config, load_data_config  # noqa: E402
from src.model.checkpoints import load_map_params  # noqa: E402
from src.model.data_loading import load_data  # noqa: E402
from src.vis.age_model_math import local_growth_lambda  # noqa: E402

ERAS = {"1902-15": (1902, 1915), "1940-55": (1940, 1955), "2010-25": (2010, 2025)}

cfg, dcfg = load_age_model_config(), load_data_config()
data = load_data(cfg["input_dir"], target_device=jax.devices()[0], precision="float64")
params, _ = load_map_params(sys.argv[1])
sim = reconstruct_map(data, params)
lam = np.asarray(local_growth_lambda(sim["Sa_flat"], sim["Sj_flat"], sim["Fmax_flat"]))
years = np.asarray(data["years"])
rows, cols = np.asarray(data["land_rows"]), np.asarray(data["land_cols"])
with rasterio.open(dcfg["grid"]["ref_raster"]) as src:
    x, y = rasterio.transform.xy(src.transform, rows, cols, offset="center")
    lon, lat = map(np.asarray, crs_transform(src.crs, "EPSG:4326", list(x), list(y)))
regions = {"all_land": np.ones(lon.shape, bool),
           "west_of_100W": lon < -100,
           "plains_100_95W": (lon >= -100) & (lon < -95),
           "central_east_95_80W_25_49N": (lon >= -95) & (lon < -80) & (lat >= 25) & (lat < 49),
           "east_of_80W_25_49N": (lon >= -80) & (lat >= 25) & (lat < 49)}
out = {}
for name, m in regions.items():
    out[name] = {"cells": int(m.sum())}
    for era, (a, b) in ERAS.items():
        e = lam[(years >= a) & (years <= b)][:, m].mean(0)
        out[name][era] = {"mean": float(e.mean()), "median": float(np.median(e)), "suitable": float((e > 1).mean())}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
for name, v in out.items():
    print(f"{name:28s} n={v['cells']:5d} " + "  ".join(f"{e}: {v[e]['mean']:.3f} ({v[e]['suitable']:.0%})" for e in ERAS))
