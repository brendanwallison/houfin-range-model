"""src/data/preprocess/drift_covariates.py: panel build on tiny synthetic rasters."""
import os
import pickle

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.data.preprocess.drift_covariates import build


def _tif(path, arr):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs="ESRI:102003", transform=from_origin(0, 0, 1, 1)) as dst:
        dst.write(arr.astype("float32")[None])


def test_panel_interpolates_sums_transforms_standardises(tmp_path):
    d = tmp_path / "data"; inp = tmp_path / "inp"; os.makedirs(inp)
    for src, var, yrs in [("hyde35_grid", "population_density", (1950, 2000)),
                          ("luh3_grid", "c3ann", (1950, 2000)), ("luh3_grid", "c4ann", (1950, 2000))]:
        os.makedirs(d / src, exist_ok=True)
        for y in yrs:
            _tif(d / src / f"{var}_{y}_grid.tif", np.full((3, 4), 10.0 if y == 1950 else 30.0))
    rows, cols = np.array([0, 1, 2]), np.array([0, 1, 3])
    years = np.arange(1940, 2011)
    pickle.dump({"land_rows": rows, "land_cols": cols, "years": years}, open(inp / "metadata.pkl", "wb"))
    panel = {"log_popd": ("hyde35_grid", ["population_density"], "log1p"),
             "cropland": ("luh3_grid", ["c3ann", "c4ann"], "none")}
    out = build(str(d), str(inp), str(tmp_path / "p.npz"), panel=panel)
    z = np.load(out)
    assert z["x"].shape == (years.size, 3, 2) and list(z["names"]) == ["log_popd", "cropland"]
    crop_raw = z["x"][..., 1] * z["sd"][1] + z["mean"][1]
    assert np.allclose(crop_raw[years == 1940], 20.0) and np.allclose(crop_raw[years == 2010], 60.0)  # ends held
    assert np.allclose(crop_raw[years == 1975], 40.0)                                                 # linear between
    assert np.allclose(z["x"].mean(axis=(0, 1)), 0.0, atol=1e-5)                                     # standardised
