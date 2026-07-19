"""Tests for the subgrid climate path pure cores (cluster/climr-free).

Covers the sub-cell centroid mesh (parent mapping, grid density, NaN-drop) and the
spatial-quantile aggregation that turns per-sub-point downscale output into the
per-cell climate_{q10,q50,q90} the downstream expects. Runs standalone or pytest.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.preprocess.subcell_centroids import build_subcell_centroids
from src.data.acquire.climatena import quantile_aggregate


def test_subcell_mesh():
    H, W, g = 2, 3, 2
    ref_tr = from_origin(0, 50000, 25000, 25000)
    fine_tr = ref_tr * rasterio.Affine.scale(1 / g, 1 / g)
    with tempfile.TemporaryDirectory() as d:
        dem = os.path.join(d, "dem.tif")
        arr = np.arange(H * g * W * g, dtype="float32").reshape(H * g, W * g)
        arr[0, 0] = np.nan  # ocean/nodata sub-point -> dropped
        prof = dict(driver="GTiff", height=H * g, width=W * g, count=1, dtype="float32",
                    crs="EPSG:3857", transform=fine_tr, nodata=np.nan)
        with rasterio.open(dem, "w", **prof) as dst:
            dst.write(arr, 1)
        cols = build_subcell_centroids(dem, ref_tr, "EPSG:3857", H, W, grid=g)
    assert cols["id"].size == H * g * W * g - 1                 # one dropped
    assert (cols["parent_id"] == (cols["row"] * W + cols["col"])).all()  # id convention
    import collections
    cnt = collections.Counter(cols["parent_id"].tolist())
    assert max(cnt.values()) <= g * g and np.isfinite(cols["elev"]).all()
    print("subcell mesh OK")


def test_spatial_quantile_aggregate():
    pts = pd.DataFrame({
        "id": [0, 1, 2, 0, 1, 2],
        "PERIOD": [2000, 2000, 2000, 2001, 2001, 2001],
        "DATASET": ["cru.gpcc"] * 6,
        "Tmax_07": [10., 20, 30, 40, 50, 60],
        "PPT_07": [1., 2, 3, 4, 5, 6]})
    idp = pd.DataFrame({"id": [0, 1, 2], "parent_id": [99, 99, 99]})
    out = quantile_aggregate(pts, idp)
    assert set(out) == {"q10", "q50", "q90"}
    q50, q10, q90 = out["q50"], out["q10"], out["q90"]
    r = q50[q50.PERIOD == 2000].iloc[0]
    assert r["id"] == 99 and abs(r["Tmax_07"] - 20) < 1e-9 and abs(r["PPT_07"] - 2) < 1e-9  # median
    assert abs(q10[q10.PERIOD == 2000].iloc[0]["Tmax_07"] - 12) < 1e-9   # np.quantile .1
    assert abs(q90[q90.PERIOD == 2000].iloc[0]["Tmax_07"] - 28) < 1e-9
    assert "DATASET" not in q50.columns                              # non-numeric dropped
    print("spatial quantile aggregate OK")


if __name__ == "__main__":
    test_subcell_mesh()
    test_spatial_quantile_aggregate()
    print("\nALL CLIMATE-SUBGRID CHECKS PASSED")
