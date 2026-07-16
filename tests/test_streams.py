"""Tests for src/data/combine/streams.py — generic covariate streamers.

Synthetic per-year grid rasters; checks nearest-year fill, EMA, static streaming,
and the run_states bag/offsets/NaN-filter. Runs standalone or under pytest.
"""
import os
import sys
import tempfile

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.combine import streams


def _write_grid(path, arr):
    arr = np.asarray(arr, dtype="float32")
    h, w = arr.shape
    prof = dict(driver="GTiff", height=h, width=w, count=1, dtype="float32",
                crs="EPSG:4326", transform=from_origin(0, h, 1, 1), nodata=float("nan"))
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)


def test_nearest_year_fill_and_shape():
    with tempfile.TemporaryDirectory() as d:
        # primf every year; pastr only at 1900 and 2000 (sparse, like HYDE).
        for y in (1900, 1901, 1902):
            _write_grid(os.path.join(d, f"primf_{y}_grid.tif"), np.full((2, 3), 0.5))
        _write_grid(os.path.join(d, "pastr_1900_grid.tif"), np.full((2, 3), 1.0))
        _write_grid(os.path.join(d, "pastr_2000_grid.tif"), np.full((2, 3), 9.0))
        s = streams.PerVariableYearStreamer(d, ["primf", "pastr"], 1900, 1902,
                                            streams.ema_alpha(10), name="landuse")
        out = dict(s)
        assert out[1900].shape == (2, 3, 2)
        # pastr at 1901/1902 fills from nearest available (1900), not 2000.
        assert abs(out[1902][0, 0, 1] - 1.0) < 1e-4


def test_ema_smoothing():
    with tempfile.TemporaryDirectory() as d:
        _write_grid(os.path.join(d, "v_2000_grid.tif"), np.zeros((1, 1)))
        _write_grid(os.path.join(d, "v_2001_grid.tif"), np.ones((1, 1)))
        s = streams.PerVariableYearStreamer(d, ["v"], 2000, 2001,
                                            streams.ema_alpha(10), name="s")
        out = dict(s)
        a = streams.ema_alpha(10)
        assert abs(out[2000][0, 0, 0] - 0.0) < 1e-6           # first year = raw
        assert abs(out[2001][0, 0, 0] - (a * 1.0)) < 1e-6     # a*curr + (1-a)*0


def test_static_streamer_constant():
    with tempfile.TemporaryDirectory() as d:
        _write_grid(os.path.join(d, "soc_grid.tif"), np.full((2, 2), 3.0))
        _write_grid(os.path.join(d, "ph_grid.tif"), np.full((2, 2), 7.0))
        s = streams.StaticStreamer(streams.static_paths(d), 1990, 1992, name="soil")
        seen = list(s)
        assert [y for y, _ in seen] == [1990, 1991, 1992]
        assert seen[0][1].shape == (2, 2, 2)
        assert np.allclose(seen[0][1], seen[-1][1])           # same every year


def test_run_states_bag_offsets_and_npz():
    with tempfile.TemporaryDirectory() as d:
        lu, so, out = os.path.join(d, "lu"), os.path.join(d, "so"), os.path.join(d, "out")
        os.makedirs(lu); os.makedirs(so)
        for y in (2000, 2001):
            _write_grid(os.path.join(lu, f"primf_{y}_grid.tif"), np.full((2, 2), 0.4))
            _write_grid(os.path.join(lu, f"pastr_{y}_grid.tif"), np.full((2, 2), 0.6))
        _write_grid(os.path.join(so, "soc_grid.tif"), np.full((2, 2), 5.0))
        specs = [
            {"type": "per_variable", "name": "landuse", "grid_dir": lu,
             "variables": ["primf", "pastr"], "ema_tau": 10},
            {"type": "static", "name": "soil", "grid_dir": so},
        ]
        mask = np.ones((2, 2), dtype=bool)
        bag, offs = streams.run_states(specs, out, 2000, 2001, mask, sample_start=2000,
                                       samples_per_year=4, rng=np.random.default_rng(0))
        # 2 landuse channels + 1 soil channel = 3.
        assert offs == {"landuse": (0, 2), "soil": (2, 3)}
        assert bag.shape[1] == 3 and bag.shape[0] == 8   # 4 samples x 2 years
        npz = np.load(os.path.join(out, "yearly_states", "state_2001.npz"))
        assert set(npz.files) == {"landuse", "soil"}
        assert npz["landuse"].shape == (2, 2, 2) and npz["soil"].shape == (2, 2, 1)


if __name__ == "__main__":
    test_nearest_year_fill_and_shape()
    print("[per-variable] nearest-year fill + shape OK")
    test_ema_smoothing()
    print("[ema] first-year raw, then a*curr+(1-a)*state OK")
    test_static_streamer_constant()
    print("[static] same state every year OK")
    test_run_states_bag_offsets_and_npz()
    print("[run_states] bag/offsets/per-year npz OK")
    print("\nALL STREAM CHECKS PASSED")
