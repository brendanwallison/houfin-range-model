"""Tests for src/data/preprocess/build_ref_grid.py — model-grid geometry.

Runs standalone (``python tests/test_build_ref_grid.py``) or under pytest.
Needs rasterio (present in the data-pipeline env).
"""
import os
import sys
import tempfile

import rasterio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.preprocess.build_ref_grid import build_ref_grid


def test_cell_count_ceils_to_cover_box():
    # A box 100 km x 50 km at 25 km res: 4 x 2 cells exactly.
    bounds = (0.0, 0.0, 100_000.0, 50_000.0)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "ref.tif")
        build_ref_grid(bounds, "ESRI:102039", 25_000, out)
        with rasterio.open(out) as r:
            assert (r.width, r.height) == (4, 2)
            assert r.crs.to_string() == "ESRI:102039"
            # Origin at (left, top); pixel size = res, north-up (negative dy).
            assert r.transform.a == 25_000 and r.transform.e == -25_000
            assert r.transform.c == 0.0 and r.transform.f == 50_000.0


def test_partial_cell_rounds_up():
    # 60 km wide at 25 km -> ceil(2.4) = 3 columns (cover the remainder).
    bounds = (0.0, 0.0, 60_000.0, 25_000.0)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "ref.tif")
        build_ref_grid(bounds, "EPSG:5070", 25_000, out)
        with rasterio.open(out) as r:
            assert (r.width, r.height) == (3, 1)


if __name__ == "__main__":
    test_cell_count_ceils_to_cover_box()
    print("[ref-grid] exact box -> exact cell count, origin/pixel-size correct OK")
    test_partial_cell_rounds_up()
    print("[ref-grid] partial cell rounds up to cover the box OK")
    print("\nALL REF-GRID CHECKS PASSED")
