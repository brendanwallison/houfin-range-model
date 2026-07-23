"""Ingestion guards for the age-model input builder (src/data/combine/model_inputs.py).

These turn silent grid/timeline mismatches — created by the 25->27 km grid migration and
the 1902-2025 year-span — into loud failures at ingest time (plan items E1/E2/E3):
  E1  the Z cube / Z_disp lattice must match the BBS/model grid (row/col gather)
  E2  the age-model ocean mask must equal the BBS npz's embedded land mask
  E3  the cube must cover the pre-invasion span or all pseudo-zeros are silently dropped
"""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.data.combine.model_inputs import (
    load_ocean_land_mask, require_mask_match, require_pseudo_zero_coverage, require_same_grid,
)


# --- E1: grid-shape agreement -------------------------------------------------------------

def test_require_same_grid_accepts_match():
    require_same_grid("Z cube", (133, 224), (133, 224))          # no raise


def test_require_same_grid_rejects_mismatch():
    with pytest.raises(ValueError, match="BBS/model grid"):
        require_same_grid("Z cube", (140, 240), (133, 224))      # e.g. stale 25 km vs 27 km


# --- E2: mask agreement -------------------------------------------------------------------

def test_require_mask_match_accepts_identical():
    land = np.array([[1, 0, 1], [0, 1, 1]], bool)
    require_mask_match(land, land.astype(int), "mask.tif")       # dtype-agnostic, no raise


def test_require_mask_match_rejects_shape():
    with pytest.raises(ValueError, match="!= BBS land grid"):
        require_mask_match(np.ones((2, 3), bool), np.ones((2, 4), bool), "mask.tif")


def test_require_mask_match_rejects_cell_diff():
    a = np.array([[1, 0, 1], [0, 1, 1]], bool)
    b = a.copy(); b[0, 0] = False
    with pytest.raises(ValueError, match="land cells differ"):
        require_mask_match(a, b, "mask.tif")


def test_load_ocean_land_mask_convention(tmp_path):
    # ocean-mask raster: water encoded nonzero, land == 0 -> load returns True on land.
    ocean = np.array([[0, 1, 0], [1, 0, 0]], dtype="uint8")      # 0=land, 1=water
    p = tmp_path / "ocean_mask.tif"
    with rasterio.open(p, "w", driver="GTiff", height=2, width=3, count=1,
                       dtype="uint8", crs="ESRI:102003",
                       transform=from_origin(0, 0, 27000, 27000)) as dst:
        dst.write(ocean, 1)
    land = load_ocean_land_mask(str(p))
    assert land.dtype == bool
    assert np.array_equal(land, ocean == 0)


# --- E3: pseudo-zero coverage -------------------------------------------------------------

def test_pseudo_zero_coverage_accepts_full_span():
    require_pseudo_zero_coverage(1902, 1902, 1940, 2025)         # cube covers pre-invasion, no raise


def test_pseudo_zero_coverage_accepts_start_at_last_preinvasion():
    require_pseudo_zero_coverage(1939, 1902, 1940, 2025)         # 1939 == invasion-1, still ok


def test_pseudo_zero_coverage_rejects_post_invasion_start():
    # cube starts 1966 -> all 1902-1939 pseudo-zeros would drop silently
    with pytest.raises(ValueError, match="pseudo-zero"):
        require_pseudo_zero_coverage(1966, 1902, 1940, 2025)
