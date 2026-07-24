"""Pure terrestrial-nodata handling for the SoilGrids preprocessing stage."""
import numpy as np

from src.data.preprocess.soilgrids import fill_terrestrial_nodata


def test_fill_terrestrial_nodata_uses_land_not_ocean_sources():
    land = np.array([[True, True, False], [True, True, False]])
    a = np.array([[1.0, np.nan, 99.0], [2.0, 3.0, 99.0]], dtype="float32")
    out, n = fill_terrestrial_nodata(a, land)
    assert n == 1
    assert np.isfinite(out[0, 1])
    assert out[0, 1] != 99.0
    # Ocean values are not altered; the routine only fixes terrestrial holes.
    assert out[0, 2] == 99.0


def test_fill_terrestrial_nodata_leaves_complete_grid_unchanged():
    land = np.ones((2, 2), bool)
    a = np.array([[1, 2], [3, 4]], dtype="float32")
    out, n = fill_terrestrial_nodata(a, land)
    assert n == 0
    assert np.array_equal(out, a)
