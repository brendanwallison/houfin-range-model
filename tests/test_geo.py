"""The shared geo context: each test pins a way a map can be silently wrong about the ground.

A map is harder to falsify than a number -- a shifted, stretched, or wrongly-masked continent still
looks like a continent. These fix the things that would not announce themselves.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "viz")))

import _geo  # noqa: E402


def test_the_grid_spec_ignores_the_local_25km_override():
    """Every validation artifact is produced on TACC at 27 km. This machine's dev config points at
    25 km, so a figure that resolved its extent through `load_data_config` would place a 27 km
    array on a 25 km box -- shifted by ~8%, and shifted is not visibly broken."""
    spec = _geo.base_grid_spec()
    assert spec["res_m"] == 27000.0
    assert spec["crs"] == "ESRI:102003"
    minx, maxx, miny, maxy = spec["extent"]
    assert maxx > minx and maxy > miny
    # extent must be imshow's order, not the box order, or every map is transposed vertically
    assert (minx, miny, maxx, maxy) == spec["box_bounds"]


def test_the_extent_matches_the_grid_it_will_carry():
    """133x224 cells at 27 km must exactly span the configured box. An off-by-one in either
    direction tilts every overlay against the raster by a cell and puts the coastline in the
    wrong place -- which looks like a projection choice, not an error."""
    spec = _geo.base_grid_spec()
    minx, miny, maxx, maxy = spec["box_bounds"]
    assert round((maxx - minx) / spec["res_m"]) == 224
    assert round((maxy - miny) / spec["res_m"]) == 133


def test_to_grid_averages_repeated_cells_and_can_count_them():
    """A validation map is usually many cell-YEARS collapsing onto one cell, and the reader has to
    know whether they are looking at a mean or a count -- so the reduction is an argument."""
    rows = [0, 0, 1]
    cols = [0, 0, 1]
    g = _geo.to_grid(rows, cols, [1.0, 3.0, 5.0], (2, 2))
    assert g[0, 0] == pytest.approx(2.0)
    assert g[1, 1] == pytest.approx(5.0)
    assert np.isnan(g[0, 1])
    n = _geo.to_grid(rows, cols, [1.0, 3.0, 5.0], (2, 2), reduce="count")
    assert n[0, 0] == 2 and n[1, 1] == 1 and n[0, 1] == 0


def test_to_grid_drops_non_finite_values_rather_than_poisoning_the_cell():
    """One NaN among a cell's observations must not erase the cell: a predictor legitimately
    declines rows it cannot reach, and a cell where it reached three times out of four still has
    an answer."""
    g = _geo.to_grid([0, 0], [0, 0], [np.nan, 4.0], (1, 1))
    assert g[0, 0] == pytest.approx(4.0)


def test_gate_nans_thin_support_and_reports_what_survived():
    """Gate hard, name the threshold, print the count -- the house precedent. A faded cell still
    invites reading its colour; a NaN cell does not."""
    grid = np.array([[1.0, 2.0], [3.0, 4.0]])
    sup = np.array([[5, 1], [9, 2]])
    out, note = _geo.gate(grid, sup, 3)
    assert np.isnan(out[0, 1]) and np.isnan(out[1, 1])
    assert out[0, 0] == 1.0 and out[1, 0] == 3.0
    assert "2 cells at >= 3" in note


def test_desaturate_fades_a_cell_with_no_room_to_transparent():
    """The scalar suite REFUSES to rank predictors where the floor-to-ceiling distance is under
    ~0.15. A map cannot refuse a cell, so it fades it -- a cell nobody can resolve should recede
    rather than shout a colour."""
    grid = np.array([0.5, 0.5, 0.5])
    room = np.array([0.0, 0.075, 0.30])
    a = _geo.desaturate(grid, room, floor=0.15)
    assert a[0] == 0.0
    assert a[1] == pytest.approx(0.5)
    assert a[2] == 1.0


def test_a_missing_optional_layer_costs_an_outline_not_the_figure():
    """Natural Earth, the zone raster and the land mask are all absent on some machines and all
    absent on HPC at various times. A basemap that raised would take the whole suite with it."""
    geo = _geo.GeoContext.__new__(_geo.GeoContext)
    spec = _geo.base_grid_spec()
    geo.extent, geo.box_bounds = spec["extent"], spec["box_bounds"]
    geo.res_m, geo.crs = spec["res_m"], spec["crs"]
    geo.land = geo.land_geom = geo.gp_zones = None
    geo.shape = (133, 224)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    geo.basemap(ax)
    geo.coastline(ax)
    assert geo.great_plains(ax) is False        # reports absence, does not raise
    geo.scalebar(ax)
    plt.close(fig)


@pytest.mark.skipif(not os.path.exists(
    os.path.join(os.path.dirname(__file__), "..", "data", "ref_grid_27km.tif")),
    reason="grid rasters are gitignored")
def test_the_great_plains_partition_matches_the_land_mask_exactly():
    """The zones are a gap-free three-way partition and the land mask is the model's own. If the
    two disagree, a zone-stratified error map is attributing cells to the wrong side of the
    barrier -- which is the one ecological claim these maps are for."""
    geo = _geo.GeoContext(shape=(133, 224), want_vectors=False)
    if geo.land is None or geo.gp_zones is None:
        pytest.skip("land mask or zone raster absent")
    counts = {k: int((v & geo.land).sum()) for k, v in geo.gp_zones.items()}
    assert counts == {"west": 5497, "barrier": 4110, "east": 7602}
    assert sum(counts.values()) == int(geo.land.sum()) == 17209
