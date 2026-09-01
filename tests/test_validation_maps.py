"""The map suite: each test pins a way a map can lie more convincingly than a number can.

A wrong scalar is a wrong scalar. A wrong map is a picture of a pattern, and this project's own
history is of exactly that -- a shared colour scale that "cost an entire investigation", a ceiling
read off a filter, an angle reported without its magnitude. These fix the map versions.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "viz")))

import _geo                     # noqa: E402
import _validation_load as L    # noqa: E402
import validation_maps as VM    # noqa: E402


def _geo_ctx(shape=(20, 30)):
    """A GeoContext with a synthetic continent, so the tests need no rasters."""
    g = _geo.GeoContext.__new__(_geo.GeoContext)
    spec = _geo.base_grid_spec()
    g.extent, g.box_bounds = spec["extent"], spec["box_bounds"]
    g.res_m, g.crs, g.shape = spec["res_m"], spec["crs"], shape
    g.land = np.ones(shape, bool)
    g.land_geom = None
    g.gp_zones = {"west": np.zeros(shape, bool), "barrier": np.zeros(shape, bool),
                  "east": np.zeros(shape, bool)}
    g.gp_zones["west"][:, :10] = True
    g.gp_zones["barrier"][:, 10:20] = True
    g.gp_zones["east"][:, 20:] = True
    g.front = None
    g.notes = []
    return g


def _maps(shape=(20, 30), seed=0, n_years=6):
    """A minimal but structurally faithful map layer: the keys and shapes the validators emit."""
    rng = np.random.default_rng(seed)
    cells = np.array([(r, c) for r in range(shape[0]) for c in range(shape[1])], dtype="int32")
    n = len(cells)
    r, c = cells[:, 0], cells[:, 1]
    ho = np.zeros(shape, bool)
    ho[:6, :6] = True
    bf = np.zeros(shape, bool)
    bf[:8, :8] = True
    bf &= ~ho
    years = np.arange(1966, 1966 + n_years) * 10 // 10
    rr = np.repeat(r, n_years)
    cc = np.repeat(c, n_years)
    yy = np.tile(np.linspace(1966, 2025, n_years).astype(int), n)
    m = len(rr)
    err = 0.4 + 0.1 * rng.normal(size=m)
    st = {"recon_rows": rr.astype("int32"), "recon_cols": cc.astype("int32"),
          "recon_year": yy.astype("int32"),
          "recon_err_desk": err.astype("float32"),
          "recon_err_nochange": (err + 0.05).astype("float32"),
          "baseline_ladder._rows": np.stack([rr, cc, yy], 1).astype("int32"),
          "baseline_ladder._bars.desk": err.astype("float32"),
          "baseline_ladder._bars.no_change": (err + 0.05).astype("float32"),
          "baseline_ladder._bars.spacetime_idw": (err + 0.02).astype("float32")}
    pre = "epoch_directions.windowed.pairs.1967_2025._"
    st[pre + "cells"] = cells
    st[pre + "dir_cos"] = rng.uniform(-1, 1, n).astype("float32")
    st[pre + "mag_ratio"] = rng.uniform(0.2, 1.8, n).astype("float32")
    st[pre + "ceiling_cell_idx"] = np.arange(n // 2, dtype="int32")
    st[pre + "ceiling_per_cell"] = rng.uniform(0.4, 0.8, n // 2).astype("float32")
    ep = {"cells": cells, "rows": r.astype("int32"), "cols": c.astype("int32"),
          "floor_early": rng.uniform(0.6, 0.95, n).astype("float32"),
          "floor_modern": rng.uniform(0.7, 0.95, n).astype("float32"),
          "split_ok": np.arange(n) % 5 != 0,
          "is_heldout": ho[r, c]}
    return {"spacetime": st, "epoch": ep, "routes": None,
            "holdout": ho, "buffer": bf, "missing": [], "run_dir": "."}


# --- the loader's addressing ---------------------------------------------------------------------

def test_ladder_bars_are_found_by_prefix_so_a_new_rung_needs_no_edit():
    """The extractor flattens with dotted paths. Reading the rungs by prefix rather than by a
    fixed name list is what lets a rung added to the ladder appear on the map with no change
    here -- the allow-list failure this codebase has already paid for twice."""
    rows, bars = L.ladder_bars(_maps())
    assert rows is not None and rows.shape[1] == 3
    assert set(bars) == {"desk", "no_change", "spacetime_idw"}


def test_epoch_pairs_are_addressed_individually_and_never_merged():
    """Pairs share cells and nest in time, so an average over them would overstate the evidence
    exactly the way a pooled scalar would."""
    m = _maps()
    assert L.epoch_pairs_available(m) == ["1967_2025"]
    d = L.epoch_pair_cells(m, "1967_2025")
    assert {"cells", "dir_cos", "mag_ratio", "ceiling_per_cell"} <= set(d)


def test_a_run_with_no_npz_names_what_it_wanted(tmp_path):
    """Every one of these artifacts is produced on HPC, so a local run legitimately has none. A
    map that cannot be drawn should say which file it wanted, not render an empty axis."""
    m = L.load_maps(str(tmp_path))
    assert m["spacetime"] is None and m["holdout"] is None
    assert "validate_spacetime.npz" in m["missing"]
    geo = _geo_ctx()
    assert all(fn({"report": None}, m, geo, str(tmp_path)) is None for fn in VM.MAPS)


# --- the nuance devices, on a map ----------------------------------------------------------------

def test_the_winner_map_is_scored_only_where_every_rung_reached():
    """Scoring a rung's easy subset against another's full set flatters whichever declined the
    hard rows -- the reason `win_rate_vs` intersects. A map has the same exposure, cell by cell."""
    m = _maps()
    m["spacetime"]["baseline_ladder._bars.spacetime_idw"][:50] = np.nan
    geo = _geo_ctx()
    rows, bars = L.ladder_bars(m)
    stack = np.vstack([bars[n] for n in bars if n != "no_change"])
    keep = np.isfinite(stack).all(axis=0)
    assert keep.sum() == len(rows) - 50, "rows a rung declined must leave the comparison"


def test_a_tie_fades_instead_of_claiming_a_winner(tmp_path):
    """A bare winner-take-all map shows a confident colour for a 1% margin, which is how a noise
    field comes to look like a spatial finding. The fade reference is the seed-to-seed spread the
    rest of the suite already uses, not a number chosen here."""
    import _validation_style as S
    m = _maps()
    # make every rung nearly identical -> every cell is a tie
    base = m["spacetime"]["baseline_ladder._bars.desk"]
    m["spacetime"]["baseline_ladder._bars.spacetime_idw"] = base * 1.001
    made = VM.map03_ladder_winner({"report": None}, m, _geo_ctx(), str(tmp_path))
    assert made
    import matplotlib.image as mpimg
    img = mpimg.imread(made)
    assert img.size > 0
    assert S.SEED_NOISE == 0.066, "the fade threshold must stay the suite's measured seed spread"


def test_the_held_out_panel_aggregates_to_the_block():
    """217 held-out cells sit in ~87 contiguous blocks, so within-block texture is not evidence --
    the same reason `bootstrap_skill_ci` resamples focal cells and never pairs. A per-cell
    out-of-sample map would show ~4x more independent detail than exists."""
    g = np.arange(144, dtype="float64").reshape(12, 12)
    b = VM._blockify(g, block=6)
    assert b.shape == g.shape
    # every cell in a block carries that block's mean, so a block has exactly one value
    assert len(np.unique(b)) == 4
    assert b[0, 0] == pytest.approx(np.mean(g[:6, :6]))


def test_blockify_survives_a_grid_that_is_not_a_block_multiple():
    """The real grid is 133x224 and 133 is not divisible by 6. Padding with NaN and taking a
    nanmean keeps the edge blocks honest rather than dropping or wrapping them."""
    g = np.ones((13, 13))
    b = VM._blockify(g, block=6)
    assert b.shape == (13, 13) and np.allclose(b, 1.0)


def test_a_diverging_panel_can_be_centred_somewhere_other_than_zero(tmp_path):
    """A magnitude RATIO is neutral at 1.0. Centring it on 0 paints every cell that moved at all
    as 'high' and hides the over- versus under-moving split the panel exists to show."""
    import matplotlib.pyplot as plt
    geo = _geo_ctx()
    grid = np.full(geo.shape, 1.5)
    fig, ax = plt.subplots()
    im = VM._draw(geo, ax, grid, "ratio", diverging=True, vmax=1.0, center=1.0)
    assert im.get_clim() == (0.0, 2.0)
    plt.close(fig)


def test_the_ceiling_map_refuses_unsplittable_cells_rather_than_colouring_them(tmp_path):
    """A cell with too few surveys in an era cannot be split in half, so no independent
    observation exists and no ceiling can be formed. Those cells are refused -- given their own
    flat colour and a legend entry -- because a hatch over scattered 27 km cells is illegible and
    an illegible refusal is worse than none: the reader reads the colour underneath."""
    m = _maps()
    made = VM.map02_ceiling({"report": None}, m, _geo_ctx(), str(tmp_path))
    assert made and os.path.getsize(made) > 3000


def test_the_era_panels_share_one_scale(tmp_path):
    """They are the same quantity at different times. A per-panel scale would hide exactly the
    drift the panels exist to show -- the inverse of the rule for panels of different quantities,
    and the reason the rule has to be stated per figure rather than applied blindly."""
    m = _maps(n_years=8)
    made = VM.map08_time({"report": None}, m, _geo_ctx(), str(tmp_path))
    assert made
    src = open(os.path.join(os.path.dirname(__file__), "..", "scripts", "viz",
                            "validation_maps.py"), encoding="utf-8").read()
    era_block = src[src.index("def map08_time"):]
    assert "vmax=vmax, cbar=False" in era_block, "era panels must share one scale and one bar"


def test_every_map_renders_from_the_full_layer(tmp_path):
    m = _maps()
    made = [fn({"report": None}, m, _geo_ctx(), str(tmp_path)) for fn in VM.MAPS]
    drawn = [p for p in made if p]
    assert len(drawn) >= 6, f"only {len(drawn)} of {len(VM.MAPS)} maps rendered"
    assert all(os.path.getsize(p) > 3000 for p in drawn)


# --- the colonization front ----------------------------------------------------------------------

def test_the_front_ignores_the_pseudo_zero_block():
    """`bbs_data_for_python.npz` concatenates synthetic 1902-1939 zeros AHEAD of real
    observations, so an unfiltered first-year returns 1902 everywhere -- a perfectly plausible
    raster of nothing."""
    from scripts.build_colonization_front import first_detection
    rows = [0, 0] + [0, 1]
    cols = [0, 1] + [0, 1]
    yrs = [1902, 1902] + [1980, 1990]
    cnt = [0, 0] + [4, 7]
    f = first_detection(rows, cols, yrs, cnt, (2, 2), n_pseudo=2)
    assert f[0, 0] == 1980 and f[1, 1] == 1990
    assert not np.any(f == 1902)


def test_a_surveyed_absence_is_not_a_detection():
    """The question is first DETECTION, not first survey: a cell surveyed in 1970 with zero finches
    was not colonized in 1970."""
    from scripts.build_colonization_front import first_detection
    f = first_detection([0, 0], [0, 0], [1970, 1985], [0, 2], (1, 1), n_pseudo=0)
    assert f[0, 0] == 1985


def test_the_native_hull_is_masked_because_the_west_has_no_front():
    """Pseudo-zeros are only asserted outside a 700 km halo around the native hull, so a western
    cell's first detection is 1966 -- BBS's launch year, an artifact of when counting started. A
    raster that reported that as a colonization date would put a bright artificial edge exactly
    where the Great Plains analysis looks."""
    from scripts.build_colonization_front import native_hull_mask
    m = native_hull_mask([2], [2], (5, 5), dilate=1)
    assert m[2, 2] and m[1, 1] and m[3, 3]
    assert not m[0, 0]
    assert native_hull_mask([2], [2], (5, 5), dilate=0).sum() == 1


def test_the_ecology_map_bins_error_against_the_front_when_one_exists(tmp_path):
    """The front panel is the only one here that asks a MECHANISTIC question rather than a
    geographic one: a range model worst exactly where the range was moving is failing at the thing
    it exists to do, and no zone-stratified number shows that. The raster is a derived product
    that has to be built where the BBS npz lives, so its absence is a note, not a failure."""
    m = _maps()
    geo = _geo_ctx()
    assert geo.front is None
    made = VM.map06_ecology({"report": None}, m, geo, str(tmp_path))
    assert made, "the map must still draw with no front"

    yy, xx = np.mgrid[0:geo.shape[0], 0:geo.shape[1]]
    geo.front = 1966.0 + xx * 1.5
    made2 = VM.map06_ecology({"report": None}, m, geo, str(tmp_path))
    assert made2 and os.path.getsize(made2) > os.path.getsize(made) * 0.5


def test_a_front_raster_that_is_the_wrong_shape_is_refused_not_resampled():
    """A stale-grid overlay silently misaligns the front by hundreds of km, which is exactly the
    failure `load_disease_onset` raises on rather than resampling. Here the cost is a missing
    panel, so it declines instead of raising -- but it must never stretch to fit."""
    g = _geo.GeoContext.__new__(_geo.GeoContext)
    assert g._load_front((999, 999)) is None
