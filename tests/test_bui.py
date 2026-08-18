"""BUI -> model grid: the pure cores of src/data/preprocess/bui.py.

Covers discovery, channel naming, per-cell quantiling over a partial-coverage mask,
the availability fraction, and the neutral fill -- the piece most worth pinning,
because getting it wrong is silent: absent cells would carry a systematic offset
(or a spurious trend) that reads as real urbanisation signal.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.preprocess.bui import (
    AVAIL_VAR, QUANTILES, availability, cell_quantiles, discover_bui_rasters,
    neutral_fill, quantile_names,
)


def test_quantile_names_match_the_configured_quantiles():
    assert quantile_names() == ["bui_q05", "bui_q25", "bui_q50", "bui_q75",
                               "bui_q90", "bui_q99"]
    assert quantile_names((0.5,)) == ["bui_q50"]
    # sorted() is what build_states falls back to without a manifest, so the names must
    # sort into a stable order and the indicator must not collide with a quantile
    names = quantile_names() + [AVAIL_VAR]
    assert sorted(names) == sorted(set(names))


def test_discover_skips_appledouble_and_sorts_by_year(tmp_path):
    (tmp_path / "nested").mkdir()
    for name in ("2020_BUI.tif", "1810_BUI.tif", "._2015_BUI.tif", "notes.txt"):
        (tmp_path / "nested" / name).write_text("x")
    (tmp_path / "1900_BUI.tif").write_text("x")
    found = discover_bui_rasters(str(tmp_path))
    assert [y for y, _ in found] == [1810, 1900, 2020]      # ._2015 skipped, sorted


def test_cell_quantiles_ignores_invalid_subcells():
    """0 inside CONUS means 'no buildings'; outside it means nothing at all. Invalid
    sub-cells must be excluded from the distribution, not counted as zeros."""
    block = 2
    fine = np.array([[10.0, 20.0, 0.0, 0.0],
                     [30.0, 40.0, 0.0, 0.0]])
    valid = np.array([[True, True, False, False],
                      [True, True, False, False]])
    q = cell_quantiles(fine, valid, block, quantiles=(0.5,))
    assert q.shape == (1, 1, 2)
    assert np.isclose(q[0, 0, 0], 25.0)                     # median of 10,20,30,40
    assert np.isnan(q[0, 0, 1])                             # no valid sub-cell -> NaN

    # a partially-valid cell uses only its valid sub-cells
    valid2 = np.array([[True, False, False, False],
                       [True, False, False, False]])
    q2 = cell_quantiles(fine, valid2, block, quantiles=(0.5,))
    assert np.isclose(q2[0, 0, 0], 20.0)                    # median of 10,30


def test_availability_is_the_valid_fraction():
    block = 2
    valid = np.array([[True, True, True, False],
                      [True, True, False, False]])
    a = availability(valid, block)
    assert a.shape == (1, 2)
    assert np.isclose(a[0, 0], 1.0)                         # 4/4
    assert np.isclose(a[0, 1], 0.25)                        # 1/4
    # always finite, even with zero coverage: a NaN here would invalidate the whole
    # cell across every concatenated channel in covariate_io.norm_grid
    none = availability(np.zeros((2, 2), bool), 2)
    assert np.isfinite(none).all() and none[0, 0] == 0.0


def test_neutral_fill_lands_at_the_in_coverage_mean_after_the_transform():
    """The fill must satisfy log1p(fill) == mean(log1p(in-coverage values)). Averaging in
    raw space instead would land somewhere else entirely, since log1p of the mean is not
    the mean of log1p."""
    vals = np.array([[[0.0, 10.0], [1000.0, 1e6]]])         # (1, 2, 2), one channel
    stacks = {2000: vals}
    fill = neutral_fill(stacks, {"type": "log1p"})
    assert fill.shape == (1,)
    want = np.mean(np.log1p(vals.ravel()))
    assert np.isclose(np.log1p(fill[0]), want)
    # and it is NOT the raw mean, which is what a naive implementation would give
    assert not np.isclose(fill[0], vals.mean())


def test_neutral_fill_ignores_absent_cells_and_pools_across_years():
    """One value per channel for ALL years. BUI grows monotonically, so a per-year fill
    would put a rising trend into exactly the region that has no data."""
    y1 = np.array([[[1.0, np.nan]]])                        # (1,1,2)
    y2 = np.array([[[9.0, np.nan]]])
    pooled = neutral_fill({1900: y1, 2000: y2}, {"type": "log1p"})
    assert np.isclose(np.log1p(pooled[0]),
                      np.mean(np.log1p([1.0, 9.0])))        # NaNs excluded, years pooled
    per_year = [neutral_fill({y: s}, {"type": "log1p"})[0] for y, s in ((1900, y1), (2000, y2))]
    assert per_year[0] < pooled[0] < per_year[1]            # the drift a pooled fill avoids


def test_neutral_fill_handles_every_supported_transform():
    vals = np.array([[[0.0, 16.0, 81.0]]])
    stacks = {2000: vals}
    assert np.isclose(neutral_fill(stacks, None)[0], vals.mean())          # identity
    p = neutral_fill(stacks, {"type": "pow", "p": 0.25})[0]
    assert np.isclose(np.power(p, 0.25), np.mean(np.power(vals.ravel(), 0.25)))
    try:
        neutral_fill(stacks, {"type": "sqrt-ish"})
    except ValueError:
        pass
    else:
        raise AssertionError("an uninvertible transform was accepted silently")


def test_neutral_fill_refuses_a_fully_absent_channel():
    """No coverage at all means the CONUS polygon and the rasters do not overlap the
    study box -- a configuration error, not a value to invent."""
    try:
        neutral_fill({2000: np.full((1, 2, 2), np.nan)}, {"type": "log1p"})
    except SystemExit:
        return
    raise AssertionError("a fully-absent channel produced a fill value")


def test_quantiles_are_monotone_in_q():
    """Quantiles are taken at target resolution from the fine values, so monotonicity in
    q is automatic -- unlike interpolating quantile bands, which needs a re-sort."""
    rng = np.random.default_rng(0)
    fine = rng.lognormal(size=(8, 8)) * 1e5
    q = cell_quantiles(fine, np.ones_like(fine, bool), 4, quantiles=QUANTILES)
    assert q.shape == (len(QUANTILES), 2, 2)
    assert np.all(np.diff(q, axis=0) >= -1e-9)


if __name__ == "__main__":
    import tempfile
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                import pathlib
                with tempfile.TemporaryDirectory() as d:
                    fn(pathlib.Path(d))
            else:
                fn()
            print(f"  ok   {name}")
        except Exception:
            fails += 1
            print(f"  FAIL {name}")
            traceback.print_exc()
    print("\n" + (f"{fails} FAILED" if fails else "ALL BUI CHECKS PASSED"))
    sys.exit(1 if fails else 0)
