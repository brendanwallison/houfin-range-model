"""Correctness tests for src/processing/regrid.py — the resolution-deferral core.

These prove, on synthetic rasters, the analytical claim behind Part C:

  * a quantile recovered at target resolution from a finest-resolution
    histogram matches the quantile of the raw values, while the *old* pipeline
    order (quantile in sub-blocks, then average) does not;
  * a pointwise nonlinearity applied to the linear aggregate (the deferred,
    correct order) differs materially from applying it before aggregation (the
    old order), and equals evaluating it on the target-resolution mean by
    construction;
  * the linear summaries we store (sum, histogram) are additive, which is what
    makes the target resolution a free choice made later.

Runs standalone (``python tests/test_regrid.py``) or under pytest. NumPy only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.processing import regrid

RNG = np.random.default_rng(0)
BLOCK = 32          # e.g. 125 m native -> 4 km target; enough samples per cell
NY_T, NX_T = 4, 4   # target grid
FINE = (NY_T * BLOCK, NX_T * BLOCK)


def _skewed_field():
    """A fine raster with strong *within-cell* heterogeneity and a heavy tail.

    A coarse cell spans a smooth spatial gradient (as a 4 km BUI cell spans
    varied fine-scale urban texture) plus an exponential tail. This is the
    honest stress case for the flagship bug: because sub-regions of a cell have
    very different distributions, the quantile of a sub-block is a poor proxy
    for the quantile of the whole cell, so averaging sub-block quantiles is
    badly biased — while a histogram over all native values is not.
    """
    yy, xx = np.mgrid[0 : FINE[0], 0 : FINE[1]]
    gradient = 0.5 * yy + 0.5 * xx
    tail = RNG.exponential(2.0, size=FINE)
    return gradient + tail


def test_quantile_deferral_beats_average_of_subquantiles():
    arr = _skewed_field()
    q = 0.9
    lo, hi = np.floor(arr.min()), np.ceil(arr.max())
    edges = np.linspace(lo, hi, 2049)  # 2048 bins
    bin_w = edges[1] - edges[0]

    # Truth: quantile of the raw native values in each target cell.
    cells = regrid._blocks(arr, BLOCK)  # (NY_T, NX_T, BLOCK*BLOCK)
    truth = np.quantile(cells, q, axis=2)

    # Deferred (correct): per-cell histogram at finest res -> quantile at target.
    counts = regrid.block_histogram(arr, BLOCK, edges)
    deferred = regrid.quantile_from_histogram(counts, edges, q)
    deferred_err = np.abs(deferred - truth).max()

    # Broken (old order): quantile in 4x4 sub-blocks, then average up to target.
    sub = 4
    subq = np.quantile(regrid._blocks(arr, sub), q, axis=2)  # quantiles at finer grid
    broken = regrid.block_reduce(subq, BLOCK // sub, how="mean")
    broken_err = np.abs(broken - truth).max()

    data_range = hi - lo
    # Deferred quantile is accurate in absolute terms (small vs the data range).
    assert deferred_err < 0.01 * data_range, (
        f"deferred error {deferred_err:.4f} large vs range {data_range:.1f}"
    )
    # The old order's error is structural (averaging quantiles under intra-cell
    # heterogeneity), not a binning artifact — it stays large as bins refine.
    assert broken_err > 0.03 * data_range, f"old-order error {broken_err:.4f} unexpectedly small"
    assert broken_err > 20 * deferred_err, (
        f"expected the old order to be far worse: broken={broken_err:.4f} "
        f"deferred={deferred_err:.4f}"
    )
    return deferred_err, broken_err, bin_w


def test_pointwise_transform_order_matters():
    arr = np.abs(_skewed_field())
    power = lambda x: x ** 0.1

    deferred = regrid.defer_pointwise(arr, BLOCK, power, how="mean")   # f(mean(x))
    old_order = regrid.block_reduce(power(arr), BLOCK, how="mean")     # mean(f(x))

    # Deferred equals evaluating f on the target-res mean, by construction.
    assert np.allclose(deferred, power(regrid.block_reduce(arr, BLOCK, "mean")))
    # And the two orders genuinely diverge (Jensen gap for a concave power).
    assert np.abs(deferred - old_order).max() > 1e-3
    return np.abs(deferred - old_order).max()


def test_linear_summaries_are_additive():
    arr = _skewed_field()
    edges = np.linspace(arr.min(), arr.max(), 65)

    # Histogram additivity: hist over a full target cell == sum of two half-cell
    # histograms. This is why histograms can be re-aggregated to any resolution.
    top, bot = arr[: FINE[0] // 2], arr[FINE[0] // 2 :]
    h_full = regrid.block_histogram(arr, BLOCK, edges).sum(axis=(0, 1))
    h_parts = (
        regrid.block_histogram(top, BLOCK, edges).sum(axis=(0, 1))
        + regrid.block_histogram(bot, BLOCK, edges).sum(axis=(0, 1))
    )
    assert np.array_equal(h_full, h_parts)

    # Sum additivity: total over target cells == global sum (no NaNs here).
    assert np.isclose(regrid.block_reduce(arr, BLOCK, "sum").sum(), arr.sum())


def test_block_factor_validation():
    assert regrid.block_factor(250, 4000) == 16
    assert regrid.block_factor(2963.0, 2963.0) == 1
    for bad in (lambda: regrid.block_factor(300, 4000),   # not commensurate
                lambda: regrid.block_factor(4000, 250)):  # would upsample
        try:
            bad()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_nan_handling():
    arr = _skewed_field()
    arr[:BLOCK, :BLOCK] = np.nan  # first target cell fully nodata (ocean)
    m = regrid.block_reduce(arr, BLOCK, "mean")
    assert np.isnan(m[0, 0]) and np.isfinite(m[0, 1])
    edges = np.linspace(np.nanmin(arr), np.nanmax(arr), 65)
    counts = regrid.block_histogram(arr, BLOCK, edges)
    assert counts[0, 0].sum() == 0  # no samples in the all-NaN cell
    qd = regrid.quantile_from_histogram(counts, edges, 0.5)
    assert np.isnan(qd[0, 0]) and np.isfinite(qd[0, 1])


if __name__ == "__main__":
    de, be, bw = test_quantile_deferral_beats_average_of_subquantiles()
    print(f"[quantile]  deferred_err={de:.4f}  old-order_err={be:.4f}  "
          f"bin_width={bw:.4f}  -> old order {be/de:.0f}x worse")
    print(f"[pointwise] |deferred - old-order| max = {test_pointwise_transform_order_matters():.4f}")
    test_linear_summaries_are_additive(); print("[additivity] histogram + sum additivity OK")
    test_block_factor_validation(); print("[block_factor] validation OK")
    test_nan_handling(); print("[nan] all-NaN cell -> NaN, valid cell finite OK")
    print("\nALL REGRID CORRECTNESS CHECKS PASSED")
