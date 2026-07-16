"""Resolution-aware regridding: aggregate at the finest resolution, defer
nonlinear transforms to the chosen target resolution.

Why this module exists
----------------------
An operation that is *nonlinear* (a quantile, a power ``x**0.1``, a log, a
z-score standardization, a thresholded mask) does **not commute** with spatial
aggregation:

    mean(f(x)) != f(mean(x))                       # power / log / standardize
    mean(quantile(sub-blocks)) != quantile(cell)   # quantiles

If such a transform runs *before* a resolution change, today's resolution gets
baked irreversibly into the values. The fix, applied uniformly here, is:

    1. store a *linear* summary at the finest resolution
         - mean / sum          -> enough to defer any pointwise transform
         - per-cell histogram  -> enough to defer any quantile
    2. apply the nonlinear transform once, at the chosen target resolution.

The linear summaries are chosen precisely because they are **additive**: a
coarse cell's mean/sum/histogram is a simple combination of the finer cells it
contains, so the target resolution can be picked (or changed) later without
recomputing from raw. Histograms in particular are additive
(``hist(A ∪ B) == hist(A) + hist(B)``), which is what lets a quantile be
recovered correctly at any coarser resolution.

The array math below is pure NumPy (no rasterio/config imports) so it is
testable anywhere; the raster/config glue lives in the small helpers at the
bottom and imports its dependencies lazily.
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np


# Block geometry

def block_factor(native_res_m: float, target_res_m: float) -> int:
    """Integer number of native cells per target cell along one axis.

    Requires the target resolution to be a near-integer multiple of the native
    resolution (raises otherwise) so aggregation is a clean block reduction.
    """
    ratio = target_res_m / native_res_m
    nearest = round(ratio)
    if nearest < 1:
        raise ValueError(
            f"target_res_m ({target_res_m}) is finer than native ({native_res_m}); "
            "upsampling is not a linear aggregation and is not supported here."
        )
    if abs(ratio - nearest) > 1e-3 * nearest:
        raise ValueError(
            f"target_res_m ({target_res_m}) is not an integer multiple of native "
            f"({native_res_m}); ratio={ratio:.4f}. Choose a commensurate target."
        )
    return int(nearest)


def _blocks(arr: np.ndarray, block: int) -> np.ndarray:
    """Reshape a 2-D array to (ny_t, nx_t, block*block) grouping each target cell.

    The trailing axis holds the ``block*block`` native values that fall inside
    one target cell. Any partial edge rows/cols that don't fill a whole block
    are cropped (with a warning) so the reduction stays exact.
    """
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {arr.shape}")
    ny, nx = arr.shape
    ny_t, nx_t = ny // block, nx // block
    if ny_t == 0 or nx_t == 0:
        raise ValueError(f"array {arr.shape} smaller than one block of {block}")
    if ny % block or nx % block:
        warnings.warn(
            f"array {arr.shape} not a multiple of block {block}; cropping "
            f"{ny - ny_t * block} rows and {nx - nx_t * block} cols.",
            stacklevel=2,
        )
    cropped = arr[: ny_t * block, : nx_t * block]
    # (ny_t, block, nx_t, block) -> (ny_t, nx_t, block, block) -> (ny_t, nx_t, K)
    return (
        cropped.reshape(ny_t, block, nx_t, block)
        .transpose(0, 2, 1, 3)
        .reshape(ny_t, nx_t, block * block)
    )


# Linear aggregation (commutes with resolution change)

def block_reduce(arr: np.ndarray, block: int, how: str = "mean") -> np.ndarray:
    """NaN-aware block reduction to the coarser grid.

    ``how`` in {"mean", "sum", "max", "min"}. ``mean``/``sum`` are linear and
    are the summaries to store when a pointwise nonlinearity will be deferred.
    ``max``/``min`` are provided for masks/presence but are themselves
    nonlinear — use them deliberately, not as a default.
    """
    vals = _blocks(arr, block)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cell -> NaN
        if how == "mean":
            return np.nanmean(vals, axis=2)
        if how == "sum":
            return np.nansum(vals, axis=2)
        if how == "max":
            return np.nanmax(vals, axis=2)
        if how == "min":
            return np.nanmin(vals, axis=2)
    raise ValueError(f"unknown reduction '{how}'")


def block_quantiles(arr: np.ndarray, block: int, quantiles) -> np.ndarray:
    """Exact per-cell quantiles of the finest-res values: (n_quantiles, ny_t, nx_t).

    NaN-aware (nodata subpixels ignored; all-NaN cell -> NaN). This is the
    deferral-correct way to summarize a within-cell distribution at target
    resolution (used for BUI quantile bands and DEM elevation quantiles).
    """
    vals = _blocks(arr, block)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cell -> NaN
        return np.nanquantile(vals, np.asarray(quantiles), axis=2)


def defer_pointwise(
    arr: np.ndarray,
    block: int,
    func: Callable[[np.ndarray], np.ndarray],
    how: str = "mean",
) -> np.ndarray:
    """Correct order for a pointwise nonlinearity: aggregate LINEARLY, then apply.

    Returns ``func(block_reduce(arr, block, how))`` — i.e. the transform is
    evaluated on the target-resolution aggregate. The *wrong* order, which this
    exists to replace, is ``block_reduce(func(arr), block)``.
    """
    return func(block_reduce(arr, block, how=how))


# Deferred quantiles via additive per-cell histograms

def block_histogram(arr: np.ndarray, block: int, edges: np.ndarray) -> np.ndarray:
    """Per-target-cell histogram of the native values (NaNs dropped).

    Returns counts of shape ``(ny_t, nx_t, len(edges)-1)``. Histograms are
    additive across space, so these can be summed to any coarser resolution and
    a quantile recovered later (see :func:`quantile_from_histogram`) — unlike a
    precomputed quantile, which cannot be re-aggregated.
    """
    edges = np.asarray(edges, dtype=float)
    nbins = len(edges) - 1
    vals = _blocks(arr, block)  # (ny_t, nx_t, K)
    ny_t, nx_t, K = vals.shape

    idx = np.digitize(vals, edges) - 1  # 0..nbins-1 inside range; else out
    valid = np.isfinite(vals) & (idx >= 0) & (idx < nbins)

    cell = np.repeat(np.arange(ny_t * nx_t)[:, None], K, axis=1).ravel()
    flat_bin = cell * nbins + idx.reshape(-1)
    counts = np.bincount(
        flat_bin[valid.reshape(-1)], minlength=ny_t * nx_t * nbins
    ).reshape(ny_t, nx_t, nbins)
    return counts


def quantile_from_histogram(
    counts: np.ndarray, edges: np.ndarray, q: float
) -> np.ndarray:
    """Recover the ``q``-quantile per cell from its histogram (linear interp).

    Assumes values are uniform within each bin, so accuracy is bounded by bin
    width. Cells with no samples return NaN.
    """
    edges = np.asarray(edges, dtype=float)
    counts = counts.astype(float)
    total = counts.sum(axis=-1)  # (ny_t, nx_t)
    cdf = np.cumsum(counts, axis=-1)  # upper cumulative count per bin
    target = q * total

    # First bin whose cumulative count reaches the target.
    b = (cdf < target[..., None]).sum(axis=-1)
    b = np.clip(b, 0, counts.shape[-1] - 1)

    lower_edge = edges[b]
    width = edges[b + 1] - edges[b]
    cdf_below = np.take_along_axis(
        np.concatenate([np.zeros(counts.shape[:-1] + (1,)), cdf[..., :-1]], axis=-1),
        b[..., None],
        axis=-1,
    ).squeeze(-1)
    in_bin = np.take_along_axis(counts, b[..., None], axis=-1).squeeze(-1)

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(in_bin > 0, (target - cdf_below) / in_bin, 0.0)
    out = lower_edge + frac * width
    out[total == 0] = np.nan
    return out


# Config / raster glue (dependencies imported lazily)

def load_grid_spec(cfg: Optional[dict] = None) -> dict:
    """Return the ``grid`` block from data_config (``target_res_m``, ``ref_raster``)."""
    if cfg is None:
        from src.config_utils import load_data_config

        cfg = load_data_config()
    grid = cfg.get("grid")
    if not grid or "target_res_m" not in grid:
        raise KeyError("data_config.json is missing a 'grid.target_res_m' entry.")
    return {"target_res_m": grid["target_res_m"], "ref_raster": grid.get("ref_raster")}


def native_res_m(raster_path: str) -> float:
    """Native pixel size (metres) read from a raster's transform."""
    import rasterio

    with rasterio.open(raster_path) as src:
        return abs(src.transform.a)


def load_ref(cfg: Optional[dict] = None):
    """Open the model-grid reference raster (``grid.ref_raster``) as a DataArray.

    Defines the CRS/transform/extent every reprojected product aligns to.
    """
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    from src.config_utils import load_data_config

    if cfg is None:
        cfg = load_data_config()
    return rioxarray.open_rasterio(cfg["grid"]["ref_raster"])


def reproject_to_ref(da, ref, resampling: str = "average"):
    """Resample a rioxarray DataArray onto the model grid via ``reproject_match``.

    Handles arbitrary native:target ratios (unlike the integer ``block_reduce``).
    Use ``average`` for continuous fields (the linear areal aggregate — apply any
    nonlinear transform afterward, at target resolution), ``nearest``/``mode``
    for categorical masks, ``sum`` where a count must be conserved.
    """
    from rasterio.enums import Resampling

    return da.rio.reproject_match(ref, resampling=Resampling[resampling])


def block_factor_for(raster_path: str, cfg: Optional[dict] = None) -> int:
    """Block factor to take one source raster from its native res to the target grid."""
    spec = load_grid_spec(cfg)
    return block_factor(native_res_m(raster_path), spec["target_res_m"])
