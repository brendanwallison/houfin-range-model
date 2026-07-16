"""De-dilated coastline: land-fraction land mask + gated observation snapping.

Replaces the old rule (ocean iff *all* BUI quantile bands were NaN), which made
any model-grid cell containing even one land subpixel "land" — dilating the
coast. Here a cell is land iff at least ``tau`` of its finest-resolution
subpixels are land, computed from a **continental** land/water source so
Canadian/Mexican land inside the bounding box is real land (not nodata).

De-dilating can strand coastal observations on newly-ocean cells, so
observations are **snapped** to the nearest land cell within a small radius;
anything farther (genuinely offshore, e.g. a pelagic eBird smear) is dropped
rather than pulled in. Snapping reuses ``scipy.ndimage.distance_transform_edt``,
the same primitive the cube gap-fill uses.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.processing import regrid


def compute_land_fraction(fine_land, block):
    """Per target cell, the fraction (0..1) of finest-res subpixels that are land.

    ``fine_land`` is a binary array (1 = land, 0 = water) at the fine source
    resolution; ``block`` = fine cells per target cell. A block-mean of the
    binary field is exactly the land fraction.
    """
    return regrid.block_reduce(np.asarray(fine_land, dtype=float), block, how="mean")


def land_mask_from_fraction(frac, tau=0.5):
    """Boolean model-grid land mask: a cell is land iff land_fraction >= tau."""
    return np.asarray(frac) >= tau


def snap_to_nearest_land(rows, cols, land_mask, max_cells=1):
    """Snap observation cells to the nearest land cell, within ``max_cells``.

    Returns ``(snapped_rows, snapped_cols, keep)``. Points already on land are
    unchanged. Ocean points within ``max_cells`` of land snap to that nearest
    land cell; ocean points farther than ``max_cells`` are marked ``keep=False``
    (genuinely offshore — drop, don't invent land for them).
    """
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    land_mask = np.asarray(land_mask, dtype=bool)
    # Distance (in cells) from every cell to the nearest land cell, plus the
    # index of that nearest land cell.
    dist, (iy, ix) = distance_transform_edt(~land_mask, return_indices=True)
    on_land = land_mask[rows, cols]
    d = dist[rows, cols]
    keep = on_land | (d <= max_cells)
    out_rows = np.where(on_land, rows, iy[rows, cols])
    out_cols = np.where(on_land, cols, ix[rows, cols])
    return out_rows, out_cols, keep
