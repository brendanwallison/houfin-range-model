"""Canonical readers for the project's ocean-mask raster convention.

Stored raster values are always ``0 = land`` and ``1 = ocean``. New rasters use
``255 = nodata`` so metadata never aliases a valid class. Older products wrote
``nodata=0`` even though zero meant land; readers deliberately trust the raw
0/1 values and warn about that legacy metadata instead of masking out all land.
"""
from __future__ import annotations

import warnings

import numpy as np
import rasterio


def read_land_mask(path, *, return_meta=False):
    """Read a semantic land mask, validating 0=land / 1=ocean without inversion ambiguity."""
    with rasterio.open(path) as src:
        raw = src.read(1)
        nodata = src.nodata
        meta = {
            "shape": (src.height, src.width),
            "transform": tuple(src.transform),
            "crs": src.crs.to_string() if src.crs else None,
            "res": tuple(float(x) for x in src.res),
            "nodata": nodata,
        }
    if nodata in (0, 1):
        warnings.warn(
            f"{path} has legacy nodata={nodata}, which aliases a semantic mask value; "
            "raw 0=land/1=ocean values are being used. Regenerate the mask to write nodata=255.",
            RuntimeWarning, stacklevel=2,
        )
    valid_values = set(np.unique(raw).tolist())
    allowed = {0, 1}
    if nodata is not None and nodata not in (0, 1):
        allowed.add(nodata)
    unexpected = valid_values - allowed
    if unexpected:
        raise ValueError(f"{path} is not a 0=land/1=ocean mask; unexpected values {unexpected}")
    land = raw == 0
    return (land, meta) if return_meta else land
