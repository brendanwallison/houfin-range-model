"""Great Plains geometry and west/barrier/east zones on the model grid.

The EPA/CEC Level I "GREAT PLAINS" ecoregion is a north-south band that TILTS
across latitude, so a single global x threshold misclassifies cells (the Pacific
Northwest is "west" but shares x-coordinates with the band's eastern edge much
further south). Every zone here is therefore computed per grid ROW.

Two zone definitions live here, deliberately, both derived from ONE pair of
per-row edges so they cannot drift:

* :func:`row_wise_gp_zones` -- the original masks (``inside_gp`` by centroid
  rasterization, west/east strictly outside it). This is what the eBird overlay
  figures use, and its behaviour is deliberately unchanged.
* :func:`corridor_zones` -- a GAP-FREE three-way partition using the x-interval
  between the per-row edges as the barrier. Required for anything that conserves
  mass: the ecoregion polygon has interior holes, so a cell can lie between the
  edges while being outside ``inside_gp``, leaving it in none of the three zones.
  A dispersal operator projected onto such a partition would silently lose mass
  into the unclassified cells. The corridor is also the better definition of a
  BARRIER: crossing means traversing that longitude band whether or not a given
  cell is technically in the ecoregion.

The source shapefile is NOT in git (``/data/`` is gitignored) and is not fetched
by ``scripts/tacc/download_all.sh``, so it exists only where it was downloaded by
hand. Anything that must run on HPC has to consume the PERSISTED RASTER written
by ``scripts/build_great_plains_mask.py``, never the shapefile -- see
:func:`read_zone_raster`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rasterio.features import rasterize
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

# Zone codes for the persisted raster. 0 is reserved for nodata so a partially
# written or wrongly-typed raster cannot masquerade as a valid zone.
ZONE_NODATA = 0
ZONE_WEST = 1
ZONE_BARRIER = 2
ZONE_EAST = 3
ZONE_NAMES = {ZONE_WEST: "west", ZONE_BARRIER: "barrier", ZONE_EAST: "east"}

ECOREGION_LEVEL1_FIELD = "NA_L1NAME"
ECOREGION_LEVEL1_VALUE = "GREAT PLAINS"


def clean_great_plains_geom(geom, tol):
    """Smooth the polygon: fill holes/notches, drop spikes, keep the main band.

    The shapefile is digitized far finer than 27 km, so its boundary carries small
    enclaves and thin protrusions that leave slivers unclassified right at the
    eastern edge. A morphological closing (dilate then erode) fills holes up to
    ``tol``; the following opening (erode then dilate) removes islands and spikes
    up to ``tol``. What survives can still include real disjunct patches larger
    than ``tol`` -- e.g. a ~6,800 km^2 outlier near San Antonio, TX -- which is why
    the largest polygon is selected explicitly rather than trusting the tolerance.
    """
    closed = geom.buffer(tol).buffer(-tol)
    opened = closed.buffer(-tol).buffer(tol)
    if opened.geom_type == "MultiPolygon":
        opened = max(opened.geoms, key=lambda g: g.area)
    return opened


def great_plains_edges(great_plains_geom, transform, ny, nx, box_bounds):
    """Per-row west/east x-extent of the polygon, plus cell x-centres and inside mask.

    THE shared core: both zone definitions in this module are built from this, so
    they always agree about where the band is. Rows the polygon does not reach
    (north or south of its extent) inherit the nearest row that does, via
    ffill/bfill -- without that they would be NaN and every cell in them would
    classify as neither west nor east.
    """
    box_minx, _, box_maxx, _ = box_bounds
    west_edge = np.full(ny, np.nan)
    east_edge = np.full(ny, np.nan)
    for i in range(ny):
        y_top = transform.f + i * transform.e
        y_bot = y_top + transform.e
        strip = shapely_box(box_minx, min(y_top, y_bot), box_maxx, max(y_top, y_bot))
        inter = great_plains_geom.intersection(strip)
        if not inter.is_empty:
            minx, _, maxx, _ = inter.bounds
            west_edge[i] = minx
            east_edge[i] = maxx
    west_edge = pd.Series(west_edge).ffill().bfill().to_numpy()
    east_edge = pd.Series(east_edge).ffill().bfill().to_numpy()

    x_centers = transform.c + (np.arange(nx) + 0.5) * transform.a
    inside_gp = rasterize(
        [(great_plains_geom, 1)], out_shape=(ny, nx), transform=transform,
        fill=0, dtype="uint8",
    ).astype(bool)
    return west_edge, east_edge, x_centers, inside_gp


def row_wise_gp_zones(great_plains_geom, transform, ny, nx, box_bounds):
    """West-of-GP / inside-GP / east-of-GP masks, per grid row.

    Behaviour preserved verbatim from ``scripts/viz/overlay_great_plains_ebird.py``,
    which is where this originally lived; the eBird overlay figures depend on it.
    Note the partition is NOT exhaustive -- see :func:`corridor_zones`.
    """
    west_edge, east_edge, x_centers, inside_gp = great_plains_edges(
        great_plains_geom, transform, ny, nx, box_bounds)
    west_mask = (x_centers[None, :] < west_edge[:, None]) & ~inside_gp
    east_mask = (x_centers[None, :] > east_edge[:, None]) & ~inside_gp
    return inside_gp, west_mask, east_mask, west_edge, east_edge


def corridor_zones(west_edge, east_edge, x_centers, ny, nx):
    """Gap-free three-way partition: west | barrier corridor | east.

    The barrier is the closed x-interval ``[west_edge, east_edge]`` at each row, so
    every cell falls in exactly one zone and a mass-conserving operator cannot leak
    into unclassified cells. Returns ``(west, barrier, east)`` boolean masks whose
    union is everything and whose pairwise intersections are empty.
    """
    x = np.broadcast_to(x_centers[None, :], (ny, nx))
    w = west_edge[:, None]
    e = east_edge[:, None]
    west = x < w
    east = x > e
    barrier = ~west & ~east
    return west, barrier, east


def build_zone_array(great_plains_geom, transform, ny, nx, box_bounds):
    """The uint8 zone raster: ZONE_WEST / ZONE_BARRIER / ZONE_EAST everywhere.

    Every cell is assigned (the corridor partition is exhaustive), so the array
    contains no ZONE_NODATA. Land masking is deliberately NOT applied here -- the
    zones are pure geography and the consumer intersects with its own land mask,
    which keeps this raster valid if the land mask is ever rebuilt.
    """
    west_edge, east_edge, x_centers, _ = great_plains_edges(
        great_plains_geom, transform, ny, nx, box_bounds)
    west, barrier, east = corridor_zones(west_edge, east_edge, x_centers, ny, nx)
    zones = np.full((ny, nx), ZONE_NODATA, dtype="uint8")
    zones[west] = ZONE_WEST
    zones[barrier] = ZONE_BARRIER
    zones[east] = ZONE_EAST
    if (zones == ZONE_NODATA).any():
        raise ValueError("zone raster has unassigned cells; the corridor partition "
                         "should be exhaustive")
    return zones


def load_great_plains_geom(shp_path, project_crs, clean_tol):
    """Read, dissolve, reproject and clean the Great Plains Level I polygon.

    Imports geopandas lazily: the diagnostics path reads the persisted raster and
    must not require geopandas (or the 44 MB shapefile) to be installed at all.
    """
    import geopandas as gpd

    ecoregions = gpd.read_file(shp_path).to_crs(project_crs)
    selected = ecoregions[ecoregions[ECOREGION_LEVEL1_FIELD] == ECOREGION_LEVEL1_VALUE]
    if selected.empty:
        raise ValueError(
            f"no {ECOREGION_LEVEL1_VALUE!r} polygon in {ECOREGION_LEVEL1_FIELD} "
            f"of {shp_path}")
    return clean_great_plains_geom(unary_union(selected.geometry), clean_tol)


def read_zone_raster(path, expected_shape=None):
    """Load the persisted zone raster into boolean masks.

    This is the ONLY entry point that should be used from a diagnostics or HPC run:
    it needs neither geopandas nor the shapefile. Returns
    ``{"west":..., "barrier":..., "east":...}``.
    """
    import rasterio

    with rasterio.open(path) as src:
        zones = src.read(1)
    if expected_shape is not None and zones.shape != tuple(expected_shape):
        raise ValueError(f"zone raster {path} is {zones.shape}, expected "
                         f"{tuple(expected_shape)}")
    unknown = set(np.unique(zones)) - set(ZONE_NAMES)
    if unknown:
        raise ValueError(f"zone raster {path} contains unexpected codes {sorted(unknown)} "
                         f"(expected {sorted(ZONE_NAMES)})")
    return {name: zones == code for code, name in ZONE_NAMES.items()}
