#!/usr/bin/env python3
"""Overlay the EPA/CEC Level I "GREAT PLAINS" ecoregion on the modern eBird
House Finch relative-abundance map, on the project's own model grid. Writes
two SEPARATE figures -- abundance (continuous) and range (occupancy, binary)
are different products and don't belong on the same axes.

Inputs (already fetched):
  data/ebird_abundance/houfin_abundance_seasonal_mean_27km_2023.tif
      eBird Status & Trends 2023 seasonal-mean relative abundance, native
      27 km / EPSG:8857 (Equal Earth). Fetched via src/data/acquire/ebird.py's
      FETCH_URL/stream_download helpers (objkey
      2023/houfin/seasonal/houfin_abundance_seasonal_mean_27km_2023.tif).
  data/ecoregions/NA_CEC_Eco_Level1.shp
      EPA/CEC Ecoregions of North America, Level I (coarsest of 4 levels).
      https://www.epa.gov/eco-research/ecoregions-north-america
  data/ebird_range/houfin_range_smooth_27km_2023.gpkg
      eBird Status & Trends 2023 smoothed range boundary (the "range" layer;
      the gpkg also has an unused "prediction_area" layer), EPSG:4326. Fetched
      via the same stream_download_raw helper (objkey
      2023/houfin/ranges/houfin_range_smooth_27km_2023.gpkg).

The raster is reprojected onto the project's model grid (``grid.ref_raster``:
ESRI:102003, 27 km, box_bounds from config/data_config.json) via the same
``regrid.reproject_to_ref`` helper ``src/data/preprocess/ebird.py`` uses, so
this matches the grid the rest of the pipeline works on -- not the raster's
native global extent. Abundance is displayed as log1p(x) since raw relative
abundance is heavily right-skewed (a small number of cells with much higher
values than the rest).

Also writes three hypothesis-illustration variants of the abundance figure
(zone classification is row-wise: for each grid row, the Great Plains
polygon's west/east edges at that row's latitude split the row into
west-of-GP / inside-GP / east-of-GP -- this follows the ecoregion's
north-south tilt rather than a single global x threshold):
  docs/img/houfin_abundance_gp_zeroed.png       (inside Great Plains -> 0)
  docs/img/houfin_abundance_gp_east_zeroed.png  (inside GP + east of GP -> 0)
  docs/img/houfin_abundance_no_overlay.png      (no Great Plains boundary drawn)

Same three variants for the range polygon (clipped by geometry difference
instead of zeroed by value, since range is a vector occupancy layer):
  docs/img/houfin_range_gp_zeroed.png       (Great Plains clipped out)
  docs/img/houfin_range_gp_east_zeroed.png  (GP + east of GP clipped out)
  docs/img/houfin_range_no_overlay.png      (no Great Plains boundary drawn)

Output:
  docs/img/houfin_abundance_great_plains.png (log1p abundance + Great Plains)
  docs/img/houfin_range_great_plains.png (range polygon + Great Plains)
"""
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.features import rasterize
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config_utils import load_data_config
from src.processing import regrid

ABUNDANCE_TIF = os.path.join(
    REPO_ROOT, "data", "ebird_abundance", "houfin_abundance_seasonal_mean_27km_2023.tif"
)
ECOREGION_SHP = os.path.join(REPO_ROOT, "data", "ecoregions", "NA_CEC_Eco_Level1.shp")
RANGE_GPKG = os.path.join(
    REPO_ROOT, "data", "ebird_range", "houfin_range_smooth_27km_2023.gpkg"
)
OUT_ABUNDANCE_PNG = os.path.join(REPO_ROOT, "docs", "img", "houfin_abundance_great_plains.png")
OUT_RANGE_PNG = os.path.join(REPO_ROOT, "docs", "img", "houfin_range_great_plains.png")
OUT_GP_ZEROED_PNG = os.path.join(REPO_ROOT, "docs", "img", "houfin_abundance_gp_zeroed.png")
OUT_GP_EAST_ZEROED_PNG = os.path.join(
    REPO_ROOT, "docs", "img", "houfin_abundance_gp_east_zeroed.png"
)
OUT_NO_OVERLAY_PNG = os.path.join(REPO_ROOT, "docs", "img", "houfin_abundance_no_overlay.png")
OUT_RANGE_GP_ZEROED_PNG = os.path.join(REPO_ROOT, "docs", "img", "houfin_range_gp_zeroed.png")
OUT_RANGE_GP_EAST_ZEROED_PNG = os.path.join(
    REPO_ROOT, "docs", "img", "houfin_range_gp_east_zeroed.png"
)
OUT_RANGE_NO_OVERLAY_PNG = os.path.join(REPO_ROOT, "docs", "img", "houfin_range_no_overlay.png")

EBIRD_CRS = "EPSG:8857"


def _plot_range(range_geom, project_crs, box_bounds, out_png, great_plains=None):
    box_minx, box_miny, box_maxx, box_maxy = box_bounds
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("white")
    if range_geom is not None and not range_geom.is_empty:
        gpd.GeoSeries([range_geom], crs=project_crs).plot(
            ax=ax, facecolor="tab:orange", edgecolor="none", alpha=0.6
        )
    if great_plains is not None:
        great_plains.boundary.plot(ax=ax, edgecolor="red", linewidth=1.5)
    ax.set_xlim(box_minx, box_maxx)
    ax.set_ylim(box_miny, box_maxy)
    ax.set_axis_off()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def _plot_abundance(log_abund, raster_extent, box_bounds, out_png, great_plains=None):
    box_minx, box_miny, box_maxx, box_maxy = box_bounds
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("white")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white", alpha=0)
    ax.imshow(log_abund, extent=raster_extent, origin="upper", cmap=cmap, vmin=0)
    if great_plains is not None:
        great_plains.boundary.plot(ax=ax, edgecolor="red", linewidth=1.5)
    ax.set_xlim(box_minx, box_maxx)
    ax.set_ylim(box_miny, box_maxy)
    ax.set_axis_off()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def _row_wise_gp_zones(great_plains_geom, transform, ny, nx, box_bounds):
    """West-of-GP / inside-GP / east-of-GP masks, computed per grid ROW.

    The Great Plains ecoregion is a north-south band that tilts across
    latitude, so a single global x threshold misclassifies cells (e.g. the
    Pacific Northwest is "west" but shares x-coordinates with the eastern
    edge of the band further south). Per row, intersect the polygon with a
    thin horizontal strip at that row's latitude and take the west/east
    extent of that intersection; rows the polygon doesn't reach (north/south
    of its extent) inherit the nearest row that does.
    """
    box_minx, _, box_maxx, _ = box_bounds
    row_h = abs(transform.e)
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
    west_mask = (x_centers[None, :] < west_edge[:, None]) & ~inside_gp
    east_mask = (x_centers[None, :] > east_edge[:, None]) & ~inside_gp
    return inside_gp, west_mask, east_mask, west_edge, east_edge


def _clean_great_plains_geom(geom, tol):
    """Smooth the Great Plains polygon: fill small interior holes/notches,
    drop small spikes/disjoint fragments (at a scale of `tol` meters), and
    keep only the single largest contiguous polygon.

    The EPA/CEC shapefile is digitized at a much finer resolution than the
    27 km model grid, so its boundary has small enclaves and thin
    protrusions that (a) look noisy as a hairline overlay and (b) leave
    slivers un-zeroed right at the eastern edge when used for the zone
    masks/clips below. A morphological closing (dilate then erode) fills
    holes/notches up to size `tol`; a following opening (erode then dilate)
    removes islands/spikes up to size `tol`. What survives that can still
    include real, non-trivial disjunct patches of the ecoregion (e.g. a
    ~6,800 km^2 outlier near San Antonio, TX) -- bigger than `tol`, so the
    opening doesn't remove it -- which is why the main polygon is picked out
    explicitly rather than relying on tolerance alone. Net effect: just the
    single main north-south band, sub-`tol` detail smoothed away.
    """
    closed = geom.buffer(tol).buffer(-tol)
    opened = closed.buffer(-tol).buffer(tol)
    if opened.geom_type == "MultiPolygon":
        opened = max(opened.geoms, key=lambda g: g.area)
    return opened


def _east_of_gp_geom(east_edge, transform, ny, box_maxx):
    """Vector polygon for 'east of GP', built from the same per-row east_edge
    used for the raster mask (rather than polygonizing the raster mask), so
    the range clip stays exact instead of picking up stairstep cell edges.
    """
    boxes = []
    for i in range(ny):
        y_top = transform.f + i * transform.e
        y_bot = y_top + transform.e
        x0 = east_edge[i]
        if x0 < box_maxx:
            boxes.append(shapely_box(x0, min(y_top, y_bot), box_maxx, max(y_top, y_bot)))
    return unary_union(boxes)


def main():
    cfg = load_data_config()
    ref = regrid.load_ref(cfg)
    box_minx, box_miny, box_maxx, box_maxy = cfg["grid"]["box_bounds"]
    project_crs = cfg["grid"]["box_crs"]

    da = rioxarray.open_rasterio(ABUNDANCE_TIF, masked=True)
    da = da.rio.write_crs(EBIRD_CRS, inplace=False)
    da = da.rio.write_nodata(float("nan"), inplace=False)
    da_grid = regrid.reproject_to_ref(da, ref, resampling="average")

    abund = np.ma.masked_invalid(da_grid.values[0])
    log_abund = np.ma.masked_array(np.log1p(abund.filled(0)), mask=abund.mask)

    transform = da_grid.rio.transform()
    ny, nx = abund.shape
    raster_extent = (
        transform.c, transform.c + nx * transform.a,
        transform.f + ny * transform.e, transform.f,
    )

    ecoregions = gpd.read_file(ECOREGION_SHP).to_crs(project_crs)
    great_plains_raw = ecoregions[ecoregions["NA_L1NAME"] == "GREAT PLAINS"]
    if great_plains_raw.empty:
        raise SystemExit("No 'GREAT PLAINS' polygon found in NA_L1NAME.")

    # Clean once, at a scale of 2 grid cells, and use this everywhere below
    # (boundary display AND the zone masks/clips) so the overlay and the
    # zeroed/clipped variants agree with each other.
    gp_clean_tol = 2 * cfg["grid"]["target_res_m"]
    gp_geom = _clean_great_plains_geom(unary_union(great_plains_raw.geometry), gp_clean_tol)
    great_plains = gpd.GeoSeries([gp_geom], crs=project_crs)

    houfin_range = gpd.read_file(RANGE_GPKG, layer="range").to_crs(project_crs)

    os.makedirs(os.path.dirname(OUT_ABUNDANCE_PNG), exist_ok=True)
    box_bounds = (box_minx, box_miny, box_maxx, box_maxy)

    # Figure 1: log1p abundance (continuous) + Great Plains boundary.
    _plot_abundance(log_abund, raster_extent, box_bounds, OUT_ABUNDANCE_PNG, great_plains)

    # Figure 2: range polygon (occupancy, binary) + Great Plains boundary --
    # a separate figure since abundance (continuous) and range (binary) are
    # different products and don't read sensibly on the same axes.
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("white")
    houfin_range.plot(ax=ax, facecolor="tab:orange", edgecolor="none", alpha=0.6)
    great_plains.boundary.plot(ax=ax, edgecolor="red", linewidth=1.5)
    ax.set_xlim(box_minx, box_maxx)
    ax.set_ylim(box_miny, box_maxy)
    ax.set_axis_off()
    fig.savefig(OUT_RANGE_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_RANGE_PNG}")

    # Hypothesis-illustration variants, all derived from the same raw `abund`
    # (zeroed before log1p, matching the transform used everywhere else).
    inside_gp, _, east_mask, _, east_edge = _row_wise_gp_zones(
        gp_geom, transform, ny, nx, box_bounds
    )

    abund_gp_zeroed = abund.filled(0).copy()
    abund_gp_zeroed[inside_gp] = 0
    log_gp_zeroed = np.ma.masked_array(np.log1p(abund_gp_zeroed), mask=abund.mask)
    _plot_abundance(log_gp_zeroed, raster_extent, box_bounds, OUT_GP_ZEROED_PNG, great_plains)

    abund_gp_east_zeroed = abund_gp_zeroed.copy()
    abund_gp_east_zeroed[east_mask] = 0
    log_gp_east_zeroed = np.ma.masked_array(
        np.log1p(abund_gp_east_zeroed), mask=abund.mask
    )
    _plot_abundance(
        log_gp_east_zeroed, raster_extent, box_bounds, OUT_GP_EAST_ZEROED_PNG, great_plains
    )

    # Same base figure as #1, just without the Great Plains boundary drawn.
    _plot_abundance(log_abund, raster_extent, box_bounds, OUT_NO_OVERLAY_PNG, great_plains=None)

    # Same three variants for the range polygon (occupancy). "Zeroing out" a
    # vector layer means clipping the polygon by the zone geometry (the GP
    # polygon itself for the interior; the row-wise east geometry, built from
    # the same east_edge used for the raster mask, for the eastern zone).
    range_geom = unary_union(houfin_range.geometry)

    range_gp_zeroed = range_geom.difference(gp_geom)
    _plot_range(range_gp_zeroed, project_crs, box_bounds, OUT_RANGE_GP_ZEROED_PNG, great_plains)

    east_geom = _east_of_gp_geom(east_edge, transform, ny, box_maxx)
    range_gp_east_zeroed = range_gp_zeroed.difference(east_geom)
    _plot_range(
        range_gp_east_zeroed, project_crs, box_bounds, OUT_RANGE_GP_EAST_ZEROED_PNG, great_plains
    )

    # Same base figure as Figure 2, just without the Great Plains boundary drawn.
    _plot_range(range_geom, project_crs, box_bounds, OUT_RANGE_NO_OVERLAY_PNG, great_plains=None)


if __name__ == "__main__":
    main()
