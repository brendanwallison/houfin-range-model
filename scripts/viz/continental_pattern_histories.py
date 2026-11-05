"""Coordinated published distributions, abundance and competing histories."""
from pathlib import Path
from copy import deepcopy
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform as project_coords
from rasterio.features import rasterize
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hypothesis_scenarios as h
from src.config_utils import load_data_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecoregions", required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    cfg = deepcopy(load_data_config(ROOT / "config/data_config.json"))
    cfg["datasets_root"] = str(ROOT / "data")
    refpath = ROOT / "data/audubon/houfin/generalists_HOFI_houfin_breeding_2010_NA_NA_range_27km.tif"
    cfg["grid"]["ref_raster"] = str(refpath)
    cfg["coastline"]["land_source"] = "naturalearth/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp"
    h.base.load_data_config = lambda: deepcopy(cfg)
    h.base.ECOREGION_SHP = args.ecoregions
    zones, crs, bounds, land, range_geom = h._load_zone_geoms()
    with rasterio.open(refpath) as ref:
        abundance = np.full(ref.shape, np.nan, dtype="float32")
        with rasterio.open(h.base.ABUNDANCE_TIF) as src:
            reproject(rasterio.band(src, 1), abundance, src_transform=src.transform,
                      src_crs=src.crs, src_nodata=src.nodata, dst_transform=ref.transform,
                      dst_crs=ref.crs, dst_nodata=np.nan, resampling=Resampling.average)
    land_mask = rasterize([(land, 1)], out_shape=abundance.shape, transform=ref.transform).astype(bool)
    abundance[~land_mask] = np.nan
    abundance_bands = np.zeros(abundance.shape, dtype="float32")
    positive = np.isfinite(abundance) & (abundance > 0)
    abundance_bands[positive] = np.searchsorted([.1, .5, 1, 2, 5, 10], abundance[positive], side="left") + 1
    abundance_bands[~np.isfinite(abundance)] = np.nan
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    fig = plt.figure(figsize=(13, 10.5))

    def geography(ax):
        h._draw_zone(ax, land, "none", edgecolor="#666666", linewidth=.45)
        h._draw_zone(ax, zones["gp"], "none", edgecolor="#404040", linestyle=(0, (4, 3)), linewidth=.8)
        ax.set(xlim=bounds[::2], ylim=bounds[1::2], aspect="equal")
        ax.set_axis_off()

    # Preserve Audubon's published binary classification; do not threshold
    # suitability or the previously averaged 27 km range raster.
    display_transform = from_origin(bounds[0], bounds[3], 3000, 3000)
    display_shape = (round((bounds[3]-bounds[1])/3000), round((bounds[2]-bounds[0])/3000))
    audubon = np.full(display_shape, 255, dtype="uint8")
    with rasterio.open(ROOT / "data/audubon/houfin/generalists_HOFI_houfin_breeding_2010_NA_NA_range.tif") as src:
        reproject(rasterio.band(src, 1), audubon, src_transform=src.transform,
                  src_crs=src.crs, src_nodata=src.nodata, dst_transform=display_transform,
                  dst_crs=crs, dst_nodata=255, resampling=Resampling.nearest)
    ebird = rasterize([(range_geom, 1)], out_shape=display_shape, transform=display_transform, dtype="uint8")
    display_land = rasterize([(land, 1)], out_shape=display_shape, transform=display_transform).astype(bool)
    range_categories = (audubon == 1).astype("uint8") + 2 * ebird
    range_colors = ["#faf9f4", "#8db8d5", "#4b6aa0", "#493567"]
    ax = fig.add_axes([.035, .565, .445, .37])
    h._draw_zone(ax, land, "#ededeb", linewidth=0)
    for patch in ax.patches:
        patch.set_zorder(0)
    ax.imshow(np.ma.masked_where((audubon == 255) | ~display_land, range_categories),
              extent=extent, origin="upper", cmap=ListedColormap(range_colors),
              norm=BoundaryNorm(np.arange(5)-.5, 4), interpolation="nearest", zorder=.5)
    geography(ax)
    ax.set_title("A  Published distribution estimates", loc="left", fontweight="bold", fontsize=15, pad=22)
    ax.text(0, 1.015, "Audubon: 2010 breeding · eBird: 2023 resident", transform=ax.transAxes,
            fontsize=9, color="#555555", va="bottom")
    fig.legend(handles=[Patch(facecolor=range_colors[k], label=label) for k, label in
                        [(1, "Audubon only"), (2, "eBird only"), (3, "Both"), (0, "Neither")]],
               loc="lower left", bbox_to_anchor=(.035, .515), ncol=4, frameon=False,
               fontsize=9, columnspacing=1.2, handlelength=1.3)

    for rect, label, values, cmap, maximum, ticks, ticklabels in [
        ([.52, .565, .445, .37], "B  Relative abundance", abundance_bands, "magma", 7,
         list(range(8)), ["0", "0–.1", ".1–.5", ".5–1", "1–2", "2–5", "5–10", ">10"]),
    ]:
        ax = fig.add_axes(rect)
        h._draw_zone(ax, land, "#ededeb", linewidth=0)
        for patch in ax.patches:
            patch.set_zorder(0)
        if label.startswith("B"):
            palette = ListedColormap(plt.get_cmap(cmap)(np.linspace(.04, .97, 8)))
            im = ax.imshow(np.ma.masked_invalid(values), extent=extent, origin="upper", cmap=palette,
                           norm=BoundaryNorm(np.arange(9)-.5, 8), interpolation="nearest", zorder=.5)
        else:
            im = ax.imshow(np.ma.masked_invalid(values), extent=extent, origin="upper", cmap=cmap,
                           vmin=0, vmax=maximum, interpolation="nearest", zorder=.5)
        geography(ax)
        ax.set_title(label, loc="left", fontweight="bold", fontsize=15, pad=12)
        margin = .025 if label.startswith("B") else .07
        cax = fig.add_axes([rect[0]+margin, .535, rect[2]-2*margin, .012])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
        cb.ax.set_xticklabels(ticklabels)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, length=3)
        cb.set_label("P(occupied)" if label.startswith("A") else "Relative abundance bands", fontsize=10, labelpad=3)

    fig.text(.035, .455, "C  Competing histories", fontsize=15, fontweight="bold")
    fig.legend(handles=[Patch(facecolor="#ededeb", label="Missing prediction (A, B)"),
                        Line2D([0], [0], color="#404040", linestyle=(0, (4, 3)), linewidth=.8,
                               label="Focal Great Plains region")],
               loc="lower left", bbox_to_anchor=(.035, .477), ncol=2, frameon=False, fontsize=9)
    colors = {0: "#e2e2df", 1: "#b4d8c7", 2: "#28775e", 3: "#d58247"}

    def scenario(rect, categories, title):
        ax = fig.add_axes(rect)
        for name in ("background", "west", "gp", "east"):
            cat = 0 if name == "background" else categories[name]
            for geom, value in h._zone_pieces(zones[name], cat, range_geom):
                h._draw_zone(ax, geom, colors[value], linewidth=0)
        geography(ax)
        ax.set_title(title, fontsize=12, pad=5)
        return ax

    scenario([.045, .17, .32, .20], h.PRE_CATEGORIES, "Hypothesized pre-introduction\narrangement")
    scenario([.64, .265, .32, .20], h.H0_POST_CATEGORIES, "Sink → source conversion")
    sink_ax = scenario([.64, .035, .32, .20], h.H1_POST_CATEGORIES, "Occupation of a persistent sink")
    for start, end, curvature in [((-112, 36), (-101, 38), .15),
                                   ((-85, 37), (-98, 39), -.15)]:
        x, y = project_coords("EPSG:4326", crs, [start[0], end[0]], [start[1], end[1]])
        arrow = FancyArrowPatch((x[0], y[0]), (x[1], y[1]),
                                connectionstyle=f"arc3,rad={curvature}",
                                arrowstyle="-|>", mutation_scale=13,
                                linewidth=1.8, color="#222222", zorder=5)
        arrow.set_path_effects([pe.Stroke(linewidth=3.5, foreground="white"), pe.Normal()])
        sink_ax.add_patch(arrow)
    for end in ((.625, .375), (.625, .15)):
        fig.add_artist(FancyArrowPatch((.39, .27), end, transform=fig.transFigure,
                       arrowstyle="-|>", mutation_scale=17, linewidth=1.5, color="#444444"))
    fig.legend(handles=[Patch(facecolor=colors[k], label=label) for k, label in
                        [(2,"Occupied source"), (3,"Occupied sink"), (1,"Unoccupied source"), (0,"Unoccupied sink")]],
               loc="lower left", bbox_to_anchor=(.035, .035), ncol=2, frameon=False, fontsize=10,
               columnspacing=1.8, handlelength=1.5)
    out = ROOT / "figures/external/continental_pattern_histories"
    for extension in ("png", "pdf"):
        fig.savefig(out.with_suffix("."+extension), dpi=300, facecolor="white")
        print(out.with_suffix("."+extension))
    plt.close(fig)
    caption = (
        "**Continental distribution and abundance admit contrasting demographic histories.** "
        "Published distribution estimates span a region with markedly lower relative abundance between western "
        "and eastern concentrations. This pattern motivates two hypotheses: the historically limiting interior "
        "became self-sustaining, or its occupation remains dependent on immigration from productive regions on "
        "either side. Demographic classifications in C are hypothesized rather than inferred from A or B.\n\n"
        "(A) Published binary distribution estimates: Audubon Survival by Degrees breeding range for the "
        "2010 baseline and eBird Status & Trends resident range for 2023 (published smoothed 27 km polygon). "
        "Colors identify Audubon only, eBird only, overlap, and neither within shared land coverage. "
        "Audubon's original binary raster is reprojected with nearest-neighbor sampling on a 3 km display "
        "lattice; its suitability values are not thresholded anew. These products differ in season, baseline "
        "period, and range-classification method; they are not ecological occupancy probabilities. "
        "(B) eBird Status & Trends 2023 resident seasonal-mean relative abundance, shown in fixed bands: "
        "zero, >0–0.1, >0.1–0.5, >0.5–1, >1–2, >2–5, >5–10, and >10. "
        "Gray land in A indicates unavailable Audubon coverage, and in B indicates missing predictions; "
        "neither published range in A is pale cream, zero abundance in B is the darkest band, "
        "and white indicates water or outside the displayed land domain. "
        "(C) Schematic hypotheses, not dated reconstructions: an occupied western source, an unoccupied "
        "eastern source, and an unoccupied interior sink precede either sink-to-source conversion or occupation "
        "of a persistent sink. Inward arrows in the persistent-sink panel depict hypothesized immigration "
        "from western and eastern productive regions; they do not estimate movement paths or rates. "
        "Historical absence alone does not establish source or sink status. "
        "The schematic uses the existing zone construction within the modern eBird range. "
        "All maps share an Albers equal-area projection and geographic extent. Dashed outlines identify the "
        "focal Great Plains region (CEC Level I), with the same geographic meaning in every panel; they are "
        "not estimated demographic boundaries. Our fitted occupancy surfaces and their covariate comparison "
        "are supplementary analyses. Their 2015–2025 years-as-replicates design assumes temporal closure; "
        "assessment of closure and temporal confounding is required before interpreting them as persistent "
        "route occupation. They do not underpin the geographic claim in this figure.\n"
    )
    out.with_suffix(".caption.md").write_text(caption, encoding="utf-8")


if __name__ == "__main__":
    main()
