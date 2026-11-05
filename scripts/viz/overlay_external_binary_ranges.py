"""Plot published Audubon breeding and eBird resident binary ranges.

Uses Audubon's original binary raster (nearest-neighbor reprojection), rather
than thresholding an averaged suitability or range raster. eBird uses its
published smoothed range polygon. Run from any directory; outputs PNG and PDF.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling


ROOT = Path(__file__).resolve().parents[2]


def main():
    # A 3 km display lattice retains the original boundaries without making
    # the figure depend on the population model's grid or configuration.
    crs = "ESRI:102003"
    left, bottom, right, top = -2976931.1039, -1523184.5953, 3071068.8961, 2067815.4047
    res = 3000
    width, height = round((right-left)/res), round((top-bottom)/res)
    transform = from_origin(left, top, res, res)
    audubon = np.full((height, width), 255, dtype="uint8")
    source = ROOT / "data/audubon/houfin/generalists_HOFI_houfin_breeding_2010_NA_NA_range.tif"
    with rasterio.open(source) as src:
        reproject(rasterio.band(src, 1), audubon, src_transform=src.transform,
                  src_crs=src.crs, src_nodata=src.nodata,
                  dst_transform=transform, dst_crs=crs, dst_nodata=255,
                  resampling=Resampling.nearest)
    ebird = gpd.read_file(
        ROOT / "data/ebird_range/houfin_range_smooth_27km_2023.gpkg", layer="range"
    ).to_crs(crs)
    assert set(ebird["season"]) == {"resident"}
    e = rasterize(((geom, 1) for geom in ebird.geometry),
                  out_shape=audubon.shape, transform=transform, dtype="uint8")
    countries = gpd.read_file(ROOT / "data/naturalearth/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp").to_crs(crs)
    land = rasterize(((geom, 1) for geom in countries.geometry),
                     out_shape=audubon.shape, transform=transform, dtype="uint8").astype(bool)
    # Do not imply absence where Audubon's raster has no coverage.
    valid = (audubon != 255) & land
    categories = (audubon == 1).astype("uint8") + 2 * e
    shown = np.ma.masked_where(~valid | (categories == 0), categories)
    colors = ["#ffffff", "#d98d36", "#5297c5", "#67518f"]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    countries.plot(ax=ax, color="#f0f0ee", edgecolor="#a4a4a4", linewidth=.45)
    ax.imshow(shown, extent=(left, right, bottom, top), origin="upper",
              cmap=ListedColormap(colors), vmin=-.5, vmax=3.5, interpolation="nearest", zorder=2)
    countries.boundary.plot(ax=ax, color="#777777", linewidth=.45, zorder=3)
    ax.set(xlim=(left, right), ylim=(bottom, top))
    ax.set_axis_off()
    fig.suptitle("House Finch: published binary ranges", fontsize=20, y=.97)
    ax.set_title("Audubon breeding range (2010) · eBird resident range (2023)", fontsize=12, pad=14, color="#555555")
    fig.legend(handles=[Patch(facecolor=colors[i], label=label) for i, label in
                         [(1, "Audubon only"), (2, "eBird only"), (3, "Both")]],
               loc="lower center", ncol=3, frameon=False, fontsize=12,
               bbox_to_anchor=(.5, .065))
    fig.text(.5, .025, "Published range classifications; seasons and baseline years differ. Compared within shared land coverage.",
             ha="center", fontsize=9, color="#666666")
    fig.subplots_adjust(left=.015, right=.985, top=.88, bottom=.14)
    out = ROOT / "figures/external/houfin_binary_range_overlay"
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suffix), dpi=220, facecolor="white")
        print(out.with_suffix(suffix))
    plt.close(fig)


if __name__ == "__main__":
    main()
