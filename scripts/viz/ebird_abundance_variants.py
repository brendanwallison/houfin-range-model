"""Render five scales of the same eBird 2023 resident relative-abundance map."""
from pathlib import Path
import json

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures/external"


def main():
    with rasterio.open(ROOT / "data/audubon/houfin/generalists_HOFI_houfin_breeding_2010_NA_NA_range_27km.tif") as ref:
        shape, transform, crs, bounds = ref.shape, ref.transform, ref.crs, ref.bounds
    abundance = np.full(shape, np.nan, dtype="float32")
    source = ROOT / "data/ebird_abundance/houfin_abundance_seasonal_mean_27km_2023.tif"
    with rasterio.open(source) as src:
        assert src.count == 1 and src.descriptions == ("resident",)
        reproject(rasterio.band(src, 1), abundance, src_transform=src.transform,
                  src_crs=src.crs, src_nodata=src.nodata, dst_transform=transform,
                  dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.average)
    countries = gpd.read_file(ROOT / "data/naturalearth/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp").to_crs(crs)
    land = rasterize(((g, 1) for g in countries.geometry), out_shape=shape,
                     transform=transform, dtype="uint8").astype(bool)
    valid = np.isfinite(abundance) & land
    assert np.all(abundance[valid] >= 0)
    positive = valid & (abundance > 0)
    quantiles = np.quantile(abundance[positive], np.linspace(0, 1, 6))
    assert np.all(np.diff(quantiles) > 0)
    fixed = np.array([0, .1, .5, 1, 2, 5, 10, np.inf])
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    OUT.mkdir(parents=True, exist_ok=True)

    variants = [
        ("absolute", "Raw values · linear scale", abundance, "Relative abundance", None),
        ("quantiles", "Positive-abundance quintiles", abundance, "Positive-cell quintile", quantiles),
        ("ranges", "Fixed abundance bands", abundance, "Relative abundance band", fixed),
        ("log1p", "Log1p scale", np.log1p(abundance), "log(1 + relative abundance)", None),
        ("sqrt", "Square-root scale", np.sqrt(abundance), "Square root of relative abundance", None),
    ]
    for key, subtitle, values, label, edges in variants:
        fig, ax = plt.subplots(figsize=(11, 7.5))
        countries.plot(ax=ax, color="#eeeeec", edgecolor="none", zorder=0)
        if edges is None:
            cmap = plt.get_cmap("viridis").copy()
            shown = np.ma.masked_where(~valid, values)
            im = ax.imshow(shown, extent=extent, origin="upper", cmap=cmap,
                           vmin=0, vmax=float(values[valid].max()), interpolation="nearest", zorder=2)
            cbar = fig.colorbar(im, ax=ax, fraction=.027, pad=.025, shrink=.85)
        else:
            # Zero has its own pale swatch; quantiles apply only to positive cells.
            count = len(edges) - 1
            categories = np.zeros(shape, dtype="uint8")
            categories[positive] = np.searchsorted(edges[1:-1], abundance[positive], side="left") + 1
            cmap = ListedColormap(["#faf9f4"] + list(plt.get_cmap("viridis")(np.linspace(.12, .98, count))))
            shown = np.ma.masked_where(~valid, categories)
            im = ax.imshow(shown, extent=extent, origin="upper", cmap=cmap,
                           norm=BoundaryNorm(np.arange(count+2)-.5, count+1),
                           interpolation="nearest", zorder=2)
            cbar = fig.colorbar(im, ax=ax, fraction=.027, pad=.025, shrink=.85,
                                ticks=np.arange(count+1))
            if key == "quantiles":
                labels = ["0"] + [f"Q{i+1}: {edges[i]:.3g}–{edges[i+1]:.3g}" for i in range(count)]
            else:
                labels = ["0", ">0–0.1", ">0.1–0.5", ">0.5–1", ">1–2", ">2–5", ">5–10", ">10"]
            cbar.ax.set_yticklabels(labels, fontsize=9)
        cbar.set_label(label, fontsize=10)
        countries.boundary.plot(ax=ax, color="#777777", linewidth=.4, zorder=3)
        ax.set(xlim=(bounds.left, bounds.right), ylim=(bounds.bottom, bounds.top))
        ax.set_axis_off()
        fig.suptitle("House Finch: eBird relative abundance", fontsize=20, y=.97)
        ax.set_title(f"2023 resident seasonal mean · {subtitle}", fontsize=12, color="#555555", pad=14)
        fig.text(.5, .055, "eBird Status & Trends · 27 km display grid · Gray: missing data / outside land mask",
                 ha="center", fontsize=9, color="#666666")
        fig.text(.5, .025, "Relative abundance is a standardized model estimate, not total population size. Continuous scales use the full range.",
                 ha="center", fontsize=8.5, color="#666666")
        fig.subplots_adjust(left=.015, right=.78, top=.88, bottom=.12)
        path = OUT / f"houfin_ebird_abundance_{key}.png"
        fig.savefig(path, dpi=220, facecolor="white")
        plt.close(fig)
        print(path)
    receipt = {"source": str(source.relative_to(ROOT)), "band": "resident", "year": 2023,
               "quantiles": "Quintiles of positive finite land cells in the displayed extent; zero separate",
               "quintile_edges": quantiles.tolist(), "fixed_upper_bounds": [.1, .5, 1, 2, 5, 10, "infinity"],
               "valid_land_cells": int(valid.sum()), "positive_land_cells": int(positive.sum()),
               "maximum": float(abundance[valid].max()), "resampling": "average onto 27 km ESRI:102003 lattice"}
    (OUT / "houfin_ebird_abundance_scales.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
