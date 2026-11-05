"""Export saved biolith occupancy probabilities, without refitting or thresholding."""
from pathlib import Path
import json
import hashlib

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
TITLES = {"standard": "Standard environmental covariates",
          "full": "Full environmental covariates", "latent": "Learned habitat basis (Z)"}


def main():
    source = ROOT / "data/sdm_benchmark"
    out = ROOT / "figures/external"
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(ROOT / "data/audubon/houfin/generalists_HOFI_houfin_breeding_2010_NA_NA_range_27km.tif") as ref:
        shape, bounds, crs = ref.shape, ref.bounds, ref.crs
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    countries = gpd.read_file(ROOT / "data/naturalearth/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp").to_crs(crs)
    surfaces, receipt = {}, {}
    for tier in TITLES:
        path = source / f"biolith_{tier}_pred.npz"
        with np.load(path) as z:
            surfaces[tier] = z["grid_psi"].copy()
        grid = surfaces[tier]
        assert grid.shape == shape
        finite = grid[np.isfinite(grid)]
        assert len(finite) and finite.min() >= 0 and finite.max() <= 1
        meta = json.loads((source / f"biolith_{tier}.json").read_text())
        receipt[tier] = {"source": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "field": "grid_psi", "window": meta["window"], "years": meta["years"],
                         "covariate_year": meta["covariate_year"], "occu_convergence": meta["occu"]["convergence"]["worst"],
                         "divergences": meta["occu"]["sampler"]["divergences"]}

    def panel(ax, tier):
        countries.plot(ax=ax, color="#eeeeec", edgecolor="none", zorder=0)
        im = ax.imshow(np.ma.masked_invalid(surfaces[tier]), extent=extent, origin="upper",
                       cmap="viridis", vmin=0, vmax=1, interpolation="nearest", zorder=2)
        countries.boundary.plot(ax=ax, color="#777777", linewidth=.4, zorder=3)
        ax.set(xlim=(bounds.left, bounds.right), ylim=(bounds.bottom, bounds.top))
        ax.set_axis_off()
        ax.set_title(TITLES[tier], fontsize=12, pad=12)
        return im

    footer = "BBS repeat surveys, 2015–2025 · 2025 covariates · 27 km grid · Gray: no prediction"
    for tier in TITLES:
        fig, ax = plt.subplots(figsize=(11, 7.5))
        im = panel(ax, tier)
        fig.colorbar(im, ax=ax, fraction=.028, pad=.025, shrink=.85).set_label("Occupancy probability ψ")
        fig.suptitle("House Finch: correlative SDM occupancy", fontsize=20, y=.97)
        fig.text(.5, .045, footer, ha="center", fontsize=9, color="#555555")
        fig.text(.5, .02, "Detection-corrected occupancy-model estimates; continuous probabilities, no binary threshold.",
                 ha="center", fontsize=9, color="#555555")
        fig.subplots_adjust(left=.015, right=.91, top=.88, bottom=.12)
        path = out / f"houfin_sdm_occupancy_{tier}.png"
        fig.savefig(path, dpi=220, facecolor="white")
        plt.close(fig)
        print(path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    for ax, tier in zip(axes, TITLES):
        im = panel(ax, tier)
    fig.subplots_adjust(left=.01, right=.91, top=.83, bottom=.15, wspace=.04)
    cax = fig.add_axes([.93, .25, .012, .48])
    fig.colorbar(im, cax=cax).set_label("Occupancy probability ψ")
    fig.suptitle("House Finch: correlative SDM occupancy", fontsize=19, y=.97)
    fig.text(.5, .08, footer, ha="center", fontsize=10, color="#555555")
    fig.text(.5, .035, "Detection-corrected occupancy estimates · Shared 0–1 scale · No binary threshold", ha="center", fontsize=10)
    for suffix in ("png", "pdf"):
        path = out / f"houfin_sdm_occupancy_comparison.{suffix}"
        fig.savefig(path, dpi=220, facecolor="white")
        print(path)
    plt.close(fig)
    (out / "houfin_sdm_occupancy_sources.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
