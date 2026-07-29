#!/usr/bin/env python3
"""Illustrative smooth spatiotemporal field of a vital rate r, plus its
source/sink binarization, on the same continental grid + Great Plains polygon
the rest of the hypothesis figures use.

This is a SCHEMATIC, not a model output: nothing here is fitted. It exists to
show what "r varies smoothly in space and time, and the Great Plains flips
from marginal sink to marginal source" would actually look like as a field,
so the source/sink maps that follow have an obvious continuous parent.

Construction (all on the project's 27 km model grid):

    r(x, t) = c0 + s * f(x) + b * g(t) * w(x)

  f(x)  spatial shape, tied to observed abundance: log1p of the eBird 2023
        seasonal-mean relative abundance, Gaussian-smoothed and rescaled to
        roughly [-1, 1] about its land mean, plus a small amount of smooth
        (heavily blurred, fixed-seed) noise so the field reads as a natural
        surface rather than a rescaled copy of the abundance map. This is
        what makes criterion (1) hold: high-abundance country is high-r.
  w(x)  where the temporal change is concentrated: a Gaussian-blurred Great
        Plains indicator (blur >> a grid cell), so the change enters as a
        broad smooth bump centered on the Plains and feathers out across the
        surrounding country -- no polygon edge is visible in the field.
  g(t)  linear ramp 0 -> 1 over the four timesteps.

  c0, b are then SOLVED (2 linear equations, 2 unknowns) so that the
  land-area mean of r inside the Great Plains equals -GP_MARGIN at the first
  timestep and +GP_MARGIN at the last -- criterion (2), marginally negative
  to marginally positive, with the crossing landing between the middle two
  panels rather than being nudged in by hand.

Both figures mask ocean (Natural Earth land, config coastline.land_source)
and draw the coastline + the cleaned Great Plains boundary for orientation.

Outputs:
  docs/img/hypothesis_r_field_continuous.png  (4 timesteps, diverging r)
  docs/img/hypothesis_r_field_sources.png     (the same 4, binarized)
  docs/img/hypothesis_r_field_combined.png    (both rows, shared timesteps)
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIZ_DIR = os.path.join(REPO_ROOT, "scripts", "viz")
for p in (REPO_ROOT, VIZ_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import overlay_great_plains_ebird as base  # noqa: E402  (path set up above)

OUT_DIR = os.path.join(REPO_ROOT, "docs", "img")

N_STEPS = 4
YEAR_LABELS = ("t₁", "t₂", "t₃", "t₄")

# How far the Great Plains mean sits from zero at the two ends of the series.
# "Marginal" is the whole point: a small number relative to the spread of r
# across the continent (SPATIAL_SCALE below), so the flip is a near-zero
# field drifting across zero, not a regime change.
GP_MARGIN = 0.02
SPATIAL_SCALE = 0.35        # amplitude of the abundance-driven spatial term
ABUND_SMOOTH_CELLS = 2.5    # blur on log1p abundance -> smooth spatial shape
NOISE_SMOOTH_CELLS = 6.0    # blur on the fixed-seed noise (>> a grid cell)
NOISE_WEIGHT = 0.30         # noise contribution, as a fraction of f's spread
GP_BLUR_CELLS = 5.0         # blur on the GP indicator -> broad change bump
NOISE_SEED = 20260726


def _build_grid_inputs():
    """Abundance, land mask, GP mask/geometry, all on the model grid."""
    cfg = base.load_data_config()
    ref = base.regrid.load_ref(cfg)
    project_crs = cfg["grid"]["box_crs"]
    box_bounds = tuple(cfg["grid"]["box_bounds"])
    box_geom = shapely_box(*box_bounds)

    da = rioxarray.open_rasterio(base.ABUNDANCE_TIF, masked=True)
    da = da.rio.write_crs(base.EBIRD_CRS, inplace=False)
    da = da.rio.write_nodata(float("nan"), inplace=False)
    da_grid = base.regrid.reproject_to_ref(da, ref, resampling="average")
    abund = np.nan_to_num(da_grid.values[0], nan=0.0)

    transform = da_grid.rio.transform()
    ny, nx = abund.shape
    extent = (
        transform.c, transform.c + nx * transform.a,
        transform.f + ny * transform.e, transform.f,
    )

    ecoregions = base.gpd.read_file(base.ECOREGION_SHP).to_crs(project_crs)
    great_plains_raw = ecoregions[ecoregions["NA_L1NAME"] == "GREAT PLAINS"]
    gp_geom = base._clean_great_plains_geom(
        unary_union(great_plains_raw.geometry), 2 * cfg["grid"]["target_res_m"]
    )

    land_path = os.path.join(cfg["datasets_root"], cfg["coastline"]["land_source"])
    land_gdf = base.gpd.read_file(land_path).to_crs(project_crs)
    # buffer(0) re-nodes the reprojected Natural Earth polygons; without it
    # unary_union can trip over tiny self-intersections (same fix-up as in
    # hypothesis_scenarios.py).
    land_geom = unary_union(land_gdf.geometry.buffer(0)).intersection(box_geom)

    def _rasterize(geom):
        return rasterize(
            [(geom, 1)], out_shape=(ny, nx), transform=transform,
            fill=0, all_touched=True, dtype="uint8",
        ).astype(bool)

    return {
        "abund": abund,
        "land": _rasterize(land_geom),
        "gp": _rasterize(gp_geom),
        "gp_geom": gp_geom,
        "land_geom": land_geom,
        "extent": extent,
        "box_bounds": box_bounds,
        "crs": project_crs,
    }


def _spatial_shape(abund, land):
    """f(x): smooth, abundance-following, roughly [-1, 1] about the land mean."""
    smooth = gaussian_filter(np.log1p(abund), ABUND_SMOOTH_CELLS, mode="nearest")

    rng = np.random.default_rng(NOISE_SEED)
    noise = gaussian_filter(rng.standard_normal(abund.shape), NOISE_SMOOTH_CELLS,
                            mode="nearest")

    def _center_scale(arr):
        vals = arr[land]
        centered = arr - vals.mean()
        spread = np.abs(centered[land]).max()
        return centered / spread if spread > 0 else centered

    f = _center_scale(smooth) + NOISE_WEIGHT * _center_scale(noise)
    return _center_scale(f)


def _change_weight(gp):
    """w(x): broad smooth bump centered on the Great Plains, peak 1."""
    w = gaussian_filter(gp.astype(float), GP_BLUR_CELLS, mode="nearest")
    return w / w.max()


def _solve_field(inputs):
    """r(x, t) for the four timesteps, with the GP endpoints pinned."""
    land, gp = inputs["land"], inputs["gp"]
    f = _spatial_shape(inputs["abund"], land)
    w = _change_weight(gp)
    g = np.linspace(0.0, 1.0, N_STEPS)

    # Great Plains land-area means of each term. r's GP mean at step k is
    #   c0 + s*<f> + b*g_k*<w>
    # so pinning k=0 to -GP_MARGIN and k=last to +GP_MARGIN is two linear
    # equations in (c0, b); g[0] == 0 makes it triangular.
    on = gp & land
    f_bar, w_bar = f[on].mean(), w[on].mean()
    c0 = -GP_MARGIN - SPATIAL_SCALE * f_bar
    b = 2 * GP_MARGIN / (g[-1] * w_bar)

    frames = [c0 + SPATIAL_SCALE * f + b * gk * w for gk in g]
    gp_means = [float(fr[on].mean()) for fr in frames]
    return frames, gp_means


def _draw_outlines(ax, inputs):
    base.gpd.GeoSeries([inputs["land_geom"]], crs=inputs["crs"]).boundary.plot(
        ax=ax, edgecolor="black", linewidth=0.5
    )
    base.gpd.GeoSeries([inputs["gp_geom"]], crs=inputs["crs"]).boundary.plot(
        ax=ax, edgecolor="black", linewidth=1.2, linestyle=(0, (3, 2))
    )
    box_minx, box_miny, box_maxx, box_maxy = inputs["box_bounds"]
    ax.set_xlim(box_minx, box_maxx)
    ax.set_ylim(box_miny, box_maxy)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _masked(frame, land):
    return np.ma.masked_array(frame, mask=~land)


def _r_norm(frames, land):
    # One diverging norm shared by every panel, centered on 0 so the
    # source/sink boundary is the color midpoint and panels are comparable.
    lo = min(fr[land].min() for fr in frames)
    hi = max(fr[land].max() for fr in frames)
    return TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)


SINK_COLOR = "#2c7bb6"
SOURCE_COLOR = "#f2c40c"
R_CMAP = LinearSegmentedColormap.from_list("YlBu_r", [SINK_COLOR, "#f7f7f7", SOURCE_COLOR])
BINARY_CMAP = ListedColormap([SINK_COLOR, SOURCE_COLOR])


def _panel_continuous(ax, frame, inputs, norm):
    im = ax.imshow(_masked(frame, inputs["land"]), extent=inputs["extent"],
                   origin="upper", cmap=R_CMAP, norm=norm)
    _draw_outlines(ax, inputs)
    return im


def _panel_binary(ax, frame, inputs):
    ax.imshow(_masked((frame > 0).astype(float), inputs["land"]),
              extent=inputs["extent"], origin="upper", cmap=BINARY_CMAP,
              vmin=0, vmax=1, interpolation="nearest")
    # The r = 0 contour: the same boundary the binarization draws, but as a
    # line, so the two rows of the combined figure are visibly the same object.
    ax.contour(_masked(frame, inputs["land"]), levels=[0.0], colors="black",
               linewidths=0.8, extent=inputs["extent"], origin="upper")
    _draw_outlines(ax, inputs)


def _titles(gp_means):
    return [f"{lab}   (Great Plains mean r = {m:+.3f})"
            for lab, m in zip(YEAR_LABELS, gp_means)]


# Every layout below places its axes in INCHES via _grid_axes rather than
# leaning on tight_layout/constrained_layout: the panels are forced to equal
# aspect (a map), so any layout engine that sizes boxes independently of the
# map's own aspect ratio leaves the content floating inside oversized boxes --
# which is what put a colorbar on top of the second row in the first draft.
PANEL_W_IN = 3.1
TITLE_H_IN = 0.32
ROW_LABEL_IN = 0.32
BAR_BLOCK_IN = 0.95   # colorbar / legend strip under the panels
MARGIN_IN = 0.12


def _grid_axes(n_rows, box_bounds, n_cols=N_STEPS, row_label=False):
    """Lay out n_rows x n_cols equal-aspect map panels; return (fig, axes)."""
    box_minx, box_miny, box_maxx, box_maxy = box_bounds
    aspect = (box_maxx - box_minx) / (box_maxy - box_miny)
    panel_h = PANEL_W_IN / aspect
    left = MARGIN_IN + (ROW_LABEL_IN if row_label else 0.0)

    fig_w = left + n_cols * PANEL_W_IN + MARGIN_IN
    fig_h = MARGIN_IN + BAR_BLOCK_IN + n_rows * (panel_h + TITLE_H_IN) + MARGIN_IN
    fig = plt.figure(figsize=(fig_w, fig_h))

    axes = []
    for row in range(n_rows):
        # Rows fill downward from the top; row r's bottom sits above every
        # row below it plus the bar block.
        y0 = MARGIN_IN + BAR_BLOCK_IN + (n_rows - 1 - row) * (panel_h + TITLE_H_IN)
        axes.append([
            fig.add_axes((
                (left + col * PANEL_W_IN) / fig_w, y0 / fig_h,
                PANEL_W_IN / fig_w, panel_h / fig_h,
            ))
            for col in range(n_cols)
        ])
    return fig, axes, (fig_w, fig_h)


def _add_colorbar(fig, im, norm, fig_size, y_in):
    fig_w, fig_h = fig_size
    cax = fig.add_axes((0.30, y_in / fig_h, 0.24, 0.16 / fig_h))
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("r  (per-capita growth rate; schematic)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    # TwoSlopeNorm's default ticks are dense and unevenly spaced (the two
    # halves are scaled independently), which ran the labels together.
    cbar.set_ticks([norm.vmin, norm.vmin / 2, 0.0, norm.vmax / 2, norm.vmax])
    cbar.ax.set_xticklabels([f"{v:+.2f}" for v in
                             (norm.vmin, norm.vmin / 2, 0.0, norm.vmax / 2, norm.vmax)])
    # Mark the source/sink threshold on the bar itself -- it is the level the
    # binarized row cuts at, and on a diverging bar it is otherwise just
    # "somewhere near the pale middle".
    cbar.ax.axvline(norm(0.0), color="black", linewidth=1.2)
    return cbar


def render_continuous(inputs, frames, gp_means, out_png):
    norm = _r_norm(frames, inputs["land"])
    fig, axes, fig_size = _grid_axes(1, inputs["box_bounds"])
    for ax, frame, title in zip(axes[0], frames, _titles(gp_means)):
        im = _panel_continuous(ax, frame, inputs, norm)
        ax.set_title(title, fontsize=9)
    _add_colorbar(fig, im, norm, fig_size, MARGIN_IN + 0.45)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def _binary_legend(fig, **kwargs):
    import matplotlib.patches as mpatches
    fig.legend(
        handles=[
            mpatches.Patch(facecolor=SOURCE_COLOR, edgecolor="black",
                           linewidth=0.5, label="source (r > 0)"),
            mpatches.Patch(facecolor=SINK_COLOR, edgecolor="black",
                           linewidth=0.5, label="sink (r < 0)"),
        ],
        loc="lower center", ncol=2, frameon=False, fontsize=10, **kwargs,
    )


def render_binary(inputs, frames, gp_means, out_png):
    fig, axes, fig_size = _grid_axes(1, inputs["box_bounds"])
    for ax, frame, title in zip(axes[0], frames, _titles(gp_means)):
        _panel_binary(ax, frame, inputs)
        ax.set_title(title, fontsize=9)
    _binary_legend(fig, bbox_to_anchor=(0.5, (MARGIN_IN + 0.25) / fig_size[1]))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def render_combined(inputs, frames, gp_means, out_png):
    norm = _r_norm(frames, inputs["land"])
    fig, axes, fig_size = _grid_axes(2, inputs["box_bounds"], row_label=True)
    for col, (frame, title) in enumerate(zip(frames, _titles(gp_means))):
        im = _panel_continuous(axes[0][col], frame, inputs, norm)
        axes[0][col].set_title(title, fontsize=9)
        _panel_binary(axes[1][col], frame, inputs)
    axes[0][0].text(-0.03, 0.5, "continuous r", transform=axes[0][0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=11)
    axes[1][0].text(-0.03, 0.5, "source / sink", transform=axes[1][0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=11)
    _add_colorbar(fig, im, norm, fig_size, MARGIN_IN + 0.62)
    _binary_legend(fig, bbox_to_anchor=(0.72, (MARGIN_IN + 0.42) / fig_size[1]))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main():
    inputs = _build_grid_inputs()
    frames, gp_means = _solve_field(inputs)
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Great Plains mean r by timestep: "
          + ", ".join(f"{m:+.4f}" for m in gp_means))
    render_continuous(inputs, frames, gp_means,
                      os.path.join(OUT_DIR, "hypothesis_r_field_continuous.png"))
    render_binary(inputs, frames, gp_means,
                  os.path.join(OUT_DIR, "hypothesis_r_field_sources.png"))
    render_combined(inputs, frames, gp_means,
                    os.path.join(OUT_DIR, "hypothesis_r_field_combined.png"))


if __name__ == "__main__":
    main()
