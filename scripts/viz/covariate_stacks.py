#!/usr/bin/env python3
"""One slideshow-ready composite per dataset (HYDE, LUH-3, climr): for each of
the three eras (1900-1915, 1950-1965, 2010-2025), a small perspective "deck"
of raster cards offset diagonally behind each other -- enough cards to read
as "many rasters" for that source, not necessarily one per actual covariate
(variables cycle to fill the deck depth if the dataset has fewer). The three
era-decks are chained left to right in a tight row, with a "..." marker
between them. The frontmost card in each deck is itself a real, recognizable
map -- there is no separate hero panel.

Reuses the exact same period-mean-on-grid pipeline as covariate_samples.py
(same variables, same three eras) so the cards and the standalone sample
PNGs agree.

Output:
  docs/img/covariate_stacks/hyde_stack.png
  docs/img/covariate_stacks/luh3_stack.png
  docs/img/covariate_stacks/climr_stack.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIZ_DIR = os.path.join(REPO_ROOT, "scripts", "viz")
for p in (REPO_ROOT, VIZ_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config_utils import load_data_config
from src.processing import regrid
import covariate_samples as cs  # noqa: E402  (path set up above)

OUT_DIR = os.path.join(REPO_ROOT, "docs", "img", "covariate_stacks")

STACKS = [
    ("HYDE", [v for v in cs.VARIABLES if v[0] == "HYDE"]),
    ("LUH-3", [v for v in cs.VARIABLES if v[0] == "LUH-3"]),
    ("climr", [v for v in cs.VARIABLES if v[0] == "climr"]),
    ("Z", [v for v in cs.VARIABLES if v[0] == "Z"]),
]

# --- Card-deck geometry (inches) ------------------------------------------
DEPTH = 4              # cards per deck, regardless of how many real variables
CARD_H_IN = 1.9
STEP_X_IN = 0.16       # per-depth-step offset (back cards shift up-right)
STEP_Y_IN = 0.13
BACK_ALPHA = 0.82      # cards behind the front one fade slightly for depth
GAP_IN = 0.18          # padding around the "..." separator
DOT_WIDTH_IN = 0.45
MARGIN_X_IN = 0.35
MARGIN_TOP_IN = 0.55
MARGIN_BOTTOM_IN = 0.55  # room for the era label under each deck


def _deck_variables(rows):
    """DEPTH variables, front-to-back, cycling through ``rows`` if fewer than
    DEPTH (front = rows[-1], then walking backwards through the list)."""
    n = len(rows)
    return [rows[(n - 1 - i) % n] for i in range(DEPTH)]


def _card_scale(row, ref, land, eras):
    """Absolute scale for one variable, pooled across every era in ``eras`` and
    shared by every era's card (so real cross-era differences in magnitude
    show up instead of being re-normalized away era by era). Uses the
    1st/99th percentile of the pooled eras rather than the literal min/max:
    it's still "whichever era holds the largest value" in effect, but a
    robust version -- a handful of extreme outlier pixels (e.g. climr's CMI
    has a few cells ~4x its own 98th percentile) shouldn't be allowed to
    wash out the whole map."""
    arrs = []
    for year_lo, year_hi in eras:
        grid = cs.period_mean(row, year_lo, year_hi, ref)
        arr = np.squeeze(cs._to_array(grid)).astype("float64")
        _tf = row[6]
        if _tf is not None:
            arr = _tf(np.clip(arr, 0, None))
        arrs.append(arr)
    pooled = np.concatenate([np.where(land, a, np.nan).ravel() for a in arrs])
    pooled = pooled[np.isfinite(pooled)]
    lo, hi = np.nanpercentile(pooled, [1, 99]) if pooled.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1e-9
    return dict(zip(eras, arrs)), (lo, hi)


def _build_stack(dataset_label, rows, land, extent, box_bounds, ref, eras=None):
    eras = eras if eras is not None else cs.PERIODS
    deck_vars = _deck_variables(rows)
    # Precompute each distinct variable's (era -> array) and shared scale once.
    cache = {}
    for row in {r[3]: r for r in deck_vars}.values():
        cache[row[3]] = _card_scale(row, ref, land, eras)

    box_w = box_bounds[2] - box_bounds[0]
    box_h = box_bounds[3] - box_bounds[1]
    aspect = box_w / box_h
    card_w_in = CARD_H_IN * aspect

    deck_w_in = card_w_in + (DEPTH - 1) * STEP_X_IN
    deck_h_in = CARD_H_IN + (DEPTH - 1) * STEP_Y_IN
    n_eras = len(eras)
    total_w_in = (
        2 * MARGIN_X_IN
        + n_eras * deck_w_in
        + (n_eras - 1) * (2 * GAP_IN + DOT_WIDTH_IN)
    )
    total_h_in = MARGIN_TOP_IN + deck_h_in + MARGIN_BOTTOM_IN

    fig = plt.figure(figsize=(total_w_in, total_h_in))
    fig.suptitle(f"{dataset_label} covariates", fontsize=13, y=0.99)

    def rect_in(x_in, y_in, w_in, h_in):
        return (x_in / total_w_in, y_in / total_h_in, w_in / total_w_in, h_in / total_h_in)

    cursor_x = MARGIN_X_IN
    deck_y0 = MARGIN_BOTTOM_IN
    for e, (year_lo, year_hi) in enumerate(eras):
        if e > 0:
            cursor_x += GAP_IN
            ax_dot = fig.add_axes(rect_in(cursor_x, deck_y0, DOT_WIDTH_IN, deck_h_in))
            ax_dot.set_axis_off()
            ax_dot.text(0.5, 0.5, "•••", ha="center", va="center", fontsize=18,
                        transform=ax_dot.transAxes)
            cursor_x += DOT_WIDTH_IN + GAP_IN

        deck_x0 = cursor_x
        for depth in reversed(range(DEPTH)):  # furthest back first, front last (on top)
            row = deck_vars[depth]
            _ds, _kind, _src, var, _resamp, _label, _tf, cmap = row
            eras_arr, (lo, hi) = cache[var]
            arr = eras_arr[(year_lo, year_hi)]
            masked = np.where(land, arr, np.nan)

            x_in = deck_x0 + depth * STEP_X_IN
            y_in = deck_y0 + depth * STEP_Y_IN
            ax = fig.add_axes(rect_in(x_in, y_in, card_w_in, CARD_H_IN))
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("white", alpha=0)
            alpha = 1.0 if depth == 0 else BACK_ALPHA
            ax.imshow(masked, extent=extent, origin="upper", cmap=cm,
                      vmin=lo, vmax=hi, alpha=alpha)
            ax.set_xlim(box_bounds[0], box_bounds[2])
            ax.set_ylim(box_bounds[1], box_bounds[3])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("black")
                spine.set_linewidth(0.8)
            ax.patch.set_facecolor("white")

        fig.text(
            (deck_x0 + deck_w_in / 2) / total_w_in, (MARGIN_BOTTOM_IN - 0.12) / total_h_in,
            f"{year_lo}-{year_hi}", ha="center", va="top", fontsize=10,
        )
        cursor_x = deck_x0 + deck_w_in

    out_png = os.path.join(OUT_DIR, f"{dataset_label.lower().replace('-', '')}_stack.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Wrote {out_png}")

    front_label = deck_vars[0][5]  # label of the depth=0 (frontmost, on-top) row
    return front_label


def main():
    cfg = load_data_config()
    ref = regrid.load_ref(cfg)
    box_bounds = tuple(cfg["grid"]["box_bounds"])
    ny, nx = ref.shape[-2], ref.shape[-1]
    transform = ref.rio.transform()
    land, _ = cs._land_mask(cfg, ref, ny, nx, transform)
    extent = (
        transform.c, transform.c + nx * transform.a,
        transform.f + ny * transform.e, transform.f,
    )

    # Z lives on its own (base-config, 27km) grid, pre-masked over water --
    # see covariate_samples.py's main() for the same reasoning.
    base_spec = cs._base_grid_spec()

    os.makedirs(OUT_DIR, exist_ok=True)
    contents_lines = []
    for dataset_label, rows in STACKS:
        if not rows:
            continue
        if dataset_label == "Z":
            sample = cs.period_mean(rows[0], *cs.PERIODS[0], ref)
            z_shape = np.squeeze(cs._to_array(sample)).shape
            row_land = np.ones(z_shape, dtype=bool)
            row_extent, row_box_bounds = base_spec["extent"], base_spec["box_bounds"]
            # Just the first/last era for Z, matching covariate_stacks_combined.py's
            # two-era convention -- three near-identical decades of a slow-moving
            # latent field add less than they cost in width.
            row_eras = (cs.PERIODS[0], cs.PERIODS[-1])
        else:
            row_land, row_extent, row_box_bounds = land, extent, box_bounds
            row_eras = None
        front_label = _build_stack(
            dataset_label, rows, row_land, row_extent, row_box_bounds, ref, eras=row_eras
        )
        png_name = f"{dataset_label.lower().replace('-', '')}_stack.png"
        contents_lines.append(f"{png_name}: front card (real, on top) = {front_label}")

    contents_path = os.path.join(OUT_DIR, "stack_contents.txt")
    with open(contents_path, "w") as f:
        f.write(
            "What's on top of each covariate stack -- the frontmost, fully-opaque "
            "card in every era-deck of a given PNG is always this one variable "
            "(the cards behind it cycle through the dataset's other variables "
            "purely for visual depth, not to be individually read).\n\n"
        )
        f.write("\n".join(contents_lines) + "\n")
    print(f"Wrote {contents_path}")


if __name__ == "__main__":
    main()
