#!/usr/bin/env python3
"""A single combined slideshow figure with all three data sources (HYDE,
LUH-3, climr) stacked one above the other, sharing one pair of era labels
between them instead of repeating the era under every dataset's own row.

Differs from covariate_stacks.py in two ways:
  - Only the first and last era are shown (1900-1915 and 2010-2025), with a
    single "..." marker between them -- the middle era (1950-1965) is
    dropped here to keep the combined figure tight.
  - The era date labels appear once, below the bottom row, lined up under
    the shared deck columns (deck geometry -- card size/offsets -- is
    identical across datasets, so the two era columns land at the same x
    position in every row without any extra alignment work).

Each dataset row still uses the exact same perspective card-deck rendering
(and the same real-data pipeline) as covariate_stacks.py; the per-dataset
PNGs there remain the reference for the individual, three-era version.

Output:
  docs/img/covariate_stacks/all_sources_stack.png
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
import covariate_stacks as st  # noqa: E402  (reuse deck geometry + helpers)

OUT_DIR = os.path.join(REPO_ROOT, "docs", "img", "covariate_stacks")
OUT_PNG = os.path.join(OUT_DIR, "all_sources_stack.png")

ERAS = (cs.PERIODS[0], cs.PERIODS[-1])  # start + end only, middle era dropped

ROW_LABEL_W_IN = 0.9    # left margin reserved for the dataset name
ROW_GAP_IN = 0.25       # vertical padding between dataset rows
BOTTOM_LABEL_IN = 0.5   # room for the single shared row of era labels


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

    box_w = box_bounds[2] - box_bounds[0]
    box_h = box_bounds[3] - box_bounds[1]
    aspect = box_w / box_h
    card_w_in = st.CARD_H_IN * aspect
    deck_w_in = card_w_in + (st.DEPTH - 1) * st.STEP_X_IN
    deck_h_in = st.CARD_H_IN + (st.DEPTH - 1) * st.STEP_Y_IN
    n_eras = len(ERAS)

    total_w_in = (
        ROW_LABEL_W_IN
        + st.MARGIN_X_IN
        + n_eras * deck_w_in
        + (n_eras - 1) * (2 * st.GAP_IN + st.DOT_WIDTH_IN)
        + st.MARGIN_X_IN
    )
    n_rows = len(st.STACKS)
    total_h_in = (
        st.MARGIN_TOP_IN
        + n_rows * deck_h_in
        + (n_rows - 1) * ROW_GAP_IN
        + BOTTOM_LABEL_IN
    )

    fig = plt.figure(figsize=(total_w_in, total_h_in))

    def rect_in(x_in, y_in, w_in, h_in):
        return (x_in / total_w_in, y_in / total_h_in, w_in / total_w_in, h_in / total_h_in)

    # Rows are drawn top to bottom, but figure y=0 is the bottom, so row 0
    # (HYDE) gets the highest y0.
    row_y0s = [
        BOTTOM_LABEL_IN + (n_rows - 1 - r) * (deck_h_in + ROW_GAP_IN)
        for r in range(n_rows)
    ]

    era_deck_x0 = None  # same for every row; captured once to place shared labels
    for r, (dataset_label, rows) in enumerate(st.STACKS):
        if not rows:
            continue
        deck_vars = st._deck_variables(rows)

        # Z is on its own (base-config, 27km) grid and comes pre-masked over
        # water -- box_bounds/extent are numerically identical either way
        # (local_data_config.json only overrides resolution/ref_raster, not
        # box_bounds), but the array SHAPE differs from the other datasets'
        # local-grid shape, so "land" must match per row (see
        # covariate_samples.py's main()/covariate_stacks.py's main() for the
        # same reasoning).
        if dataset_label == "Z":
            sample = cs.period_mean(rows[0], *cs.PERIODS[0], ref)
            z_shape = np.squeeze(cs._to_array(sample)).shape
            row_land = np.ones(z_shape, dtype=bool)
        else:
            row_land = land

        cache = {}
        for row in {rr[3]: rr for rr in deck_vars}.values():
            cache[row[3]] = st._card_scale(row, ref, row_land, ERAS)

        row_y0 = row_y0s[r]
        fig.text(
            0.01, (row_y0 + deck_h_in / 2) / total_h_in, dataset_label,
            fontsize=11, ha="left", va="center", weight="bold",
        )

        cursor_x = ROW_LABEL_W_IN + st.MARGIN_X_IN
        deck_x0s = []
        for e, (year_lo, year_hi) in enumerate(ERAS):
            if e > 0:
                cursor_x += st.GAP_IN
                ax_dot = fig.add_axes(rect_in(cursor_x, row_y0, st.DOT_WIDTH_IN, deck_h_in))
                ax_dot.set_axis_off()
                ax_dot.text(0.5, 0.5, "•••", ha="center", va="center", fontsize=18,
                            transform=ax_dot.transAxes)
                cursor_x += st.DOT_WIDTH_IN + st.GAP_IN

            deck_x0 = cursor_x
            deck_x0s.append(deck_x0)
            for depth in reversed(range(st.DEPTH)):
                row = deck_vars[depth]
                _ds, _kind, _src, var, _resamp, _label, _tf, cmap = row
                eras_arr, (lo, hi) = cache[var]
                arr = eras_arr[(year_lo, year_hi)]
                masked = np.where(row_land, arr, np.nan)

                x_in = deck_x0 + depth * st.STEP_X_IN
                y_in = row_y0 + depth * st.STEP_Y_IN
                ax = fig.add_axes(rect_in(x_in, y_in, card_w_in, st.CARD_H_IN))
                cm = plt.get_cmap(cmap).copy()
                cm.set_bad("white", alpha=0)
                alpha = 1.0 if depth == 0 else st.BACK_ALPHA
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

            cursor_x = deck_x0 + deck_w_in

        era_deck_x0 = deck_x0s  # identical across rows -- last write is fine

    # Single shared row of era labels, under the bottom-most dataset row.
    for (year_lo, year_hi), deck_x0 in zip(ERAS, era_deck_x0):
        fig.text(
            (deck_x0 + deck_w_in / 2) / total_w_in, (BOTTOM_LABEL_IN - 0.12) / total_h_in,
            f"{year_lo}-{year_hi}", ha="center", va="top", fontsize=11,
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
