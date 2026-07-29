#!/usr/bin/env python3
"""Perspective card-deck view of the hypothesis_r_field.py schematic, matching
the same tight "..." two-era convention used elsewhere (covariate_stacks.py's
Z stack, covariate_stacks_combined.py): first and last timestep only (t1, t4),
each a 2-card deck with the source/sink binarization stacked OVER the raw
continuous r values (front = binary, back = continuous) rather than showing
them as two separate rows.

Does NOT modify hypothesis_r_field.py -- reuses its grid-building/field-solving
functions and its exact color conventions (R_CMAP, BINARY_CMAP, shared diverging
norm) so this is guaranteed to agree with that script's own continuous/binary
figures. Also reuses covariate_stacks.py's card-deck geometry constants, with
its own DEPTH=2 (one schematic "dataset", two representations of it, not
multiple covariates).

Output:
  docs/img/covariate_stacks/r_field_stack.png
"""
import os
import sys

import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIZ_DIR = os.path.join(REPO_ROOT, "scripts", "viz")
for p in (REPO_ROOT, VIZ_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import hypothesis_r_field as hrf  # noqa: E402  (path set up above; not modified)
import covariate_stacks as st     # noqa: E402  (reuse deck geometry constants)

OUT_DIR = os.path.join(REPO_ROOT, "docs", "img", "covariate_stacks")
OUT_PNG = os.path.join(OUT_DIR, "r_field_stack.png")

DEPTH = 2               # continuous r (back) + source/sink binary (front, "over" it)
STEPS = (0, -1)          # first and last of hrf.N_STEPS timesteps only


def main():
    inputs = hrf._build_grid_inputs()
    frames, gp_means = hrf._solve_field(inputs)
    land = inputs["land"]
    norm = hrf._r_norm(frames, land)
    box_bounds, extent = inputs["box_bounds"], inputs["extent"]

    box_w = box_bounds[2] - box_bounds[0]
    box_h = box_bounds[3] - box_bounds[1]
    aspect = box_w / box_h
    card_w_in = st.CARD_H_IN * aspect
    deck_w_in = card_w_in + (DEPTH - 1) * st.STEP_X_IN
    deck_h_in = st.CARD_H_IN + (DEPTH - 1) * st.STEP_Y_IN
    n_eras = len(STEPS)

    total_w_in = (
        2 * st.MARGIN_X_IN
        + n_eras * deck_w_in
        + (n_eras - 1) * (2 * st.GAP_IN + st.DOT_WIDTH_IN)
    )
    total_h_in = st.MARGIN_TOP_IN + deck_h_in + st.MARGIN_BOTTOM_IN

    fig = plt.figure(figsize=(total_w_in, total_h_in))
    fig.suptitle("Hypothesis r field: source/sink over raw r", fontsize=13, y=0.99)

    def rect_in(x_in, y_in, w_in, h_in):
        return (x_in / total_w_in, y_in / total_h_in, w_in / total_w_in, h_in / total_h_in)

    r_cmap = hrf.R_CMAP.copy()
    r_cmap.set_bad("white", alpha=0)
    binary_cmap = hrf.BINARY_CMAP.copy()
    binary_cmap.set_bad("white", alpha=0)

    cursor_x = st.MARGIN_X_IN
    deck_y0 = st.MARGIN_BOTTOM_IN
    for e, step in enumerate(STEPS):
        if e > 0:
            cursor_x += st.GAP_IN
            ax_dot = fig.add_axes(rect_in(cursor_x, deck_y0, st.DOT_WIDTH_IN, deck_h_in))
            ax_dot.set_axis_off()
            ax_dot.text(0.5, 0.5, "•••", ha="center", va="center", fontsize=18,
                        transform=ax_dot.transAxes)
            cursor_x += st.DOT_WIDTH_IN + st.GAP_IN

        deck_x0 = cursor_x
        frame = frames[step]
        # depth 1 (back) = continuous r; depth 0 (front, drawn "over" it) = binary.
        for depth in (1, 0):
            x_in = deck_x0 + depth * st.STEP_X_IN
            y_in = deck_y0 + depth * st.STEP_Y_IN
            ax = fig.add_axes(rect_in(x_in, y_in, card_w_in, st.CARD_H_IN))
            alpha = 1.0 if depth == 0 else st.BACK_ALPHA
            if depth == 1:
                ax.imshow(hrf._masked(frame, land), extent=extent, origin="upper",
                          cmap=r_cmap, norm=norm, alpha=alpha)
            else:
                ax.imshow(hrf._masked((frame > 0).astype(float), land),
                          extent=extent, origin="upper", cmap=binary_cmap,
                          vmin=0, vmax=1, interpolation="nearest", alpha=alpha)
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
            (deck_x0 + deck_w_in / 2) / total_w_in, (st.MARGIN_BOTTOM_IN - 0.12) / total_h_in,
            f"{hrf.YEAR_LABELS[step]}  (GP mean r = {gp_means[step]:+.3f})",
            ha="center", va="top", fontsize=10,
        )
        cursor_x = deck_x0 + deck_w_in

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
