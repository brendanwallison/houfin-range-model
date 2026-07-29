#!/usr/bin/env python3
"""Simple, non-perspective early-vs-modern collage for the first few Z latent
components: a plain grid (one row per component, one column per era), no
card-deck offset styling -- a direct side-by-side comparison rather than the
"many rasters" deck effect in covariate_stacks.py.

Reuses covariate_samples.py's Z loader/grid-spec helpers, so this stays in
sync with any change to which components or eras are used elsewhere.

Output:
  docs/img/covariate_stacks/z_early_vs_modern_collage.png
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

import covariate_samples as cs  # noqa: E402  (path set up above)

OUT_DIR = os.path.join(REPO_ROOT, "docs", "img", "covariate_stacks")
OUT_PNG = os.path.join(OUT_DIR, "z_early_vs_modern_collage.png")

ERAS = (cs.PERIODS[0], cs.PERIODS[-1])  # early, modern -- same convention as
                                         # covariate_stacks_combined.py


def main():
    z_rows = [v for v in cs.VARIABLES if v[0] == "Z"]
    base_spec = cs._base_grid_spec()
    extent, box_bounds = base_spec["extent"], base_spec["box_bounds"]

    n_rows = len(z_rows)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(9, 2.6 * n_rows), squeeze=False,
    )

    for r, row in enumerate(z_rows):
        _dataset, _kind, _source, var, _resamp, label, _tf, cmap = row
        arrs = {era: np.squeeze(cs._to_array(cs.period_mean(row, *era, None))) for era in ERAS}

        pooled = np.concatenate([a[np.isfinite(a)].ravel() for a in arrs.values()])
        lo, hi = np.nanpercentile(pooled, [1, 99]) if pooled.size else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1e-9

        for c, era in enumerate(ERAS):
            ax = axes[r][c]
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("white", alpha=0)
            ax.imshow(arrs[era], extent=extent, origin="upper", cmap=cm, vmin=lo, vmax=hi)
            ax.set_xlim(box_bounds[0], box_bounds[2])
            ax.set_ylim(box_bounds[1], box_bounds[3])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("black")
                spine.set_linewidth(0.8)
            if r == 0:
                ax.set_title(f"{era[0]}-{era[1]}", fontsize=12)
        axes[r][0].set_ylabel(label, fontsize=10, rotation=0, ha="right", va="center")

    fig.suptitle("Z: early vs. modern, first few latent components", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
