"""Visualize the DESK-vs-BBS spatiotemporal comparison (from validate_spacetime.npz).

Two figures, both about community *similarity* structure (not suitability):
  1. Turnover maps — per-site temporal turnover magnitude, DESK-predicted vs
     BBS-observed, side by side + difference. "Do the models agree WHERE communities
     changed most?"
  2. Analog arrows — for a sample of historical sites, the geographic direction each
     site's community "points" toward among present-day cells (similarity-weighted
     analog centroid), DESK (blue) vs BBS (red), overlaid. "Do they agree WHICH WAY
     communities are moving?" (poleward warming, E-W precipitation).

    python scripts/viz/visualize_validate_spacetime.py [--max-arrows 250]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from src.config_utils import load_config, load_data_config


def _grid_shape(ref_raster):
    import rasterio
    with rasterio.open(ref_raster) as src:
        return src.height, src.width


def _turnover_maps(z, ref_raster, out_dir, cmap):
    rows, cols = z["turn_rows"], z["turn_cols"]
    if rows.size == 0:
        print("[viz] no turnover data; skipping turnover maps")
        return
    H, W = _grid_shape(ref_raster)
    def _grid(vals):
        g = np.full((H, W), np.nan, np.float32); g[rows, cols] = vals; return g
    gp, go = _grid(z["turnover_pred"]), _grid(z["turnover_obs"])
    fin = np.isfinite(np.r_[z["turnover_pred"], z["turnover_obs"]])
    vmax = float(np.nanpercentile(np.r_[z["turnover_pred"], z["turnover_obs"]][fin], 98)) if fin.any() else 1.0
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, g, t in zip(ax, [gp, go, gp - go],
                       ["DESK turnover (predicted)", "BBS turnover (observed)", "pred − obs"]):
        cm = cmap if t != "pred − obs" else "RdBu_r"
        vm = vmax if t != "pred − obs" else np.nanpercentile(np.abs(gp - go)[np.isfinite(gp - go)], 98)
        im = a.imshow(g, cmap=cm, vmin=(0 if t != "pred − obs" else -vm), vmax=vm)
        a.set_title(t); a.axis("off"); fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    p = os.path.join(out_dir, "turnover_maps.png"); fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[viz] turnover maps -> {p}")


def _analog_arrows(z, out_dir, max_arrows, seed=0):
    xy, dp, do = z["xy_hist"], z["d_pred"], z["d_obs"]
    if xy.shape[0] == 0:
        print("[viz] no analog data; skipping arrows")
        return
    rng = np.random.default_rng(seed)
    idx = rng.choice(xy.shape[0], min(max_arrows, xy.shape[0]), replace=False)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.quiver(xy[idx, 0], xy[idx, 1], dp[idx, 0], dp[idx, 1], color="tab:blue",
              angles="xy", scale_units="xy", scale=1, width=0.003, alpha=0.7, label="DESK")
    ax.quiver(xy[idx, 0], xy[idx, 1], do[idx, 0], do[idx, 1], color="tab:red",
              angles="xy", scale_units="xy", scale=1, width=0.003, alpha=0.7, label="BBS")
    ax.set_aspect("equal"); ax.legend(loc="upper right")
    ax.set_title("Analog displacement: which present-day community a past site resembles\n"
                 "(Albers x=E–W, y=N–S; up = north)")
    ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")
    fig.tight_layout()
    p = os.path.join(out_dir, "analog_arrows.png"); fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[viz] analog arrows -> {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=None, help="validate_spacetime.npz (default: desk_output_dir)")
    ap.add_argument("--out", default=None, help="output dir (default: desk_output_dir/validate_viz)")
    ap.add_argument("--max-arrows", type=int, default=250)
    ap.add_argument("--cmap", default="magma")
    args = ap.parse_args()

    cfg = load_config()
    desk_dir = cfg["paths"]["desk_output_dir"]
    npz = args.npz or os.path.join(desk_dir, "validate_spacetime.npz")
    z = np.load(npz, allow_pickle=True)
    ref_raster = str(z["ref_raster"]) if "ref_raster" in z else load_data_config()["grid"]["ref_raster"]
    out_dir = args.out or os.path.join(desk_dir, "validate_viz")
    os.makedirs(out_dir, exist_ok=True)

    _turnover_maps(z, ref_raster, out_dir, args.cmap)
    _analog_arrows(z, out_dir, args.max_arrows)
    print(f"[viz] done -> {out_dir} (scp the PNGs)")


if __name__ == "__main__":
    main()
