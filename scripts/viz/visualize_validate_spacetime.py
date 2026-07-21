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


def _analog_arrows(z, out_dir, nbins=18, min_per_bin=3):
    """Binned mean-displacement field (one arrow per coarse bin) — legible where a raw
    per-point quiver is not. DESK (blue) vs BBS (red) mean analog direction per bin."""
    xy, dp, do = z["xy_hist"], z["d_pred"], z["d_obs"]
    if xy.shape[0] == 0:
        print("[viz] no analog data; skipping arrows")
        return
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    bx = np.clip(((xy[:, 0] - xmin) / (xmax - xmin + 1e-9) * nbins).astype(int), 0, nbins - 1)
    by = np.clip(((xy[:, 1] - ymin) / (ymax - ymin + 1e-9) * nbins).astype(int), 0, nbins - 1)
    cx, cy, up, vp, uo, vo = [], [], [], [], [], []
    for ix in range(nbins):
        for iy in range(nbins):
            m = (bx == ix) & (by == iy)
            if m.sum() < min_per_bin:
                continue
            cx.append(xmin + (ix + 0.5) / nbins * (xmax - xmin))
            cy.append(ymin + (iy + 0.5) / nbins * (ymax - ymin))
            up.append(dp[m, 0].mean()); vp.append(dp[m, 1].mean())
            uo.append(do[m, 0].mean()); vo.append(do[m, 1].mean())
    if not cx:
        print("[viz] no bins meet min_per_bin; skipping arrows")
        return
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.quiver(cx, cy, up, vp, color="tab:blue", angles="xy", scale_units="xy", scale=1,
              width=0.004, label="DESK (bin mean)")
    ax.quiver(cx, cy, uo, vo, color="tab:red", angles="xy", scale_units="xy", scale=1,
              width=0.004, label="BBS (bin mean)")
    ax.set_aspect("equal"); ax.legend(loc="upper right")
    ax.set_title("Analog displacement (binned mean): which present-day community a past\n"
                 "site resembles — DESK vs BBS. Albers x=E–W, y=N–S; up = north.")
    ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")
    fig.tight_layout()
    p = os.path.join(out_dir, "analog_arrows.png"); fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[viz] analog arrows ({len(cx)} bins) -> {p}")


def _support_maps(support_npz, out_dir, cmap):
    """BBS smoothed-effort (support) maps — early / mid / recent year + max-over-years.
    Shows where BBS actually backs the estimate (answers 'is the blank real?')."""
    if not os.path.exists(support_npz):
        print(f"[viz] no support field at {support_npz}; skipping support maps")
        return
    s = np.load(support_npz)
    sup, years = s["support"], s["years"]
    T = sup.shape[0]
    idx = sorted({0, T // 2, T - 1})
    vmax = float(np.nanpercentile(sup, 99)) or 1.0
    fig, ax = plt.subplots(1, len(idx) + 1, figsize=(5 * (len(idx) + 1), 5))
    for a, i in zip(ax, idx):
        im = a.imshow(sup[i], cmap=cmap, vmin=0, vmax=vmax)
        a.set_title(f"support {int(years[i])}"); a.axis("off"); fig.colorbar(im, ax=a, fraction=0.046)
    im = ax[-1].imshow(sup.max(0), cmap=cmap, vmin=0, vmax=vmax)
    ax[-1].set_title("support max over years"); ax[-1].axis("off"); fig.colorbar(im, ax=ax[-1], fraction=0.046)
    fig.tight_layout()
    p = os.path.join(out_dir, "support_maps.png"); fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[viz] support maps -> {p}")


def _reconstruction_maps(z, ref_raster, out_dir, cmap):
    """Z-space per-cell reconstruction error: DESK vs no-change, and their difference.
    Positive difference (no-change err - DESK err) => DESK reconstructs that cell better."""
    rows = z["recon_rows"] if "recon_rows" in z else np.array([])
    if rows.size == 0:
        print("[viz] no reconstruction data (re-run esk to save the projection); skipping")
        return
    cols, ed, en = z["recon_cols"], z["recon_err_desk"], z["recon_err_nochange"]
    H, W = _grid_shape(ref_raster)
    lin = rows.astype(int) * W + cols.astype(int)

    def _cell_mean(vals):
        s = np.bincount(lin, weights=vals, minlength=H * W)
        c = np.bincount(lin, minlength=H * W).astype(float)
        g = np.full(H * W, np.nan); m = c > 0; g[m] = s[m] / c[m]
        return g.reshape(H, W)

    gd, gn = _cell_mean(ed), _cell_mean(en)
    diff = gn - gd                                   # >0 => DESK error smaller => DESK better
    vmax = float(np.nanpercentile(np.r_[ed, en], 98)) or 1.0
    fin = np.isfinite(diff)
    vd = float(np.nanpercentile(np.abs(diff[fin]), 98)) if fin.any() else 1.0
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, g, t in zip(ax[:2], [gd, gn], ["DESK reconstruction error", "no-change error"]):
        im = a.imshow(g, cmap=cmap, vmin=0, vmax=vmax); a.set_title(t); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046)
    im = ax[2].imshow(diff, cmap="RdBu", vmin=-vd, vmax=vd)
    ax[2].set_title("no-change − DESK  (blue = DESK better)"); ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout()
    p = os.path.join(out_dir, "reconstruction_maps.png"); fig.savefig(p, dpi=110); plt.close(fig)
    win = float(np.mean(ed < en))
    print(f"[viz] reconstruction maps -> {p}  (DESK beats no-change at {win:.1%} of points)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=None, help="validate_spacetime.npz (default: desk_output_dir)")
    ap.add_argument("--out", default=None, help="output dir (default: desk_output_dir/validate_viz)")
    ap.add_argument("--nbins", type=int, default=18, help="analog arrow grid bins per axis")
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
    _reconstruction_maps(z, ref_raster, out_dir, args.cmap)
    _analog_arrows(z, out_dir, nbins=args.nbins)
    # support field lives with the amplitude point set (bbs.z_dir)
    support_npz = os.path.join(cfg["bbs"]["z_dir"], "support_field.npz")
    _support_maps(support_npz, out_dir, args.cmap)
    print(f"[viz] done -> {out_dir} (scp the PNGs)")


if __name__ == "__main__":
    main()
