"""Raw-BBS community turnover vs smoothed -- separates genuine change from smoothing.

Question this answers: our pipeline reports the community barely changed since the
1960s. Is that because communities are genuinely inertial, or because our spatial
smoothing (smooth_sigma_s = 5 cells = 125 km stdev) blurs the change away?

Method: take the cells BBS actually surveyed in BOTH an early decade and the recent
decade (raw route coverage in each -- no borrowing). For that FIXED cell set, compute
each cell's community turnover = 1 - Ruzicka(early per-species abundance vector,
recent per-species abundance vector), at several smoothing levels:

  (sigma_t, sigma_s) = (0,0)  -> RAW: what BBS says actually changed at that cell,
                                 no spatial borrowing. The genuine-change ceiling.
                       (1,5)  -> the current pipeline smoothing.
  ...and a few in between, so you see the flattening as sigma_s grows.

Same cells at every level, so the ONLY thing changing is the blur. If raw turnover is
large but the (1,5) turnover is small, the smoothing is hiding real change. If raw
turnover is itself small, the near-stability is genuine inertia and no smoothing
setting will reveal change that isn't there.

This is BBS-only (no eBird fixed-shape, no anomaly cap/shrink), so it isolates the
SMOOTHING. The remaining suppressor -- the fixed-2023 eBird shape + anomaly cap -- is a
separate question; compare this raw ceiling to the amplitude turnover to gauge that.

    python scripts/viz/raw_bbs_turnover.py [--min-routes 3] [--early-decade 1966]
"""
import argparse
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from src.config_utils import load_config
from src.community_encoder.train_DESK.spacetime_community import _scatter_dense


def _ruzicka_rows(a, b):
    """Row-wise Ruzicka similarity between (N, S) matrices a and b."""
    mn = np.minimum(a, b).sum(1)
    mx = np.maximum(a, b).sum(1)
    return np.where(mx > 0, mn / mx, 1.0)


def _grid_shape(config):
    from src.config_utils import load_data_config
    import rasterio
    with rasterio.open(load_data_config()["grid"]["ref_raster"]) as src:
        return src.height, src.width


def turnover_by_level(cm, T, H, W, yr_ix, early_idx, recent_idx, cellset, levels):
    """Per-cell community turnover on the fixed cell set, at each (sigma_t, sigma_s) level.

    Aggregates each species' smoothed numerator/denominator over the early and recent
    year-windows (effort-weighted window means), assembles per-cell community vectors,
    and returns {level: turnover_array_over_cellset}.
    """
    cov_t = np.array([yr_ix[int(y)] for y in cm["cov_year"]])
    weight = _scatter_dense(cm["cov_n"].astype(float), cm["cov_row"], cm["cov_col"], cov_t, T, H, W)
    codes = [str(c) for c in cm["species_codes"]]
    S = len(codes)
    rr, cc = cellset[:, 0], cellset[:, 1]
    n = len(cellset)

    # Smoothed effort per level (denominator), summed over each window.
    sm_w = {}
    for lv in levels:
        w = gaussian_filter(weight, (lv[0], lv[1], lv[1]), mode="constant")
        sm_w[lv] = (w[early_idx].sum(0)[rr, cc], w[recent_idx].sum(0)[rr, cc])

    early = {lv: np.zeros((n, S)) for lv in levels}
    recent = {lv: np.zeros((n, S)) for lv in levels}
    mean_t = np.array([yr_ix[int(y)] for y in cm["year"]])
    for sp in range(S):
        sel = cm["species_index"] == sp
        if not sel.any():
            continue
        obs = _scatter_dense(cm["mean_count"][sel].astype(float), cm["row"][sel],
                             cm["col"][sel], mean_t[sel], T, H, W)   # mean abundance placed on grid
        for lv in levels:
            o = gaussian_filter(obs, (lv[0], lv[1], lv[1]), mode="constant")
            we, wr = sm_w[lv]
            ne = o[early_idx].sum(0)[rr, cc]
            nr = o[recent_idx].sum(0)[rr, cc]
            early[lv][:, sp] = np.divide(ne, we, out=np.zeros(n), where=we > 0)
            recent[lv][:, sp] = np.divide(nr, wr, out=np.zeros(n), where=wr > 0)
    return {lv: 1.0 - _ruzicka_rows(early[lv], recent[lv]) for lv in levels}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-routes", type=float, default=3.0,
                    help="min summed route-coverage in EACH window for a cell to count")
    ap.add_argument("--early-decade", type=int, default=None,
                    help="first year of the early 10-yr window (default: earliest in data)")
    ap.add_argument("--out", default=None, help="output PNG (default: alongside community_matrix)")
    args = ap.parse_args()

    cfg = load_config()
    bc = cfg["bbs"]
    cm = np.load(bc["community_matrix"], allow_pickle=True)
    H, W = _grid_shape(cfg)
    years = np.arange(int(cm["cov_year"].min()), int(cm["cov_year"].max()) + 1)
    yr_ix = {int(y): i for i, y in enumerate(years)}
    T = len(years)

    e0 = args.early_decade if args.early_decade is not None else int(years.min())
    early_yrs = [y for y in years if e0 <= y < e0 + 10]
    rec = bc["anomaly_ref_years"]
    recent_yrs = [y for y in years if rec[0] <= y <= rec[1]]
    early_idx = np.array([yr_ix[int(y)] for y in early_yrs])
    recent_idx = np.array([yr_ix[int(y)] for y in recent_yrs])
    print(f"early window {early_yrs[0]}-{early_yrs[-1]} vs recent {recent_yrs[0]}-{recent_yrs[-1]}")

    # Fixed cell set: raw route coverage >= min_routes in BOTH windows (no borrowing).
    cov_t = np.array([yr_ix[int(y)] for y in cm["cov_year"]])
    weight = _scatter_dense(cm["cov_n"].astype(float), cm["cov_row"], cm["cov_col"], cov_t, T, H, W)
    eff_e = weight[early_idx].sum(0); eff_r = weight[recent_idx].sum(0)
    cellset = np.argwhere((eff_e >= args.min_routes) & (eff_r >= args.min_routes))
    print(f"cells with raw coverage in both windows (>= {args.min_routes} route-yr each): {len(cellset)}")
    if len(cellset) < 10:
        print("too few raw-supported cells; lower --min-routes or widen windows"); return

    sig_t = float(bc["smooth_sigma_t"]); sig_s = float(bc["smooth_sigma_s"])
    levels = [(0.0, 0.0), (sig_t, 1.0), (sig_t, 2.0), (sig_t, sig_s), (0.0, sig_s)]
    turns = turnover_by_level(cm, T, H, W, yr_ix, early_idx, recent_idx, cellset, levels)

    raw = turns[(0.0, 0.0)]
    # The cells that genuinely changed (top raw decile). Smoothing REDISTRIBUTES turnover
    # (lowers changed cells, raises unchanged neighbors), so a whole-cell mean is not
    # monotonic -- track the changed cells to see how much real change survives each blur.
    changed = raw >= np.percentile(raw, 90)
    raw_changed = float(np.mean(raw[changed]))
    print(f"\nRAW ceiling: median turnover {np.median(raw):.3f}, 90th pct {np.percentile(raw,90):.3f}, "
          f"top-decile mean {raw_changed:.3f}  ({int(changed.sum())} genuinely-changed cells)")
    print(f"\n{'(sigma_t, sigma_s)':<20}{'median':>10}{'90th pct':>10}"
          f"{'top-decile mean':>18}{'% of raw change kept':>22}")
    for lv in levels:
        t = turns[lv]
        tag = "  <- RAW" if lv == (0.0, 0.0) else ("  <- PIPELINE" if lv == (sig_t, sig_s) else "")
        print(f"{str(lv):<20}{np.median(t):>10.3f}{np.percentile(t,90):>10.3f}"
              f"{np.mean(t[changed]):>18.3f}{100*np.mean(t[changed])/raw_changed:>21.0f}%{tag}")

    print(f"\nRead: 'top-decile mean' tracks the SAME genuinely-changed cells across blurs. "
          f"If the\nPIPELINE row keeps only a small % of the raw change, spatial smoothing is "
          f"hiding it.\nIf the RAW ceiling is itself small, the stability is genuine (and "
          f"enrich has little\nto gain). Note the whole-cell mean is not shown -- smoothing "
          f"smears change into\nneighbors, so only the changed-cell view is honest.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot([turns[lv] for lv in levels], labels=[str(lv) for lv in levels], showfliers=False)
        ax.set_ylabel("per-cell community turnover (1 - Ruzicka)")
        ax.set_xlabel("(sigma_t, sigma_s)  -- (0,0)=raw, rightmost two = pipeline sigma_s")
        ax.set_title(f"BBS community turnover vs smoothing ({len(cellset)} raw-supported cells)\n"
                     f"{early_yrs[0]}-{early_yrs[-1]} vs {recent_yrs[0]}-{recent_yrs[-1]}")
        fig.tight_layout()
        out = args.out or os.path.join(os.path.dirname(bc["community_matrix"]), "raw_bbs_turnover.png")
        fig.savefig(out, dpi=110); print(f"\n[plot] -> {out}")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
