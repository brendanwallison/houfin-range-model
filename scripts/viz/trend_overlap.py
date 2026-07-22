"""Explore the overlap between the two trend products (BBS long-term vs eBird recent).

The trend-product community target blends a BBS 1966-2022 %/yr rate with an eBird
2012-2022 %/yr rate. This visual QC asks: where and how much do the two agree? They
measure different temporal domains, so low cell-level agreement is expected and is
exactly why blending them adds information -- but it must be looked at.

Outputs (PNG) under ``--out`` (default ``{processed}/encoder/analysis/trend_overlap``):
  1. maps.png      -- per-cell community-median %/yr: BBS, eBird, difference, sign-agreement
  2. scatter.png   -- cell x species %/yr, BBS vs eBird (2D hist) + corr / sign-agreement
  3. per_species.png -- per-species correlation + median-rate comparison (agreement ranking)

A later z-space overlap (BBS-only vs eBird-only vs blended reconstructions projected
into the ESK basis) belongs with validate_spacetime, once the ESK basis exists.

    python scripts/viz/trend_overlap.py
"""
import argparse
import os

import numpy as np


def _load(dcfg):
    bbs = np.load(dcfg["trends"]["bbs_trend_grid"], allow_pickle=True)
    eb = np.load(dcfg["trends"]["ebird_trend_grid"], allow_pickle=True)
    bcodes = [str(c) for c in bbs["species_code"]]
    ecodes = [str(c) for c in eb["species_code"]]
    common = [c for c in bcodes if c in set(ecodes)]
    bi = {c: i for i, c in enumerate(bcodes)}
    ei = {c: i for i, c in enumerate(ecodes)}
    B = np.stack([bbs["rate"][bi[c]] for c in common])          # (S, H, W) %/yr
    E = np.stack([eb["abd_ppy"][ei[c]] for c in common])
    return common, B, E


def maps_figure(B, E, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bmed = np.nanmedian(B, axis=0)                              # community-median %/yr per cell
    emed = np.nanmedian(E, axis=0)
    both = np.isfinite(bmed) & np.isfinite(emed)
    diff = np.where(both, bmed - emed, np.nan)
    agree = np.where(both, (np.sign(bmed) == np.sign(emed)).astype(float), np.nan)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for a, (arr, title, vlim, cmap) in zip(ax.ravel(), [
        (bmed, "BBS long-term median %/yr", 8, "RdBu_r"),
        (emed, "eBird recent median %/yr", 8, "RdBu_r"),
        (diff, "BBS - eBird (%/yr)", 8, "PuOr_r"),
        (agree, "sign agreement (1=same direction)", None, "Greys"),
    ]):
        kw = dict(cmap=cmap) if vlim is None else dict(cmap=cmap, vmin=-vlim, vmax=vlim)
        im = a.imshow(arr, **kw); a.set_title(title); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.035)
    fig.suptitle("Community-median trend, BBS long-term vs eBird recent")
    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


def scatter_figure(B, E, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    both = np.isfinite(B) & np.isfinite(E)
    b, e = B[both], E[both]
    fig, a = plt.subplots(figsize=(6.5, 6))
    lim = np.nanpercentile(np.abs(np.concatenate([b, e])), 99) if b.size else 15
    a.hist2d(b, e, bins=120, range=[[-lim, lim], [-lim, lim]], cmap="magma", cmin=1)
    a.plot([-lim, lim], [-lim, lim], "c--", lw=1)
    a.axhline(0, color="w", lw=0.4); a.axvline(0, color="w", lw=0.4)
    corr = float(np.corrcoef(b, e)[0, 1]) if b.size > 2 else float("nan")
    sign = float(np.mean(np.sign(b) == np.sign(e))) if b.size else float("nan")
    a.set_xlabel("BBS long-term %/yr"); a.set_ylabel("eBird recent %/yr")
    a.set_title(f"cell x species trend: corr={corr:.2f}, sign-agree={sign:.2f}  (n={b.size})")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def per_species_figure(common, B, E, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for i, c in enumerate(common):
        m = np.isfinite(B[i]) & np.isfinite(E[i])
        if m.sum() < 20:
            continue
        corr = float(np.corrcoef(B[i][m], E[i][m])[0, 1])
        rows.append((c, corr, float(np.nanmedian(B[i])), float(np.nanmedian(E[i]))))
    if not rows:
        return
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, max(4, 0.28 * len(rows))))
    y = np.arange(len(rows))
    ax[0].barh(y, [r[1] for r in rows], color="steelblue")
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=7)
    ax[0].set_xlabel("corr(BBS, eBird) %/yr"); ax[0].set_title("per-species cell-level agreement")
    ax[0].axvline(0, color="k", lw=0.5)
    ax[1].scatter([r[2] for r in rows], [r[3] for r in rows], s=14)
    lim = max(1.0, max(abs(v) for r in rows for v in r[2:]))
    ax[1].plot([-lim, lim], [-lim, lim], "r--", lw=1); ax[1].axhline(0, lw=0.4); ax[1].axvline(0, lw=0.4)
    ax[1].set_xlabel("BBS median %/yr"); ax[1].set_ylabel("eBird median %/yr")
    ax[1].set_title("per-species median rate")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def main():
    from src.config_utils import load_data_config

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dcfg = load_data_config()
    out = args.out or os.path.join(dcfg["processed_root"], "encoder", "analysis", "trend_overlap")
    os.makedirs(out, exist_ok=True)

    common, B, E = _load(dcfg)
    both = np.isfinite(B) & np.isfinite(E)
    corr = float(np.corrcoef(B[both], E[both])[0, 1]) if both.sum() > 2 else float("nan")
    print(f"[trend-overlap] {len(common)} species in both grids; {int(both.sum())} cell x species "
          f"overlap; pooled corr={corr:.3f}")
    maps_figure(B, E, os.path.join(out, "maps.png"))
    scatter_figure(B, E, os.path.join(out, "scatter.png"))
    per_species_figure(common, B, E, os.path.join(out, "per_species.png"))
    print(f"[trend-overlap] wrote maps.png, scatter.png, per_species.png -> {out}")


if __name__ == "__main__":
    main()
