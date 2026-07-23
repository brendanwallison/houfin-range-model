#!/usr/bin/env python
"""Post-fit diagnostics for fused trend communities -> ESK -> DESK.

The three representations are compared on the same selected ``(cell, year)``
points. ESK coordinates are pinned by the saved Nyström basis, so coordinate-wise
ESK-vs-DESK comparisons are meaningful (no Procrustes rotation is needed).

Outputs
-------
``component_fidelity.png``
    Per-component global, within-year spatial, and within-cell temporal fidelity.
``structure_retention.png``
    Spatial-neighbour and temporal-change RMS retained by DESK relative to ESK.
``kernel_by_dimension.png``
    Fused Ružička similarity reconstruction as progressively more Z dimensions
    are retained; reveals a natural truncation point or high-dimension failure.
``turnover_maps.png``
    Deep-to-recent turnover in fused community, ESK, DESK, and DESK residual,
    all measured as one minus the same Ružička-kernel quantity.
``component_atlas_<year>.png``
    ESK, DESK, and residual maps for representative low/high components.
``presentation_*.png``
    Three standalone, plain-language figures: kernel calibration, geographic
    community-similarity maps, and temporal-turnover agreement.
``metrics.json`` / ``component_metrics.csv``
    Machine-readable values behind every plot.

Run after ``spacetime-esk -> desk -> cube``. Projection of the selected fused
points is cached under the output directory.

    python scripts/viz/encoder_diagnostics.py
    python scripts/viz/encoder_diagnostics.py --years 1966,1980,1995,2012,2020,2025
"""
import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from src.config_utils import load_config, load_data_config
from src.community_encoder.train_DESK.esk_kernel import project_into_z, smooth_abundances


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _group_demean(A, groups):
    """Demean rows of ``A`` within integer/string group labels."""
    out = np.full_like(A, np.nan, dtype="float64")
    order = np.argsort(groups, kind="stable")
    sorted_groups = np.asarray(groups)[order]
    starts = np.r_[0, np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        if end - start >= 2:
            ix = order[start:end]
            out[ix] = A[ix] - np.nanmean(A[ix], axis=0, keepdims=True)
    return out


def component_metrics(Z_esk, Z_desk, rows, cols, years):
    """Coordinate fidelity, separating spatial and temporal deviations."""
    cells = rows.astype(np.int64) * (int(cols.max()) + 1) + cols
    Es = _group_demean(Z_esk, years)       # within-year spatial structure
    Ds = _group_demean(Z_desk, years)
    Et = _group_demean(Z_esk, cells)       # within-cell temporal structure
    Dt = _group_demean(Z_desk, cells)
    records = []
    for d in range(Z_esk.shape[1]):
        scale = float(np.nanstd(Z_esk[:, d]))
        records.append({
            "dimension": d + 1,
            "global_corr": _corr(Z_esk[:, d], Z_desk[:, d]),
            "spatial_corr": _corr(Es[:, d], Ds[:, d]),
            "temporal_corr": _corr(Et[:, d], Dt[:, d]),
            "nrmse": float(np.sqrt(np.nanmean((Z_desk[:, d] - Z_esk[:, d]) ** 2)) /
                           max(scale, 1e-12)),
            "variance_ratio": float(np.nanvar(Z_desk[:, d]) / max(np.nanvar(Z_esk[:, d]), 1e-12)),
        })
    return pd.DataFrame(records)


def structure_retention(Z_esk, Z_desk, rows, cols, years):
    """RMS spatial-neighbour and annualized temporal change, per component."""
    lookup = {(int(y), int(r), int(c)): i for i, (r, c, y) in
              enumerate(zip(rows, cols, years))}
    spatial_pairs = []
    for key, i in lookup.items():
        y, r, c = key
        for q in ((y, r + 1, c), (y, r, c + 1)):
            if q in lookup:
                spatial_pairs.append((i, lookup[q], 1.0))

    by_cell = {}
    for i, (r, c, y) in enumerate(zip(rows, cols, years)):
        by_cell.setdefault((int(r), int(c)), []).append((int(y), i))
    temporal_pairs = []
    for vals in by_cell.values():
        vals.sort()
        for (y0, i), (y1, j) in zip(vals[:-1], vals[1:]):
            temporal_pairs.append((i, j, float(y1 - y0)))

    def rms(A, pairs):
        if not pairs:
            return np.full(A.shape[1], np.nan)
        i = np.array([p[0] for p in pairs]); j = np.array([p[1] for p in pairs])
        gap = np.array([p[2] for p in pairs])[:, None]
        return np.sqrt(np.nanmean(((A[j] - A[i]) / gap) ** 2, axis=0))

    se, sd = rms(Z_esk, spatial_pairs), rms(Z_desk, spatial_pairs)
    te, td = rms(Z_esk, temporal_pairs), rms(Z_desk, temporal_pairs)
    return {
        "spatial_esk": se, "spatial_desk": sd,
        "temporal_esk": te, "temporal_desk": td,
        "spatial_ratio": sd / np.maximum(se, 1e-12),
        "temporal_ratio": td / np.maximum(te, 1e-12),
        "n_spatial_pairs": len(spatial_pairs), "n_temporal_pairs": len(temporal_pairs),
    }


def _ruzicka_pairs(X, i, j):
    mn = np.minimum(X[i], X[j]).sum(1); mx = np.maximum(X[i], X[j]).sum(1)
    return np.divide(mn, mx, out=np.zeros_like(mn), where=mx > 0)


def _feature_kernel_pairs(Z, i, j):
    """Approximate Ružička similarities represented by uncentered ESK features."""
    return np.sum(Z[i] * Z[j], axis=1)


def kernel_dimension_curve(X, Z_esk, Z_desk, dims, seed=0, n_pairs=50000):
    """Pairwise fused-kernel reconstruction as latent dimensions accumulate."""
    rng = np.random.default_rng(seed); N = len(X)
    i = rng.integers(0, N, n_pairs); j = rng.integers(0, N, n_pairs)
    keep = i != j; i, j = i[keep], j[keep]
    obs = _ruzicka_pairs(X, i, j); scale = max(float(np.sqrt(np.mean(obs ** 2))), 1e-12)
    out = []
    for k in dims:
        pe = _feature_kernel_pairs(Z_esk[:, :k], i, j)
        pdesk = _feature_kernel_pairs(Z_desk[:, :k], i, j)
        out.append({
            "dimension": int(k),
            "esk_rmse_norm": float(np.sqrt(np.mean((pe - obs) ** 2)) / scale),
            "desk_rmse_norm": float(np.sqrt(np.mean((pdesk - obs) ** 2)) / scale),
            "esk_pair_corr": _corr(pe, obs), "desk_pair_corr": _corr(pdesk, obs),
            "desk_vs_esk_corr": _corr(pdesk, pe),
        })
    return pd.DataFrame(out)


def paired_turnover(X, Ze, Zd, rows, cols, years, deep, recent):
    """Matched deep-to-recent turnover under the common Ružička kernel.

    ESK/DESK are trained so ``Z @ Z.T`` approximates the *uncentered* Ružička
    similarity.  Cosine-normalizing Z would instead change the kernel and make
    its turnover incomparable with ``1 - Ruzicka(X0, X1)``.
    """
    ix = {(int(y), int(r), int(c)): i for i, (r, c, y) in enumerate(zip(rows, cols, years))}
    cells = sorted({(int(r), int(c)) for r, c, y in zip(rows, cols, years)
                    if int(y) == deep and (recent, int(r), int(c)) in ix})
    a = np.array([ix[(deep, r, c)] for r, c in cells]); b = np.array([ix[(recent, r, c)] for r, c in cells])
    fused = 1.0 - _ruzicka_pairs(X, a, b)
    esk = 1.0 - _feature_kernel_pairs(Ze, a, b)
    desk = 1.0 - _feature_kernel_pairs(Zd, a, b)
    return np.array([r for r, _ in cells]), np.array([c for _, c in cells]), fused, esk, desk


def load_comparison(cfg, selected_years, out_dir, recompute=False):
    cache = out_dir / "comparison_points.npz"
    if cache.exists() and not recompute:
        z = np.load(cache)
        cached_years = sorted(set(z["years"].astype(int).tolist()))
        if cached_years == sorted(selected_years) and "is_eval" in z.files:
            print(f"[encoder-viz] using cached ESK projections -> {cache}")
            return {k: z[k] for k in z.files}

    point_dir = Path(cfg["bbs"]["z_dir"])
    X_all = np.load(point_dir / "X_points.npy", mmap_mode="r")
    pidx_all = np.load(point_dir / "point_index.npy")
    take = np.isin(pidx_all[:, 2], selected_years)
    pidx = pidx_all[take].astype(int); X = np.asarray(X_all[take], dtype="float32")

    zdir = Path(cfg["desk"]["z_dir"])
    with open(zdir / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("kernel") != "ruzicka" or bool(meta.get("centered", True)):
        raise ValueError(f"expected uncentered Ružička ESK basis, got {meta}")
    landmarks = np.load(zdir / "esk_landmarks.npy")
    proj = np.load(zdir / "esk_projmat.npy")
    print(f"[encoder-viz] projecting {len(X):,} fused points into pinned ESK basis...")
    sigma = float(meta.get("sigma", 0.0))
    n_weeks = int(meta.get("n_weeks", 1))
    X_basis = smooth_abundances(X, n_weeks, sigma) if sigma > 0 else X
    Z_esk = project_into_z(X_basis, landmarks, proj, batch_size=2000)

    cube = Path(cfg["latent_cube"]["output_dir"])
    Z_desk = np.empty_like(Z_esk)
    r_max, c_max = int(pidx[:, 0].max()), int(pidx[:, 1].max())
    for year in sorted(set(pidx[:, 2])):
        g = np.load(cube / f"Z_latent_{int(year)}.npy", mmap_mode="r")
        # The cube and the ESK points must be on the SAME grid + latent width. A stale cube
        # (e.g. a pre-27km/25km build, or a narrower latent_dim) would IndexError cryptically
        # or silently misalign, so fail loudly with what to rebuild.
        if g.shape[0] <= r_max or g.shape[1] <= c_max:
            raise ValueError(
                f"DESK cube {cube.name}/Z_latent_{int(year)}.npy grid {tuple(g.shape[:2])} is "
                f"smaller than the ESK point grid (needs > {r_max}x{c_max}). The cube is stale / "
                f"on a different grid than the ESK basis — rebuild desk->cube at the current grid.")
        if g.shape[2] < Z_esk.shape[1]:
            raise ValueError(
                f"DESK cube latent_dim {g.shape[2]} < ESK basis {Z_esk.shape[1]}; rebuild "
                f"spacetime-esk -> desk -> cube at a matching latent_dim.")
        sel = pidx[:, 2] == year
        Z_desk[sel] = g[pidx[sel, 0], pidx[sel, 1], :Z_esk.shape[1]]
    holdout_path = Path(cfg["paths"]["desk_output_dir"]) / "holdout_cells.npy"
    if holdout_path.exists():
        holdout = np.load(holdout_path)
        is_eval = holdout[pidx[:, 0], pidx[:, 1]].astype(bool)
        print(f"[encoder-viz] honest spatial holdout: {is_eval.sum():,}/{len(is_eval):,} points")
    else:
        is_eval = np.ones(len(pidx), dtype=bool)
        print("[encoder-viz] no holdout_cells.npy; metrics use all matched points")
    ok = np.isfinite(Z_esk).all(1) & np.isfinite(Z_desk).all(1) & np.isfinite(X).all(1)
    payload = dict(X=X[ok], rows=pidx[ok, 0], cols=pidx[ok, 1], years=pidx[ok, 2],
                   Z_esk=Z_esk[ok], Z_desk=Z_desk[ok], is_eval=is_eval[ok])
    np.savez_compressed(cache, **payload)
    print(f"[encoder-viz] cached {ok.sum():,} matched points -> {cache}")
    return payload


def _grid(H, W, rows, cols, vals):
    g = np.full((H, W), np.nan, dtype="float32"); g[rows, cols] = vals; return g


def plot_component_fidelity(df, out):
    d = df["dimension"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(d, df["global_corr"], label="global", lw=2)
    ax[0].plot(d, df["spatial_corr"], label="within-year spatial", lw=2)
    ax[0].plot(d, df["temporal_corr"], label="within-cell temporal", lw=2)
    ax[0].axhline(0, color="0.5", lw=.7); ax[0].set_ylim(-.05, 1.02)
    ax[0].set(xlabel="ESK component", ylabel="ESK–DESK correlation",
              title="Pinned-coordinate fidelity by component"); ax[0].legend()
    ax[1].plot(d, df["nrmse"], label="normalized RMSE", lw=2)
    ax[1].plot(d, df["variance_ratio"], label="DESK / ESK variance", lw=2)
    ax[1].axhline(1, color="0.5", lw=.7); ax[1].set(xlabel="ESK component",
              title="Amplitude loss or inflation at high dimensions"); ax[1].legend()
    for a in ax: a.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def plot_structure(ret, out):
    d = np.arange(1, len(ret["spatial_ratio"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(d, ret["spatial_ratio"], lw=2); ax[0].axhline(1, color="0.5", lw=.7)
    ax[0].set(xlabel="ESK component", ylabel="DESK / ESK neighbour-difference RMS",
              title=f"Spatial detail retained ({ret['n_spatial_pairs']:,} neighbour pairs)")
    ax[1].plot(d, ret["temporal_ratio"], lw=2); ax[1].axhline(1, color="0.5", lw=.7)
    ax[1].set(xlabel="ESK component", ylabel="DESK / ESK annualized-change RMS",
              title=f"Temporal variation retained ({ret['n_temporal_pairs']:,} cell intervals)")
    for a in ax: a.grid(alpha=.2); a.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def plot_kernel_curve(df, out):
    best_dim = int(df.loc[df.desk_rmse_norm.idxmin(), "dimension"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(df.dimension, df.esk_rmse_norm, "o-", label="ESK")
    ax[0].plot(df.dimension, df.desk_rmse_norm, "o-", label="DESK")
    ax[0].set(xlabel="retained dimensions", ylabel="normalized Ružička RMSE",
              title="Kernel error versus truncation"); ax[0].legend()
    ax[1].plot(df.dimension, df.esk_pair_corr, "o-", label="fused vs ESK")
    ax[1].plot(df.dimension, df.desk_pair_corr, "o-", label="fused vs DESK")
    ax[1].plot(df.dimension, df.desk_vs_esk_corr, "o-", label="ESK vs DESK")
    ax[1].set(xlabel="retained dimensions", ylabel="pair-similarity correlation",
              title="Does added dimensionality help DESK?"); ax[1].legend()
    for a in ax:
        a.axvline(best_dim, color="0.45", lw=.8, ls="--")
        a.grid(alpha=.2)
    ax[0].annotate(f"best DESK RMSE: {best_dim}D", (best_dim, df.desk_rmse_norm.min()),
                   xytext=(5, 8), textcoords="offset points")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def plot_turnover(data, H, W, deep, recent, out):
    r, c, fused, esk, desk = paired_turnover(data["X"], data["Z_esk"], data["Z_desk"],
                                              data["rows"], data["cols"], data["years"], deep, recent)
    maps = [_grid(H, W, r, c, v) for v in (fused, esk, desk, desk - fused)]
    vmax = float(np.nanpercentile(np.concatenate([fused, esk, desk]), 98))
    vd = float(np.nanpercentile(np.abs(desk - fused), 98))
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.8))
    titles = ["fused trend community", "ESK target", "DESK prediction", "DESK − fused"]
    for i, (a, g, title) in enumerate(zip(ax, maps, titles)):
        if i < 3: im = a.imshow(g, cmap="inferno", vmin=0, vmax=vmax)
        else: im = a.imshow(g, cmap="RdBu_r", vmin=-vd, vmax=vd)
        a.set_title(title); a.axis("off"); fig.colorbar(im, ax=a, fraction=.04)
    fig.suptitle(f"Temporal turnover {deep} → {recent}: fused vs ESK vs DESK ({len(r):,} matched cells)")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def _normalized_rmse(pred, obs):
    return float(np.sqrt(np.mean((pred - obs) ** 2)) /
                 max(float(np.sqrt(np.mean(obs ** 2))), 1e-12))


def plot_similarity_calibration(X, Z_esk, Z_desk, out, seed=0, n_pairs=30000):
    """Presentation figure: are encoded similarities calibrated to communities?"""
    rng = np.random.default_rng(seed)
    n = len(X)
    i, j = rng.integers(0, n, (2, min(n_pairs, max(n * 2, 1))))
    keep = i != j; i, j = i[keep], j[keep]
    observed = _ruzicka_pairs(X, i, j)
    predicted = [_feature_kernel_pairs(Z_esk, i, j), _feature_kernel_pairs(Z_desk, i, j)]
    labels = ["ESK target", "DESK prediction from covariates"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    lo = min(0.0, *(float(np.nanpercentile(x, .5)) for x in predicted))
    hi = max(1.0, *(float(np.nanpercentile(x, 99.5)) for x in predicted))
    for a, pred, label in zip(ax, predicted, labels):
        a.hexbin(observed, pred, gridsize=42, mincnt=1, cmap="viridis", linewidths=0)
        a.plot([lo, hi], [lo, hi], color="white", lw=1.3, alpha=.9)
        a.set(title=label, xlabel="Fused-community Ružička similarity", xlim=(lo, hi), ylim=(lo, hi))
        a.text(.03, .96, f"r = {_corr(observed, pred):.3f}\nnormalized RMSE = {_normalized_rmse(pred, observed):.3f}",
               transform=a.transAxes, va="top", color="white",
               bbox={"facecolor": "black", "alpha": .55, "pad": 3, "edgecolor": "none"})
    ax[0].set_ylabel("Similarity represented by Z · Z′")
    fig.suptitle(f"Community similarity learned by ESK and predicted by DESK ({len(observed):,} pairs)")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def _spatially_spread_anchors(rows, cols, n=3):
    """Choose reproducible occupied cells spread across the available grid."""
    unique = np.unique(np.column_stack([rows, cols]), axis=0)
    rscale = max(float(np.ptp(unique[:, 0])), 1.0); cscale = max(float(np.ptp(unique[:, 1])), 1.0)
    targets = np.array([[.20, .22], [.52, .55], [.80, .76]])[:n]
    scaled = np.column_stack([(unique[:, 0] - unique[:, 0].min()) / rscale,
                              (unique[:, 1] - unique[:, 1].min()) / cscale])
    chosen = []
    for target in targets:
        order = np.argsort(np.sum((scaled - target) ** 2, axis=1))
        chosen.append(next((tuple(unique[k]) for k in order if tuple(unique[k]) not in chosen),
                           tuple(unique[order[0]])))
    return chosen


def plot_similarity_atlas(data, H, W, year, out):
    """Presentation maps of three reference communities and their analogues."""
    sel = data["years"] == year
    X, E, D = data["X"][sel], data["Z_esk"][sel], data["Z_desk"][sel]
    rows, cols = data["rows"][sel], data["cols"][sel]
    anchors = _spatially_spread_anchors(rows, cols)
    fig, ax = plt.subplots(3, len(anchors), figsize=(4.1 * len(anchors), 10), squeeze=False)
    for col_i, (ar, ac) in enumerate(anchors):
        anchor = np.flatnonzero((rows == ar) & (cols == ac))[0]
        ii = np.full(len(rows), anchor, dtype=int); jj = np.arange(len(rows))
        fields = [_ruzicka_pairs(X, ii, jj), _feature_kernel_pairs(E, ii, jj),
                  _feature_kernel_pairs(D, ii, jj)]
        for row_i, (field, label) in enumerate(zip(fields, ["Fused community", "ESK target", "DESK prediction"])):
            image = ax[row_i, col_i].imshow(_grid(H, W, rows, cols, field), cmap="viridis", vmin=0, vmax=1)
            ax[row_i, col_i].plot(ac, ar, marker="*", ms=9, mfc="white", mec="black", mew=.7)
            ax[row_i, col_i].axis("off")
            if col_i == 0:
                ax[row_i, col_i].set_ylabel(label)
            if row_i == 0:
                ax[row_i, col_i].set_title(f"Reference community {col_i + 1}")
            fig.colorbar(image, ax=ax[row_i, col_i], fraction=.04, pad=.02)
    fig.suptitle(f"Geographic analogues of three reference communities — {year}\nstar = reference cell; colours = Ružička similarity")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def plot_turnover_agreement(data, deep, recent, out):
    """Presentation figure: true versus represented deep-to-recent turnover."""
    _, _, observed, esk, desk = paired_turnover(data["X"], data["Z_esk"], data["Z_desk"],
                                                 data["rows"], data["cols"], data["years"], deep, recent)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True)
    lo = min(0.0, float(np.nanpercentile(np.r_[esk, desk], .5)))
    hi = max(1.0, float(np.nanpercentile(np.r_[observed, esk, desk], 99.5)))
    for a, pred, label in zip(ax, [esk, desk], ["ESK target", "DESK prediction"]):
        a.hexbin(observed, pred, gridsize=42, mincnt=1, cmap="magma", linewidths=0)
        a.plot([lo, hi], [lo, hi], color="white", lw=1.3, alpha=.9)
        a.set(title=label, xlabel="Fused-community turnover", xlim=(lo, hi), ylim=(lo, hi))
        a.text(.03, .96, f"r = {_corr(observed, pred):.3f}\nnormalized RMSE = {_normalized_rmse(pred, observed):.3f}",
               transform=a.transAxes, va="top", color="white",
               bbox={"facecolor": "black", "alpha": .55, "pad": 3, "edgecolor": "none"})
    ax[0].set_ylabel("Turnover represented by 1 − Z · Z′")
    fig.suptitle(f"Temporal community turnover: {deep} → {recent} ({len(observed):,} matched cells)")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def plot_component_atlas(data, H, W, year, dims, out):
    sel = data["years"] == year; r, c = data["rows"][sel], data["cols"][sel]
    E, D = data["Z_esk"][sel], data["Z_desk"][sel]
    dims = [d for d in dims if d <= E.shape[1]]
    fig, ax = plt.subplots(len(dims), 3, figsize=(11, 2.7 * len(dims)), squeeze=False)
    for row, dim in enumerate(dims):
        e, d = E[:, dim - 1], D[:, dim - 1]; q = np.nanpercentile(np.abs(np.r_[e, d]), 98) or 1
        residual = d - e; qr = np.nanpercentile(np.abs(residual), 98) or 1
        for j, (v, title, lim) in enumerate(((e, "ESK", q), (d, "DESK", q), (residual, "DESK − ESK", qr))):
            im = ax[row, j].imshow(_grid(H, W, r, c, v), cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[row, j].axis("off"); ax[row, j].set_title(f"Z{dim} {title}")
            fig.colorbar(im, ax=ax[row, j], fraction=.035)
    fig.suptitle(f"Pinned latent component maps — {year}")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="1966,1980,1995,2012,2020,2025")
    ap.add_argument("--out", default=None)
    ap.add_argument("--recompute-projection", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg, dcfg = load_config(), load_data_config()
    years = sorted({int(x) for x in args.years.split(",") if x.strip()})
    out = Path(args.out or os.path.join(cfg["paths"]["desk_output_dir"], "encoder_diagnostics"))
    out.mkdir(parents=True, exist_ok=True)
    data = load_comparison(cfg, years, out, args.recompute_projection)
    available = sorted(set(data["years"].astype(int).tolist()))
    if len(available) < 2:
        raise ValueError(f"need at least two matched years, found {available}")

    with rasterio.open(dcfg["grid"]["ref_raster"]) as src:
        H, W = src.height, src.width
    # Report fidelity on the spatial holdout when the trainer saved one. Maps use
    # every matched cell so spatial patterns remain geographically legible.
    ev = data["is_eval"].astype(bool)
    eval_data = {k: v[ev] for k, v in data.items() if k != "is_eval"}
    comp = component_metrics(eval_data["Z_esk"], eval_data["Z_desk"], eval_data["rows"],
                             eval_data["cols"], eval_data["years"])
    ret = structure_retention(eval_data["Z_esk"], eval_data["Z_desk"], eval_data["rows"],
                              eval_data["cols"], eval_data["years"])
    maxdim = data["Z_esk"].shape[1]
    dims = sorted({d for d in (1, 2, 4, 8, 16, 24, 32, 48, 64) if d <= maxdim})
    curve = kernel_dimension_curve(eval_data["X"], eval_data["Z_esk"], eval_data["Z_desk"],
                                   dims, args.seed)

    comp.to_csv(out / "component_metrics.csv", index=False)
    curve.to_csv(out / "kernel_dimension_curve.csv", index=False)
    plot_component_fidelity(comp, out / "component_fidelity.png")
    plot_structure(ret, out / "structure_retention.png")
    plot_kernel_curve(curve, out / "kernel_by_dimension.png")
    plot_turnover(data, H, W, available[0], available[-1], out / "turnover_maps.png")
    plot_similarity_calibration(eval_data["X"], eval_data["Z_esk"], eval_data["Z_desk"],
                                out / "presentation_similarity_calibration.png", args.seed)
    plot_similarity_atlas(data, H, W, available[-1], out / "presentation_similarity_atlas.png")
    plot_turnover_agreement(data, available[0], available[-1], out / "presentation_turnover_agreement.png")
    atlas_dims = sorted({1, 2, 8, 16, 32, 48, maxdim})
    for year in (available[0], available[-1]):
        plot_component_atlas(data, H, W, year, atlas_dims, out / f"component_atlas_{year}.png")

    metrics = {
        "years": available, "n_points": int(len(data["X"])),
        "n_evaluation_points": int(ev.sum()),
        "evaluation_subset": "spatial holdout" if not ev.all() else "all matched points",
        "latent_dim": int(maxdim),
        "structure": {k: (int(v) if k.startswith("n_") else np.asarray(v).tolist())
                      for k, v in ret.items()},
        "component_metrics": comp.to_dict(orient="records"),
        "kernel_dimension_curve": curve.to_dict(orient="records"),
        "turnover_contract": {
            "fused": "1 - Ruzicka(X_deep, X_recent)",
            "esk": "1 - Z_esk(deep) dot Z_esk(recent)",
            "desk": "1 - Z_desk(deep) dot Z_desk(recent)",
            "note": "All three quantities use the same uncentered Ružička-kernel geometry; cosine-normalized Z is intentionally not used.",
        },
        "suggested_truncation": {
            "criterion": "minimum held-out DESK-to-fused normalized kernel RMSE",
            "dimension": int(curve.loc[curve.desk_rmse_norm.idxmin(), "dimension"]),
            "note": "treat as diagnostic evidence, not an automatic model-config change",
        },
        "interpretation": {
            "spatial_ratio": "<1 means DESK smooths away ESK neighbour-scale spatial detail",
            "temporal_ratio": "<1 means DESK under-reproduces ESK temporal change",
            "dimension_curve": "a DESK plateau/degradation while ESK improves indicates a natural model truncation",
        },
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[encoder-viz] complete -> {out}")


if __name__ == "__main__":
    main()
