#!/usr/bin/env python3
"""Post-MAP ecological conclusions and sanity checks.

This is deliberately separate from the legacy ``visualize_age_model.py``.  It
reconstructs the *current* age-structured model and uses its census-order
algebra.  The principal niche quantity is post-establishment, local,
density-independent intrinsic growth:

    lambda_potential = dominant_eigenvalue([[Sa, Sj], [Fmax * Sa, 0]])

It excludes dispersal, realized occupancy, density limitation, and the Allee
factor.  Those are important for realized range expansion, but including them
would make this a realized-distribution map rather than a fundamental-niche
map.  The Allee threshold is instead reported as a separate fitted mechanism.
The habitat manifolds (``H_s_local``/``H_r_local``, and hence Sa/Sj/Fmax) are
now purely covariate-driven (Z.beta only) -- an earlier design mixed a shared
smooth spatiotemporal term into both manifolds, but that has been replaced by
a K-only latent multiplicative correction (see ``age_fields.py``'s
``_K_CORRECTION_OFFSET``), so this niche quantity no longer carries even the
minor non-covariate caveat that used to apply. Sa/Sj/Fmax themselves are not
approximated for this purpose: they are the exact fitted per-cell fields the
full model uses (via ``reconstruct_map`` below), with only the dispersal
(``Q``) and density-dependence/Allee (``K``, ``c``, ``allee_gamma``) fields
dropped from the niche calculation itself.

``07_realized_source_sink.png`` is the deliberate REALIZED counterpart --
same Sa/Sj/Fmax but WITH density-dependence, the Allee effect, AND the K-only
latent correction (meant to capture disease-shaped dynamics with no covariate
of their own) -- so the two can be compared directly; see
``src/vis/age_model_math.py`` for the shared, samples-axis-agnostic math both
draw on (also the seam for a future MCMC-sample version of this script).

Outputs are written under the selected MAP run directory in ``map_diagnostics/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import transform as crs_transform
from numpyro.infer import Predictive

from src.config_utils import load_age_model_config, load_data_config
from src.model.age_fields import _K_CORRECTION_OFFSET
from src.model.age_priors import build_model_2d
from src.model.checkpoints import auto_delta_params_to_latents, load_map_params
from src.model.data_loading import load_data
from src.model.runtime_diagnostics import memory_snapshot, require_gpu
from src.vis.age_model_math import (
    add_timeline_markers, local_growth_lambda, realized_equilibrium,
    response_curve_fields, scatter_to_grid, window_mean,
)

# Back-compat local aliases (this file's plot functions historically used
# these private names; kept so the diffs below stay small).
_grid = scatter_to_grid
_window_mean = window_mean


def _run_dir(cfg, profile, precision):
    name = cfg["run_names"]["map"].format(precision=precision)
    if profile != "standard":
        name = f"{name}_{profile}"
    return Path(cfg["results_dir"]) / name


def reconstruct_map(data, params):
    """Evaluate deterministic model fields at a verified AutoDelta MAP point."""
    latents = auto_delta_params_to_latents(params)
    posterior = {name: jnp.expand_dims(value, 0) for name, value in latents.items()}
    needed = ["simulated_density", "Sa_flat", "Sj_flat", "Fmax_flat", "K_flat",
              "Q_flat", "expected_obs", "allee_gamma", "n50_raw", "w_env", "rho",
              "st_weights"]
    predictive = Predictive(build_model_2d, posterior_samples=posterior, return_sites=needed)
    result = predictive(jax.random.PRNGKey(104), data=data, prior_scale=1.0)
    result = jax.block_until_ready(result)
    sim = {name: np.asarray(value[0]) for name, value in result.items()}
    sim["latents"] = latents
    return sim


def plot_modern_niche(lam, years, rows, cols, shape, out, window):
    modern, early, n = _window_mean(lam, window)
    modern_g, early_g = _grid(modern[None], rows, cols, shape)[0], _grid(early[None], rows, cols, shape)[0]
    change = modern_g - early_g
    early_ok, modern_ok = early_g > 1.0, modern_g > 1.0
    transition = np.full(shape, np.nan)
    transition[(~early_ok) & (~modern_ok) & np.isfinite(modern_g)] = 0
    transition[(~early_ok) & modern_ok] = 1
    transition[early_ok & (~modern_ok)] = -1
    transition[early_ok & modern_ok] = 2
    lo, hi = np.nanpercentile(np.r_[early_g[np.isfinite(early_g)], modern_g[np.isfinite(modern_g)]], [2, 98])
    lo, hi = min(lo, 1.0), max(hi, 1.0)
    delta_lim = max(float(np.nanpercentile(np.abs(change), 98)), .02)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    im = ax[0, 0].imshow(modern_g, cmap="viridis", vmin=lo, vmax=hi)
    ax[0, 0].contour(modern_g, [1.0], colors="white", linewidths=1.0)
    ax[0, 0].set_title(f"Modern intrinsic growth λ ({years[-n]}–{years[-1]} mean)")
    fig.colorbar(im, ax=ax[0, 0], fraction=.046, label="Post-establishment λ")
    im = ax[0, 1].imshow(early_g, cmap="viridis", vmin=lo, vmax=hi)
    ax[0, 1].contour(early_g, [1.0], colors="white", linewidths=1.0)
    ax[0, 1].set_title(f"Early intrinsic growth λ ({years[0]}–{years[n - 1]} mean)")
    fig.colorbar(im, ax=ax[0, 1], fraction=.046, label="Post-establishment λ")
    im = ax[1, 0].imshow(change, cmap="RdBu_r", vmin=-delta_lim, vmax=delta_lim)
    ax[1, 0].set_title("Change in intrinsic growth (modern − early)")
    fig.colorbar(im, ax=ax[1, 0], fraction=.046, label="Δλ")
    # Transition codes are -1=lost, 0=persistently unsuitable, 1=gained,
    # 2=persistently suitable. Use explicit bins: imshow's default continuous
    # scaling silently mapped code 0 to the second (blue) colour before.
    cmap = mcolors.ListedColormap(["#d7301f", "#bdbdbd", "#2c7fb8", "#238443"])
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    im = ax[1, 1].imshow(transition, cmap=cmap, norm=norm)
    ax[1, 1].set_title("Fundamental-niche transition")
    ax[1, 1].legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#bdbdbd", label="Persistently λ ≤ 1"),
        plt.Rectangle((0, 0), 1, 1, color="#238443", label="Persistently λ > 1"),
        plt.Rectangle((0, 0), 1, 1, color="#2c7fb8", label="Gained λ > 1"),
        plt.Rectangle((0, 0), 1, 1, color="#d7301f", label="Lost λ > 1"),
    ], loc="lower left", fontsize=8, frameon=True)
    for a in ax.flat:
        a.axis("off")
    fig.suptitle("House Finch fundamental niche: local demographic potential", y=.98)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)
    return modern, early, transition


def plot_niche_trajectory(lam, years, rows, cols, ref_raster, out):
    suitable = lam > 1.0
    fraction = suitable.mean(axis=1)
    mean_lambda = lam.mean(axis=1)
    centroid_lat = np.full(len(years), np.nan)
    with rasterio.open(ref_raster) as src:
        for t in range(len(years)):
            where = np.flatnonzero(suitable[t])
            if not len(where):
                continue
            x, y = rasterio.transform.xy(src.transform, rows[where], cols[where], offset="center")
            _, lat = crs_transform(src.crs, "EPSG:4326", list(x), list(y))
            centroid_lat[t] = np.mean(lat)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(years, fraction * 100, color="#238443", lw=2, label="Land cells with λ > 1")
    ax[0].set(ylabel="Suitable land (%)", title="Trajectory of local demographic potential")
    ax0b = ax[0].twinx(); ax0b.plot(years, mean_lambda, color="#54278f", lw=1.8, label="Mean λ")
    ax0b.set_ylabel("Mean λ")
    ax[1].plot(years, centroid_lat, color="#2c7fb8", lw=2)
    ax[1].set(xlabel="Year", ylabel="Mean latitude (°N)", title="Centroid of land with λ > 1")
    for a in ax:
        a.grid(alpha=.25)
        add_timeline_markers(a)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)
    return fraction, mean_lambda, centroid_lat


def plot_modern_rate_maps(sim, years, rows, cols, shape, out, window):
    fields = [("Adult survival", sim["Sa_flat"], "viridis", None),
              ("Juvenile survival", sim["Sj_flat"], "viridis", None),
              ("Fecundity ceiling", sim["Fmax_flat"], "magma", None),
              ("Carrying capacity", sim["K_flat"], "magma", "relative units")]
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    for axis, (label, field, cmap, unit) in zip(ax.flat, fields):
        avg, _, n = _window_mean(field, window)
        grid = _grid(avg[None], rows, cols, shape)[0]
        lo, hi = np.nanpercentile(grid, [2, 98])
        image = axis.imshow(grid, cmap=cmap, vmin=lo, vmax=hi)
        axis.set_title(label); axis.axis("off")
        fig.colorbar(image, ax=axis, fraction=.046, label=unit or label)
    fig.suptitle(f"Demographic ingredients of the modern niche ({years[-n]}–{years[-1]} mean)")
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def plot_fit_diagnostics(sim, data, years, out):
    observed = np.asarray(data["observed_results"])
    predicted = np.asarray(sim["expected_obs"])
    t = np.asarray(data["obs_time_indices"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    hb = ax[0].hexbin(np.log1p(observed), np.log1p(predicted), gridsize=50, mincnt=1, cmap="viridis")
    lim = max(ax[0].get_xlim()[1], ax[0].get_ylim()[1]); ax[0].plot([0, lim], [0, lim], color="white", lw=1)
    ax[0].set(xlabel="log(1 + observed BBS count)", ylabel="log(1 + fitted mean)", title="Observation-scale calibration")
    fig.colorbar(hb, ax=ax[0], label="Routes")
    obs_mean = np.array([observed[t == i].mean() if np.any(t == i) else np.nan for i in range(len(years))])
    pred_mean = np.array([predicted[t == i].mean() if np.any(t == i) else np.nan for i in range(len(years))])
    ax[1].plot(years, obs_mean, label="Observed", color="#252525", lw=1.8)
    ax[1].plot(years, pred_mean, label="Fitted mean", color="#d95f0e", lw=1.8)
    ax[1].set(xlabel="Year", ylabel="Mean BBS count", title="Observed versus fitted annual mean")
    ax[1].legend(); ax[1].grid(alpha=.25)
    add_timeline_markers(ax[1])
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)
    residual = np.log1p(predicted) - np.log1p(observed)
    return {"n_observations": int(len(observed)), "log1p_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "log1p_correlation": float(np.corrcoef(np.log1p(observed), np.log1p(predicted))[0, 1])}


def plot_response_curves(sim, out, top_n=6):
    """Sweep the top-|weight| Z features and plot Sa/Sj/Fmax/K response curves.

    Corrects a stale bug in the deprecated ``visualize_age_model.py`` (which
    used ``exp`` instead of ``softplus`` for Fmax) and adds the K response
    curve that script never plotted. See ``age_model_math.response_curve_fields``.
    """
    latents = sim["latents"]
    w_env = np.asarray(latents["w_env"])
    importance = np.abs(w_env).sum(axis=1)
    top_idx = np.argsort(importance)[::-1][:min(top_n, w_env.shape[0])]
    z_sweep = np.linspace(-3.0, 3.0, 60)

    ncols = 3
    nrows = int(np.ceil(len(top_idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.6 * nrows), squeeze=False)
    for axis, idx in zip(axes.flat, top_idx):
        curves = response_curve_fields(latents, z_sweep, int(idx))
        axis.plot(z_sweep, curves["Sa"], color="navy", lw=1.8)
        axis.plot(z_sweep, curves["Sj"], color="royalblue", lw=1.8, linestyle="--")
        axis.set_ylim(0, 1)
        axis.set_title(f"Z_{idx}  (|w_env|={importance[idx]:.2f})", fontsize=9)
        axis2 = axis.twinx()
        axis2.plot(z_sweep, curves["Fmax"], color="darkorange", lw=1.6)
        axis2.plot(z_sweep, curves["K"], color="seagreen", lw=1.6, linestyle=":")
    for axis in axes.flat[len(top_idx):]:
        axis.axis("off")

    handles = [
        plt.Line2D([], [], color="navy", label="Adult survival (Sa)"),
        plt.Line2D([], [], color="royalblue", linestyle="--", label="Juvenile survival (Sj)"),
        plt.Line2D([], [], color="darkorange", label="Fecundity ceiling (Fmax)"),
        plt.Line2D([], [], color="seagreen", linestyle=":", label="Carrying capacity (K)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Demographic response curves (top Z features by |w_env|)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=180); plt.close(fig)
    return {"response_curve_top_features": [int(i) for i in top_idx]}


def plot_environmental_drivers_limits(data, sim, years, rows, cols, shape, out, window):
    """Which Z feature contributes most to the survival/reproduction manifold, per cell."""
    latents = sim["latents"]
    w_env = np.asarray(latents["w_env"])
    beta_s, beta_r = w_env[:, 0], w_env[:, 1]
    # Z_gathered is (time, N_land, M) and typically device-resident; slice the
    # window BEFORE pulling to host so a full-array transfer is never needed.
    Z_full = data["Z_gathered"]
    n = min(window, Z_full.shape[0])
    Z = np.asarray(Z_full[-n:])
    Z_early = np.asarray(Z_full[:n])

    def dominant_feature(Z_window, beta):
        contrib_mean = (Z_window * beta[None, None, :]).mean(axis=0)  # (N_land, M)
        return np.argmax(contrib_mean, axis=1).astype("float32")

    panels = [
        (dominant_feature(Z, beta_s), f"Survival driver ({years[-1] - n + 1}–{years[-1]})"),
        (dominant_feature(Z, beta_r), f"Reproduction driver ({years[-1] - n + 1}–{years[-1]})"),
        (dominant_feature(Z_early, beta_s), f"Survival driver ({years[0]}–{years[0] + n - 1})"),
        (dominant_feature(Z_early, beta_r), f"Reproduction driver ({years[0]}–{years[0] + n - 1})"),
    ]
    M = w_env.shape[0]
    cmap = plt.get_cmap("tab20", M)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    im = None
    for axis, (idx_flat, title) in zip(ax.flat, panels):
        grid = _grid(idx_flat[None], rows, cols, shape)[0]
        im = axis.imshow(grid, cmap=cmap, vmin=-0.5, vmax=M - 0.5)
        axis.set_title(title, fontsize=10); axis.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=.025, ticks=range(M))
    cbar.set_label("Z feature index")
    fig.suptitle("Dominant environmental driver by cell (modern vs. early)")
    fig.savefig(out, dpi=180); plt.close(fig)


def plot_realized_source_sink(sim, lam_fundamental, years, rows, cols, shape, out, window):
    """Realized (density-dependent + Allee) counterpart to the fundamental-niche map.

    Contrasts directly against ``01_modern_fundamental_niche.png``: same
    Sa/Sj/Fmax, but with K and the Allee effect included, so
    ``lambda_realized <= lambda_fundamental`` everywhere.
    """
    _, _, lam_realized, _ = realized_equilibrium(
        sim["Sa_flat"], sim["Sj_flat"], sim["Fmax_flat"], sim["K_flat"], sim["allee_gamma"]
    )
    modern, _, n = _window_mean(lam_realized, window)
    modern_g = _grid(modern[None], rows, cols, shape)[0]
    fund_modern, _, _ = _window_mean(lam_fundamental, window)
    fund_g = _grid(fund_modern[None], rows, cols, shape)[0]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    binary = np.where(np.isfinite(modern_g), (modern_g > 1.0).astype(float), np.nan)
    ax[0].imshow(binary, cmap=mcolors.ListedColormap(["#d73027", "#4575b4"]), vmin=0, vmax=1)
    ax[0].set_title(f"Realized source/sink ({years[-n]}–{years[-1]})")
    ax[0].legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#4575b4", label="Source (λ_realized > 1)"),
        plt.Rectangle((0, 0), 1, 1, color="#d73027", label="Sink (λ_realized ≤ 1)"),
    ], loc="lower left", fontsize=7, frameon=True)

    lo, hi = np.nanpercentile(modern_g, [2, 98]); lo, hi = min(lo, 1.0), max(hi, 1.0)
    im = ax[1].imshow(modern_g, cmap="RdYlBu_r", vmin=lo, vmax=hi)
    ax[1].contour(modern_g, [1.0], colors="black", linewidths=1.0)
    ax[1].set_title("Realized λ (density-dependent + Allee)")
    fig.colorbar(im, ax=ax[1], fraction=.046, label="λ_realized")

    gap = fund_g - modern_g
    lim = max(float(np.nanpercentile(np.abs(gap[np.isfinite(gap)]), 98)), .02)
    im2 = ax[2].imshow(gap, cmap="magma", vmin=0, vmax=lim)
    ax[2].set_title("Gap: fundamental − realized λ")
    fig.colorbar(im2, ax=ax[2], fraction=.046, label="Δλ (≥ 0)")

    for a in ax:
        a.axis("off")
    fig.suptitle("Realized demographic potential (density-dependence + Allee included)")
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)
    return {"realized_modern_mean_lambda": float(np.mean(modern)),
            "realized_modern_source_fraction": float(np.mean(modern > 1.0))}


def plot_spatial_residuals(sim, data, shape, out):
    """Mean per-route log-scale residual (fitted − observed), scattered to the grid."""
    observed = np.asarray(data["observed_results"])
    predicted = np.asarray(sim["expected_obs"])
    obs_rows, obs_cols = np.asarray(data["obs_rows"]), np.asarray(data["obs_cols"])
    residual = np.log1p(predicted) - np.log1p(observed)

    grid_sum = np.zeros(shape, dtype="float64")
    grid_cnt = np.zeros(shape, dtype="int32")
    np.add.at(grid_sum, (obs_rows, obs_cols), residual)
    np.add.at(grid_cnt, (obs_rows, obs_cols), 1)
    grid_mean = np.where(grid_cnt > 0, grid_sum / np.maximum(grid_cnt, 1), np.nan)

    lim = max(float(np.nanpercentile(np.abs(grid_mean[np.isfinite(grid_mean)]), 98)), .05)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(grid_mean, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title("Mean log-scale residual per route (log(1+fitted) − log(1+observed))")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=.04, label="Residual (log1p scale)")
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def plot_spatiotemporal_diagnostics(data, sim, out, window):
    """'Escape hatch' check: how much is the K-only latent correction actually doing?

    st_basis/st_weights no longer touch H_s/H_r (an earlier design mixed a
    shared spatiotemporal term into both manifolds; that's been replaced by a
    K-only multiplicative correction, see age_fields.py's
    _K_CORRECTION_OFFSET). A runaway correction (multiplier far from 1 nearly
    everywhere) would mean this "latent disease dynamics" term is substituting
    for genuine covariate signal rather than capturing something real and
    localized; this plots the weight distribution and the actual per-cell
    correction values over a trailing window against that 1.0 no-effect line.
    """
    st_weights = np.asarray(sim["st_weights"])
    st_basis_full = data["st_basis"]  # (N_basis, time_post_invasion, N_land), device-resident

    n = min(window, st_basis_full.shape[1])
    st_basis = np.asarray(st_basis_full[:, -n:, :])  # slice before host transfer
    k_smooth = np.einsum("bnl,b->nl", st_basis, st_weights)
    k_multiplier = np.log1p(np.exp(-np.abs(_K_CORRECTION_OFFSET + k_smooth))) + \
        np.maximum(_K_CORRECTION_OFFSET + k_smooth, 0.0)  # numerically-stable softplus

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].hist(st_weights, bins=40, color="#6a51a3")
    ax[0].axvline(0, color="black", lw=.8)
    ax[0].set(title="K-correction basis weights (st_weights)", xlabel="Weight", ylabel="Count")

    ax[1].hist(k_multiplier.ravel(), bins=60, color="#238443")
    ax[1].axvline(1.0, color="black", lw=1.2, linestyle="--", label="no effect")
    ax[1].set(title=f"K multiplier, last {n} yr (median={np.median(k_multiplier):.2f})",
              xlabel="K multiplier (1.0 = no correction)", ylabel="Cell-years")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)
    return {"k_correction_median_multiplier": float(np.median(k_multiplier)),
            "k_correction_p05_multiplier": float(np.percentile(k_multiplier, 5)),
            "k_correction_p95_multiplier": float(np.percentile(k_multiplier, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default=os.environ.get("HOUFIN_MAP_PROFILE", "standard"))
    parser.add_argument("--precision", default=os.environ.get("HOUFIN_MODEL_PRECISION", "float32"), choices=["float32", "float64"])
    parser.add_argument("--window-years", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.window_years < 1:
        raise ValueError("--window-years must be positive")

    cfg, dcfg = load_age_model_config(), load_data_config()
    run_dir = _run_dir(cfg, args.profile, args.precision)
    out = Path(args.out) if args.out else run_dir / "map_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    device = require_gpu("post-MAP diagnostics")
    data = load_data(cfg["input_dir"], target_device=device, precision=args.precision)
    params, checkpoint = load_map_params(str(run_dir))
    print(f"[map-viz] profile={args.profile}; checkpoint step={checkpoint['step']}; output={out}")
    sim = reconstruct_map(data, params)
    memory_snapshot("map-viz-reconstructed", device)

    rows, cols = np.asarray(data["land_rows"]), np.asarray(data["land_cols"])
    years = np.asarray(data["years"])
    shape = tuple(np.asarray(data["land_mask"]).shape)
    lam = local_growth_lambda(sim["Sa_flat"], sim["Sj_flat"], sim["Fmax_flat"])
    modern, early, transition = plot_modern_niche(lam, years, rows, cols, shape,
                                                   out / "01_modern_fundamental_niche.png", args.window_years)
    fraction, mean_lambda, centroid_lat = plot_niche_trajectory(
        lam, years, rows, cols, dcfg["grid"]["ref_raster"], out / "02_niche_trajectory.png")
    plot_modern_rate_maps(sim, years, rows, cols, shape, out / "03_modern_demographic_rates.png", args.window_years)
    fit_metrics = plot_fit_diagnostics(sim, data, years, out / "04_map_fit_diagnostics.png")
    response_metrics = plot_response_curves(sim, out / "05_demographic_response_curves.png")
    plot_environmental_drivers_limits(data, sim, years, rows, cols, shape,
                                       out / "06_environmental_drivers_limits.png", args.window_years)
    source_sink_metrics = plot_realized_source_sink(
        sim, lam, years, rows, cols, shape, out / "07_realized_source_sink.png", args.window_years)
    plot_spatial_residuals(sim, data, shape, out / "08_spatial_residuals.png")
    st_metrics = plot_spatiotemporal_diagnostics(data, sim, out / "09_spatiotemporal_weight_diagnostics.png",
                                                  args.window_years)
    n50_raw = float(np.asarray(sim["n50_raw"])); n50 = float(np.logaddexp(0.0, n50_raw))
    transition_land = np.isfinite(transition)
    metrics = {
        "profile": args.profile, "checkpoint_step": int(checkpoint["step"]),
        "years": [int(years[0]), int(years[-1])], "window_years": args.window_years,
        "fundamental_niche_definition": "post-establishment, density-independent local dominant eigenvalue of [[Sa, Sj], [Fmax*Sa, 0]]; excludes dispersal, density limitation, realized occupancy, and Allee limitation",
        "modern_mean_lambda": float(np.mean(modern)), "early_mean_lambda": float(np.mean(early)),
        "modern_suitable_fraction": float(np.mean(modern > 1.0)), "early_suitable_fraction": float(np.mean(early > 1.0)),
        "gained_suitable_fraction": float(np.mean(transition[transition_land] == 1)),
        "lost_suitable_fraction": float(np.mean(transition[transition_land] == -1)),
        "final_suitable_centroid_latitude": float(centroid_lat[-1]),
        "allee_n50_bbs_count": n50, "fit": fit_metrics,
        "realized_source_sink": source_sink_metrics,
        "spatiotemporal_diagnostics": st_metrics,
        **response_metrics,
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[map-viz] complete -> {out}")


if __name__ == "__main__":
    main()
