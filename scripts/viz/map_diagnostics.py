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
an onset-gated, sign-constrained disease depression of K alone (see
``age_fields.py``), so this niche quantity no longer carries even the
minor non-covariate caveat that used to apply. Sa/Sj/Fmax themselves are not
approximated for this purpose: they are the exact fitted per-cell fields the
full model uses (via ``reconstruct_map`` below), with only the dispersal
(``Q``) and density-dependence/Allee (``K``, ``c``, ``allee_gamma``) fields
dropped from the niche calculation itself.

``07_source_sink_fields.npz`` (and a georeferenced ``.tif`` beside it) persists the
grids figure 07 draws -- modern realized lambda, modern fundamental lambda, modern
K, the source mask, and the averaging window's year span. These used to exist only
as pixels in the PNG, so nothing could compare source/sink structure between runs
(e.g. across a dispersal-distance sweep) or re-plot it without a GPU and a full
model reconstruction.

``07_realized_source_sink.png`` is the deliberate REALIZED counterpart --
same Sa/Sj/Fmax but WITH density-dependence, the Allee effect, AND the K-only
disease depression (mycoplasmal conjunctivitis, which has no covariate of its
own) -- so the two can be compared directly; see
``src/vis/age_model_math.py`` for the shared, samples-axis-agnostic math both
draw on (also the seam for a future MCMC-sample version of this script).

``10_age_structure.png`` compares theoretical equilibrium age structure
(local rho implied by the fitted vital rates, assuming the system has
settled -- no invasion-front/transient history) against REALIZED age
structure (Nj/(Na+Nj) from the actual forward-simulated Na_grid/Nj_grid age
pools, which does carry that history); a gap between them, especially near a
still-advancing range edge, is the expected signature of non-equilibrium age
structure at an invasion front. Na_grid/Nj_grid cost nothing extra during
MAP/SVI optimization -- see forward_sim_age_structured's docstring for why.

``13_niche_change_since_invasion.png`` and
``14_environmental_drivers_since_invasion.png`` repeat figures 01 and 06 with the
baseline window anchored at ``invasion_year`` (1940) instead of the start of the
model timeline (1902). The 1902 baseline is "before anything happened," but it is
also 38 years of climate change removed from the release, so change measured
against it is not change the invasion experienced; the 1940-anchored pair is the
one to read for invasion-relative statements. ``metrics.json`` carries both.

``11_invasion_progression.png`` (small-multiple maps) and
``12_invasion_animation.mp4`` (side-by-side simulated-vs-observed animation,
GIF fallback if FFMpeg is unavailable) visualize the spread of the modeled
invasion over the full timeline using ``simulated_density`` -- reconstructed
by ``reconstruct_map`` like everything else here, but previously never
plotted despite being the model's most direct depiction of "an invasion."

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
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import transform as crs_transform
from numpyro.infer import Predictive

from src.config_utils import load_age_model_config, load_data_config
from src.temporal import load_timeline
from src.model.age_priors import build_model_2d
from src.model.checkpoints import auto_delta_params_to_latents, load_map_params
from src.model.data_loading import load_data
from src.model.runtime_diagnostics import memory_snapshot, require_gpu
from src.vis.age_model_math import (
    add_timeline_markers, baseline_window_mean, local_growth_lambda,
    realized_equilibrium, response_curve_fields, scatter_to_grid, window_mean,
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
              "env_corr_repro_capacity", "env_corr_survival_capacity",
              "manifold_loadings", "w_k_trend",
              "disease_severity_map", "disease_mu_sev", "disease_b_late",
              "disease_w_lag", "disease_lag0", "disease_tau", "disease_rec",
              "disease_tau_rec", "Na_grid", "Nj_grid"]
    predictive = Predictive(build_model_2d, posterior_samples=posterior, return_sites=needed)
    result = predictive(jax.random.PRNGKey(104), data=data, prior_scale=1.0)
    result = jax.block_until_ready(result)
    sim = {name: np.asarray(value[0]) for name, value in result.items()}
    sim["latents"] = latents
    return sim


def plot_modern_niche(lam, years, rows, cols, shape, out, window, ref_year=None):
    """Modern vs baseline fundamental niche, and the transition between them.

    ``ref_year=None`` anchors the baseline at the start of the model timeline
    (1902). Pass ``invasion_year`` to anchor it at the release instead -- the
    1902 baseline also carries 38 years of climate change that has nothing to do
    with the invasion, so the two figures answer genuinely different questions
    and are both produced (see ``baseline_window_mean``).
    """
    modern, _, n = _window_mean(lam, window)
    early, n_base, base_span = baseline_window_mean(lam, years, window, ref_year)
    n = min(n, n_base)
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
    ax[0, 1].set_title(f"Baseline intrinsic growth λ ({base_span[0]}–{base_span[1]} mean)")
    fig.colorbar(im, ax=ax[0, 1], fraction=.046, label="Post-establishment λ")
    im = ax[1, 0].imshow(change, cmap="RdBu_r", vmin=-delta_lim, vmax=delta_lim)
    ax[1, 0].set_title(f"Change in intrinsic growth (modern − {base_span[0]}–{base_span[1]})")
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
    baseline_label = "timeline start" if ref_year is None else f"invasion ({base_span[0]})"
    fig.suptitle(f"House Finch fundamental niche: local demographic potential "
                 f"(baseline = {baseline_label})", y=.98)
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


def plot_environmental_drivers_limits(data, sim, years, rows, cols, shape, out, window,
                                      ref_year=None):
    """Which Z feature contributes most to the survival/reproduction manifold, per cell.

    ``ref_year`` anchors the baseline pair of panels, exactly as in
    ``plot_modern_niche``: None = timeline start (1902), or the invasion year to
    ask what the drivers looked like when the species actually arrived.
    """
    latents = sim["latents"]
    w_env = np.asarray(latents["w_env"])
    beta_s, beta_r = w_env[:, 0], w_env[:, 1]
    # Z_gathered is (time, N_land, M) and typically device-resident; slice the
    # window BEFORE pulling to host so a full-array transfer is never needed.
    Z_full = data["Z_gathered"]
    n = min(window, Z_full.shape[0])
    Z = np.asarray(Z_full[-n:])
    # Slice the baseline window on device too, for the same reason.
    years_arr = np.asarray(years)
    i0 = 0 if ref_year is None else int(np.flatnonzero(years_arr == int(ref_year))[0])
    n_base = min(n, Z_full.shape[0] - i0)
    Z_early = np.asarray(Z_full[i0:i0 + n_base])
    base_span = (int(years_arr[i0]), int(years_arr[i0 + n_base - 1]))

    def dominant_feature(Z_window, beta):
        contrib_mean = (Z_window * beta[None, None, :]).mean(axis=0)  # (N_land, M)
        return np.argmax(contrib_mean, axis=1).astype("float32")

    panels = [
        (dominant_feature(Z, beta_s), f"Survival driver ({years[-1] - n + 1}–{years[-1]})"),
        (dominant_feature(Z, beta_r), f"Reproduction driver ({years[-1] - n + 1}–{years[-1]})"),
        (dominant_feature(Z_early, beta_s), f"Survival driver ({base_span[0]}–{base_span[1]})"),
        (dominant_feature(Z_early, beta_r), f"Reproduction driver ({base_span[0]}–{base_span[1]})"),
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
    fig.suptitle(f"Dominant environmental driver by cell "
                 f"(modern vs. {base_span[0]}–{base_span[1]} baseline)")
    fig.savefig(out, dpi=180); plt.close(fig)


def _write_source_sink_fields(npz_path, lam_realized, lam_fundamental, K_modern,
                              year_first, year_last, n_years, ref_raster=None):
    """Persist the modern source/sink grids as .npz (+ a GeoTIFF beside it).

    Kept deliberately small (a handful of MB): the three window-mean grids that
    ``07_realized_source_sink.png`` draws, plus the window's year span so a
    consumer can verify two runs were averaged over the same years before
    differencing them. The GeoTIFF is georeferenced from the same
    ``grid.ref_raster`` the trajectory plot uses, so the field drops straight into
    GIS; band order matches ``band_names``.
    """
    npz_path = Path(npz_path)
    np.savez_compressed(
        npz_path,
        lam_realized_modern=lam_realized.astype("float32"),
        lam_fundamental_modern=lam_fundamental.astype("float32"),
        K_modern=K_modern.astype("float32"),
        source_mask=np.where(np.isfinite(lam_realized), lam_realized > 1.0, False),
        window_years=np.array([int(year_first), int(year_last)]),
        window_n=np.int32(n_years),
    )
    if ref_raster is None:
        return
    bands = [lam_realized, lam_fundamental, K_modern]
    with rasterio.open(ref_raster) as src:
        transform, crs = src.transform, src.crs
        if (src.height, src.width) != lam_realized.shape:
            print(f"[map-viz] ref raster {src.height}x{src.width} != field "
                  f"{lam_realized.shape}; skipped GeoTIFF")
            return
    with rasterio.open(npz_path.with_suffix(".tif"), "w", driver="GTiff",
                       height=lam_realized.shape[0], width=lam_realized.shape[1],
                       count=len(bands), dtype="float32", crs=crs, transform=transform,
                       nodata=np.float32(np.nan), compress="deflate") as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(band.astype("float32"), i)
        dst.update_tags(band_names="lam_realized_modern,lam_fundamental_modern,K_modern",
                        window=f"{int(year_first)}-{int(year_last)}")


def plot_realized_source_sink(sim, lam_fundamental, years, rows, cols, shape, out, window,
                              fields_out=None, ref_raster=None):
    """Realized (density-dependent + Allee) counterpart to the fundamental-niche map.

    Contrasts directly against ``01_modern_fundamental_niche.png``: same
    Sa/Sj/Fmax, but with K and the Allee effect included, so
    ``lambda_realized <= lambda_fundamental`` everywhere.

    ``fields_out`` additionally persists the underlying grids (see
    ``_write_source_sink_fields``). Without it these rasters exist only as pixels
    in the PNG, so nothing downstream -- a dispersal sweep comparing runs, or any
    re-plot -- can get at them without a GPU and a full model reconstruction.
    """
    _, _, lam_realized, _ = realized_equilibrium(
        sim["Sa_flat"], sim["Sj_flat"], sim["Fmax_flat"], sim["K_flat"], sim["allee_gamma"]
    )
    modern, _, n = _window_mean(lam_realized, window)
    modern_g = _grid(modern[None], rows, cols, shape)[0]
    fund_modern, _, _ = _window_mean(lam_fundamental, window)
    fund_g = _grid(fund_modern[None], rows, cols, shape)[0]
    if fields_out is not None:
        K_modern, _, _ = _window_mean(np.asarray(sim["K_flat"]), window)
        _write_source_sink_fields(
            fields_out, modern_g, fund_g,
            _grid(K_modern[None], rows, cols, shape)[0],
            years[-n], years[-1], n, ref_raster,
        )

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


def _k_trend_metrics(data, sim):
    """Report the continental K trend against its own (deliberately tight) prior.

    The point of the prior-SD multiple is that the safety valve reads itself: no
    one has to remember what the budget was to know whether the term has been
    pushed somewhere it should not go.
    """
    w = np.asarray(sim["w_k_trend"])
    basis = np.asarray(data["k_trend_basis"])
    spec = load_age_model_config()["population_model"]["k_trend"]
    prior_sd = float(spec["budget"]) / np.sqrt(int(spec["n_basis"]))
    trend = basis.T @ w                      # (time,) on K's pre-softplus scale
    mult = np.exp(trend)
    z = float(np.abs(w).max() / prior_sd)
    return {"weights": [float(x) for x in w],
            "prior_sd_per_weight": prior_sd,
            "max_prior_sd_multiple": z,
            "k_multiplier_first_year": float(mult[0]),
            "k_multiplier_last_year": float(mult[-1]),
            "k_multiplier_max_deviation": float(np.abs(mult - 1.0).max()),
            # True = a real temporal signal is going unexplained by the mechanisms.
            "safety_valve_tripped": bool(z > 3.0 or np.abs(mult - 1.0).max() > 0.15)}


def plot_disease_diagnostics(data, sim, years, rows, cols, shape, out, window):
    """Is the disease term describing an epizootic, or absorbing spatial misfit?

    The term is K = K_base * (1 - severity(x) * gate(x,t) * (1 - recovery)); see
    age_fields.py. Its predecessor was a generic 967-coefficient spatiotemporal
    field subtracted from K's pre-softplus argument, which annihilated eastern K
    (softplus is effectively exp there, so an additive penalty was an unbounded
    multiplicative one) and, being the model's only non-covariate spatial degree of
    freedom, soaked up every kind of spatial mismatch.

    Three panels, each checking one thing the structured form claims:

    1. **The severity map** -- the falsifiable claim. The hypothesis is ~50% of
       capacity removed in the east and less in the west (more genetic diversity).
       If this comes out uniform, or saturated near 1, the term is still doing
       non-disease work.
    2. **The onset profile** -- mean removed fraction against years since the
       modeled arrival. Must be ~0 left of zero and rise through it; flat means the
       arrival map is not actually structuring the term.
    3. **The recovery trajectory** -- removed fraction over calendar years for
       early- vs late-arriving thirds of the range, which is where "slowly
       increasing resilience" is visible (or absent).
    """
    sev = np.asarray(sim["disease_severity_map"])          # (N_land,) peak fraction
    ceiling = float(load_age_model_config()["population_model"]
                    ["disease_prior"]["severity_ceiling"])
    onset = np.asarray(data["disease_onset"])              # timestep units
    dis_t0 = int(data["disease_timestep"])
    lag0 = float(np.asarray(sim["disease_lag0"]))
    tau = float(np.asarray(sim["disease_tau"]))
    rec = float(np.asarray(sim["disease_rec"]))
    tau_rec = float(np.asarray(sim["disease_tau_rec"]))
    w_lag = np.asarray(sim["disease_w_lag"])
    onset_t = onset + lag0 + np.asarray(data["disease_lag_basis"]).T @ w_lag

    # Reproduce the model's fraction over the whole epizootic window on the host.
    t_abs = np.arange(dis_t0, len(years))[:, None]
    gate = 1.0 / (1.0 + np.exp(-(t_abs - onset_t[None, :]) / tau))
    age = np.maximum(t_abs - onset_t[None, :], 0.0)
    frac = sev[None, :] * gate * (1.0 - rec * (-np.expm1(-age / tau_rec)))

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

    sev_grid = _grid(sev[None], rows, cols, shape)[0]
    at_ceiling = float((sev > 0.99 * ceiling).mean())
    im = ax[0].imshow(sev_grid, cmap="inferno_r", vmin=0.0, vmax=ceiling)
    ax[0].set(title=f"Peak severity: fraction of K removed\n"
                    f"(median {np.median(sev):.0%}, range {sev.min():.0%}-{sev.max():.0%}; "
                    f"{at_ceiling:.0%} at the {ceiling:.0%} ceiling)")
    ax[0].axis("off")
    fig.colorbar(im, ax=ax[0], fraction=.046, label="Fraction of K removed at peak")

    since = (t_abs - onset_t[None, :]).ravel()
    bins = np.arange(-15, 26)
    which = np.digitize(since, bins) - 1
    flat = frac.ravel()
    prof = np.array([flat[which == i].mean() if (which == i).any() else np.nan
                     for i in range(len(bins) - 1)])
    ax[1].plot(bins[:-1], prof, color="#cc4c02", lw=2)
    ax[1].axvline(0, color="black", lw=.8)
    ax[1].axhline(0, color="0.7", lw=.8)
    ax[1].set(title=f"Onset profile (lag0={lag0:+.1f} yr, tau={tau:.2f} yr)",
              xlabel="Years since modeled arrival", ylabel="Mean fraction of K removed")

    # Early vs late thirds of the arrival distribution: recovery should show up as
    # a decline after each group's own onset, and the late group should be milder
    # if the western-diversity hypothesis holds.
    q1, q2 = np.percentile(onset, [33, 67])
    groups = [("earliest-arriving third", onset <= q1, "#08519c"),
              ("latest-arriving third", onset >= q2, "#a50f15")]
    cal = np.asarray(years)[dis_t0:]
    for label, mask, color in groups:
        if mask.any():
            ax[2].plot(cal, frac[:, mask].mean(axis=1), color=color, lw=2, label=label)
    ax[2].set(title=f"Recovery (rec={rec:.0%} of the hit, tau_rec={tau_rec:.0f} yr)",
              xlabel="Year", ylabel="Mean fraction of K removed")
    ax[2].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)

    pre_front = flat[since < -5]
    return {"disease_severity_median": float(np.median(sev)),
            "disease_severity_ceiling": ceiling,
            # THE diagnostic: cells pinned at the ceiling mean misfit is still being
            # routed to the disease term rather than to H_k or the K trend.
            "disease_severity_fraction_at_ceiling": at_ceiling,
            "disease_severity_p05": float(np.percentile(sev, 5)),
            "disease_severity_p95": float(np.percentile(sev, 95)),
            "disease_fraction_median_modern": float(np.median(frac[-min(window, frac.shape[0]):])),
            # Must stay ~0: nonzero means the exogenous gate has been defeated and
            # the term is acting where the front had not yet arrived.
            "disease_fraction_pre_front_mean": float(pre_front.mean()) if pre_front.size else 0.0,
            "disease_severity_mu_logit": float(np.asarray(sim["disease_mu_sev"])),
            "disease_late_arrival_coef": float(np.asarray(sim["disease_b_late"])),
            "disease_lag0_years": lag0,
            "disease_tau_years": tau,
            "disease_recovered_fraction": rec,
            "disease_recovery_tau_years": tau_rec}


def plot_age_structure(sim, years, rows, cols, shape, land_mask, out, window):
    """Theoretical (equilibrium) vs realized juvenile-fraction maps.

    Theoretical: rho from age_model_math.realized_equilibrium -- the LOCAL,
    density-dependent+Allee equilibrium juvenile fraction implied by the
    fitted Sa/Sj/Fmax/K/allee_gamma at each cell/year, assuming the system has
    settled there (no transient/dispersal history). Realized: the ACTUAL
    simulated Nj/(Na+Nj) from the forward age-structured dynamics (Na_grid/
    Nj_grid), which does carry transient/invasion-front history. A gap
    between the two -- especially near a still-advancing range edge -- is the
    expected signature of non-equilibrium age structure at the invasion front.
    """
    _, _, _, rho_theory = realized_equilibrium(
        sim["Sa_flat"], sim["Sj_flat"], sim["Fmax_flat"], sim["K_flat"], sim["allee_gamma"]
    )
    rho_theory_grid = _grid(rho_theory, rows, cols, shape)  # (time, Ny, Nx)

    land = land_mask.astype(bool)
    Na_grid, Nj_grid = sim["Na_grid"], sim["Nj_grid"]
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_realized_grid = Nj_grid / np.maximum(Na_grid + Nj_grid, 1e-9)
    rho_realized_grid = np.where(land[None], rho_realized_grid, np.nan)
    rho_theory_grid = np.where(land[None], rho_theory_grid, np.nan)

    modern_theory, _, n = _window_mean(rho_theory_grid, window)
    modern_realized, _, _ = _window_mean(rho_realized_grid, window)
    gap = modern_realized - modern_theory

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    im0 = ax[0].imshow(modern_theory, cmap="viridis", vmin=0, vmax=1)
    ax[0].set_title("Theoretical equilibrium ρ")
    fig.colorbar(im0, ax=ax[0], fraction=.046, label="Juvenile fraction ρ")
    im1 = ax[1].imshow(modern_realized, cmap="viridis", vmin=0, vmax=1)
    ax[1].set_title("Realized ρ (simulated Nj/(Na+Nj))")
    fig.colorbar(im1, ax=ax[1], fraction=.046, label="Juvenile fraction ρ")
    lim = max(float(np.nanpercentile(np.abs(gap[np.isfinite(gap)]), 98)), .02)
    im2 = ax[2].imshow(gap, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[2].set_title("Gap: realized − theoretical")
    fig.colorbar(im2, ax=ax[2], fraction=.046, label="Δρ")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"Age structure ({years[-n]}–{years[-1]} mean)")
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)
    return {"modern_mean_theoretical_juvenile_fraction": float(np.nanmean(modern_theory)),
            "modern_mean_realized_juvenile_fraction": float(np.nanmean(modern_realized))}


def plot_invasion_progression(sim, years, land_mask, out, n_panels=6):
    """Small multiples of total simulated density across the invasion era."""
    density = np.where(land_mask.astype(bool)[None], sim["simulated_density"], np.nan)
    log_density = np.log1p(density)

    idx = np.linspace(0, len(years) - 1, n_panels).round().astype(int)
    vmax = float(np.nanpercentile(log_density[idx], 99))

    ncols = 3
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.6 * nrows), squeeze=False)
    im = None
    for axis, i in zip(axes.flat, idx):
        im = axis.imshow(log_density[i], cmap="magma", vmin=0, vmax=vmax)
        axis.set_title(str(years[i]), fontsize=10)
        axis.axis("off")
    for axis in axes.flat[len(idx):]:
        axis.axis("off")
    fig.colorbar(im, ax=axes, fraction=.025, label="log(1 + simulated density)")
    fig.suptitle("Invasion progression: simulated density over time")
    fig.savefig(out, dpi=180); plt.close(fig)


def create_invasion_animation(sim, data, years, land_mask, out):
    """Animated side-by-side: simulated density vs. observed BBS counts, all years.

    Falls back from mp4 (FFMpeg) to GIF (pillow) if FFMpeg isn't available on
    the run environment -- mirrors src/vis/_age_vis_common.py's create_animation.
    """
    shape = land_mask.shape
    obs_grid = np.full((len(years), *shape), np.nan)
    obs_rows = np.asarray(data["obs_rows"])
    obs_cols = np.asarray(data["obs_cols"])
    obs_t = np.asarray(data["obs_time_indices"])
    obs_grid[obs_t, obs_rows, obs_cols] = np.asarray(data["observed_results"])

    density = np.where(land_mask.astype(bool)[None], sim["simulated_density"], np.nan)
    vmax_sim = float(np.nanpercentile(density, 99))
    vmax_obs = float(np.nanpercentile(obs_grid[np.isfinite(obs_grid)], 99))

    fig, (ax_sim, ax_obs) = plt.subplots(1, 2, figsize=(13, 6))
    im_sim = ax_sim.imshow(density[0], cmap="magma", vmin=0, vmax=vmax_sim)
    ax_sim.set_title("Simulated density"); ax_sim.axis("off")
    im_obs = ax_obs.imshow(obs_grid[0], cmap="magma", vmin=0, vmax=vmax_obs)
    ax_obs.set_title("Observed BBS counts"); ax_obs.axis("off")
    title = fig.suptitle(f"Year {years[0]}", fontsize=14, fontweight="bold")

    def update(frame):
        title.set_text(f"Year {years[frame]}")
        im_sim.set_data(density[frame])
        im_obs.set_data(obs_grid[frame])
        return im_sim, im_obs, title

    ani = animation.FuncAnimation(fig, update, frames=len(years), interval=120, blit=False)
    try:
        ani.save(str(out), writer=animation.FFMpegWriter(fps=8, bitrate=1800))
    except Exception as exc:
        gif_out = str(out).rsplit(".", 1)[0] + ".gif"
        print(f"[map-viz] FFMpeg unavailable ({exc}); falling back to GIF: {gif_out}")
        ani.save(gif_out, writer="pillow", fps=8)
    plt.close(fig)


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
    # Invasion-anchored counterparts (13/14). The default baseline is the start of
    # the model timeline (1902), which mixes 38 years of pre-invasion climate
    # change into every "change since the beginning" statement; anchoring at 1940
    # instead measures change relative to what the species actually met on
    # arrival. Both are kept because they answer different questions.
    inv_year = int(load_timeline(dcfg)["invasion_year"])
    _, early_inv, transition_inv = plot_modern_niche(
        lam, years, rows, cols, shape, out / "13_niche_change_since_invasion.png",
        args.window_years, ref_year=inv_year)
    fraction, mean_lambda, centroid_lat = plot_niche_trajectory(
        lam, years, rows, cols, dcfg["grid"]["ref_raster"], out / "02_niche_trajectory.png")
    plot_modern_rate_maps(sim, years, rows, cols, shape, out / "03_modern_demographic_rates.png", args.window_years)
    fit_metrics = plot_fit_diagnostics(sim, data, years, out / "04_map_fit_diagnostics.png")
    response_metrics = plot_response_curves(sim, out / "05_demographic_response_curves.png")
    plot_environmental_drivers_limits(data, sim, years, rows, cols, shape,
                                       out / "06_environmental_drivers_limits.png", args.window_years)
    plot_environmental_drivers_limits(data, sim, years, rows, cols, shape,
                                       out / "14_environmental_drivers_since_invasion.png",
                                       args.window_years, ref_year=inv_year)
    source_sink_metrics = plot_realized_source_sink(
        sim, lam, years, rows, cols, shape, out / "07_realized_source_sink.png", args.window_years,
        fields_out=out / "07_source_sink_fields.npz", ref_raster=dcfg["grid"]["ref_raster"])
    plot_spatial_residuals(sim, data, shape, out / "08_spatial_residuals.png")
    disease_metrics = plot_disease_diagnostics(
        data, sim, years, rows, cols, shape, out / "09_disease_diagnostics.png",
        args.window_years)
    land_mask_arr = np.asarray(data["land_mask"])
    age_structure_metrics = plot_age_structure(
        sim, years, rows, cols, shape, land_mask_arr, out / "10_age_structure.png", args.window_years)
    plot_invasion_progression(sim, years, land_mask_arr, out / "11_invasion_progression.png")
    create_invasion_animation(sim, data, years, land_mask_arr, out / "12_invasion_animation.mp4")
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
        # Same quantities against the invasion-year baseline (figures 13/14).
        "invasion_baseline_year": inv_year,
        "invasion_baseline_mean_lambda": float(np.mean(early_inv)),
        "invasion_baseline_suitable_fraction": float(np.mean(early_inv > 1.0)),
        "gained_suitable_fraction_since_invasion": float(
            np.mean(transition_inv[np.isfinite(transition_inv)] == 1)),
        "lost_suitable_fraction_since_invasion": float(
            np.mean(transition_inv[np.isfinite(transition_inv)] == -1)),
        "final_suitable_centroid_latitude": float(centroid_lat[-1]),
        "allee_n50_bbs_count": n50, "fit": fit_metrics,
        # The three manifolds' fitted coupling. K having its own manifold is what
        # lets covariates -- rather than the disease term -- explain why capacity's
        # spatial pattern differs from fecundity's; these say how much it used that
        # freedom. Prior medians: F-S 0.70, F-K 0.85, S-K 0.70.
        "manifold": {
            "corr_survival_repro": float(np.asarray(sim["rho"])),
            "corr_repro_capacity": float(np.asarray(sim["env_corr_repro_capacity"])),
            "corr_survival_capacity": float(np.asarray(sim["env_corr_survival_capacity"])),
            "loadings": [float(x) for x in np.asarray(sim["manifold_loadings"])],
        },
        # Continental capacity drift. This term is a SAFETY VALVE, not a modeled
        # mechanism -- its prior is strongly concentrated on zero. A fitted trend
        # that moves K by more than ~10-15%, or that sits more than ~3 prior SDs
        # out, means a real temporal signal is not being explained by any mechanism
        # in the model. That is a finding about the model, not a result to report,
        # and the response is to find the missing mechanism rather than loosen the
        # prior. Reported as the multiplier on K (exp of the trend, since K sits
        # near softplus's exponential regime) at each end of the timeline.
        "k_trend": _k_trend_metrics(data, sim),
        "realized_source_sink": source_sink_metrics,
        "disease": disease_metrics,
        "age_structure": age_structure_metrics,
        **response_metrics,
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[map-viz] complete -> {out}")


if __name__ == "__main__":
    main()
