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
from src.model.age_priors import build_model_2d
from src.model.checkpoints import auto_delta_params_to_latents, load_map_params
from src.model.data_loading import load_data
from src.model.runtime_diagnostics import memory_snapshot, require_gpu


def local_growth_lambda(Sa, Sj, F):
    """Dominant local eigenvalue for the forward model's census order.

    Adults survive before reproducing, hence the fecundity entry is ``F*Sa``.
    This differs from the older ``F*Sj/(1-Sa)`` surrogate.
    """
    Sa, Sj, F = np.asarray(Sa), np.asarray(Sj), np.asarray(F)
    return (Sa + np.sqrt(np.maximum(Sa ** 2 + 4.0 * F * Sa * Sj, 0.0))) / 2.0


def _grid(flat, rows, cols, shape):
    grid = np.full((flat.shape[0], *shape), np.nan, dtype="float32")
    grid[:, rows, cols] = flat
    return grid


def _window_mean(values, n):
    n = min(n, values.shape[0])
    return np.nanmean(values[-n:], axis=0), np.nanmean(values[:n], axis=0), n


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
              "Q_flat", "expected_obs", "allee_gamma", "n50_raw", "w_env", "rho"]
    predictive = Predictive(build_model_2d, posterior_samples=posterior, return_sites=needed)
    result = predictive(jax.random.PRNGKey(104), data=data, prior_scale=1.0)
    result = jax.block_until_ready(result)
    return {name: np.asarray(value[0]) for name, value in result.items()}


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
    cmap = mcolors.ListedColormap(["#bdbdbd", "#2c7fb8", "#d7301f", "#238443"])
    im = ax[1, 1].imshow(transition, cmap=cmap, vmin=-1, vmax=2)
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
    for a in ax: a.grid(alpha=.25)
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
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)
    residual = np.log1p(predicted) - np.log1p(observed)
    return {"n_observations": int(len(observed)), "log1p_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "log1p_correlation": float(np.corrcoef(np.log1p(observed), np.log1p(predicted))[0, 1])}


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
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[map-viz] complete -> {out}")


if __name__ == "__main__":
    main()
