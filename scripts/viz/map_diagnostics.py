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
grids figure 07 draws -- modern lambda at N=K, modern fundamental lambda, modern
K, the viability mask, the Allee suppression at K, and the averaging window's year
span. These used to exist only as pixels in the PNG, so nothing could compare
viability structure between runs (e.g. across a dispersal-distance sweep) or
re-plot it without a GPU and a full model reconstruction.

``source_mask`` in that file is the FOLD criterion -- a positive equilibrium exists
-- not ``lambda > 1``; see ``age_model_math.allee_viability`` for the derivation and
``_write_source_sink_fields`` for why the key name was kept.

``07_realized_source_sink.png`` is the deliberate REALIZED counterpart --
same Sa/Sj/Fmax but WITH density-dependence, the Allee effect, AND the K-only
disease depression (mycoplasmal conjunctivitis, which has no covariate of its
own) -- so the two can be compared directly; see
``src/vis/age_model_math.py`` for the shared, samples-axis-agnostic math both
draw on (also the seam for a future MCMC-sample version of this script).

``15_counterfactual_no_invasion.png`` / ``16_counterfactual_animation`` re-run the
forward model at the SAME MAP point with the 1940 NYC release deleted
(``inv_pop -> 0``), so the difference is attributable to the release alone. Two
counterfactual arms are simulated because one cannot answer both questions: the
epizootic is held fixed for the attribution map (only ``inv_pop`` differs, so the
difference is cleanly signed) and removed for the coherent "no release" world (no
dense eastern population means no 1994 outbreak). Differencing against the
no-epizootic arm alone would confound two interventions with opposite signs on
density. Both arms are gradient-free forward passes, which is also why none of this
touches a file ``age_run_map._run_fingerprint`` hashes. See
``simulate_no_invasion_counterfactual`` for the full argument and the extrapolation
caveat.

``17_barrier_crossing.png`` is the directed cost of crossing the Great Plains, both
ways, from the linearized annual operator restricted to the barrier -- see
``src/vis/barrier_crossing`` for the derivation. It is OPTIONAL: it needs a zone
raster built by ``scripts/build_great_plains_mask.py`` from an ecoregion shapefile
that is not in git and not in ``download_all.sh``, so ``plot_barrier_crossing``
returns None and says why when the raster is absent, and ``metrics.json`` carries
``barrier_crossing: null``. A missing optional input must never cost a whole
diagnostics run.

``10_age_structure.png`` compares theoretical equilibrium age structure
(local rho implied by the fitted vital rates, assuming the system has
settled -- no invasion-front/transient history) against REALIZED age
structure (Nj/(Na+Nj) from the actual forward-simulated Na_grid/Nj_grid age
pools, which does carry that history); a gap between them, especially near a
still-advancing range edge, is the expected signature of non-equilibrium age
structure at an invasion front. Na_grid/Nj_grid cost nothing extra during
MAP/SVI optimization -- see forward_sim_age_structured's docstring for why.

``01b_niche_change_since_invasion.png`` and
``06b_environmental_drivers_since_invasion.png`` repeat 01a and 06a with the
baseline era anchored at the release (1940-1955) instead of the start of the
model timeline (1902-1915). The 1902 baseline is "before anything happened," but it
is also 38 years of climate change removed from the release, so change measured
against it is not change the invasion experienced; the 1940-anchored pair is the
one to read for invasion-relative statements. ``metrics.json`` carries both.

The ``a``/``b`` suffixes exist so each pair sorts ADJACENT in the output listing:
they are the same figure differing only in baseline era, and numbering them 01/13
and 06/14 put twelve unrelated figures between the two halves of one comparison.

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
import matplotlib.image as mimage
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import transform as crs_transform
from numpyro.infer import Predictive

from src.config_utils import load_age_model_config, load_data_config
from src.temporal import load_timeline
from src.model.age_priors import build_model_2d
from src.model.age_forward import forward_sim_age_structured
from src.model.checkpoints import auto_delta_params_to_latents, load_map_params
from src.model.data_loading import load_data
from src.model.runtime_diagnostics import memory_snapshot, require_gpu
from src.data.preprocess.great_plains import read_zone_raster
from src.vis.age_model_math import (
    ERAS, add_timeline_markers, allee_viability, demographic_params, era_mean, era_span,
    eras_from_window, local_growth_lambda, rates_from_manifolds, realized_equilibrium,
    response_curve_fields, scatter_to_grid,
)
from src.vis.barrier_crossing import (
    DIRECTIONS, crossing_gain, directional_q_contrast, edge_correction_summary,
    low_density_departure_probability, modern_dispersal_fields, propagule_pressure,
    q_asymmetry_attribution,
)

# Back-compat local alias (this file's plot functions historically used this
# private name; kept so the diffs below stay small).
_grid = scatter_to_grid


def _map_grid(nrows, ncols, shape, panel_w=3.4, header=0.55, right_pad=0.0,
              cbar_frac=0.0):
    """A map-panel grid sized to the RASTER's aspect, packed tightly.

    ``imshow`` fixes the image's aspect, so whenever the AXES box is taller than
    ``width * ny/nx`` the surplus becomes dead space inside the axes -- padding
    that no outer layout engine can reclaim, because as far as it is concerned
    the axes is fully occupied. Sizing the figure from the grid's own aspect
    (133x224 => 0.59) removes it at the source.

    ``cbar_frac`` is the share of each panel's width that a per-panel colorbar
    consumes: constrained_layout takes the colorbar out of the panel, so the map
    is drawn in the REMAINDER and the height must be computed from that
    remainder, not from ``panel_w``. Getting this wrong is what left a tall white
    band above and below every map in the first pass. Pass 0.0 when the figure
    has no per-panel colorbar (use ``right_pad`` instead for one shared bar).
    """
    ny, nx = shape
    image_w = panel_w * (1.0 - cbar_frac)
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, layout="constrained",
                           figsize=(panel_w * ncols + right_pad,
                                    image_w * (ny / nx) * nrows + header))
    # Tighter than the constrained_layout defaults: these panels share a frame and
    # a colour scale, so the gaps between them carry no information.
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.01, hspace=0.03)
    return fig, ax


def _snap_map_height(fig, iters=4, tol=0.02):
    """Shrink the figure until each map row's allocated cell matches its image.

    ``_map_grid`` predicts the height a row needs, but the prediction depends on
    how much width the colorbars actually take, which is only known once the
    figure is laid out. Whatever surplus remains shows up as a grid cell taller
    than the image drawn in it -- the axes then shrinks to the image (aspect is
    fixed) and the COLORBAR, sized to the cell, ends up taller than the map. That
    is the "colorbar taller than the figure" waste, and it cannot be cropped away
    afterwards because it sits between panels.

    Measuring ``get_position(original=True)`` (the allocated cell) against
    ``get_position(original=False)`` (the box after aspect was applied) recovers
    the surplus directly, so this removes it by construction instead of by tuning
    a per-figure fudge factor. Iterates because shrinking the figure slightly
    changes the colorbars' relative width.
    """
    for _ in range(iters):
        fig.canvas.draw()
        per_row = {}
        for a in fig.axes:
            if not any(isinstance(c, mimage.AxesImage) for c in a.get_children()):
                continue
            alloc, drawn = a.get_position(original=True), a.get_position(original=False)
            # Group by the row's allocated top edge; panels in a row share a cell
            # height, and the TALLEST image in the row is what constrains the shrink.
            key = round(alloc.y1, 3)
            surplus = (alloc.height - drawn.height) * fig.get_figheight()
            per_row[key] = min(per_row.get(key, surplus), surplus)
        total = sum(v for v in per_row.values() if v > 0)
        if total <= tol:
            return
        fig.set_size_inches(fig.get_figwidth(), fig.get_figheight() - total)


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
              "manifold_loadings", "w_k_trend", "k_level", "k_level_route_counts",
              "gamma_a", "gamma_j", "gamma_f", "gamma_k",
              # Needed to re-run the forward simulator for the no-invasion
              # counterfactual (figures 15/16): the t=0 native seed, the 1940 pulse
              # to zero out, and the pre-disease capacity.
              "initpop_seeded", "inv_pop_relative", "K_base_flat",
              "disease_k_half_route_counts", "disease_hill_n",
              "disease_severity_map", "disease_mu_sev", "disease_b_late",
              "disease_w_lag", "disease_lag0", "disease_tau", "disease_rec",
              "disease_tau_rec", "Na_grid", "Nj_grid"]
    predictive = Predictive(build_model_2d, posterior_samples=posterior, return_sites=needed)
    result = predictive(jax.random.PRNGKey(104), data=data, prior_scale=1.0)
    result = jax.block_until_ready(result)
    sim = {name: np.asarray(value[0]) for name, value in result.items()}
    # auto_delta_params_to_latents returns only SAMPLED sites, but several things the
    # response curves need are numpyro.deterministic: w_env is built from
    # manifold_factor/manifold_idio/loadings, k_level is softplus(alpha_k), and the
    # gamma_* slopes are constants fixed at 1 (their amplitude moved into w_scale).
    # The gammas are the reason this list must be kept in sync with what
    # age_model_math reads -- omitting them raised KeyError('gamma_a_raw') from
    # response_curve_fields, which killed diagnostics BEFORE the source/sink fields
    # and metrics.json were written. Fold the deterministic values in so plotting
    # code has a single place to look and cannot silently read a stale name.
    #
    # It happened a SECOND time with gamma_j_diff, which moved sampled -> deterministic when
    # juvenile survival got its own manifold. Any name that makes that migration has to be added
    # here, or every reader pulling it out of this dict gets a KeyError.
    sim["latents"] = dict(latents)
    for name in ("w_env", "k_level", "gamma_a", "gamma_j", "gamma_f", "gamma_k",
                 "gamma_j_diff", "manifold_loadings", "manifold_communality",
                 "env_corr_survival_adult_juv"):
        if name in sim:
            sim["latents"][name] = sim[name]
    return sim


def plot_modern_niche(lam, years, rows, cols, shape, out, modern_era, baseline_era):
    """Modern vs baseline fundamental niche, and the transition between them.

    ``baseline_era="early"`` (1902-1915) anchors the comparison at the start of
    the model timeline. ``"invasion"`` (1940-1955) anchors it at the release
    instead -- the 1902 baseline also carries 38 years of climate change that has
    nothing to do with the invasion, so the two figures answer genuinely
    different questions and are both produced (see ``age_model_math.ERAS``).
    """
    modern, modern_span, _ = era_mean(lam, years, modern_era)
    early, base_span, _ = era_mean(lam, years, baseline_era)
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
    fig, ax = _map_grid(2, 2, shape, panel_w=5.4, header=1.05, cbar_frac=0.17)
    im = ax[0, 0].imshow(modern_g, cmap="viridis", vmin=lo, vmax=hi)
    ax[0, 0].contour(modern_g, [1.0], colors="white", linewidths=1.0)
    ax[0, 0].set_title(f"Modern intrinsic growth λ ({modern_span[0]}–{modern_span[1]} mean)")
    fig.colorbar(im, ax=ax[0, 0], fraction=.046, label="Post-establishment λ")
    im = ax[0, 1].imshow(early_g, cmap="viridis", vmin=lo, vmax=hi)
    ax[0, 1].contour(early_g, [1.0], colors="white", linewidths=1.0)
    ax[0, 1].set_title(f"Baseline intrinsic growth λ ({base_span[0]}–{base_span[1]} mean)")
    fig.colorbar(im, ax=ax[0, 1], fraction=.046, label="Post-establishment λ")
    im = ax[1, 0].imshow(change, cmap="RdBu_r", vmin=-delta_lim, vmax=delta_lim)
    ax[1, 0].set_title(f"Change in intrinsic growth ({modern_span[0]}–{modern_span[1]} "
                       f"− {base_span[0]}–{base_span[1]})")
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
    fig.suptitle(f"House Finch fundamental niche: local demographic potential "
                 f"({modern_span[0]}–{modern_span[1]} vs {base_span[0]}–{base_span[1]})")
    _snap_map_height(fig)
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
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
    ax[0].plot(years, fraction * 100, color="#238443", lw=2,
               label="Suitable land (% of cells with λ > 1)")
    ax[0].set(ylabel="Suitable land (%)", title="Trajectory of local demographic potential")
    ax0b = ax[0].twinx()
    ax0b.plot(years, mean_lambda, color="#54278f", lw=1.8, label="Mean λ over all land")
    ax0b.set_ylabel("Mean λ")
    ax[1].plot(years, centroid_lat, color="#2c7fb8", lw=2,
               label="Mean latitude of cells with λ > 1")
    ax[1].set(xlabel="Year", ylabel="Mean latitude (°N)", title="Centroid of land with λ > 1")
    # The top panel's two lines live on DIFFERENT axes, so neither axis's own
    # legend() would name both. Colour each axis to match its line as well, so the
    # left/right scales are unambiguous even before reading the legend.
    ax[0].yaxis.label.set_color("#238443")
    ax[0].tick_params(axis="y", colors="#238443")
    ax0b.yaxis.label.set_color("#54278f")
    ax0b.tick_params(axis="y", colors="#54278f")
    top_lines = ax[0].get_lines() + ax0b.get_lines()
    ax[0].legend(top_lines, [ln.get_label() for ln in top_lines],
                 loc="upper left", fontsize=8, frameon=False)
    ax[1].legend(loc="upper left", fontsize=8, frameon=False)
    for a in ax:
        a.grid(alpha=.25)
        add_timeline_markers(a)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)
    return fraction, mean_lambda, centroid_lat


def plot_modern_rate_maps(sim, years, rows, cols, shape, out, era):
    # ``unit`` is the ONLY source of a colorbar label. It used to fall back to the
    # panel title, which printed the same words twice per panel; a colorbar whose
    # quantity is already named by the title above it needs no label at all.
    fields = [("Adult survival", sim["Sa_flat"], "viridis", None),
              ("Juvenile survival", sim["Sj_flat"], "viridis", None),
              ("Fecundity ceiling", sim["Fmax_flat"], "magma", "juveniles adult⁻¹ yr⁻¹"),
              ("Carrying capacity", sim["K_flat"], "magma", "relative units")]
    fig, ax = _map_grid(2, 2, shape, panel_w=5.0, header=0.85, cbar_frac=0.17)
    for axis, (label, field, cmap, unit) in zip(ax.flat, fields):
        avg, span, _ = era_mean(field, years, era)
        grid = _grid(avg[None], rows, cols, shape)[0]
        lo, hi = np.nanpercentile(grid, [2, 98])
        image = axis.imshow(grid, cmap=cmap, vmin=lo, vmax=hi)
        axis.set_title(label); axis.axis("off")
        fig.colorbar(image, ax=axis, fraction=.046, label=unit)
    fig.suptitle(f"House Finch Vital Rates ({span[0]}–{span[1]} mean)")
    _snap_map_height(fig)
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_fit_diagnostics(sim, data, years, out):
    observed = np.asarray(data["observed_results"])
    predicted = np.asarray(sim["expected_obs"])
    t = np.asarray(data["obs_time_indices"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    lo, lp = np.log1p(observed), np.log1p(predicted)
    # LOG counts. Route-years pile up near zero by orders of magnitude, so a linear
    # count scale renders the whole informative range as one flat colour and the
    # panel reads as an undifferentiated blob.
    hb = ax[0].hexbin(lo, lp, gridsize=(60, 45), bins="log", mincnt=1,
                      cmap="magma", linewidths=0)
    lim = float(max(np.nanmax(lo), np.nanmax(lp))) * 1.02
    ax[0].plot([0, lim], [0, lim], color="0.35", lw=1, ls="--", zorder=3)
    # The thing the panel actually exists to show: conditional bias. Median fitted
    # value per decile of observed, so systematic over/under-prediction is legible
    # independently of how many route-years sit in each bin.
    edges = np.unique(np.nanpercentile(lo, np.linspace(0, 100, 11)))
    if edges.size > 2:
        which = np.clip(np.digitize(lo, edges[1:-1]), 0, edges.size - 2)
        ctr, med, q1, q3 = [], [], [], []
        for b in range(edges.size - 1):
            sel = lp[which == b]
            if sel.size < 5:
                continue
            ctr.append(0.5 * (edges[b] + edges[b + 1]))
            med.append(np.median(sel)); q1.append(np.percentile(sel, 25)); q3.append(np.percentile(sel, 75))
        if ctr:
            # Cyan, not white: magma's dense end is pale, and a white line vanished
            # into exactly the region with the most data.
            ax[0].fill_between(ctr, q1, q3, color="#00d9ff", alpha=.22, zorder=4)
            ax[0].plot(ctr, med, color="#00d9ff", lw=2.0, marker="o", ms=3.5, zorder=5,
                       path_effects=[pe.Stroke(linewidth=3.6, foreground="#08306b"), pe.Normal()],
                       label="Median fitted per observed decile (IQR band)")
            ax[0].legend(loc="upper left", fontsize=7, framealpha=.8, facecolor="white",
                         edgecolor="0.7", labelcolor="0.15")
    ax[0].set(xlabel="log(1 + observed BBS count)", ylabel="log(1 + fitted mean)",
              title="Observation-scale calibration", xlim=(0, lim), ylim=(0, lim))
    ax[0].set_aspect("equal")
    _resid = np.log1p(predicted) - np.log1p(observed)
    ax[0].annotate(f"RMSE={np.sqrt(np.mean(_resid ** 2)):.3f}\n"
                   f"r={np.corrcoef(lo, lp)[0, 1]:.3f}\nn={len(observed):,}",
                   xy=(.97, .03), xycoords="axes fraction", ha="right", va="bottom",
                   fontsize=7.5, color="0.15",
                   bbox=dict(fc="white", ec="0.7", alpha=.75, boxstyle="round,pad=0.3"))
    fig.colorbar(hb, ax=ax[0], label="Route-years per hex (log scale)")
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


def plot_response_curves(sim, out, top_n=6, names=None):
    """Sweep the top-|weight| Z features and plot Sa/Sj/Fmax/K response curves.

    Corrects a stale bug in the deprecated ``visualize_age_model.py`` (which
    used ``exp`` instead of ``softplus`` for Fmax) and adds the K response
    curve that script never plotted. See ``age_model_math.response_curve_fields``.
    """
    latents = sim["latents"]
    w_env = np.asarray(latents["w_env"])
    importance = np.abs(w_env).sum(axis=1)
    top_idx = np.argsort(importance)[::-1][:min(top_n, w_env.shape[0])]
    names = list(names) if names is not None else [f"Z_{i}" for i in range(w_env.shape[0])]
    z_sweep = np.linspace(-3.0, 3.0, 60)

    ncols = 3
    nrows = int(np.ceil(len(top_idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.6 * nrows), squeeze=False)
    for pos, (axis, idx) in enumerate(zip(axes.flat, top_idx)):
        curves = response_curve_fields(latents, z_sweep, int(idx))
        axis.plot(z_sweep, curves["Sa"], color="navy", lw=1.8)
        axis.plot(z_sweep, curves["Sj"], color="royalblue", lw=1.8, linestyle="--")
        axis.set_ylim(0, 1)
        axis.set_title(f"{names[idx]}  (|w_env|={importance[idx]:.2f})", fontsize=9)
        axis2 = axis.twinx()
        axis2.plot(z_sweep, curves["Fmax"], color="darkorange", lw=1.6)
        axis2.plot(z_sweep, curves["K"], color="seagreen", lw=1.6, linestyle=":")
        # Label the axes once per edge rather than on every panel: the twin axis
        # carried no label at all, so the right-hand scale was unreadable.
        if pos % ncols == 0:
            axis.set_ylabel("survival probability", fontsize=8)
        if pos % ncols == ncols - 1:
            axis2.set_ylabel("Fmax / K (relative units)", fontsize=8)
        if pos >= len(top_idx) - ncols:
            axis.set_xlabel("Z feature value (SD)", fontsize=8)
    for axis in axes.flat[len(top_idx):]:
        axis.axis("off")

    handles = [
        plt.Line2D([], [], color="navy", label="Adult survival (Sa)"),
        plt.Line2D([], [], color="royalblue", linestyle="--", label="Juvenile survival (Sj)"),
        plt.Line2D([], [], color="darkorange", label="Fecundity ceiling (Fmax)"),
        plt.Line2D([], [], color="seagreen", linestyle=":", label="Carrying capacity (K)"),
    ]
    # A NEGATIVE bbox_to_anchor puts the legend outside the figure, and tight_layout
    # (which does not account for fig.legend) then cropped it away. Reserve the strip
    # explicitly with subplots_adjust and anchor the legend inside it instead.
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("Demographic response curves (top Z features by |w_env|)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.subplots_adjust(bottom=max(0.10, 0.34 / nrows))
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    return {"response_curve_top_features": [int(i) for i in top_idx]}


def z_feature_names(cfg, n_features):
    """Human-readable Z feature labels, or ``Z_0..Z_{M-1}`` when unavailable.

    Labels live alongside the path-integrated dispersal covariates
    (``Z_disp_*.npz``, key ``labels``), which is the same source
    ``analysis/analyze_svi.py`` reads. Padded/truncated to ``n_features`` rather
    than raising: a label mismatch must never cost a figure.
    """
    names = []
    try:
        disp = sorted(Path(cfg["path_diagnostics_dir"]).glob("Z_disp_*.npz"))
        if disp:
            with np.load(disp[0], allow_pickle=True) as loader:
                if "labels" in loader:
                    names = [str(x) for x in loader["labels"]]
    except (KeyError, OSError, ValueError):
        names = []
    names = list(names[:n_features])
    names += [f"Z_{i}" for i in range(len(names), n_features)]
    return names


def plot_w_env_ranking(sim, out, names=None):
    """Signed environmental weights per Z feature, ranked by total magnitude.

    Figure 05 shows the SHAPE of each top feature's response but never its sign
    or its standing relative to the others; ``|w_env|`` appears only as a number
    in a panel title. This is the ranking itself: every feature, all three
    manifolds, signed, on one axis.

    ``w_env`` columns are ``beta_s`` (adult survival), ``beta_r`` (reproduction),
    ``beta_k`` (capacity) and ``beta_sj`` (juvenile survival, APPENDED 4th column under the
    rank-2 manifold prior; absent in older checkpoints). Because the links are monotone
    increasing (sigmoid, softplus), the SIGN here is the sign of the effect on the
    corresponding vital rate -- positive beta_s means more of this feature raises survival.
    """
    w_env = np.asarray(sim["latents"]["w_env"])
    M = w_env.shape[0]
    beta = {"Adult survival (β_s)": w_env[:, 0], "Reproduction (β_r)": w_env[:, 1],
            "Capacity (β_k)": w_env[:, 2] if w_env.shape[1] > 2 else w_env[:, 1]}
    # Juvenile survival is the APPENDED 4th column; absent in pre-rank-2 checkpoints.
    if w_env.shape[1] > 3:
        beta["Juvenile survival (β_sj)"] = w_env[:, 3]
    names = list(names) if names is not None else [f"Z_{i}" for i in range(M)]
    importance = np.abs(w_env).sum(axis=1)
    order = np.argsort(importance)          # ascending: barh draws bottom-up
    colors = {"Adult survival (β_s)": "#2166ac", "Reproduction (β_r)": "#d95f0e",
              "Capacity (β_k)": "#238443", "Juvenile survival (β_sj)": "#6a51a3"}

    y = np.arange(M)
    # Bar group height and offset both derive from len(beta): it is 3 for a pre-rank-2
    # checkpoint and 4 once juvenile survival has its own column, and a hardcoded (k - 1)
    # offset silently mis-centred the group in the 4-bar case.
    n_bars = len(beta)
    height = 0.78 / n_bars
    fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.42 * M + 1.6)))
    for k, (label, values) in enumerate(beta.items()):
        ax.barh(y + (k - (n_bars - 1) / 2.0) * height, values[order], height=height,
                color=colors[label], label=label, edgecolor="none")
    ax.axvline(0, color="0.25", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{names[i]}  ({importance[i]:.2f})" for i in order], fontsize=8)
    ax.set_xlabel("Fitted weight (signed; manifold units)")
    ax.set_title("Environmental weights by Z feature\n"
                 "sorted by Σ|w_env|, shown in parentheses; positive = raises that rate")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)
    return {"w_env_ranking": [
        {"index": int(i), "name": names[i], "abs_total": float(importance[i]),
         "beta_s": float(w_env[i, 0]), "beta_r": float(w_env[i, 1]),
         "beta_sj": float(w_env[i, 3]) if w_env.shape[1] > 3 else None,
         "beta_k": float(beta["Capacity (β_k)"][i])}
        for i in order[::-1]]}


def _lambda_from_manifolds(p, H_s, H_r):
    """Fundamental λ implied by survival/reproduction manifold values."""
    r = rates_from_manifolds(p, H_s, H_r)
    return local_growth_lambda(r["Sa"], r["Sj"], r["Fmax"])


def plot_z_feature_attribution(data, sim, years, rows, cols, shape, out,
                               modern_era, baseline_era, top_n=3, names=None):
    """Top Z features in each era, how they moved, and what that did to λ.

    Figures 06/14 say WHICH feature dominates a cell but not whether it changed;
    figure 01a says λ changed but not why. This joins the two: one row per top
    feature, columns ``[baseline Z] [modern Z] [ΔZ] [Δλ attributable to ΔZ]``.

    THE λ COLUMN IS A SINGLE-FEATURE COUNTERFACTUAL, not a decomposition. For
    feature m it moves ONLY that feature from its baseline to its modern value,
    holding every other feature at baseline, and re-evaluates the model's own
    links::

        H_s = Z_base . beta_s + (Z_m^modern - Z_m^base) * beta_s[m]

    Because sigmoid/softplus are nonlinear, the per-feature Δλ do NOT sum to the
    total Δλ -- interactions live in the residual, which is reported in
    ``metrics.json`` rather than hidden. A large residual means the features move
    λ jointly and no single-feature attribution is trustworthy on its own.

    Both λ fields here are rebuilt from ERA-MEAN Z, so they differ slightly from
    the era mean of the per-year λ in figure 01a (Jensen's inequality); the
    comparison within this figure is self-consistent, which is what it is for.
    """
    p = demographic_params(sim["latents"])
    beta_s, beta_r = p["beta_s"], p["beta_r"]
    i0, i1, modern_span = era_span(modern_era, years)
    b0, b1, base_span = era_span(baseline_era, years)
    Z_full = data["Z_gathered"]
    Z_mod = np.asarray(Z_full[i0:i1]).mean(axis=0)     # (N_land, M)
    Z_base = np.asarray(Z_full[b0:b1]).mean(axis=0)
    dZ = Z_mod - Z_base

    H_s_base, H_r_base = Z_base @ beta_s, Z_base @ beta_r
    lam_base = _lambda_from_manifolds(p, H_s_base, H_r_base)
    lam_mod = _lambda_from_manifolds(p, Z_mod @ beta_s, Z_mod @ beta_r)

    def attributed(m):
        return _lambda_from_manifolds(
            p, H_s_base + dZ[:, m] * beta_s[m], H_r_base + dZ[:, m] * beta_r[m]) - lam_base

    M = Z_mod.shape[1]
    names = list(names) if names is not None else [f"Z_{i}" for i in range(M)]
    # Rank by how much this feature actually MOVED λ here, not by |w_env|: a
    # heavily-weighted feature that never changed explains none of the change.
    per_feature = [attributed(m) for m in range(M)]
    moved = np.array([np.nanmean(np.abs(a)) for a in per_feature])
    top_idx = np.argsort(moved)[::-1][:min(top_n, M)]
    residual = (lam_mod - lam_base) - np.sum(per_feature, axis=0)

    def g(flat):
        return _grid(np.asarray(flat)[None], rows, cols, shape)[0]

    zlim = float(np.nanpercentile(np.abs(np.r_[Z_base[:, top_idx], Z_mod[:, top_idx]]), 98)) or 1.0
    nrows = len(top_idx)
    fig, axes = _map_grid(nrows, 4, shape, panel_w=4.7, header=1.45, cbar_frac=0.19)
    for r, m in enumerate(top_idx):
        dz_lim = max(float(np.nanpercentile(np.abs(dZ[:, m]), 98)), 1e-6)
        dl = per_feature[m]
        dl_lim = max(float(np.nanpercentile(np.abs(dl), 98)), 1e-6)
        panels = [
            (g(Z_base[:, m]), "RdYlBu_r", -zlim, zlim,
             f"{names[m]}: {base_span[0]}–{base_span[1]}", "Z (SD)"),
            (g(Z_mod[:, m]), "RdYlBu_r", -zlim, zlim,
             f"{names[m]}: {modern_span[0]}–{modern_span[1]}", "Z (SD)"),
            (g(dZ[:, m]), "RdBu_r", -dz_lim, dz_lim, f"Δ{names[m]} (modern − baseline)", "ΔZ (SD)"),
            (g(dl), "RdBu_r", -dl_lim, dl_lim,
             f"Δλ attributable to Δ{names[m]}\n(this feature moved alone)", "Δλ"),
        ]
        for c, (grid, cmap, lo, hi, title, cbl) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(grid, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(title, fontsize=9); ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=.042, label=cbl)
        axes[r, 3].set_title(
            axes[r, 3].get_title() + f"\nmean |Δλ| = {moved[m]:.3g}, "
            f"β_s={beta_s[m]:+.2f} β_r={beta_r[m]:+.2f}", fontsize=8.5)

    tot = float(np.nanmean(np.abs(lam_mod - lam_base)))
    res = float(np.nanmean(np.abs(residual)))
    fig.suptitle(
        f"Z-feature attribution of niche change, {base_span[0]}–{base_span[1]} → "
        f"{modern_span[0]}–{modern_span[1]}\n"
        f"single-feature counterfactuals; nonlinear links mean these do NOT sum to the "
        f"total (mean |Δλ| total = {tot:.3g}, unattributed interaction residual = {res:.3g})",
        fontsize=11)
    _snap_map_height(fig)
    fig.savefig(out, dpi=170); plt.close(fig)
    return {
        "baseline_era": list(base_span), "modern_era": list(modern_span),
        "top_features": [{"index": int(m), "name": names[m],
                          "mean_abs_delta_lambda": float(moved[m])} for m in top_idx],
        "mean_abs_total_delta_lambda": tot,
        # Interaction term the single-feature counterfactuals cannot capture. Large
        # relative to the total => read the per-feature maps as indicative only.
        "mean_abs_unattributed_residual": res,
        "residual_fraction_of_total": float(res / tot) if tot > 0 else float("nan"),
    }


def plot_environmental_drivers_limits(data, sim, years, rows, cols, shape, out,
                                      modern_era, baseline_era, names=None):
    """Which Z feature contributes most to the survival/reproduction manifold, per cell.

    ``baseline_era`` anchors the lower pair of panels, exactly as in
    ``plot_modern_niche``: "early" (1902-1915) or "invasion" (1940-1955), to ask
    what the drivers looked like when the species actually arrived.
    """
    latents = sim["latents"]
    w_env = np.asarray(latents["w_env"])
    beta_s, beta_r = w_env[:, 0], w_env[:, 1]
    # Z_gathered is (time, N_land, M) and typically device-resident; slice each era
    # BEFORE pulling to host so a full-array transfer is never needed.
    Z_full = data["Z_gathered"]
    i0, i1, modern_span = era_span(modern_era, years)
    b0, b1, base_span = era_span(baseline_era, years)
    Z = np.asarray(Z_full[i0:i1])
    Z_early = np.asarray(Z_full[b0:b1])

    def dominant_feature(Z_window, beta):
        contrib_mean = (Z_window * beta[None, None, :]).mean(axis=0)  # (N_land, M)
        return np.argmax(contrib_mean, axis=1).astype("float32")

    panels = [
        (dominant_feature(Z, beta_s), f"Survival driver ({modern_span[0]}–{modern_span[1]})"),
        (dominant_feature(Z, beta_r), f"Reproduction driver ({modern_span[0]}–{modern_span[1]})"),
        (dominant_feature(Z_early, beta_s), f"Survival driver ({base_span[0]}–{base_span[1]})"),
        (dominant_feature(Z_early, beta_r), f"Reproduction driver ({base_span[0]}–{base_span[1]})"),
    ]
    M = w_env.shape[0]
    cmap = plt.get_cmap("tab20", M)
    fig, ax = _map_grid(2, 2, shape, panel_w=4.6, header=0.75, right_pad=1.1)
    im = None
    for axis, (idx_flat, title) in zip(ax.flat, panels):
        grid = _grid(idx_flat[None], rows, cols, shape)[0]
        im = axis.imshow(grid, cmap=cmap, vmin=-0.5, vmax=M - 0.5)
        axis.set_title(title, fontsize=10); axis.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=.025, ticks=range(M))
    # Name the categories. A bare index forced the reader to cross-reference 05b
    # to learn what any colour on this map actually means.
    if names is not None:
        cbar.ax.set_yticklabels(list(names)[:M], fontsize=7)
    cbar.set_label("Dominant Z feature")
    fig.suptitle(f"Dominant environmental driver by cell "
                 f"({modern_span[0]}–{modern_span[1]} vs {base_span[0]}–{base_span[1]})")
    _snap_map_height(fig)
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def _write_source_sink_fields(npz_path, lam_realized, lam_fundamental, K_modern,
                              year_first, year_last, n_years, ref_raster=None,
                              viable=None, allee_suppression=None, fundamental_viable=None):
    """Persist the modern source/sink grids as .npz (+ a GeoTIFF beside it).

    Kept deliberately small (a handful of MB): the three window-mean grids that
    ``07_realized_source_sink.png`` draws, plus the window's year span so a
    consumer can verify two runs were averaged over the same years before
    differencing them. The GeoTIFF is georeferenced from the same
    ``grid.ref_raster`` the trajectory plot uses, so the field drops straight into
    GIS; band order matches ``band_names``.
    """
    npz_path = Path(npz_path)
    # source_mask is the FOLD criterion (age_model_math.allee_viability), not
    # lam_realized > 1. The key name is unchanged because consumers key on it
    # (scripts/viz/juv_mdd_sweep_summary.py), but the semantics changed: it used to
    # threshold lambda at N = K, which c pins at exactly 1, so the old classification
    # turned on the 1e-6 guard inside c rather than on demography. equilibrium_exists
    # is the same array under a name that says what it means.
    finite = np.isfinite(lam_realized)
    mask = np.where(finite, viable, False) if viable is not None else \
        np.where(finite, lam_realized > 1.0, False)
    extra = {}
    if allee_suppression is not None:
        extra["allee_suppression_at_K"] = allee_suppression.astype("float32")
    if fundamental_viable is not None:
        # Cells the fundamental niche calls suitable but that cannot hold a
        # population at any density -- the sanity check this figure exists for.
        extra["allee_dead_mask"] = np.where(finite, fundamental_viable & ~mask, False)
    np.savez_compressed(
        npz_path,
        lam_realized_modern=lam_realized.astype("float32"),
        lam_fundamental_modern=lam_fundamental.astype("float32"),
        K_modern=K_modern.astype("float32"),
        source_mask=mask,
        equilibrium_exists=mask,
        window_years=np.array([int(year_first), int(year_last)]),
        window_n=np.int32(n_years),
        **extra,
    )
    if ref_raster is None:
        return
    bands = [("lam_realized_modern", lam_realized,
              "Growth rate at carrying capacity, lambda(N=K), era mean. "
              "DO NOT THRESHOLD AT 1: c is solved so lambda=1 at N=K, so this is "
              "pinned at 1 wherever the Allee factor saturates. The informative "
              "content is the SHORTFALL below 1 (the Allee cost). For source/sink "
              "classification use the fold criterion in the sibling .npz."),
             ("lam_fundamental_modern", lam_fundamental,
              "Fundamental-niche lambda: density-independent, dispersal-free, "
              "Allee-free dominant eigenvalue of [[Sa, Sj], [Fmax*Sa, 0]], era mean. "
              "This one IS meaningfully thresholded at 1."),
             ("K_modern", K_modern,
              "Carrying capacity in relative units (route counts / pop_scalar), era "
              "mean, disease effect INCLUDED.")]
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
        for i, (name, band, description) in enumerate(bands, start=1):
            dst.write(band.astype("float32"), i)
            # set_band_description is what QGIS/gdalinfo actually show in the band
            # list. Without it the only clue to band order was a dataset-level
            # band_names tag that no GIS surfaces, so the file was unreadable.
            dst.set_band_description(i, name)
            dst.update_tags(i, name=name, description=description)
        dst.update_tags(band_names=",".join(n for n, _, _ in bands),
                        era=f"{int(year_first)}-{int(year_last)}",
                        era_n_years=str(int(n_years)),
                        # Kept for readers written against the old tag name.
                        window=f"{int(year_first)}-{int(year_last)}")
    _write_source_sink_readme(npz_path, bands, year_first, year_last, n_years)


def _write_source_sink_readme(npz_path, bands, year_first, year_last, n_years):
    """Markdown sidecar naming the bands, the era, and the classification rule.

    The GeoTIFF's own band descriptions cover a GIS session; this covers the case
    someone finds the file on disk months later and needs to know which lambda is
    which and why one of them must not be thresholded at 1.
    """
    npz_path = Path(npz_path)
    lines = [f"# {npz_path.stem}", "",
             f"Era averaged: **{int(year_first)}-{int(year_last)}** ({int(n_years)} years).",
             "", f"Written by `scripts/viz/map_diagnostics.py` alongside "
             f"`{npz_path.stem.replace('_fields', '')}_realized_source_sink.png`.", "",
             "## GeoTIFF bands", ""]
    for i, (name, _, description) in enumerate(bands, start=1):
        lines += [f"{i}. **`{name}`** — {description}", ""]
    lines += [
        "## Only in the sibling `.npz`", "",
        "- `source_mask` / `equilibrium_exists` — the **fold criterion** "
        "(`age_model_math.allee_viability`): does a positive equilibrium exist at "
        "all, i.e. does `max_N F(N)` clear replacement? This, NOT a threshold on "
        "band 1, is the source/sink classification.",
        "- `allee_dead_mask` — fundamentally suitable (band 2 > 1) but no positive "
        "equilibrium: K sits in the mate-finding regime.",
        "- `allee_suppression_at_K` — `1 - allee_factor(K)`, how close a cell is to "
        "the fold.", "",
        "NaN is nodata throughout; CRS and transform are copied from "
        "`grid.ref_raster`.", ""]
    npz_path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def modern_viability(sim, years, era, k_key="K_flat"):
    """Era-mean K and the fold-criterion viability for the modern era.

    One definition, shared by figure 07 (which classifies on it) and figure 15
    (which asks which viable cells the counterfactual never reaches), so the two
    can never disagree about what "viable" means.

    Viability is evaluated on the window-MEAN environment rather than per-year and
    then averaged: the scan in ``allee_viability`` costs n_grid floats per cell, so
    the full (time, cell) stack would be ~100x the memory for a question about the
    modern window. Consistent with how every other "modern" field is built here.

    ``k_key`` selects the capacity: ``K_flat`` (as fitted, disease included) for the
    actual world, ``K_base_flat`` for the no-epizootic counterfactual. Viability is a
    function of K, so a counterfactual world with no disease has its own -- larger --
    viable area, and asking "which viable cells went unreached" only makes sense
    against the viability of the world being simulated.
    """
    K_modern, _, _ = era_mean(np.asarray(sim[k_key]), years, era)
    Sa_m, _, _ = era_mean(np.asarray(sim["Sa_flat"]), years, era)
    Sj_m, _, _ = era_mean(np.asarray(sim["Sj_flat"]), years, era)
    Fmax_m, _, _ = era_mean(np.asarray(sim["Fmax_flat"]), years, era)
    return K_modern, allee_viability(Sa_m, Sj_m, Fmax_m, K_modern,
                                     np.asarray(sim["allee_gamma"]))


def plot_realized_source_sink(sim, lam_fundamental, years, rows, cols, shape, out, era,
                              fields_out=None, ref_raster=None):
    """Realized (density-dependent + Allee) counterpart to the fundamental-niche map.

    Contrasts directly against ``01a_niche_change_since_1902.png``: same
    Sa/Sj/Fmax, but with K and the Allee effect included, so
    ``lambda_realized <= lambda_fundamental`` everywhere.

    ``fields_out`` additionally persists the underlying grids (see
    ``_write_source_sink_fields``). Without it these rasters exist only as pixels
    in the PNG, so nothing downstream -- a dispersal sweep comparing runs, or any
    re-plot -- can get at them without a GPU and a full model reconstruction.

    CLASSIFICATION. The binary panel is the FOLD criterion
    (``age_model_math.allee_viability``): does a positive equilibrium exist at all,
    i.e. does ``max_N F(N)`` clear replacement? It is NOT ``lambda_realized > 1``.
    ``c`` is solved so lambda = 1 at N = K, so that test is degenerate -- it
    classified on the ``1e-6`` guard inside ``c``, which put the contour at
    K ~ 3.7 route counts against a true fold near 0.5-1.2, and it condemned the
    typical occupied cell (fitted level ~2.1 counts) as sink. The third class,
    "Allee-dead", is what the figure exists to expose: fundamentally suitable
    habitat that still cannot hold a population because K sits in the mate-finding
    regime.

    ``lambda_realized`` is still drawn as a continuous field -- it is a meaningful
    quantity (growth rate AT carrying capacity, so <= 1 by construction, with the
    shortfall measuring the Allee cost) as long as it is not thresholded at 1.
    """
    _, _, lam_realized, _ = realized_equilibrium(
        sim["Sa_flat"], sim["Sj_flat"], sim["Fmax_flat"], sim["K_flat"], sim["allee_gamma"]
    )
    modern, span, n = era_mean(lam_realized, years, era)
    modern_g = _grid(modern[None], rows, cols, shape)[0]
    fund_modern, _, _ = era_mean(lam_fundamental, years, era)
    fund_g = _grid(fund_modern[None], rows, cols, shape)[0]

    K_modern, viab = modern_viability(sim, years, era)
    viable_g = _grid(viab["viable"].astype(float)[None], rows, cols, shape)[0]
    fund_ok_g = _grid(viab["fundamental_viable"].astype(float)[None], rows, cols, shape)[0]
    supp_g = _grid(viab["suppression_at_K"][None], rows, cols, shape)[0]
    if fields_out is not None:
        _write_source_sink_fields(
            fields_out, modern_g, fund_g,
            _grid(K_modern[None], rows, cols, shape)[0],
            span[0], span[1], n, ref_raster,
            viable=viable_g > 0.5, allee_suppression=supp_g,
            fundamental_viable=fund_ok_g > 0.5,
        )

    fig, ax = _map_grid(1, 3, shape, panel_w=5.0, header=1.30, cbar_frac=0.17)
    ax = ax[0]
    # 0 = unsuitable (fails replacement at any density), 1 = Allee-dead (suitable
    # but K too small for a positive equilibrium), 2 = viable source.
    klass = np.where(viable_g > 0.5, 2.0, np.where(fund_ok_g > 0.5, 1.0, 0.0))
    klass = np.where(np.isfinite(modern_g), klass, np.nan)
    ax[0].imshow(klass, cmap=mcolors.ListedColormap(["#d73027", "#fdae61", "#4575b4"]),
                 norm=mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3))
    n_dead = float(np.nansum(klass == 1.0)); n_fund = float(np.nansum(klass >= 1.0))
    ax[0].set_title(f"Viability, fold criterion ({span[0]}–{span[1]})\n"
                    f"{n_dead / max(n_fund, 1):.1%} of suitable habitat is Allee-dead")
    ax[0].legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#4575b4",
                      label="Viable source: max$_N$ F(N) clears replacement"),
        plt.Rectangle((0, 0), 1, 1, color="#fdae61",
                      label="Allee-dead: λ$_{fund}$ > 1 but no positive equilibrium"),
        plt.Rectangle((0, 0), 1, 1, color="#d73027", label="Unsuitable: λ$_{fund}$ ≤ 1"),
    ], loc="lower left", fontsize=7, frameon=True)

    lo, hi = np.nanpercentile(modern_g, [2, 98]); lo, hi = min(lo, 1.0), max(hi, 1.0)
    im = ax[1].imshow(modern_g, cmap="RdYlBu_r", vmin=lo, vmax=hi)
    ax[1].contour(modern_g, [1.0], colors="black", linewidths=1.0)
    # NOT thresholded at 1 -- see the docstring. c pins this at 1 wherever the Allee
    # factor saturates, so the informative content is the shortfall below 1.
    ax[1].set_title("λ at carrying capacity (N = K)\nshortfall below 1 = Allee cost")
    fig.colorbar(im, ax=ax[1], fraction=.046, label="λ(N=K)")

    gap = fund_g - modern_g
    lim = max(float(np.nanpercentile(np.abs(gap[np.isfinite(gap)]), 98)), .02)
    im2 = ax[2].imshow(gap, cmap="magma", vmin=0, vmax=lim)
    ax[2].set_title("Gap: fundamental − realized λ")
    fig.colorbar(im2, ax=ax[2], fraction=.046, label="Δλ (≥ 0)")

    for a in ax:
        a.axis("off")
    fig.suptitle("Realized demographic potential (density-dependence + Allee included)")
    # Say which figure this is the counterpart to, and which panel is which band of
    # the GeoTIFF written beside it -- the .tif was previously unlabelled anywhere.
    fig.supxlabel(
        "Fundamental λ (no density-dependence, no Allee) is figure 01a. Panels (b) and (c) "
        "are bands 1–2 of 07_source_sink_fields.tif; the classification in (a) lives only in "
        "the .npz (source_mask). See 07_source_sink_fields.md.",
        fontsize=7.5, color="0.35")
    _snap_map_height(fig)
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    # realized_modern_source_fraction keeps its name (juv_mdd_sweep_summary.py reads
    # it) but is now the fold criterion rather than mean(lam > 1), which was an
    # artifact of the 1e-6 guard in c. allee_dead_fraction is the number the figure
    # exists to report: how much fundamentally-suitable habitat K rules out.
    fund_ok = viab["fundamental_viable"]
    return {"realized_modern_mean_lambda": float(np.mean(modern)),
            "realized_modern_source_fraction": float(np.mean(viab["viable"])),
            "allee_dead_fraction_of_suitable": float(
                np.mean(fund_ok & ~viab["viable"]) / max(float(np.mean(fund_ok)), 1e-12)),
            "median_allee_suppression_at_K": float(np.median(viab["suppression_at_K"])),
            "lambda_at_K_source_fraction_deprecated": float(np.mean(modern > 1.0))}


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
    fig, ax_grid = _map_grid(1, 1, shape, panel_w=10.0, header=0.6, cbar_frac=0.10)
    ax = ax_grid[0, 0]
    im = ax.imshow(grid_mean, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title("Mean log-scale residual per route (log(1+fitted) − log(1+observed))")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=.04, label="Residual (log1p scale)")
    _snap_map_height(fig)
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def _k_range_metrics(sim):
    """Is K pressed against its bounded dynamic range?

    K = softplus(alpha_k) * exp(L*tanh(...) + trend) * (1 - disease). The tanh bound
    stops covariate-route annihilation, but saturation against it is a diagnostic in
    exactly the way a saturated disease ceiling is: it says the likelihood wants K
    lower than the model permits, and the reason deserves finding rather than
    accommodating.
    """
    cfg = load_age_model_config()["population_model"]
    max_fold = float(cfg["k_range"]["max_fold_deviation"])
    K = np.asarray(sim["K_flat"])
    modern = K[-10:].mean(axis=0)
    level = float(np.median(modern))
    floor = level / max_fold
    return {"max_fold_deviation": max_fold,
            "modern_median_K": level,
            "modern_min_K": float(modern.min()),
            "modern_max_K": float(modern.max()),
            "modern_fold_range": float(modern.max() / max(modern.min(), 1e-12)),
            # Fraction of land within 10% of the permitted floor.
            "fraction_near_floor": float((modern < floor * 1.1).mean()),
            "fraction_near_ceiling": float((modern > level * max_fold * 0.9).mean())}


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


def plot_disease_diagnostics(data, sim, years, rows, cols, shape, out, era):
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

    _era_i0, _era_i1, _ = era_span(era, years)
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
    ceiling = float(load_age_model_config()["population_model"]
                    ["disease_prior"]["severity_ceiling"])
    return {"disease_severity_median": float(np.median(sev)),
            # Saturation against the ceiling is the "is this still absorbing misfit?"
            # readout. With density dependence in place, severity should reach the
            # ceiling only in the densest cells, if anywhere.
            "disease_severity_fraction_at_ceiling": float((sev > 0.95 * ceiling).mean()),
            "disease_severity_ceiling": ceiling,
            # The learned shape of the density dependence.
            "disease_k_half_route_counts": float(np.asarray(sim["disease_k_half_route_counts"])),
            "disease_hill_n": float(np.asarray(sim["disease_hill_n"])),
            "capacity_level_route_counts": float(np.asarray(sim["k_level_route_counts"])),
            "disease_severity_ceiling": ceiling,
            # THE diagnostic: cells pinned at the ceiling mean misfit is still being
            # routed to the disease term rather than to H_k or the K trend.
            "disease_severity_fraction_at_ceiling": at_ceiling,
            "disease_severity_p05": float(np.percentile(sev, 5)),
            "disease_severity_p95": float(np.percentile(sev, 95)),
            # frac's rows are absolute timesteps dis_t0..T-1, so the era window has to
            # be shifted by dis_t0 before indexing it (clipped, since an era that
            # starts before the epizootic has no rows here).
            "disease_fraction_median_modern": float(np.median(
                frac[max(_era_i0 - dis_t0, 0):max(_era_i1 - dis_t0, 1)])),
            # Must stay ~0: nonzero means the exogenous gate has been defeated and
            # the term is acting where the front had not yet arrived.
            "disease_fraction_pre_front_mean": float(pre_front.mean()) if pre_front.size else 0.0,
            "disease_severity_mu_logit": float(np.asarray(sim["disease_mu_sev"])),
            "disease_late_arrival_coef": float(np.asarray(sim["disease_b_late"])),
            "disease_lag0_years": lag0,
            "disease_tau_years": tau,
            "disease_recovered_fraction": rec,
            "disease_recovery_tau_years": tau_rec}


def plot_age_structure(sim, years, rows, cols, shape, land_mask, out, era):
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

    modern_theory, span, _ = era_mean(rho_theory_grid, years, era)
    modern_realized, _, _ = era_mean(rho_realized_grid, years, era)
    gap = modern_realized - modern_theory

    fig, ax = _map_grid(1, 3, shape, panel_w=5.0, header=0.85, cbar_frac=0.17)
    ax = ax[0]
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
    fig.suptitle(f"Age structure ({span[0]}–{span[1]} mean)")
    _snap_map_height(fig)
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    return {"modern_mean_theoretical_juvenile_fraction": float(np.nanmean(modern_theory)),
            "modern_mean_realized_juvenile_fraction": float(np.nanmean(modern_realized))}


def plot_invasion_progression(sim, years, land_mask, out, n_panels=6, logscale=True):
    """Small multiples of total simulated density across the invasion era.

    ``logscale`` picks log1p (which is what makes the low-density invasion FRONT
    visible at all -- on a linear scale the saturated eastern core takes the whole
    colour range) or raw density (which is what makes the core's magnitude
    readable). Both are emitted; neither answers the other's question.
    """
    density = np.where(land_mask.astype(bool)[None], sim["simulated_density"], np.nan)
    field = np.log1p(density) if logscale else density
    label = "log(1 + simulated density)" if logscale else "simulated density"

    idx = np.linspace(0, len(years) - 1, n_panels).round().astype(int)
    vmax = float(np.nanpercentile(field[idx], 99))
    vmax = vmax if vmax > 0 else 1.0

    ncols = 3
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = _map_grid(nrows, ncols, field.shape[1:], panel_w=4.0, header=0.7, right_pad=0.9)
    im = None
    for axis, i in zip(axes.flat, idx):
        im = axis.imshow(field[i], cmap="magma", vmin=0, vmax=vmax)
        axis.set_title(str(years[i]), fontsize=10)
        axis.axis("off")
    for axis in axes.flat[len(idx):]:
        axis.axis("off")
    fig.colorbar(im, ax=axes, fraction=.025, label=label)
    fig.suptitle(f"Invasion progression: simulated density over time "
                 f"({'log1p' if logscale else 'linear'} scale)")
    _snap_map_height(fig)
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_barrier_crossing(sim, data, cfg, dcfg, years, rows, cols, shape, out, era):
    """Directed cost of crossing the Great Plains, both ways.

    Returns a metrics dict, or None (having printed why) when the prerequisites are
    missing. This figure is deliberately OPTIONAL: it needs a zone raster built from
    an ecoregion shapefile that is not in git and not in download_all.sh, so on a
    machine without it the rest of the diagnostics must still complete. A missing
    input here must never cost a whole run -- the failure mode that lost a five-point
    sweep once already.

    See ``src/vis/barrier_crossing`` for the derivation. Eight panels: the raw ``Q``
    east/west contrast; the two accumulated-flux corridors on a SHARED scale plus
    their ratio; the per-year arrival profiles; the propagule pressure
    ``P = arrivals / N_crit`` each way; and a verdict panel.

    TWO THINGS THIS FIGURE MUST NOT OVERSTATE, both consequences of ``rho(A_P)``:

    * The corridors are ``sum_k A_P^k b`` for one shared ``A_P`` and two initial
      conditions (see ``crossing_gain``'s docstring). Unless ``rho`` is small they
      converge to the same dominant eigenvector and differ only by a scalar -- which
      a per-panel LogNorm would divide out, making two identical-looking panels of a
      quantity that genuinely carries no directional shape information. They are
      drawn on a SHARED norm here, and the ratio panel is the honest place to look
      for directional structure.
    * When ``rho >= 1`` the barrier self-sustains under the Allee-optimistic
      linearization, ``G`` diverges and ``P = arrivals / N_crit`` runs away to ~1e10,
      so "P >= 1 on 100% of cells" is arithmetic, not biology, and the ``P = 1``
      contour cannot exist anywhere on the map. In that regime the ``G``/``P`` numbers
      are suppressed in favour of the verdict, and ``metrics.json`` reports ``None``
      rather than a number that reads as a result.
    """
    zones_path = (dcfg.get("regions") or {}).get("great_plains_zones")
    if not zones_path or not Path(zones_path).exists():
        print(f"[map-viz] skipping barrier crossing: no zone raster at {zones_path}. "
              "Build it with scripts/build_great_plains_mask.py where the ecoregion "
              "shapefile lives, then copy it over.")
        return None
    meta_path = Path(cfg["raw_z_dir"]) / "path_feature_meta.json"
    if not meta_path.exists():
        print(f"[map-viz] skipping barrier crossing: no {meta_path} (need kernel labels)")
        return None
    labels = json.loads(meta_path.read_text()).get("kernel_labels")
    if not labels:
        print(f"[map-viz] skipping barrier crossing: no kernel_labels in {meta_path}")
        return None

    zones = read_zone_raster(zones_path, expected_shape=shape)
    fields = modern_dispersal_fields(sim, data, years, era, rows, cols, shape)
    p0 = low_density_departure_probability(
        sim["latents"], float(data["dispersal_target_fraction"]), years, era)
    edge = edge_correction_summary(data, zones, fields["land"])
    contrast = directional_q_contrast(fields, labels)
    gains = {d: crossing_gain(fields, data, zones, d, p0) for d in DIRECTIONS}
    allee_gamma = np.asarray(sim["allee_gamma"])
    pressure = {d: propagule_pressure(g["arrivals_field"], fields, rows, cols, shape,
                                      allee_gamma)[0]
                for d, g in gains.items()}

    barrier = zones["barrier"] & fields["land"].astype(bool)
    edges = _barrier_outline(barrier)
    ew, we = gains["east_to_west"], gains["west_to_east"]
    # rho belongs to the barrier-restricted operator alone, so the two directions
    # agree; this is the single number that decides whether G and P mean anything.
    rho = 0.5 * (ew["rho"] + we["rho"])
    supercritical = bool(ew["barrier_self_sustaining"] or we["barrier_self_sustaining"]
                         or rho >= 1.0)

    # constrained_layout: the shared colorbar spanning two axes is not
    # tight_layout-compatible (it warns that results may be incorrect).
    fig, ax = plt.subplots(2, 4, figsize=(21.0, 9.2), layout="constrained")

    dq = np.where(fields["land"].astype(bool), contrast["q_west_minus_east"], np.nan)
    lim = float(np.nanpercentile(np.abs(dq), 98)) or 1e-3
    im = ax[0, 0].imshow(dq, cmap="PiYG", vmin=-lim, vmax=lim)
    ax[0, 0].set_title("Journey survival asymmetry\nmean Q(to WEST) − Q(to EAST)", fontsize=10)
    fig.colorbar(im, ax=ax[0, 0], fraction=.04, label="ΔQ  (green = westward cheaper)")

    # SHARED norm across both corridors. Per-panel limits divide out the scalar that
    # is the only difference between them once the transient dies, which is precisely
    # how two panels of one eigenvector came to look like two independent results.
    corridors = {d: np.where(barrier, gains[d]["corridor_horizon"], np.nan) for d in DIRECTIONS}
    allpos = np.concatenate([c[np.isfinite(c) & (c > 0)].ravel() for c in corridors.values()])
    shared = mcolors.LogNorm(vmin=max(allpos.min(), allpos.max() * 1e-4),
                             vmax=allpos.max()) if allpos.size else None
    for col, d in enumerate(("east_to_west", "west_to_east"), start=1):
        g = gains[d]
        im = ax[0, col].imshow(corridors[d], cmap="magma", norm=shared)
        gt = "∞" if not np.isfinite(g["G_total"]) else f"{g['G_total']:.3g}"
        gline = "G suppressed (ρ ≥ 1)" if supercritical else \
            f"G(124 yr)={g['G_horizon']:.3g}  G(∞)={gt}"
        ax[0, col].set_title(f"Crossing corridor: {d.replace('_', ' ')}\n{gline}", fontsize=10)
    # ONE colorbar for the pair -- they share a norm, so two would just assert twice
    # that the scales are the same while implying they were set independently.
    fig.colorbar(im, ax=[ax[0, 1], ax[0, 2]], fraction=.028,
                 label="cumulative lineage density (shared scale)")

    # The only map that can show directional structure: both corridors normalized to
    # unit mass, then their ratio. Flat zero here means the two directions genuinely
    # traverse the barrier the same way and only the SCALE differs.
    def _unit(c):
        tot = np.nansum(c)
        return c / tot if tot > 0 else c
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log2(_unit(corridors["east_to_west"]) / _unit(corridors["west_to_east"]))
    ratio = np.where(np.isfinite(ratio), ratio, np.nan)
    rlim = float(np.nanpercentile(np.abs(ratio), 98)) if np.isfinite(ratio).any() else 1.0
    rlim = max(rlim, 1e-3)
    im = ax[0, 3].imshow(ratio, cmap="RdBu_r", vmin=-rlim, vmax=rlim)
    spread = float(np.nanmax(np.abs(ratio))) if np.isfinite(ratio).any() else 0.0
    ax[0, 3].set_title("Corridor shape asymmetry\n"
                       f"log₂(E→W / W→E), mass-normalized (max |Δ| = {spread:.2g})",
                       fontsize=10)
    fig.colorbar(im, ax=ax[0, 3], fraction=.04, label="log₂ ratio (0 = same route)")

    for a in ax[0, :]:
        a.contour(edges, [0.5], colors="cyan", linewidths=.6)
        a.set_xticks([]); a.set_yticks([])

    axp = ax[1, 0]
    for d, color in (("east_to_west", "#54278f"), ("west_to_east", "#e08214")):
        g = gains[d]
        # NORMALIZED by the mass entering the barrier, so the area under the curve
        # really is G and the visual ordering matches the reported numbers. Plotting
        # absolute arrivals would rank the directions by source size instead.
        p = (np.asarray(g["per_year_arrivals"])[:g["horizon_years"]]
             / g["total_entering_barrier"])
        # G is not quoted when it diverges -- a legend entry reading "G=2e+09" is the
        # same overstatement the panel titles were fixed for.
        lbl = d.replace("_", " ") if supercritical else \
            f"{d.replace('_', ' ')}  (G={g['G_horizon']:.3g})"
        axp.semilogy(np.arange(1, p.size + 1), np.maximum(p, 1e-300), color=color, label=lbl)
    axp.set(xlabel="Years after entering the barrier",
            ylabel="Arrivals per unit entering the barrier",
            title="Arrival timing\n" + ("rising curve = diverging sum (ρ ≥ 1)"
                                        if supercritical else "area under each curve = G"))
    axp.legend(fontsize=8)

    for col, d in enumerate(("east_to_west", "west_to_east"), start=1):
        a = ax[1, col]
        p = np.where(fields["land"].astype(bool), pressure[d], np.nan)
        frac, n_considered = _establishing_fraction(p)
        med = _finite_median(p)
        if supercritical:
            # Do NOT draw it. With rho >= 1 the numerator is ~1e16 and P saturates
            # everywhere; a viridis map under a P=1 contour that cannot exist reads
            # as a spatial result when it is an artifact of the linearization.
            a.imshow(np.where(np.isfinite(p), 0.0, np.nan), cmap="Greys", vmin=0, vmax=1)
            a.text(.5, .5, "Propagule pressure not reported\n\n"
                           "ρ ≥ 1: the linearized barrier self-sustains,\n"
                           f"so arrivals diverge (median P ≈ {med:.1g})\n"
                           "and the P = 1 contour cannot exist.",
                   transform=a.transAxes, ha="center", va="center", fontsize=8.5,
                   color="0.25")
            a.set_title(f"Propagule pressure: {d.replace('_', ' ')} — suppressed", fontsize=9)
        else:
            finite = p[np.isfinite(p) & (p > 0)]
            norm = mcolors.LogNorm(vmin=max(finite.min(), finite.max() * 1e-6),
                                   vmax=max(finite.max(), 1.0)) if finite.size else None
            im = a.imshow(p, cmap="viridis", norm=norm)
            # P = 1 is the pinning boundary: one year's immigration clears N_crit.
            a.contour(np.nan_to_num(p, nan=0.0), [1.0], colors="red", linewidths=.8)
            a.set_title(f"Propagule pressure: {d.replace('_', ' ')}\n"
                        f"P≥1 on {frac:.1%} of {n_considered} reached cells; "
                        f"median P={med:.2g}", fontsize=9)
            fig.colorbar(im, ax=a, fraction=.04, label="P (red contour = 1)")
        a.contour(edges, [0.5], colors="cyan", linewidths=.6)
        a.set_xticks([]); a.set_yticks([])

    # Verdict panel: rho decides how the rest of the figure may be read, so it gets
    # stated rather than left to be inferred from an infinity in metrics.json.
    av = ax[1, 3]
    av.axis("off")
    if supercritical:
        verdict = (f"ρ(A_P) = {rho:.3f}  ≥ 1\n\n"
                   "The Great Plains contain self-sustaining habitat under the\n"
                   "Allee-optimistic linearization (Allee factor forced to 1,\n"
                   "crowding removed). The Neumann series diverges, so G is\n"
                   "infinite and P saturates — neither is reported.\n\n"
                   "This IS the result: under the optimistic bound there is no\n"
                   "barrier at all, so the observed barrier must be an Allee/K\n"
                   "phenomenon rather than a habitat-quality one.\n\n"
                   "Directional signal still lives in ΔQ (top left) and in the\n"
                   "corridor-shape ratio (top right), neither of which depends\n"
                   "on the diverging sum.")
        box = dict(fc="#fff4e6", ec="#e08214", boxstyle="round,pad=0.6")
    else:
        verdict = (f"ρ(A_P) = {rho:.3f}  < 1\n\n"
                   "Lineages confined to the barrier die out, so the barrier is a\n"
                   "genuine barrier and G is finite and comparable.\n\n"
                   f"G(124 yr) E→W = {ew['G_horizon']:.3g}\n"
                   f"G(124 yr) W→E = {we['G_horizon']:.3g}\n"
                   f"asymmetry E/W ÷ W/E = "
                   f"{ew['G_horizon'] / we['G_horizon']:.3g}\n\n"
                   "G is an UPPER bound: crossing that fails here fails in the\n"
                   "real model.")
        box = dict(fc="#eef6ff", ec="#2166ac", boxstyle="round,pad=0.6")
    av.text(0.02, 0.98, verdict, transform=av.transAxes, ha="left", va="top",
            fontsize=8.5, family="monospace", bbox=box)
    av.set_title("Verdict", fontsize=10)

    fig.suptitle("Directed cost of crossing the Great Plains "
                 f"(Allee-optimistic upper bound; p₀={p0:.3f}; ρ={rho:.3f})", fontsize=13)
    fig.savefig(out, dpi=170); plt.close(fig)

    ew, we = gains["east_to_west"], gains["west_to_east"]
    ratio = (ew["G_horizon"] / we["G_horizon"]) if we["G_horizon"] > 0 else float("nan")

    def _suppress(value):
        """None when the operator is supercritical, else the number.

        Under rho >= 1 these quantities are divergent-series artifacts, not results.
        Emitting them as JSON Infinity (or as 1e11 / 1.0) made a sweep summary read
        them as measurements; None forces a consumer to handle "not defined here".
        """
        return None if supercritical else value

    verdict = ("barrier_self_sustaining: rho >= 1 under the Allee-optimistic "
               "linearization, so the Neumann series diverges. G and propagule "
               "pressure are NOT DEFINED in this regime and are reported as null. "
               "The reportable conclusion is that the Great Plains are not a "
               "habitat-quality barrier under the optimistic bound, so the observed "
               "barrier is an Allee/K phenomenon. Directional signal lives in "
               "q_asymmetry and in corridor_shape_asymmetry_max_log2."
               if supercritical else
               "subcritical: rho < 1, lineages confined to the barrier die out, so G "
               "is finite and comparable across sweep points.")
    metrics = {
        "_comment": ("Linearized annual operator restricted to the Great Plains "
                     "corridor; see src/vis/barrier_crossing. The Allee factor is set "
                     "to 1, so these are UPPER bounds: crossing that fails here fails "
                     "in the real model. G_horizon (124 yr) is the comparable metric."),
        "barrier_crossing_verdict": verdict,
        "p0_low_density_departure": p0,
        "horizon_years": ew["horizon_years"],
        "G_horizon_east_to_west": _suppress(ew["G_horizon"]),
        "G_horizon_west_to_east": _suppress(we["G_horizon"]),
        "asymmetry_ratio_ew_over_we": _suppress(ratio),
        # G_total is +inf whenever the barrier self-sustains, which is not JSON;
        # _suppress already nulls exactly that case.
        "G_total_east_to_west": _suppress(ew["G_total"]),
        "G_total_west_to_east": _suppress(we["G_total"]),
        # rho is a property of the barrier-restricted operator alone, so the two
        # directions must agree; the gap is a convergence diagnostic, not a result.
        # This one is ALWAYS reported -- it is the headline result when >= 1.
        "rho_barrier": rho,
        "rho_direction_discrepancy": abs(ew["rho"] - we["rho"]),
        "barrier_self_sustaining": supercritical,
        "years_to_half_of_G_east_to_west": _suppress(ew["years_to_half_of_G"]),
        "years_to_half_of_G_west_to_east": _suppress(we["years_to_half_of_G"]),
        # Fraction of REACHED, VIABLE far-side cells clearing N_crit -- see
        # _establishing_fraction for why the denominator is not simply "land".
        "establishing_fraction_east_to_west": _suppress(_establishing_fraction(
            pressure["east_to_west"])[0]),
        "establishing_fraction_west_to_east": _suppress(_establishing_fraction(
            pressure["west_to_east"])[0]),
        "reached_viable_cells_east_to_west": _establishing_fraction(
            pressure["east_to_west"])[1],
        "reached_viable_cells_west_to_east": _establishing_fraction(
            pressure["west_to_east"])[1],
        "median_propagule_pressure_east_to_west": _suppress(
            _finite_median(pressure["east_to_west"])),
        "median_propagule_pressure_west_to_east": _suppress(
            _finite_median(pressure["west_to_east"])),
        "mean_q_to_east_in_barrier": float(contrast["q_to_east"][barrier].mean()),
        "mean_q_to_west_in_barrier": float(contrast["q_to_west"][barrier].mean()),
        # How differently the two directions actually route through the barrier.
        # ~0 means the corridors are one eigenvector and only their SCALE differs.
        "corridor_shape_asymmetry_max_log2": float(spread),
        **edge,
    }
    return metrics


def _pretty_band(key):
    """``331-1000000000`` -> ``331+ km``; the 1e9 upper edge is an open-ended sentinel."""
    text = str(key)
    if "-" not in text:
        return text
    lo, hi = text.split("-", 1)
    try:
        if float(hi) >= 1e8:
            return f"{float(lo):.0f}+ km"
        return f"{float(lo):.0f}\u2013{float(hi):.0f} km"
    except ValueError:
        return text


def plot_q_asymmetry_attribution(sim, data, cfg, dcfg, years, rows, cols, shape, out,
                                 era, names=None):
    """Why westward journeys cost more -- resolved by dispersal distance and covariate.

    Figure 17's ΔQ panel pools all three radial bands into one map, so it cannot say
    whether the asymmetry is a short-hop or a long-jump phenomenon -- and the bands
    (0-155, 155-483, 483+ km) are exactly the scales the juvenile-MDD sweep moves.
    Top row: per-band ΔQ maps. Bottom left: the barrier-mean contrast per band, with
    the edge-correction contrast beside it as the GEOMETRY CONTROL (if ΔQ tracks
    Δf_j, the asymmetry is the domain boundary, not habitat). Bottom right: the Z
    features driving it, decomposed on the pre-sigmoid scale where the split is exact.

    Optional in exactly the same way as figure 17, and for the same missing inputs.
    """
    zones_path = (dcfg.get("regions") or {}).get("great_plains_zones")
    meta_path = Path(cfg["raw_z_dir"]) / "path_feature_meta.json"
    if not zones_path or not Path(zones_path).exists() or not meta_path.exists():
        return None
    labels = json.loads(meta_path.read_text()).get("kernel_labels")
    if not labels:
        return None

    zones = read_zone_raster(zones_path, expected_shape=shape)
    fields = modern_dispersal_fields(sim, data, years, era, rows, cols, shape)
    attr = q_asymmetry_attribution(fields, data, sim, zones, labels, rows, cols, shape,
                                   years=years, era=era)
    bands = attr["bands"]
    if not bands:
        return None
    barrier = zones["barrier"] & fields["land"].astype(bool)
    edges = _barrier_outline(barrier)
    land = fields["land"].astype(bool)

    n = len(bands)
    # constrained_layout, not tight_layout: the top row's per-axes colorbars are not
    # tight_layout-compatible and it warned that results might be incorrect.
    fig = plt.figure(figsize=(4.6 * max(n, 3), 8.8), layout="constrained")
    gs = fig.add_gridspec(2, max(n, 3), height_ratios=[1.25, 1.0])

    lim = max(float(np.nanpercentile(np.abs(np.stack(
        [np.where(land, b["delta"], np.nan) for b in bands])), 98)), 1e-4)
    band_axes = []
    for i, b in enumerate(bands):
        a = fig.add_subplot(gs[0, i])
        im = a.imshow(np.where(land, b["delta"], np.nan), cmap="PiYG", vmin=-lim, vmax=lim)
        a.contour(edges, [0.5], colors="cyan", linewidths=.6)
        a.set_title(f"{_pretty_band(b['band'])} band\n"
                    f"ΔQ = {b['mean_delta']:+.4f} in barrier", fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        band_axes.append(a)
    # One colorbar: all three panels share vmin/vmax, so per-panel bars would imply
    # independently-scaled maps that must not be compared.
    fig.colorbar(im, ax=band_axes, fraction=.03, label="Q(→W) − Q(→E), shared scale")

    # Per-band contrast beside its geometry control.
    axb = fig.add_subplot(gs[1, 0])
    y = np.arange(n)
    axb.barh(y + .19, [b["mean_delta"] for b in bands], height=.36,
             color="#4d9221", label="ΔQ (journey survival)")
    axb.barh(y - .19, [b["mean_edge_corr_delta"] for b in bands], height=.36,
             color="#999999", label="Δf_j (edge correction) — geometry control")
    axb.axvline(0, color="0.25", lw=1)
    axb.set_yticks(y)
    axb.set_yticklabels([_pretty_band(b["band"]) for b in bands], fontsize=8)
    axb.set_xlabel("westward − eastward (barrier mean)")
    axb.set_title("Asymmetry by dispersal distance\nvs. the wedge-geometry control", fontsize=10)
    axb.legend(fontsize=7, frameon=False, loc="best")
    axb.grid(axis="x", alpha=.25)

    axf = fig.add_subplot(gs[1, 1:])
    feats = attr["features"]
    if feats:
        nm = list(names) if names is not None else [f"Z_{i}" for i in range(len(attr["beta_s"]))]
        yy = np.arange(len(feats))[::-1]
        band_keys = [b["band"] for b in bands]
        cmap = plt.get_cmap("viridis", max(len(band_keys), 2))
        left_pos = np.zeros(len(feats)); left_neg = np.zeros(len(feats))
        for bi, key in enumerate(band_keys):
            vals = np.array([f["by_band"][key] for f in feats])
            base = np.where(vals >= 0, left_pos, left_neg)
            axf.barh(yy, vals, left=base, height=.62, color=cmap(bi),
                     label=_pretty_band(key))
            left_pos = left_pos + np.maximum(vals, 0)
            left_neg = left_neg + np.minimum(vals, 0)
        axf.axvline(0, color="0.25", lw=1)
        axf.set_yticks(yy)
        axf.set_yticklabels([f"{nm[f['index']]}" for f in feats], fontsize=8)
        axf.set_xlabel("contribution to the pre-sigmoid W−E contrast  "
                       "(mean ΔZ_disp × β_s), stacked by band")
        axf.set_title("Which covariates make westward journeys costlier", fontsize=10)
        axf.legend(fontsize=7, frameon=False, ncol=len(band_keys))
        axf.grid(axis="x", alpha=.25)
    else:
        axf.axis("off")
        axf.text(.5, .5, "Z_disp_gathered not available;\nper-feature attribution skipped",
                 ha="center", va="center", fontsize=10, color="0.4")

    fig.suptitle("Journey-survival asymmetry across the Great Plains: "
                 "by dispersal distance and by covariate", fontsize=13)
    fig.savefig(out, dpi=170); plt.close(fig)
    return {
        "bands": [{k: v for k, v in b.items()
                   if k not in ("q_to_east", "q_to_west", "delta")} for b in bands],
        "top_features": [
            {"name": (list(names)[f["index"]] if names is not None else f"Z_{f['index']}"),
             **f} for f in feats],
    }


def _barrier_outline(barrier):
    """Float field that contours to the barrier boundary (for overlay outlines)."""
    return barrier.astype(float)


def _finite_median(pressure):
    """Median propagule pressure over reached, viable cells; NaN if there are none.

    Reported alongside the establishing fraction because a fraction of 0 is
    ambiguous: it can mean "arrivals fall just short everywhere" or "arrivals are
    orders of magnitude short". The median says which.
    """
    v = pressure[np.isfinite(pressure) & (pressure > 0.0)]
    return float(np.median(v)) if v.size else float("nan")


def _establishing_fraction(pressure):
    """Share of REACHED, VIABLE far-side cells where propagule pressure clears N_crit.

    The denominator matters and is easy to get wrong. ``pressure >= 1.0`` on the raw
    field would be averaged over the whole grid, where NaN (cell not viable, so no
    N_crit exists) silently compares False and is pooled with 0 (cell outside the
    target zone, so it receives nothing) and with genuine sub-threshold arrivals.
    Those are three different statements. Restricting to finite AND strictly positive
    pressure asks the one question worth asking: of the viable far-side habitat that
    receives any immigration at all, how much receives enough to establish.
    """
    considered = np.isfinite(pressure) & (pressure > 0.0)
    n = int(considered.sum())
    if n == 0:
        return float("nan"), 0
    return float((pressure[considered] >= 1.0).mean()), n


def simulate_no_invasion_counterfactual(sim, data, drop_disease=True):
    """Re-run the forward model with the 1940 release deleted.

    The intervention is exactly one array: ``inv_pop -> 0``. Everything else --
    fitted Sa/Sj/Fmax, the dispersal kernels, the per-year dispersal random effect,
    the native-range seed at t=0 -- is the MAP point unchanged, so the difference
    between this and ``sim["simulated_density"]`` is attributable to the release
    alone. No refit and no gradient: ``forward_sim_age_structured`` is a pure
    function of its arguments, which is also why this touches nothing that
    ``age_run_map._run_fingerprint`` hashes.

    The counterfactual is NOT "the East stays empty". The native western population
    is seeded in 1902 as usual and disperses for the full timeline, so what this
    measures is how far the species would have got on its own -- the release's
    contribution is the difference, not the whole eastern range.

    ``drop_disease=True`` substitutes ``K_base_flat`` for ``K_flat``, i.e. removes
    the mycoplasmal-conjunctivitis depression. That is the causally coherent choice:
    the 1994 epizootic swept the dense eastern population the release created, so a
    world without the release is a world without that epizootic. Keeping ``K_flat``
    instead would have westward-spreading birds arrive to find capacity suppressed
    in the East by an outbreak that never happened, which suppresses the very
    colonization the counterfactual is trying to measure. The continental ``k_trend``
    stays in either case -- it is not disease.

    CAVEAT worth carrying into any figure caption: essentially all of the
    information about spread rate in the likelihood comes from the observed eastern
    invasion. The counterfactual's westward-spread dynamics are therefore an
    extrapolation of parameters fitted elsewhere, not a validated prediction.
    """
    K_cf = sim["K_base_flat"] if drop_disease else sim["K_flat"]
    latents = sim["latents"]
    # c does NOT depend on K (c = Fmax*Sa*Sj/(1-Sa) - 1), so it is identical in both
    # worlds even when K_base is substituted; recompute rather than plumb it through.
    c_flat, _, _, _ = realized_equilibrium(
        sim["Sa_flat"], sim["Sj_flat"], sim["Fmax_flat"], K_cf, sim["allee_gamma"])
    inv_pop_zero = jnp.zeros_like(jnp.asarray(sim["inv_pop_relative"]))
    density, Na, Nj = forward_sim_age_structured(
        jnp.asarray(sim["Sa_flat"]), jnp.asarray(sim["Sj_flat"]),
        jnp.asarray(sim["Fmax_flat"]), jnp.asarray(K_cf), jnp.asarray(c_flat),
        jnp.asarray(sim["Q_flat"]),
        data["land_rows"], data["land_cols"], data["land_mask"],
        data["adult_fft_kernel"], data["juvenile_fft_kernel_stack"],
        data["adult_edge_correction"], data["juvenile_edge_correction_stack"],
        jnp.asarray(sim["initpop_seeded"]), jnp.asarray(latents["dispersal_random"]),
        inv_pop_zero,
        int(data["time"]), data["inv_locations"], data["inv_timestep"],
        float(np.asarray(latents["dispersal_logit_intercept"])),
        float(np.asarray(latents["dispersal_logit_slope"])),
        jnp.asarray(sim["allee_gamma"]),
        target_fraction=data["dispersal_target_fraction"],
    )
    return np.asarray(jax.block_until_ready(density))


def plot_counterfactual_attribution(sim, cf_release_only, cf_no_disease, viab_cf,
                                    years, rows, cols, shape, land_mask, out, era,
                                    pop_scalar=1.0, logscale=True):
    """What the 1940 release caused: attribution, and the range it bought.

    THREE SIMULATIONS, because one counterfactual cannot answer both questions:

    * **A** = actual (release + epizootic) -- ``sim["simulated_density"]``
    * **B** = no release, epizootic KEPT (``cf_release_only``)
    * **C** = no release, no epizootic (``cf_no_disease``)

    Attribution uses **A vs B**, the minimal intervention: only ``inv_pop`` differs,
    so ``(A - B)/A`` is the share of today's density caused by the release alone.
    Differencing A against C instead would confound two interventions with OPPOSING
    signs on density -- deleting the release removes birds, removing the epizootic
    adds capacity -- so they partially cancel and the map can even go negative. That
    is not a subtle bias: on a synthetic test C carried 25% MORE population than A.

    C is the causally coherent "world where the release never happened" (no dense
    eastern population means no 1994 epizootic), so it supplies the counterfactual
    density map, the unreached-range overlay, and the animation. Both B and C appear
    in the time series so the two effects can be read apart.

    Panels: (1) modern density A, (2) modern density C, (3) attribution (A vs B),
    (4) viable range unreached in C, (5) occupied area over time, (6) total
    population over time with the release year marked.

    Attribution is computed on era means and only where A exceeds
    ``occupancy_floor``; elsewhere the ratio is 0/0 and is left NaN rather than
    rendered as a spurious 0 or 1. ``viab_cf`` must be the viability of C's world
    (i.e. built from ``K_base_flat``) -- see :func:`modern_viability`.
    """
    mask = land_mask.astype(bool)
    # Arithmetic on the RAW arrays: the simulator already multiplies by land_mask
    # every step, so ocean cells are exactly 0 and need no NaN. Masking first would
    # make every ocean column an all-NaN slice and have nanmean warn on all of them.
    # NaN is introduced only for display, below.
    act, rel, cfa = (np.asarray(x) for x in
                     (sim["simulated_density"], cf_release_only, cf_no_disease))
    act_m, span, _ = era_mean(act, years, era)
    cf_m, _, _ = era_mean(cfa, years, era)
    rel_m, _, _ = era_mean(rel, years, era)
    act_m, cf_m, rel_m = (np.where(mask, x, np.nan) for x in (act_m, cf_m, rel_m))

    # Occupancy floor in DENSITY units from a route-count threshold: 0.05 expected
    # counts is well under a single bird on a single route, so it separates "present"
    # from numerical dust without asserting anything about detectability.
    floor = 0.05 / float(pop_scalar)
    occupied_act, occupied_cf = act_m > floor, cf_m > floor
    with np.errstate(invalid="ignore", divide="ignore"):
        # A vs B: the release is the ONLY difference, so this is signed the way the
        # title claims. Small negatives are possible where the extra eastern birds
        # redistribute dispersers away from a cell; they are clipped, and the share
        # clipped is reported so the clip can never hide a systematic sign problem.
        attrib_raw = np.where(occupied_act, (act_m - rel_m) / act_m, np.nan)
    finite_attrib = np.isfinite(attrib_raw)
    n_fin = max(float(finite_attrib.sum()), 1.0)
    # Tolerance, not zero. Where the release changes nothing (the West, or a fully
    # saturated landscape) the difference is float noise and lands either side of 0
    # at random -- counting those as "negative" would flag a sign problem in exactly
    # the case where the honest answer is "no effect". 1e-3 = the counterfactual
    # exceeding actual by more than 0.1% of actual, which noise does not reach.
    clipped_frac = float(np.sum(attrib_raw[finite_attrib] < -1e-3)) / n_fin
    attrib = np.clip(attrib_raw, 0.0, 1.0)

    viable_g = _grid(viab_cf["viable"].astype(float)[None], rows, cols, shape)[0]
    viable = np.isfinite(viable_g) & (viable_g > 0.5)
    unreached = viable & ~occupied_cf          # viable, but empty without the release

    fig = plt.figure(figsize=(16.5, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=.18, wspace=.12)
    tf, _, scale_name = _abundance_scale(logscale)
    log_act, log_cf = tf(act_m * pop_scalar), tf(cf_m * pop_scalar)
    density_label = "log(1 + route counts)" if logscale else "route counts"
    # Every reduction here must survive an ALL-NaN / all-zero input. A degenerate or
    # very early checkpoint can simulate a fully extinct landscape, and an unguarded
    # nanpercentile/nanmedian raises -- which would kill the whole diagnostic run
    # (and, in a sweep, every point's metrics.json) over a cosmetic colour limit.
    pooled = np.concatenate([log_act[np.isfinite(log_act)], log_cf[np.isfinite(log_cf)]])
    vmax = float(np.nanpercentile(pooled, 99.5)) if pooled.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(log_act, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(f"Actual ({span[0]}–{span[1]}, {scale_name} scale)", fontsize=10); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=.035, label=density_label)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(log_cf, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title("Counterfactual: no release, no epizootic", fontsize=10); ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(attrib, cmap="viridis", vmin=0, vmax=1)
    ax.contour(np.nan_to_num(attrib, nan=0.0), [0.5], colors="white", linewidths=.7)
    ax.set_title("Attribution to the release alone\n(actual − no-release) / actual, "
                 "epizootic held fixed", fontsize=10); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=.035, label="fraction of density from the release")

    ax = fig.add_subplot(gs[1, 0])
    layers = np.full(shape, np.nan)
    layers[mask] = 0.0
    layers[viable & occupied_cf] = 1.0
    layers[unreached] = 2.0
    ax.imshow(layers, cmap=mcolors.ListedColormap(["#f0f0f0", "#4575b4", "#e08214"]),
              norm=mcolors.BoundaryNorm([-.5, .5, 1.5, 2.5], 3))
    frac = float(np.sum(unreached)) / max(float(np.sum(viable)), 1.0)
    ax.set_title(f"Viable range unreached without the release\n{frac:.1%} of viable habitat",
                 fontsize=10); ax.axis("off")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#e08214", label="viable, unoccupied in counterfactual"),
        plt.Rectangle((0, 0), 1, 1, color="#4575b4", label="viable, reached anyway"),
        plt.Rectangle((0, 0), 1, 1, color="#f0f0f0", label="not viable"),
    ], loc="lower left", fontsize=6.5, frameon=True)

    n_land = float(mask.sum())
    series = (("actual", act, "#54278f", "-"),
              ("no release, epizootic kept", rel, "#2166ac", ":"),
              ("no release, no epizootic", cfa, "#e08214", "--"))
    ax = fig.add_subplot(gs[1, 1])
    for label, arr, color, ls in series:
        ax.plot(years, (arr > floor).sum(axis=(1, 2)) / n_land, color=color, ls=ls, label=label)
    ax.set(xlabel="Year", ylabel="Fraction of land occupied", title="Occupied area")
    add_timeline_markers(ax); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 2])
    for label, arr, color, ls in series:
        ax.plot(years, np.nansum(arr, axis=(1, 2)) * pop_scalar, color=color, ls=ls, label=label)
    ax.set(xlabel="Year", ylabel="Σ density (route-count units)", title="Total population",
           yscale="log")
    add_timeline_markers(ax); ax.legend(fontsize=7)

    fig.suptitle("Counterfactual history: the 1940 NYC release deleted, everything else "
                 "at the fitted MAP point", y=.99, fontsize=13)
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)

    tot_act = float(np.nansum(act_m))
    tot_cf = float(np.nansum(cf_m))
    tot_rel = float(np.nansum(rel_m))
    return {
        "_comment": ("Three arms: A = actual, B = no release with the epizootic kept, "
                     "C = no release and no epizootic. Attribution is A vs B (the "
                     "release is the only difference); C is the coherent 'no release' "
                     "world and drives the maps and the animation."),
        "occupancy_floor_route_counts": 0.05,
        "modern_total_density_actual": tot_act * float(pop_scalar),
        "modern_total_density_no_release_disease_kept": tot_rel * float(pop_scalar),
        "modern_total_density_no_release_no_disease": tot_cf * float(pop_scalar),
        # A vs B: the release's own contribution to modern population.
        "fraction_of_modern_population_from_release": (
            (tot_act - tot_rel) / tot_act if tot_act > 0 else float("nan")),
        # C vs B: how much the epizootic itself costs, inside the counterfactual.
        "epizootic_cost_fraction_in_counterfactual": (
            (tot_cf - tot_rel) / tot_cf if tot_cf > 0 else float("nan")),
        "modern_occupied_fraction_actual": float(occupied_act.sum()) / n_land,
        "modern_occupied_fraction_no_release": float(occupied_cf.sum()) / n_land,
        "viable_fraction_unreached_without_release": frac,
        "median_attribution_where_occupied": (
            float(np.median(attrib[np.isfinite(attrib)])) if finite_attrib.any()
            else float("nan")),
        # If this is not small, the A-vs-B difference is not cleanly signed and the
        # attribution map should not be read as a causal share.
        "attribution_negative_fraction_clipped": clipped_frac,
        # Distinguishes "the release did nothing" (max ~ 0) from "the release did
        # something the median cell didn't see" (max ~ 1, median ~ 0).
        "max_attribution": (float(np.max(attrib[np.isfinite(attrib)]))
                            if finite_attrib.any() else float("nan")),
    }


def _save_animation(ani, out, fps=8):
    """Write mp4 via FFMpeg, falling back to GIF via pillow.

    FFMpeg is absent on some run environments (and on TACC compute nodes), and an
    animation is never worth losing the rest of a diagnostics run over.
    """
    try:
        ani.save(str(out), writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
    except Exception as exc:
        gif_out = str(out).rsplit(".", 1)[0] + ".gif"
        print(f"[map-viz] FFMpeg unavailable ({exc}); falling back to GIF: {gif_out}")
        ani.save(gif_out, writer="pillow", fps=fps)


def _abundance_scale(logscale):
    """``(transform, colorbar label, scale name)`` for an abundance display."""
    if logscale:
        return np.log1p, "log(1 + density)", "log1p"
    return (lambda x: x), "density (route counts)", "linear"


def create_counterfactual_animation(sim, cf_density, years, land_mask, out, logscale=True):
    """Animated actual vs no-release density, side by side on one shared scale."""
    mask = land_mask.astype(bool)
    act = np.where(mask[None], np.asarray(sim["simulated_density"]), np.nan)
    cfa = np.where(mask[None], cf_density, np.nan)
    tf, cbar_label, scale_name = _abundance_scale(logscale)
    la, lc = tf(act), tf(cfa)
    # ONE scale for both panels: independent scaling would make an empty
    # counterfactual look as populated as the actual invasion.
    vmax = float(np.nanpercentile(la[np.isfinite(la)], 99.5))
    vmax = vmax if vmax > 0 else 1.0

    fig, _axg = _map_grid(1, 2, la.shape[1:], panel_w=6.0, header=0.85, right_pad=1.0)
    ax_a, ax_c = _axg[0]
    im_a = ax_a.imshow(la[0], cmap="magma", vmin=0, vmax=vmax)
    ax_a.set_title("Actual"); ax_a.axis("off")
    im_c = ax_c.imshow(lc[0], cmap="magma", vmin=0, vmax=vmax)
    ax_c.set_title("Counterfactual: no 1940 release"); ax_c.axis("off")
    fig.colorbar(im_a, ax=[ax_a, ax_c], fraction=.025, label=cbar_label)
    title = fig.suptitle(f"Year {years[0]}  ({scale_name} scale)", fontsize=14, fontweight="bold")

    def update(frame):
        title.set_text(f"Year {years[frame]}  ({scale_name} scale)")
        im_a.set_data(la[frame]); im_c.set_data(lc[frame])
        return im_a, im_c, title

    ani = animation.FuncAnimation(fig, update, frames=len(years), interval=120, blit=False)
    _save_animation(ani, out)
    plt.close(fig)


def create_invasion_animation(sim, data, years, land_mask, out, logscale=False):
    """Animated side-by-side: simulated density vs. observed BBS counts, all years.

    ``logscale`` emits the log1p counterpart, on which the advancing low-density
    front is visible; the linear version keeps the core's magnitude readable.
    Simulated and observed keep INDEPENDENT scales here (unlike the counterfactual
    animation) because they are in different units.
    """
    shape = land_mask.shape
    obs_grid = np.full((len(years), *shape), np.nan)
    obs_rows = np.asarray(data["obs_rows"])
    obs_cols = np.asarray(data["obs_cols"])
    obs_t = np.asarray(data["obs_time_indices"])
    obs_grid[obs_t, obs_rows, obs_cols] = np.asarray(data["observed_results"])

    density = np.where(land_mask.astype(bool)[None], sim["simulated_density"], np.nan)
    tf, _, scale_name = _abundance_scale(logscale)
    density, obs_grid = tf(density), tf(obs_grid)
    sim_label = "log(1 + simulated density)" if logscale else "simulated density"
    obs_label = "log(1 + observed count)" if logscale else "observed BBS count"
    vmax_sim = float(np.nanpercentile(density, 99)) or 1.0
    vmax_obs = float(np.nanpercentile(obs_grid[np.isfinite(obs_grid)], 99)) or 1.0

    fig, _axg = _map_grid(1, 2, density.shape[1:], panel_w=6.0, header=0.85, cbar_frac=0.15)
    ax_sim, ax_obs = _axg[0]
    im_sim = ax_sim.imshow(density[0], cmap="magma", vmin=0, vmax=vmax_sim)
    ax_sim.set_title("Simulated density"); ax_sim.axis("off")
    fig.colorbar(im_sim, ax=ax_sim, fraction=.035, label=sim_label)
    im_obs = ax_obs.imshow(obs_grid[0], cmap="magma", vmin=0, vmax=vmax_obs)
    ax_obs.set_title("Observed BBS counts"); ax_obs.axis("off")
    fig.colorbar(im_obs, ax=ax_obs, fraction=.035, label=obs_label)
    title = fig.suptitle(f"Year {years[0]}  ({scale_name} scale)", fontsize=14, fontweight="bold")

    def update(frame):
        title.set_text(f"Year {years[frame]}  ({scale_name} scale)")
        im_sim.set_data(density[frame])
        im_obs.set_data(obs_grid[frame])
        return im_sim, im_obs, title

    ani = animation.FuncAnimation(fig, update, frames=len(years), interval=120, blit=False)
    _save_animation(ani, out)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default=os.environ.get("HOUFIN_MAP_PROFILE", "standard"))
    parser.add_argument("--precision", default=os.environ.get("HOUFIN_MODEL_PRECISION", "float32"), choices=["float32", "float64"])
    # Era-named windows are the default (age_model_math.ERAS): modern 2010-2025 vs
    # early 1902-1915 or invasion 1940-1955. --window-years is an OVERRIDE that
    # restores the old trailing-N behaviour, for reproducing a pre-era run.
    parser.add_argument("--window-years", type=int, default=None,
                        help="override the named eras with trailing/anchored N-year windows")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.window_years is not None and args.window_years < 1:
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
    # Resolve the three comparison eras ONCE, so every figure, the metrics file and
    # the sweep summary all describe the same spans. --window-years rebuilds
    # era-shaped spans from a trailing/anchored N-year window instead.
    eras = eras_from_window(years, args.window_years) if args.window_years else dict(ERAS)
    era_spans = {name: era_span(span, years)[2] for name, span in eras.items()}
    print("[map-viz] eras: " + ", ".join(f"{k}={v[0]}-{v[1]}" for k, v in era_spans.items()))
    modern_era, early_era, invasion_era = eras["modern"], eras["early"], eras["invasion"]

    lam = local_growth_lambda(sim["Sa_flat"], sim["Sj_flat"], sim["Fmax_flat"])
    modern, early, transition = plot_modern_niche(
        lam, years, rows, cols, shape, out / "01a_niche_change_since_1902.png",
        modern_era, early_era)
    # Invasion-anchored counterparts (01b/06b). The "early" baseline is the start of
    # the model timeline (1902-1915), which mixes ~38 years of pre-invasion climate
    # change into every "change since the beginning" statement; anchoring at
    # 1940-1955 instead measures change relative to what the species actually met on
    # arrival. Both are kept because they answer different questions.
    inv_year = int(load_timeline(dcfg)["invasion_year"])
    _, early_inv, transition_inv = plot_modern_niche(
        lam, years, rows, cols, shape, out / "01b_niche_change_since_invasion.png",
        modern_era, invasion_era)
    fraction, mean_lambda, centroid_lat = plot_niche_trajectory(
        lam, years, rows, cols, dcfg["grid"]["ref_raster"], out / "02_niche_trajectory.png")
    plot_modern_rate_maps(sim, years, rows, cols, shape,
                          out / "03_modern_demographic_rates.png", modern_era)
    fit_metrics = plot_fit_diagnostics(sim, data, years, out / "04_map_fit_diagnostics.png")
    z_names = z_feature_names(cfg, np.asarray(sim["latents"]["w_env"]).shape[0])
    response_metrics = plot_response_curves(
        sim, out / "05_demographic_response_curves.png", names=z_names)
    ranking_metrics = plot_w_env_ranking(sim, out / "05b_w_env_ranking.png", names=z_names)
    # Which Z features moved, and what their movement did to lambda -- the "why"
    # behind figures 01/13. Two baselines, matching the 01/13 pair.
    z_attr_invasion = plot_z_feature_attribution(
        data, sim, years, rows, cols, shape, out / "05c_z_attribution_since_invasion.png",
        modern_era, invasion_era, names=z_names)
    z_attr_early = plot_z_feature_attribution(
        data, sim, years, rows, cols, shape, out / "05d_z_attribution_since_1902.png",
        modern_era, early_era, names=z_names)
    plot_environmental_drivers_limits(data, sim, years, rows, cols, shape,
                                       out / "06a_environmental_drivers_since_1902.png",
                                       modern_era, early_era, names=z_names)
    plot_environmental_drivers_limits(data, sim, years, rows, cols, shape,
                                       out / "06b_environmental_drivers_since_invasion.png",
                                       modern_era, invasion_era, names=z_names)
    source_sink_metrics = plot_realized_source_sink(
        sim, lam, years, rows, cols, shape, out / "07_realized_source_sink.png", modern_era,
        fields_out=out / "07_source_sink_fields.npz", ref_raster=dcfg["grid"]["ref_raster"])
    plot_spatial_residuals(sim, data, shape, out / "08_spatial_residuals.png")
    disease_metrics = plot_disease_diagnostics(
        data, sim, years, rows, cols, shape, out / "09_disease_diagnostics.png",
        modern_era)
    land_mask_arr = np.asarray(data["land_mask"])
    age_structure_metrics = plot_age_structure(
        sim, years, rows, cols, shape, land_mask_arr, out / "10_age_structure.png", modern_era)
    # Every abundance display is emitted on BOTH scales. log1p is the only one on
    # which the low-density invasion front is visible at all; linear is the only one
    # on which the saturated core's magnitude is readable. Neither substitutes.
    plot_invasion_progression(sim, years, land_mask_arr,
                              out / "11_invasion_progression.png", logscale=True)
    plot_invasion_progression(sim, years, land_mask_arr,
                              out / "11_invasion_progression_linear.png", logscale=False)
    create_invasion_animation(sim, data, years, land_mask_arr,
                              out / "12_invasion_animation.mp4", logscale=False)
    create_invasion_animation(sim, data, years, land_mask_arr,
                              out / "12_invasion_animation_log1p.mp4", logscale=True)
    # No-invasion counterfactual (15/16). TWO extra forward passes at the same MAP
    # point: one with the epizootic held fixed (isolates the release, so attribution
    # is cleanly signed) and one with it removed (the causally coherent world, since
    # no dense eastern population means no 1994 epizootic). Both are gradient-free
    # and refit-free, so they run unconditionally.
    cf_release_only = simulate_no_invasion_counterfactual(sim, data, drop_disease=False)
    cf_no_disease = simulate_no_invasion_counterfactual(sim, data, drop_disease=True)
    memory_snapshot("map-viz-counterfactual", device)
    # Viability of the counterfactual's own world: no epizootic means K_base, hence a
    # larger viable area than the actual world's.
    _, viab_cf = modern_viability(sim, years, modern_era, k_key="K_base_flat")
    counterfactual_metrics = plot_counterfactual_attribution(
        sim, cf_release_only, cf_no_disease, viab_cf, years, rows, cols, shape,
        land_mask_arr, out / "15_counterfactual_no_invasion.png", modern_era,
        pop_scalar=float(np.asarray(data["pop_scalar"])), logscale=True)
    plot_counterfactual_attribution(
        sim, cf_release_only, cf_no_disease, viab_cf, years, rows, cols, shape,
        land_mask_arr, out / "15_counterfactual_no_invasion_linear.png", modern_era,
        pop_scalar=float(np.asarray(data["pop_scalar"])), logscale=False)
    create_counterfactual_animation(sim, cf_no_disease, years, land_mask_arr,
                                    out / "16_counterfactual_animation.mp4", logscale=True)
    create_counterfactual_animation(sim, cf_no_disease, years, land_mask_arr,
                                    out / "16_counterfactual_animation_linear.mp4",
                                    logscale=False)
    # Optional: needs the Great Plains zone raster, which is built from a shapefile
    # that is not in git. Returns None and explains itself when unavailable.
    barrier_metrics = plot_barrier_crossing(
        sim, data, cfg, dcfg, years, rows, cols, shape,
        out / "17_barrier_crossing.png", modern_era)
    # Companion to 17: resolves the pooled Q contrast by dispersal distance and by
    # covariate, and is the only directional result that survives rho >= 1.
    q_asym_metrics = plot_q_asymmetry_attribution(
        sim, data, cfg, dcfg, years, rows, cols, shape,
        out / "17b_q_asymmetry_attribution.png", modern_era, names=z_names)
    n50_raw = float(np.asarray(sim["n50_raw"])); n50 = float(np.logaddexp(0.0, n50_raw))
    transition_land = np.isfinite(transition)
    metrics = {
        "profile": args.profile, "checkpoint_step": int(checkpoint["step"]),
        "years": [int(years[0]), int(years[-1])],
        # The spans every "modern"/"baseline" number below was actually averaged
        # over -- named eras by default, or reconstructed from --window-years.
        "eras": {k: list(v) for k, v in era_spans.items()},
        "window_years_override": args.window_years,
        "fundamental_niche_definition": "post-establishment, density-independent local dominant eigenvalue of [[Sa, Sj], [Fmax*Sa, 0]]; excludes dispersal, density limitation, realized occupancy, and Allee limitation",
        "modern_mean_lambda": float(np.mean(modern)), "early_mean_lambda": float(np.mean(early)),
        "modern_suitable_fraction": float(np.mean(modern > 1.0)), "early_suitable_fraction": float(np.mean(early > 1.0)),
        "gained_suitable_fraction": float(np.mean(transition[transition_land] == 1)),
        "lost_suitable_fraction": float(np.mean(transition[transition_land] == -1)),
        # Same quantities against the invasion-year baseline (figures 01b/06b).
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
        # freedom. Prior medians: within-pair (Sa-Sj, F-K) 0.85, cross-group 0.70.
        # `loadings` is now the (4, 2) rank-2 factor matrix, not a 4-vector -- flattened
        # rowwise via .ravel(), since float() on a (2,) row raises TypeError.
        "manifold": {
            "corr_survival_repro": float(np.asarray(sim["rho"])),
            "corr_repro_capacity": float(np.asarray(sim["env_corr_repro_capacity"])),
            "corr_survival_capacity": float(np.asarray(sim["env_corr_survival_capacity"])),
            "corr_survival_adult_juv": (
                float(np.asarray(sim["env_corr_survival_adult_juv"]))
                if "env_corr_survival_adult_juv" in sim else None),
            "loadings": [float(x) for x in np.asarray(sim["manifold_loadings"]).ravel()],
            "loadings_shape": list(np.asarray(sim["manifold_loadings"]).shape),
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
        # K's bounded dynamic range. If a large share of land sits at the floor, the
        # covariates are pushing capacity as low as the bound permits -- the same
        # pathology as a saturated disease ceiling, one route over.
        "k_range": _k_range_metrics(sim),
        "realized_source_sink": source_sink_metrics,
        "counterfactual_no_invasion": counterfactual_metrics,
        # None when the Great Plains zone raster is absent; the key is always present
        # so a consumer can distinguish "not computed" from "computed as zero".
        "barrier_crossing": barrier_metrics,
        # Per-band / per-covariate breakdown of the journey-survival asymmetry.
        # None on the same missing inputs that skip barrier_crossing.
        "q_asymmetry": q_asym_metrics,
        "disease": disease_metrics,
        "age_structure": age_structure_metrics,
        # Which Z features carry the fitted environmental signal (05b), and which of
        # them actually moved lambda between eras (05c/05d).
        "z_attribution_since_invasion": z_attr_invasion,
        "z_attribution_since_early": z_attr_early,
        **ranking_metrics,
        **response_metrics,
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[map-viz] complete -> {out}")


if __name__ == "__main__":
    main()
