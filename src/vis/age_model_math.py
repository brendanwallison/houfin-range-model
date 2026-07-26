"""Shared, samples-axis-agnostic math for post-fit age-model diagnostics.

Every function here operates on plain arrays (numpy, or anything numpy-
broadcast-compatible) with no assumption about a leading "posterior sample"
axis -- call once for a MAP point estimate today, or in a loop/vmap per
posterior draw and stack the results once MCMC (ADVI/HMC) runs exist,
without changing this module. Plotting code owns the leading-axis
summarization (mean/percentile), not this module -- mirrors the pattern
``src/vis/_age_vis_common.py`` already uses for the legacy model's
sample-based visualizers.

Consumed today by ``scripts/viz/map_diagnostics.py``.
"""
import numpy as np

from src.config_utils import load_age_model_config
import jax.numpy as jnp

from src.model.age_priors import equilibrium_age_quantities
from src.temporal import load_timeline


def local_growth_lambda(Sa, Sj, F):
    """Dominant local eigenvalue for the forward model's census order.

    Adults survive before reproducing, hence the fecundity entry is ``F*Sa``:
    ``dominant_eigenvalue([[Sa, Sj], [F*Sa, 0]])``. Density-independent,
    dispersal-free, Allee-free -- the FUNDAMENTAL-niche quantity.
    """
    Sa, Sj, F = np.asarray(Sa), np.asarray(Sj), np.asarray(F)
    return (Sa + np.sqrt(np.maximum(Sa ** 2 + 4.0 * F * Sa * Sj, 0.0))) / 2.0


def realized_equilibrium(Sa, Sj, Fmax, K, allee_gamma):
    """Realized (density-dependent + Allee) equilibrium quantities.

    Thin numpy-facing wrapper around ``age_priors.equilibrium_age_quantities``
    (reused, not reimplemented, so this always matches the fitted model's own
    identifiability constraint). Returns ``(c, F_at_K, lambda_realized, rho)``.
    Unlike :func:`local_growth_lambda`, this includes crowding (``c``) and the
    Allee factor -- it is the REALIZED counterpart, always <= the fundamental
    niche's lambda for the same Sa/Sj/Fmax (K and the Allee effect only ever
    shrink, never expand, the demographically viable area).
    """
    c, F_at_K, lam, rho = equilibrium_age_quantities(
        jnp.asarray(Sa), jnp.asarray(Sj), jnp.asarray(Fmax),
        jnp.asarray(K), jnp.asarray(allee_gamma),
    )
    return np.asarray(c), np.asarray(F_at_K), np.asarray(lam), np.asarray(rho)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def softplus(x):
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _gamma_slope(latents, name, raw_name):
    """One demographic slope, tolerant of both the current and the pre-d7db319 names.

    Current model: ``name`` is a deterministic equal to 1.0. Older checkpoints:
    ``raw_name`` was a sampled site read through softplus. Neither present (a
    caller that folded in no deterministics at all) falls back to the fixed 1.0
    rather than raising, because the value is not actually in question.
    """
    if name in latents:
        return float(np.asarray(latents[name]))
    if raw_name in latents:
        return float(softplus(latents[raw_name]))
    return 1.0


def response_curve_fields(latents, z_sweep, target_idx):
    """Sweep one Z feature and return Sa/Sj/Fmax/K response curves.

    ``latents`` is a dict of raw MAP (or single-sample) parameter values as
    returned by ``checkpoints.auto_delta_params_to_latents`` -- the model's
    fitted alpha/gamma/w_env values, NOT the per-cell rate fields. ``z_sweep``
    is a 1D array of feature values; ``target_idx`` selects which of the M
    ``w_env`` rows is swept (every other feature held at 0, matching
    ``age_fields.project_and_scatter_age_structured``'s per-feature dot
    product).

    Link functions mirror ``age_fields.py`` exactly: sigmoid for Sa/Sj,
    softplus for Fmax/K. This is a deliberate correctness fix versus the
    deprecated ``src/vis/visualize_age_model.py``, whose own local response-
    curve code used ``exp`` for Fmax (stale -- the already-corrected version
    lives in ``src/vis/_age_vis_common.py``). H_s/H_r here are purely
    covariate-driven (Z.beta only), which matches age_fields.py exactly: an
    earlier model design mixed a shared spatiotemporal term into H_s/H_r that
    this synthetic single-point sweep never included, and that term is long gone
    from the real model too.

    K reads its own manifold ``H_k = Z.beta_k`` (``w_env[:, 2]``), not H_r; a
    2-column ``w_env`` from an older checkpoint falls back to beta_r with the old
    meaning. The K curve is BASE carrying capacity, before the continental time
    trend and before the mycoplasmal-conjunctivitis effect. That effect is ``K_base * (1 - severity(x)*gate(x,t)*(1-recovery))``,
    all three factors being functions of location and year, so a synthetic single
    Z point has no well-defined value for it. Read the fitted severity map in
    ``09_disease_diagnostics.png`` for that piece; a curve here that looks high
    versus the fitted K field is expected in post-arrival regions, not a bug.
    """
    if "w_env" not in latents:
        raise KeyError(
            "response_curve_fields needs 'w_env', which is a numpyro.deterministic "
            "under the one-factor manifold prior -- auto_delta_params_to_latents "
            "returns only SAMPLED sites, so the caller must fold the deterministic "
            "value in (see map_diagnostics.reconstruct_map)")
    w_env = np.asarray(latents["w_env"])  # (M, 3): beta_s, beta_r, beta_k
    beta_s, beta_r = w_env[:, 0], w_env[:, 1]
    # Capacity has its OWN manifold now; reusing beta_r here would silently plot a
    # curve the model does not use.
    beta_k = w_env[:, 2] if w_env.shape[1] > 2 else beta_r

    z_sweep = np.asarray(z_sweep)
    H_s = z_sweep * beta_s[target_idx]
    H_r = z_sweep * beta_r[target_idx]
    H_k = z_sweep * beta_k[target_idx]

    alpha_a = float(latents["alpha_a"])
    alpha_j = float(latents["alpha_j"])
    alpha_f = float(latents["alpha_f"])
    # K = softplus(alpha_k + gamma_k*H_k + trend) in DENSITY space. Earlier revisions
    # used a log link with the level sampled in route counts (log_k_level_counts), and
    # before that softplus over an unbounded argument; both names are accepted so an
    # older checkpoint still plots.
    pop = float(load_age_model_config()["population_model"].get(
        "population_scale_route_counts_per_relative_unit",
        load_age_model_config()["population_model"].get(
            "population_scale_birds_per_relative_unit", 1.0)))
    # Intercept on the softplus link's own scale (route counts), inverted from the
    # reported level when only the transformed value is available.
    _lvl_cfg = load_age_model_config()["population_model"]["capacity_level_prior"]
    if "k_level" in latents:                      # deterministic, already in density
        k_level = float(np.asarray(latents["k_level"]))
    elif "alpha_k" in latents:                     # softplus link, DENSITY space
        k_level = float(softplus(latents["alpha_k"]))
    elif "log_k_level_counts" in latents:          # exp link (previous run)
        k_level = float(np.exp(latents["log_k_level_counts"])) / pop
    else:
        k_level = float(softplus(latents["alpha_k"]))  # pre-run_11 checkpoint
    from src.model.age_priors import _ALPHA_K_LOC
    alpha_k = float(latents.get("alpha_k", _ALPHA_K_LOC))
    # The demographic slopes are dimensionless and FIXED AT 1 (their amplitude moved
    # into w_scale to remove three flat ridges -- see age_priors.py). They are emitted
    # as numpyro.deterministic under the friendly names, plus *_raw constants equal to
    # softplus^-1(1) for older readers. Deterministic sites are absent from
    # auto_delta_params_to_latents, so the caller must fold them in (see
    # map_diagnostics.reconstruct_map); the fallbacks below keep pre-d7db319
    # checkpoints, where the *_raw names were genuinely sampled, plotting correctly.
    gamma_a = _gamma_slope(latents, "gamma_a", "gamma_a_raw")
    gamma_j = gamma_a + float(latents["gamma_j_diff"])  # sampled; always present
    gamma_f = _gamma_slope(latents, "gamma_f", "gamma_f_raw")
    gamma_k = _gamma_slope(latents, "gamma_k", "gamma_k_raw")

    return {
        "Sa": sigmoid(alpha_a + gamma_a * H_s),
        "Sj": sigmoid(alpha_j + gamma_j * H_s),
        "Fmax": softplus(alpha_f + gamma_f * H_r),
        # Matches age_fields: softplus(alpha_k + gamma_k*H_k) in DENSITY space. The
        # disease effect is omitted -- it is a function of location and year, so a
        # synthetic single-Z sweep has no value for it.
        "K": softplus(alpha_k + gamma_k * H_k),
    }


def scatter_to_grid(flat, rows, cols, shape):
    """Scatter a ``(..., N_land)`` flat array onto a ``(..., *shape)`` grid.

    Cells outside ``rows``/``cols`` are NaN. The leading ``...`` batch axis
    (e.g. time, or a future posterior-sample axis) is preserved untouched.
    """
    flat = np.asarray(flat)
    grid = np.full((*flat.shape[:-1], *shape), np.nan, dtype="float32")
    grid[..., rows, cols] = flat
    return grid


def window_mean(values, n):
    """Trailing- and leading-``n``-step means of a ``(time, ...)`` array."""
    n = min(n, values.shape[0])
    return np.nanmean(values[-n:], axis=0), np.nanmean(values[:n], axis=0), n


def baseline_window_mean(values, years, n, ref_year=None):
    """Mean of the ``n``-year window STARTING at ``ref_year``, plus its year span.

    ``window_mean``'s leading window is anchored at the start of the model
    timeline (1902), which is a defensible "before anything happened" baseline
    but a poor one for questions about the invasion itself: the 1902 climate is
    also 38 years of climate change away from the release. Anchoring the
    baseline at ``invasion_year`` instead measures change relative to the
    conditions the species actually encountered when it arrived.

    ``ref_year=None`` reproduces ``window_mean``'s leading window exactly. A
    ref_year past the end of the timeline raises rather than silently sliding
    the window backward. Returns ``(mean, n_used, (first_year, last_year))``.
    """
    years = np.asarray(years)
    if ref_year is None:
        i0 = 0
    else:
        matches = np.flatnonzero(years == int(ref_year))
        if not matches.size:
            raise ValueError(f"ref_year {ref_year} is not in the model timeline "
                             f"{years[0]}-{years[-1]}")
        i0 = int(matches[0])
    n = int(min(n, values.shape[0] - i0))
    if n < 1:
        raise ValueError(f"no timeline left after ref_year {ref_year}")
    return (np.nanmean(values[i0:i0 + n], axis=0), n,
            (int(years[i0]), int(years[i0 + n - 1])))


def add_timeline_markers(ax, tl=None, show_invasion=True, show_bbs_start=True, **line_kwargs):
    """Draw reference vertical lines for the invasion year and BBS data start.

    A light dashed line + small rotated label at ``invasion_year`` (1940, the
    modeled NYC release pulse) and/or ``bbs_start_year`` (1966, the first
    year with any real observational constraint on the model). Call this
    AFTER the axis's data is plotted so ``ax.get_xlim()`` reflects the actual
    year range (a marker outside the current x-limits is skipped).
    """
    tl = tl or load_timeline()
    style = dict(color="0.4", linestyle="--", linewidth=1.0, alpha=0.7, zorder=0)
    style.update(line_kwargs)
    xlo, xhi = ax.get_xlim()
    for show, year, label in (
        (show_invasion, tl["invasion_year"], "invasion (1940)"),
        (show_bbs_start, tl["bbs_start_year"], "BBS start (1966)"),
    ):
        if not show or not (xlo <= year <= xhi):
            continue
        ax.axvline(year, **style)
        ax.annotate(
            label, xy=(year, 1.0), xycoords=("data", "axes fraction"),
            xytext=(3, -3), textcoords="offset points",
            fontsize=7, color="0.35", va="top", ha="left", rotation=90,
        )
