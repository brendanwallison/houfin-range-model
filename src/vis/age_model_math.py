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
    covariate-driven (Z.beta only), which now matches age_fields.py exactly --
    an earlier model design mixed a shared spatiotemporal term into H_s/H_r
    that this synthetic single-point sweep never included; that term has
    since been removed from the real model too (replaced by a K-only latent
    correction, see age_fields.py's _K_CORRECTION_OFFSET), so this sweep is no
    longer an approximation on that front. The K curve returned here is the
    BASE carrying capacity (before that K-only correction), since a synthetic
    single Z point has no associated spatiotemporal-basis value to plug in.
    """
    w_env = np.asarray(latents["w_env"])  # (M, 2): [:, 0]=beta_s, [:, 1]=beta_r
    beta_s, beta_r = w_env[:, 0], w_env[:, 1]

    z_sweep = np.asarray(z_sweep)
    H_s = z_sweep * beta_s[target_idx]
    H_r = z_sweep * beta_r[target_idx]

    alpha_a = float(latents["alpha_a"])
    alpha_j = float(latents["alpha_j"])
    alpha_f = float(latents["alpha_f"])
    alpha_k = float(latents["alpha_k"])
    gamma_a = float(softplus(latents["gamma_a_raw"]))
    gamma_j = gamma_a + float(latents["gamma_j_diff"])
    gamma_f = float(softplus(latents["gamma_f_raw"]))
    gamma_k = float(softplus(latents["gamma_k_raw"]))

    return {
        "Sa": sigmoid(alpha_a + gamma_a * H_s),
        "Sj": sigmoid(alpha_j + gamma_j * H_s),
        "Fmax": softplus(alpha_f + gamma_f * H_r),
        "K": softplus(alpha_k + gamma_k * H_r),
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
