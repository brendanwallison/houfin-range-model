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

    DO NOT THRESHOLD ``lambda_realized`` AT 1 to classify source vs sink. ``c`` is
    solved so lambda = 1 at N = K, so this returns exactly 1 wherever the Allee
    factor saturates -- the bound above is an equality there, not slack. Which side
    of 1 such a cell lands on is decided by the ``1e-6`` guard in ``1 - Sa + 1e-6``,
    not by demography. Use :func:`allee_viability` for that classification. What
    ``lambda_realized`` legitimately measures is the SHORTFALL below 1: the
    fractional growth-rate cost the Allee effect imposes at carrying capacity.
    """
    c, F_at_K, lam, rho = equilibrium_age_quantities(
        jnp.asarray(Sa), jnp.asarray(Sj), jnp.asarray(Fmax),
        jnp.asarray(K), jnp.asarray(allee_gamma),
    )
    return np.asarray(c), np.asarray(F_at_K), np.asarray(lam), np.asarray(rho)


def allee_viability(Sa, Sj, Fmax, K, allee_gamma, n_grid=512):
    """Does a positive equilibrium exist? The Allee sanity check on the niche.

    THE QUESTION. ``local_growth_lambda`` gives the FUNDAMENTAL niche: is habitat
    good enough to grow, ignoring density. But a cell can pass that test and still
    be uninhabitable because K is so small the population never escapes the
    mate-finding Allee regime. This function answers "given this K, can the cell
    sustain a population at ANY density?"

    THE DERIVATION, entirely from the forward model's own one-year update
    (``age_forward.reproduction_age_structured``). Realized fecundity at density N:

        F(N) = Fmax / (1 + c*N/K) * (1 - exp(-allee_gamma*N))
               \\_____crowding____/   \\______mate finding______/

    Freezing N makes the update linear with projection matrix
    ``[[Sa, Sj], [F(N)*Sa, 0]]`` (the census order documented in
    ``local_growth_lambda``), whose characteristic polynomial is
    ``p(lam) = lam^2 - Sa*lam - F(N)*Sa*Sj``. It opens upward with a single
    positive root, so that root exceeds 1 exactly when ``p(1) < 0``:

        lambda(N) > 1   <=>   F(N) > (1 - Sa) / (Sa * Sj)  ==  F_replacement

    which is just R0 > 1: an adult lives ``1/(1-Sa)`` expected years, makes F
    juveniles a year, each recruiting with probability Sj, so lifetime recruits
    per adult is ``F*Sa*Sj/(1-Sa)``.

    F(N) is NON-MONOTONIC -- rising in N through mate finding, falling through
    crowding -- so it peaks at an interior density. The cell is viable iff even
    its best density clears replacement:

        viable   <=>   max_N F(N) > F_replacement

    That is the saddle-node (fold) condition: above it there are two equilibria,
    an unstable Allee threshold N_crit and a stable one near K; at it they collide
    and annihilate; below it the population declines from every starting density,
    however many birds arrive. When K is large the Allee factor saturates (scale
    ``1/allee_gamma``) long before crowding bites, so ``max F ~ Fmax`` and any
    fundamentally-suitable cell is viable. As K shrinks toward ``1/allee_gamma``
    the brake engages while mate finding is still poor, ``max F`` never reaches
    replacement, and the cell is Allee-dead despite ``lambda_fundamental > 1``.

    WHY NOT EVALUATE AT N = K, as ``realized_equilibrium`` does. ``c`` is defined
    by ``Fmax/(1+c) = F_replacement`` (see ``equilibrium_age_quantities``), i.e.
    it is solved so lambda = 1 at N = K. So ``lambda(K) == 1`` identically, up to
    the ``1e-6`` guard in ``1 - Sa + 1e-6``, and thresholding it at 1 classifies
    on that regularizer rather than on biology: it demanded the Allee suppression
    at K fall below ~2.5e-6, putting the contour at K ~ 3.7 route counts when the
    true fold sits near 0.5-1.2. This criterion has no such free constant.

    Returns a dict of arrays broadcast to the inputs' shape: ``viable``,
    ``F_peak``, ``N_peak``, ``F_replacement``, ``suppression_at_K``
    (``1 - allee_factor(K)``, i.e. the fractional fecundity cost of the Allee
    effect at carrying capacity -- how close the cell sits to the edge),
    ``fundamental_viable`` (``c > 0``, the density-free test, for contrast), and
    the two equilibria bracketing the viable interval:

    * ``N_crit`` -- the LOWER root of ``F(N) = F_replacement``. Unstable: below it
      the population declines to zero, above it grows toward ``N_star``. This is
      the Allee threshold a propagule must exceed to establish, so it is the
      denominator of any propagule-pressure or invasion-pinning calculation.
    * ``N_star`` -- the UPPER root. Stable, and slightly BELOW K rather than at it,
      because ``F(K) = F_replacement * allee(K)`` and ``allee(K) < 1``.

    Both are NaN where the cell is not viable (no roots exist -- the fold). They
    are read off the same scan by locating the sign changes of ``F - F_repl`` and
    interpolating linearly in ``log N``, so their accuracy is the grid's ~1.4%
    spacing in N; that is far finer than the uncertainty in K itself.

    The maximization is a log-spaced scan of ``N/K`` over six decades rather than
    a root-find: the stationarity condition is transcendental, the scan is
    vectorized and allocation-bounded, and the fold is a smooth maximum so modest
    grid error cannot flip a cell that is not already within a hair of the
    boundary. Costs ``n_grid`` floats per cell -- call it on window-MEAN fields,
    not on the full (time, cell) stack.
    """
    Sa, Sj, Fmax, K = (np.asarray(x, dtype=float) for x in (Sa, Sj, Fmax, K))
    allee_gamma = np.asarray(allee_gamma, dtype=float)
    Sa, Sj, Fmax, K, allee_gamma = np.broadcast_arrays(Sa, Sj, Fmax, K, allee_gamma)

    F_repl = (1.0 - Sa) / (Sa * Sj)
    # Same c as the fitted model, including the 1e-6 guard: it belongs in the
    # forward simulation (it protects Sa -> 1) and must stay bit-identical here.
    # It just no longer decides a class boundary.
    c = np.maximum((Fmax * Sa * Sj) / (1.0 - Sa + 1e-6) - 1.0, 0.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # N = u*K, u log-spaced over [1e-5, 10]. The peak lies in (0, ~K]: for
        # K >> 1/gamma it sits where crowding starts to bite, for K <~ 1/gamma at
        # a fraction of K. Six decades covers both without a per-cell grid.
        u = np.logspace(-5.0, 1.0, int(n_grid)).reshape((-1,) + (1,) * Sa.ndim)
        N = u * K
        F = Fmax / (1.0 + c * u) * (-np.expm1(-allee_gamma * N))
        F = np.where(np.isfinite(F), F, -np.inf)
        k_peak = np.argmax(F, axis=0)
        F_peak = np.take_along_axis(F, k_peak[None], axis=0)[0]
        N_peak = np.take_along_axis(N, k_peak[None], axis=0)[0]
        suppression = np.exp(-allee_gamma * K)

    fundamental_viable = c > 0.0
    # A cell with c == 0 fails replacement at ANY density (F <= Fmax <= F_repl),
    # so the scan already excludes it; the explicit conjunction just documents that
    # viability is fundamental suitability AND Allee escape, never Allee alone.
    viable = fundamental_viable & (F_peak > F_repl) & (K > 0.0)
    n_crit, n_star = _bracket_roots(N, F, F_repl, viable)
    return {"viable": viable, "F_peak": F_peak, "N_peak": N_peak,
            "F_replacement": F_repl, "suppression_at_K": suppression,
            "fundamental_viable": fundamental_viable,
            "N_crit": n_crit, "N_star": n_star}


def _bracket_roots(N, F, F_repl, viable):
    """Lower/upper roots of ``F(N) = F_repl`` from the scan, by sign change.

    ``F`` is ``(n_grid, ...)`` evaluated at densities ``N`` on a log grid, and is
    unimodal in the grid index (rising through mate-finding, falling through
    crowding), so ``g = F - F_repl`` has at most two sign changes: up at ``N_crit``
    and down at ``N_star``. Interpolating linearly in ``log N`` rather than ``N``
    matches the grid's own spacing, so the error is set by the grid, not by
    curvature over a wide interval.
    """
    g = F - F_repl
    pos = g > 0.0
    # argmax on a boolean gives the FIRST True (and 0 if none, screened by `viable`).
    first = np.argmax(pos, axis=0)
    last = pos.shape[0] - 1 - np.argmax(pos[::-1], axis=0)

    def _interp(hi_idx, lo_idx):
        """Root between grid points lo_idx (g<0) and hi_idx (g>0), in log N."""
        hi = np.clip(hi_idx, 0, g.shape[0] - 1)
        lo = np.clip(lo_idx, 0, g.shape[0] - 1)
        g_hi = np.take_along_axis(g, hi[None], axis=0)[0]
        g_lo = np.take_along_axis(g, lo[None], axis=0)[0]
        n_hi = np.take_along_axis(N, hi[None], axis=0)[0]
        n_lo = np.take_along_axis(N, lo[None], axis=0)[0]
        denom = g_hi - g_lo
        # Degenerate bracket (equal g, or the root sits at the grid edge so there is
        # no straddling pair): fall back to the in-bracket endpoint rather than 0/0.
        w = np.where(np.abs(denom) > 0.0, -g_lo / np.where(denom == 0.0, 1.0, denom), 0.0)
        w = np.clip(w, 0.0, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_n = np.log(n_lo) + w * (np.log(n_hi) - np.log(n_lo))
        return np.exp(log_n)

    n_crit = np.where(viable & (first > 0), _interp(first, first - 1), np.nan)
    n_star = np.where(viable & (last < g.shape[0] - 1), _interp(last, last + 1), np.nan)
    return n_crit, n_star


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


def demographic_params(latents):
    """Unpack the fitted link parameters shared by every synthetic-Z diagnostic.

    Returns a dict with ``beta_s``/``beta_r``/``beta_k`` (the three ``w_env``
    columns), the four ``alpha_*`` intercepts and the four ``gamma_*`` slopes,
    all on the links' own scales, ready to feed ``sigmoid``/``softplus`` exactly
    as ``age_fields.py`` does.

    Split out of :func:`response_curve_fields` so the Z-feature attribution maps
    share one copy of the checkpoint-compatibility logic (the K link changed
    twice, and the gamma slopes moved from sampled to deterministic); two copies
    would drift and silently plot a curve the model does not use.
    """
    if "w_env" not in latents:
        raise KeyError(
            "demographic_params needs 'w_env', which is a numpyro.deterministic "
            "under the one-factor manifold prior -- auto_delta_params_to_latents "
            "returns only SAMPLED sites, so the caller must fold the deterministic "
            "value in (see map_diagnostics.reconstruct_map)")
    w_env = np.asarray(latents["w_env"])  # (M, 3): beta_s, beta_r, beta_k
    beta_s, beta_r = w_env[:, 0], w_env[:, 1]
    # Capacity has its OWN manifold now; reusing beta_r here would silently plot a
    # curve the model does not use. Two-column checkpoints predate the split.
    beta_k = w_env[:, 2] if w_env.shape[1] > 2 else beta_r
    # Juvenile survival is the APPENDED 4th column (rank-2 manifold prior). Older
    # checkpoints have 3 columns, where Sj shared beta_s outright -- fall back to that.
    beta_sj = w_env[:, 3] if w_env.shape[1] > 3 else beta_s

    # K = softplus(alpha_k + gamma_k*H_k + trend) in DENSITY space. Earlier revisions
    # used a log link with the level sampled in route counts (log_k_level_counts);
    # those checkpoints have no alpha_k, so fall back to the prior location rather
    # than inverting a level that the softplus link no longer uses.
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
    return {
        "beta_sj": beta_sj,
        "w_env": w_env,
        "beta_s": beta_s, "beta_r": beta_r, "beta_k": beta_k,
        "alpha_a": float(latents["alpha_a"]),
        "alpha_j": float(latents["alpha_j"]),
        "alpha_f": float(latents["alpha_f"]),
        "alpha_k": alpha_k,
        "gamma_a": gamma_a,
        # gamma_j: prefer the DETERMINISTIC `gamma_j` site, which is the quantity itself.
        # gamma_j_diff used to be sampled ("always present"), but it became a deterministic 0.0
        # when juvenile survival got its own manifold and gamma_j was fixed at 1 -- and
        # auto_delta_params_to_latents returns SAMPLED sites only, so reading gamma_j_diff here
        # raised KeyError and killed map_diagnostics after figure 04, before metrics.json. Fall
        # back to the old reconstruction so pre-rank-2 checkpoints still load.
        "gamma_j": (float(np.asarray(latents["gamma_j"])) if "gamma_j" in latents
                    else gamma_a + float(latents["gamma_j_diff"])),
        "gamma_f": _gamma_slope(latents, "gamma_f", "gamma_f_raw"),
        "gamma_k": _gamma_slope(latents, "gamma_k", "gamma_k_raw"),
    }


def rates_from_manifolds(p, H_s, H_r, H_k=None):
    """Apply the model's link functions to already-projected manifold values.

    ``p`` is a :func:`demographic_params` dict; ``H_s``/``H_r``/``H_k`` are
    ``Z . beta_*`` for survival, reproduction and capacity. Links mirror
    ``age_fields.py`` exactly: sigmoid for Sa/Sj, softplus for Fmax/K. K is BASE
    capacity -- before the continental time trend and before the mycoplasmal-
    conjunctivitis effect, which are functions of location and year and so have
    no value for a synthetic Z point.
    """
    out = {
        "Sa": sigmoid(p["alpha_a"] + p["gamma_a"] * H_s),
        "Sj": sigmoid(p["alpha_j"] + p["gamma_j"] * H_s),
        "Fmax": softplus(p["alpha_f"] + p["gamma_f"] * H_r),
    }
    if H_k is not None:
        out["K"] = softplus(p["alpha_k"] + p["gamma_k"] * H_k)
    return out


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
    p = demographic_params(latents)
    z_sweep = np.asarray(z_sweep)
    return rates_from_manifolds(
        p,
        H_s=z_sweep * p["beta_s"][target_idx],
        H_r=z_sweep * p["beta_r"][target_idx],
        H_k=z_sweep * p["beta_k"][target_idx],
    )


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


# Named comparison eras, as inclusive (first_year, last_year) spans. Diagnostics
# that contrast "now" against "before" should name an era rather than take a
# trailing-N window: a trailing window silently moves when the timeline is
# extended, and the leading window ("timeline start") answers a different
# question from "at the release". These three are the spans the figures actually
# compare -- 1902-1915 is pre-release baseline climate, 1940-1955 is the
# conditions the species met when it arrived, 2010-2025 is the modern period.
ERAS = {
    "early": (1902, 1915),
    "invasion": (1940, 1955),
    "modern": (2010, 2025),
}


def era_span(era, years):
    """Resolve an era name (or an explicit ``(first, last)`` pair) to indices.

    Returns ``(i0, i1_exclusive, (first_year, last_year))`` clipped to the
    available timeline. Raises if the era does not overlap ``years`` at all --
    silently returning an empty or slid window would put a wrong year range in a
    figure title, which is exactly the failure this replaces.
    """
    lo, hi = ERAS[era] if isinstance(era, str) else tuple(int(v) for v in era)
    years = np.asarray(years)
    idx = np.flatnonzero((years >= lo) & (years <= hi))
    if not idx.size:
        raise ValueError(
            f"era {era!r} ({lo}-{hi}) does not overlap the model timeline "
            f"{int(years[0])}-{int(years[-1])}")
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    return i0, i1, (int(years[i0]), int(years[i1 - 1]))


def era_mean(values, years, era):
    """Mean of a ``(time, ...)`` array over the inclusive year span of ``era``.

    The era-named sibling of :func:`baseline_window_mean`. Returns
    ``(mean, (first_year, last_year), n_used)`` where the span is the ACTUAL
    clipped span, so callers can title a figure with what was really averaged
    rather than with what they asked for.
    """
    i0, i1, span = era_span(era, years)
    return np.nanmean(np.asarray(values)[i0:i1], axis=0), span, i1 - i0


def eras_from_window(years, window):
    """Rebuild era-like spans from a legacy trailing-``window`` size.

    Preserves the pre-era ``--window-years`` behaviour as an override: "modern"
    becomes the trailing ``window`` years, and each historical era becomes a
    ``window``-year span anchored at its own start year (which is what
    ``baseline_window_mean(ref_year=...)`` did). Returns a dict shaped like
    :data:`ERAS`, suitable for passing straight back in as an explicit span.
    """
    years = np.asarray(years)
    window = int(window)
    out = {}
    for name, (lo, _) in ERAS.items():
        if name == "modern":
            out[name] = (int(years[max(0, len(years) - window)]), int(years[-1]))
            continue
        i0 = int(np.flatnonzero(years >= lo)[0])
        i1 = min(i0 + window, len(years)) - 1
        out[name] = (int(years[i0]), int(years[i1]))
    return out


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
