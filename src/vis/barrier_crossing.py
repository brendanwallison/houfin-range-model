"""Directed cost of crossing a barrier, from the fitted dispersal operator.

WHAT THIS MEASURES. Juvenile dispersal in this model carries a spatially varying,
directionally asymmetric survival cost: ``Q_k(x) = sigmoid(alpha_j + gamma_j *
(Z_disp[x,:,k] . beta_s))`` is applied AFTER the convolution and is never
renormalized (``age_forward.juvenile_dispersal_vectorized``), so it is genuine
transit mortality. ``Q`` is indexed by the ARRIVAL cell and by cohort, where a
cohort is a (direction, radial band) pair -- 4 wedges x 3 bands = 12. The journey
east out of a cell and the journey west out of the same cell therefore carry
different weights, drawn from different ``Z_disp`` slices. That is the asymmetry
this module quantifies.

``Q`` has no free parameters of its own: it reuses the juvenile survival law. So
the asymmetry is a PREDICTION from path geometry plus locally-fitted survival,
never something fitted to crossing data.

WHY NOT JUST REPORT LONG-DISTANCE KERNEL MASS. Across a juvenile-MDD sweep the
kernel is a different object at every point, so any functional of the kernel alone
is incomparable. The measure below is in units of expected descendants, and it
optimizes over routes: one long jump versus many short hops that reproduce inside
the barrier. Which strategy wins is exactly what changes with MDD.

THE MEASURE. Linearize the model's own annual update at low density and restrict it
to the barrier. Writing ``a``/``j`` for adult/juvenile density and ``p0`` for the
low-density departure probability, with ``a_tilde`` the adult field after dispersal:

    a'(x) = Sa(x)*a_tilde(x) + Sj(x)*(1-p0)*j(x) + sum_y D(x<-y)*p0*j(y)
    j'(x) = Sa(x)*Fmax(x)*a_tilde(x)
    D(x<-y) = sum_k kappa_k(x-y) * Q_k(x) / f_k(y)

Every term is the model's own. ``D`` is sub-stochastic -- ``sum_x D(x<-y) < 1`` --
and the deficit IS the transit mortality. Two substitutions turn the exact update
into this linear one: crowding ``1/(1+cN/K) -> 1``, which is the correct ``N -> 0``
limit at a front, and the Allee factor ``(1-exp(-gamma*N)) -> 1``, which is NOT
(its true limit is 0). The second is a deliberate OPTIMISTIC assumption, so the
operator is an UPPER BOUND on real low-density dynamics:

* crossing fails under it  => crossing fails in the real model (a rigorous
  one-sided conclusion, and the useful direction);
* crossing succeeds under it => succeeds only if the Allee threshold is also
  cleared, which is the separate propagule-pressure check against ``N_crit``
  from ``age_model_math.allee_viability``.

Consistency: with ``p0 = 0`` the single-cell map has dominant eigenvalue > 1 exactly
when ``Fmax*Sa*Sj/(1-Sa) > 1``, the same replacement condition ``allee_viability``
uses for the fold. The two analyses agree by construction.

CROSSING GAIN. With ``A_P`` the operator projected onto the barrier after each
application, and ``b`` the adults established in the barrier in one year by the
source-side population,

    G = 1'_T . A . (I - A_P)^-1 . b  /  1'.b

``(I - A_P)^-1 = sum_k A_P^k`` is the fundamental matrix of an absorbing process
(absorption = leaving the barrier, to the target, back to the source, or by death);
term ``k`` is the lineage still inside after ``k`` years, so the sum counts every
route of every length. ``G`` is dimensionless: expected adult arrivals on the far
side per adult established in the barrier. The series converges iff the spectral
radius ``rho(A_P) < 1``, which is exactly the statement that any lineage confined to
the barrier dies out -- i.e. that it IS a barrier. ``rho >= 1`` means the barrier
contains self-sustaining habitat; that is a reportable verdict, not a failure.

Nothing is ever formed as a matrix: the iteration applies the existing FFT operator.
"""
from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from src.model.age_forward import juvenile_dispersal_vectorized, rightpad_convolution
from src.vis.age_model_math import allee_viability, era_mean, era_span

# Direction -> (source zone, target zone) in the west|barrier|east partition.
DIRECTIONS = {"east_to_west": ("east", "west"), "west_to_east": ("west", "east")}

# Iterate magnitude, relative to the mass entering the barrier, below which the decay
# ratio is float32 rounding rather than the operator. Sampling rho below this reports
# noise; above it the ratio converges to the spectral radius.
_RATIO_SIGNAL_FLOOR = 1e-7


def low_density_departure_probability(latents, target_fraction, years=None, era=None):
    """``p0 = sigmoid(beta0 + mean(r_t) - beta1 * tau)``, a scalar.

    The model's departure probability is
    ``sigmoid(beta0 + beta1*(N/K - tau) + r_t)`` and is shared by both age classes
    (``age_forward.dispersal_step_age_structured``). At an invasion front ``N/K -> 0``,
    so the density term collapses to ``-beta1*tau`` and ``p0`` is the same number
    everywhere -- it is not a field. ``r_t`` (``dispersal_random``) has prior scale
    0.001 so it is negligible; its mean over the analysis era is used rather than
    any single year, to avoid a year choice mattering. Omit ``years``/``era`` to
    average over the whole timeline.
    """
    beta0 = float(np.asarray(latents["dispersal_logit_intercept"]))
    beta1 = float(np.asarray(latents["dispersal_logit_slope"]))
    r = np.asarray(latents["dispersal_random"], dtype=float)
    if r.size and years is not None and era is not None:
        i0, i1, _ = era_span(era, years)
        r_bar = float(np.mean(r[i0:i1]))
    else:
        r_bar = float(np.mean(r)) if r.size else 0.0
    z = beta0 + r_bar - beta1 * float(target_fraction)
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -10.0, 10.0))))


def modern_dispersal_fields(sim, data, years, era, rows, cols, shape):
    """Era-mean grids the operator needs, plus the state it starts from.

    Everything is reduced to the modern era first, matching how every other
    "modern" field in the diagnostics is built, and scattered to the full grid
    because the FFT operator works on grids, not on the land-cell vector.
    """
    def grid(flat_t):
        m, _, _ = era_mean(np.asarray(flat_t), years, era)
        g = np.zeros(shape, dtype="float64")
        g[rows, cols] = m
        return g

    def grid_stack(flat_t):
        """(time, N_land, K) -> (K, Ny, Nx) era mean."""
        m, _, _ = era_mean(np.asarray(flat_t), years, era)   # (N_land, K)
        g = np.zeros((m.shape[-1], *shape), dtype="float64")
        g[:, rows, cols] = m.T
        return g

    a_m, _, _ = era_mean(np.asarray(sim["Na_grid"]), years, era)
    j_m, _, _ = era_mean(np.asarray(sim["Nj_grid"]), years, era)
    return {
        "Sa": grid(sim["Sa_flat"]), "Sj": grid(sim["Sj_flat"]),
        "Fmax": grid(sim["Fmax_flat"]), "K": grid(sim["K_flat"]),
        "Q": grid_stack(sim["Q_flat"]),
        "a0": np.asarray(a_m, dtype="float64"), "j0": np.asarray(j_m, dtype="float64"),
        "land": np.asarray(data["land_mask"], dtype="float64"),
    }


def annual_operator(fields, data, p0, use_edge_correction=True, q_override=None):
    """Return ``step((a, j)) -> (a', j')``: the linearized annual operator.

    ``use_edge_correction=False`` replaces the source-side ``1/f`` terms with 1.
    That term is itself strongly directional -- it inflates outgoing mass from cells
    whose wedge is mostly ocean -- and is anti-correlated with the geography ``Q``
    measures, so any asymmetry result should be checked both ways (see
    :func:`edge_correction_summary`). ``q_override`` replaces ``Q`` wholesale, which
    is how the ``Q == 1`` symmetry test isolates genuine ``Z_disp`` asymmetry from an
    index-convention error.
    """
    Sa, Sj, Fmax = (jnp.asarray(fields[k]) for k in ("Sa", "Sj", "Fmax"))
    land = jnp.asarray(fields["land"])
    Q = jnp.asarray(fields["Q"] if q_override is None else q_override)
    adult_kernel = jnp.asarray(data["adult_fft_kernel"])
    juv_kernels = jnp.asarray(data["juvenile_fft_kernel_stack"])
    if use_edge_correction:
        f_a = jnp.asarray(data["adult_edge_correction"])
        f_j = jnp.asarray(data["juvenile_edge_correction_stack"])
    else:
        f_a = jnp.ones_like(land)
        f_j = jnp.ones_like(Q)

    def raw_step(a, j):
        a, j = jnp.asarray(a), jnp.asarray(j)
        # Adults: isotropic kernel, and NO Q -- adults pay no journey survival cost
        # in this model (age_forward has no Q factor on adult_arriving).
        a_tilde = (1.0 - p0) * a + rightpad_convolution(a * p0 / (f_a + 1e-6), adult_kernel)
        # Juveniles: the model's own call, so the Q alignment and the mass-losing
        # multiply are inherited rather than re-derived here.
        j_arriving = juvenile_dispersal_vectorized(j * p0, juv_kernels, Q, f_j, 1e-6)
        # Arrivers are NOT multiplied by Sj again: Q replaced local juvenile survival
        # for movers (age_forward.reproduction_age_structured).
        a_new = Sa * a_tilde + Sj * (1.0 - p0) * j + j_arriving
        j_new = Sa * Fmax * a_tilde
        return a_new * land, j_new * land

    # LINEARIZATION. age_forward.rightpad pads the population grid with
    # pad_value=1e-9, NOT zero, so every convolution injects a constant fictitious
    # population that convolves back into the domain. The model's operator is
    # therefore AFFINE, not linear: raw_step(u) = A*u + c. That is harmless in the
    # forward simulation, where real densities are many orders of magnitude above
    # 1e-9, but it is fatal here -- the resolvent iteration deliberately runs the
    # mass down toward zero, and the offset floors it, so the decay ratio stalls at
    # exactly 1.0 and both rho and the Neumann sum become meaningless.
    #
    # c is recovered exactly as raw_step(0, 0) and subtracted, which recovers the
    # true linear A while still using the model's own operator verbatim rather than
    # reimplementing the convolution with a different pad. Clamping at zero is safe:
    # A maps the non-negative cone into itself, so anything negative after the
    # subtraction is float error, not signal.
    zero = jnp.zeros_like(land)
    c_a, c_j = raw_step(zero, zero)

    def step(a, j):
        a_new, j_new = raw_step(a, j)
        return jnp.maximum(a_new - c_a, 0.0), jnp.maximum(j_new - c_j, 0.0)

    return step


def crossing_gain(fields, data, zones, direction, p0, horizon_years=124,
                  max_years=2000, tol=1e-9, use_edge_correction=True, q_override=None):
    """Crossing gain ``G``, ``rho(A_P)``, the arrival field, and the corridor.

    TWO gains are returned, because the infinite-horizon one can legitimately be
    infinite. Under the Allee-optimistic linearization a barrier cell with
    ``R0 = Fmax*Sa*Sj/(1-Sa) > 1`` self-sustains, so if any patch inside the barrier
    is fundamentally suitable then ``rho(A_P) >= 1``, the Neumann series diverges, and
    the honest reading is "under the optimistic assumption this is not a barrier at
    all -- the barrier is entirely an Allee/K phenomenon". That is a real result, but
    it leaves nothing to plot against MDD, so:

    * ``G_horizon`` -- arrivals within ``horizon_years`` (default 124, the model
      timeline) per adult entering the barrier. ALWAYS finite, always directional,
      always comparable across sweep points. This is the primary metric.
    * ``G_total`` -- the infinite-horizon sum. Equals ``G_horizon`` plus the tail when
      ``rho < 1``; ``inf`` when the barrier self-sustains. Valid only if ``converged``.

    ``rho`` is a property of the barrier-restricted operator ALONE, so it does not
    depend on ``direction``; both directions returning the same value is itself a
    useful consistency check, which is why it is computed per call rather than shared.

    ``zones`` maps ``"west"``/``"barrier"``/``"east"`` to boolean grids forming a
    gap-free partition (``data.preprocess.great_plains.corridor_zones``). Gap-free
    matters: mass landing in an unclassified cell would be neither counted as an
    arrival nor carried forward, so it would silently vanish.

    Accounting. ``b`` is the ADULT field established inside the barrier by one year
    of the fitted source-side population; the juvenile component is deliberately left
    at zero because the operator generates offspring from those adults itself, and
    injecting both would double-count a generation. Each iteration counts ``a'`` in
    the target -- ``a'`` is where arrivals materialize, while ``j'`` there is
    production, not arrival -- then removes it. Mass reaching the target or returning
    to the source is therefore counted exactly once, so this is a genuine
    first-passage accounting rather than a residence-time sum.

    ``rho`` is estimated as the asymptotic ratio ``total(v_{k+1})/total(v_k)``, which
    converges to the spectral radius for a non-negative operator. If it does not fall
    below 1, ``G`` is returned as inf with ``converged=False``: the barrier contains
    self-sustaining habitat and the Neumann series genuinely diverges.

    WHY THE TWO CORRIDORS CAN LOOK IDENTICAL -- read this before interpreting them.
    ``target`` and ``barrier`` are disjoint zones, so ``a_next * barrier`` below
    already absorbs arrivals: the recursion is ``v_{k+1} = P_barrier A v_k`` with a
    genuinely absorbing ``A_P``. But ``A_P`` contains no reference to ``target`` at
    all, so it is THE SAME OPERATOR for both directions -- correctly so; the cost of
    moving through the barrier does not depend on which way you eventually exit.
    Direction enters only through the injected mass ``b`` and through which exit is
    counted as an arrival. Hence ``corridor = sum_k A_P^k b``: one operator, two
    initial conditions. Whether that leaves any directional SHAPE depends entirely on
    ``rho``, and the two regimes are covered by tests:

    * ``rho < 1`` -- the sum is dominated by its EARLY terms, the memory of ``b``
      survives, and the corridors genuinely differ (test TVD > 0.05).
    * ``rho >= 1`` -- the sum is dominated by its LATE terms, which are ``A_P``'s
      dominant eigenvector regardless of ``b``. Both corridors converge onto that one
      eigenvector and differ only by a SCALAR (test TVD < 0.02). This is the fitted
      case, and it is why the two panels looked identical: per-panel normalization
      then divided out the only surviving difference.

    So in the supercritical regime directional information lives in ``G``, in
    ``per_year_arrivals`` and in ``directional_q_contrast`` / ``q_asymmetry_attribution``
    -- NOT in the corridor shape. ``corridor_horizon`` accumulates the transient only,
    which is where any directional signal is concentrated; the plot normalizes both
    corridors and maps their ratio so the presence or absence of it is visible rather
    than assumed.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}, got {direction!r}")
    src_name, tgt_name = DIRECTIONS[direction]
    land = fields["land"].astype(bool)
    source = zones[src_name] & land
    barrier = zones["barrier"] & land
    target = zones[tgt_name] & land

    step = annual_operator(fields, data, p0, use_edge_correction, q_override)

    # Injection: one year of the fitted source population, keeping only what
    # establishes inside the barrier.
    a_src = np.where(source, fields["a0"], 0.0)
    j_src = np.where(source, fields["j0"], 0.0)
    a_inj, _ = step(a_src, j_src)
    b = np.asarray(a_inj) * barrier
    total_in = float(b.sum())

    if total_in <= 0.0:
        raise ValueError(
            f"no adults establish inside the barrier from the {src_name} source, so "
            "the crossing gain is 0/0. Either the source zone is unoccupied in the "
            "fitted state or the zone masks do not overlap the land mask.")

    arrivals_field = np.zeros_like(b)          # all years
    arrivals_field_horizon = np.zeros_like(b)  # within horizon_years
    corridor = np.zeros_like(b)
    corridor_horizon = np.zeros_like(b)        # transient only -- see the docstring
    per_year = []
    ratios = []
    a, j = b.copy(), np.zeros_like(b)
    converged = False
    for year in range(int(max_years)):
        prev = float(a.sum() + j.sum())
        corridor += a
        a_next, j_next = (np.asarray(x) for x in step(a, j))
        arrived = a_next * target
        arrivals_field += arrived
        if year < horizon_years:
            arrivals_field_horizon += arrived
            corridor_horizon += a
        per_year.append(float(arrived.sum()))
        a, j = a_next * barrier, j_next * barrier
        cur = float(a.sum() + j.sum())
        # Only sample the decay ratio while the iterate still carries signal. The
        # affine-offset subtraction leaves float32 rounding behind, and once the mass
        # has decayed into that noise the ratio drifts back toward 1 and would corrupt
        # the rho estimate with a value that says nothing about the operator.
        if prev > _RATIO_SIGNAL_FLOOR * total_in:
            ratios.append(cur / prev)
        if cur <= tol * total_in:
            converged = True
            break
        # Genuine divergence: break before float overflow turns rho into inf/inf=nan.
        if cur > 1e20 * total_in:
            break
    # Late ratios only: the transient from a boundary-hugging b takes some years to
    # settle onto the dominant eigenvector, so early ratios are not rho.
    rho = float(np.mean(ratios[-5:])) if len(ratios) >= 5 else (
        float(ratios[-1]) if ratios else float("nan"))
    # Distinguish real growth from a float-noise floor. After the affine offset is
    # removed the residual is float32 rounding on a decayed iterate, which can stall
    # the ratio near 1 while the mass is already negligible. That is exhaustion, not
    # self-sustenance -- the discriminator is the magnitude, not the ratio.
    negligible = cur <= 1e-6 * total_in
    if negligible and not converged:
        converged = True
    self_sustaining = bool(not converged and rho >= 1.0)

    g_horizon = float(arrivals_field_horizon.sum() / total_in)
    g_total = float("inf") if self_sustaining else float(arrivals_field.sum() / total_in)
    cum = np.cumsum(per_year[:horizon_years])
    half = int(np.searchsorted(cum, 0.5 * cum[-1]) + 1) if cum.size and cum[-1] > 0 else -1
    return {"direction": direction, "G_horizon": g_horizon, "G_total": g_total,
            "rho": rho, "converged": converged, "horizon_years": int(horizon_years),
            "total_entering_barrier": total_in, "per_year_arrivals": per_year,
            "arrivals_field": arrivals_field_horizon, "corridor": corridor,
            "corridor_horizon": corridor_horizon,
            "n_years": len(per_year), "years_to_half_of_G": half,
            "barrier_self_sustaining": self_sustaining}


def propagule_pressure(arrivals_field, fields, rows, cols, shape, allee_gamma):
    """``P(x) = cumulative arrivals / N_crit(x)`` -- the invasion-pinning ratio.

    With an Allee effect, arrival is not establishment: a propagule must exceed the
    unstable lower equilibrium ``N_crit`` (the lower root of ``F(N) = F_replacement``,
    from :func:`age_model_math.allee_viability`). ``P >= 1`` means the immigration
    that accumulates over ``arrivals_field``'s own window clears the threshold, so
    the front is not pinned there; ``P < 1`` means it is.

    UNITS. ``arrivals_field`` as returned by :func:`crossing_gain` is the arrival
    density summed over ``horizon_years``, NOT a per-year rate, so ``P`` is a
    cumulative pressure over that horizon. This is deliberate -- accumulated pressure
    over the study period is the quantity that decides establishment, since
    sub-threshold propagules can be topped up by later arrivals -- but it means ``1/P``
    is a multiple of the horizon, not a number of years. Pass a single year's arrivals
    if a per-year reading is wanted.

    Also note that ``P`` inherits the Allee-optimistic bias of the operator that
    produced ``arrivals_field``: the numerator is an upper bound. A cell with ``P < 1``
    here is pinned under an assumption generous to crossing, which makes that the
    robust direction of the conclusion.

    NaN where the cell is not viable at all (no ``N_crit`` exists) -- a different
    statement from "flux too small", and not to be conflated in a fraction.
    """
    flat = {k: fields[k][rows, cols] for k in ("Sa", "Sj", "Fmax", "K")}
    viab = allee_viability(flat["Sa"], flat["Sj"], flat["Fmax"], flat["K"], allee_gamma)
    n_crit = np.full(shape, np.nan)
    n_crit[rows, cols] = viab["N_crit"]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = arrivals_field / n_crit
    return p, n_crit, viab


def edge_correction_summary(data, zones, land):
    """Source-side ``1/f`` inflation inside the barrier -- the gating check.

    The edge correction divides outgoing mass by the fraction of the kernel's
    footprint on land, so it inflates emigration from cells near coasts or the grid
    frame. That inflation is anti-correlated with exactly the geography ``Q``
    measures, so if it is materially below 1 inside the barrier the asymmetry must be
    reported both with and without it (``crossing_gain(use_edge_correction=False)``).
    In the continental interior it should be ~1.

    A bare minimum is not the statistic to judge this on: ``f`` can be 0 for a cohort
    whose entire footprint is off-land, giving a nominal 1e6-fold inflation, and that
    is harmless if the cell carries no dispersers. What matters is how much of the
    barrier is materially inflated, so the fraction below 0.5 and the 1st percentile
    are reported alongside the mean.
    """
    barrier = zones["barrier"] & land.astype(bool)
    f_j = np.asarray(data["juvenile_edge_correction_stack"])[:, barrier]
    f_a = np.asarray(data["adult_edge_correction"])[barrier]
    return {
        "mean_juvenile_edge_correction_in_barrier": float(f_j.mean()),
        "p01_juvenile_edge_correction_in_barrier": float(np.percentile(f_j, 1)),
        "frac_juvenile_edge_correction_below_half": float((f_j < 0.5).mean()),
        "mean_adult_edge_correction_in_barrier": float(f_a.mean()),
        "frac_adult_edge_correction_below_half": float((f_a < 0.5).mean()),
    }


def directional_q_contrast(fields, labels, window_unused=None):
    """Mean ``Q`` over the ``to_EAST`` vs ``to_WEST`` cohorts, and their difference.

    A per-cell view of the raw asymmetry, before any dispersal or demography, so a
    reader can see whether ``G``'s asymmetry traces to ``Q`` itself or to the
    interaction of ``Q`` with the kernel geometry and the habitat inside the barrier.
    Cohort membership is read from the labels rather than assumed from index
    arithmetic, because the stack ordering (direction outer, band inner) is a
    convention of ``build_kernels.make_radial_directional_kernels``.
    """
    labels = [str(x) for x in labels]
    east = [i for i, s in enumerate(labels) if s.startswith("to_EAST")]
    west = [i for i, s in enumerate(labels) if s.startswith("to_WEST")]
    if not east or not west:
        raise ValueError(f"no to_EAST/to_WEST cohorts in labels: {labels}")
    q = fields["Q"]
    q_east, q_west = q[east].mean(axis=0), q[west].mean(axis=0)
    return {"q_to_east": q_east, "q_to_west": q_west, "q_west_minus_east": q_west - q_east,
            "east_cohorts": east, "west_cohorts": west}


def _band_key(label):
    """Radial band suffix of a kernel label (``to_WEST_155-483`` -> ``155-483``)."""
    parts = str(label).split("_")
    return parts[-1] if len(parts) >= 3 else str(label)


def q_asymmetry_attribution(fields, data, sim, zones, labels, rows, cols, shape,
                            years=None, era=None, top_n=6):
    """Why westward journeys cost more: per radial band, and per Z feature.

    ``directional_q_contrast`` pools all three radial bands into one number, which
    cannot say whether the asymmetry is a short-hop or a long-jump phenomenon --
    and the bands (0-155, 155-483, 483+ km) are exactly the spatial scales the
    juvenile-MDD sweep moves. This resolves the contrast by band, then attributes
    it to the covariates that produce it.

    ATTRIBUTION. ``Q_k(x) = sigmoid(alpha_j + gamma_j * (Z_disp[x,:,k] . beta_s))``,
    so the entire E/W contrast comes from ``Z_disp`` differing between the east-
    and west-pointing wedges of the same cell. Per feature ``m`` and band ``b`` the
    contribution to the pre-sigmoid contrast is exactly
    ``mean_x (Z_disp[x,m,W_b] - Z_disp[x,m,E_b]) * beta_s[m]``, which is additive
    and therefore a genuine decomposition (the sigmoid is applied afterwards, so
    only the RANKING is exact on the probability scale, not the magnitudes).

    GEOMETRY CONTROL. The wedges partition the kernel without per-wedge
    renormalization, so a west-pointing wedge that runs off the domain integrates a
    different amount of landscape than its east-pointing twin, and that alone would
    produce a contrast with no habitat gradient behind it. The edge-correction stack
    ``f_j`` measures exactly that -- the share of each wedge's kernel mass landing on
    valid domain -- so the E/W contrast IN ``f_j`` is a direct, fitted-data measure of
    the geometric component. Reported per band beside the ``Q`` contrast: if the two
    track each other, the asymmetry is geometry; if ``Q``'s contrast survives where
    ``f_j``'s is ~0, it is habitat.
    """
    labels = [str(x) for x in labels]
    east = [i for i, s in enumerate(labels) if s.startswith("to_EAST")]
    west = [i for i, s in enumerate(labels) if s.startswith("to_WEST")]
    if not east or not west:
        raise ValueError(f"no to_EAST/to_WEST cohorts in labels: {labels}")
    # Pair east/west cohorts by radial band so a contrast is never taken across
    # different distance classes.
    by_band = {}
    for i in east:
        by_band.setdefault(_band_key(labels[i]), {})["east"] = i
    for i in west:
        by_band.setdefault(_band_key(labels[i]), {})["west"] = i
    bands = [(k, v["east"], v["west"]) for k, v in by_band.items()
             if "east" in v and "west" in v]
    # Sort by the band's lower edge so the panels read short-hop -> long-jump.
    def _lo(key):
        try:
            return float(str(key).split("-")[0])
        except ValueError:
            return float("inf")
    bands.sort(key=lambda t: _lo(t[0]))

    barrier = zones["barrier"] & fields["land"].astype(bool)
    q = fields["Q"]
    # Edge correction is (K_kernels, Ny, Nx) -- the geometry control, see the docstring.
    f_j = np.asarray(data["juvenile_edge_correction_stack"])
    per_band = [{"band": k,
                 "q_to_east": q[ie], "q_to_west": q[iw], "delta": q[iw] - q[ie],
                 "mean_q_to_east": float(q[ie][barrier].mean()),
                 "mean_q_to_west": float(q[iw][barrier].mean()),
                 "mean_delta": float((q[iw] - q[ie])[barrier].mean()),
                 "mean_edge_corr_delta": float((f_j[iw] - f_j[ie])[barrier].mean())}
                for k, ie, iw in bands]

    # Z_disp_gathered is (time, N_land, K_kernels, M) -- kernel axis BEFORE feature
    # axis (see age_fields.py:280-301). Era-reduce it exactly as Q was reduced.
    z_disp = np.asarray(data["Z_disp_gathered"]) if "Z_disp_gathered" in data else None
    beta_s = np.asarray(sim["latents"]["w_env"])[:, 0]
    features = []
    if z_disp is not None:
        if z_disp.ndim == 4:
            if years is not None and era is not None:
                i0, i1, _ = era_span(era, years)
                z_disp = z_disp[i0:i1].mean(axis=0)
            else:
                z_disp = z_disp.mean(axis=0)      # (N_land, K, M)
        # Restrict to barrier cells, in the same flat land ordering as rows/cols.
        zb = z_disp[barrier[rows, cols]]          # (n_barrier, K, M)
        # Additive on the PRE-SIGMOID scale, which is where the decomposition is exact.
        contrib = {k: (zb[:, iw, :] - zb[:, ie, :]).mean(axis=0) * beta_s
                   for k, ie, iw in bands}
        total = np.sum(list(contrib.values()), axis=0)
        order = np.argsort(np.abs(total))[::-1]
        features = [{"index": int(m), "total": float(total[m]),
                     "by_band": {k: float(contrib[k][m]) for k, _, _ in bands}}
                    for m in order[:top_n]]

    return {"bands": per_band, "features": features, "beta_s": beta_s}
