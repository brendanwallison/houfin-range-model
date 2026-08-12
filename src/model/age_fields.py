"""Per-cell demographic fields from the latent habitat manifolds.

Local Z is projected onto four manifolds -- adult survival H_s, juvenile survival
H_sj, reproduction H_r, and capacity H_k, one w_env column each -- then passed
through link functions to per-cell adult/juvenile survival (S_a, S_j), max fecundity
(F_max), carrying capacity (K), and journey survival (Q). Runs as a checkpointed
``lax.scan`` over years to bound memory when differentiated.

K uses a SOFTPLUS link, applied in DENSITY space:

    K_base = softplus(alpha_k + gamma_k*H_k + k_trend_t)

``alpha_k`` is solved so the post-transformation prior MEDIAN of capacity equals the
measured 2.6183 route counts (``age_priors._softplus_loc_for_median``). Solving on
the median rather than the mean is deliberate: a mean target landed the realized
median 8.6% above the measured value. The measured spread is log sd 1.592, from
``scripts/diagnostics/bbs_abundance_quantiles.py``; ``config.population_model``'s
``_link_comment`` and ``capacity_level_prior`` carry the full derivation.

Each rate reads its OWN manifold. K previously reused H_r, making it a strictly
monotone function of F_max, so the disease term below was the ONLY way the two could
differ spatially -- and it duly saturated absorbing that disagreement. Likewise S_j
previously reused H_s. The four manifolds share a RANK-2 prior whose correlations are
ordered on purpose: Corr(F_max, K) ~ 0.85 sits above Corr(F_max, survival) ~ 0.70,
since fecundity and capacity are both productivity/resource axes. See
``age_priors.sample_priors``.
"""
import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, checkpoint


def disease_severity(disease, K_base):
    """Per-cell PEAK fraction of carrying capacity the epizootic removes.

    DENSITY-DEPENDENT, via a Hill function of local capacity::

        severity = ceiling * modifier(x) * K_base^n / (K_base^n + K_half^n)

    Epidemics need hosts. Mass-action transmission makes R0 scale with host
    density, and below a threshold density the pathogen cannot persist -- so
    severity goes to 0 as ``K_base -> 0`` and saturates at the ceiling in dense
    populations. This matters beyond realism: the model carries an Allee effect, so
    a small ABSOLUTE reduction in an already-small K can push a sparse population
    to local extinction. A density-independent severity therefore does its most
    violent damage exactly where it is least biologically justified.

    Both shape parameters are FITTED, so the shape is learned rather than
    asserted: ``k_half`` is the density at half-maximum severity and ``n`` the
    steepness (n~1 smooth saturation, n~3 a sharp invasion threshold).

    ``modifier(x)`` retains the smooth regional field, the epidemic-attenuation
    coefficient, and the native-lineage coefficient, as a multiplicative
    adjustment in (0, 1] around the density-driven value. ``b_late`` (on decades
    since ``disease_start_year``, NOT since a cell's own local arrival -- a
    calendar-time pathogen-attenuation/host-adaptation trend, see the module
    docstring) is now strongly collinear with the density term (the east is both
    early-arriving and dense) and is priored tightly for that reason; it is kept
    because epidemic-history timing and density are genuinely distinct
    mechanisms, so a strong fitted ``b_late`` ALONGSIDE the density term would be
    informative rather than redundant. ``b_native`` is collinear with BOTH --
    the west is native, late-arriving, and sparse -- but states a mechanistically
    distinct claim again (population origin/genetic diversity, not epidemic
    timing or density) and is the only one of the three constrained in SIGN:
    native lineage can only lower severity (see the module docstring).

    The ``ceiling`` is load-bearing for identifiability, not just plausibility:
    ``K = K_base * (1 - severity(K_base))`` must be monotone in ``K_base``, else
    two habitat qualities map to one realized capacity. At 0.5 that holds for every
    n tested; at 0.7 it fails.

    Severity keys off ``K_base``, the potential density, rather than realized N --
    partly because all demographic fields are precomputed before the forward
    simulation runs, so N does not exist yet, and partly because epidemic
    establishment depends on sustained density rather than a single year's count.
    """
    # Regional/arrival-order modifier, a plain sigmoid in (0, 1). NOT clipped: an
    # earlier version used min(1, 2*sigmoid(.)) so that a zero logit meant "no
    # modification", but that put a zero-gradient boundary exactly where mu_sev's
    # prior is centered, and the clip also broke the land-centered field's
    # guarantee that it cannot shift the continental level. With a plain sigmoid the
    # ceiling stays a strict upper bound (severity <= ceiling * density_term), the
    # field remains level-neutral in logit space, and mu_sev's prior is instead
    # centered on the modifier value we actually believe.
    modifier = jnn.sigmoid(
        disease["mu_sev"]
        + disease["b_late"] * disease["onset_decades"]
        + disease["b_native"] * disease["native_shape"]
        + jnp.dot(disease["sev_basis"].T, disease["w_sev"]))
    n = disease["hill_n"]
    density_term = K_base ** n / (K_base ** n + disease["k_half"] ** n)
    return disease["ceiling"] * modifier * density_term


def disease_onset_timestep(disease):
    """Per-cell front arrival in model-timestep units, with fitted slack.

    The arrival map (``scripts/build_disease_arrival_map.py``) is a smoothed
    reconstruction from the documented spread history, so its timing is coarse
    both continentally (``lag0``) and regionally (a smooth ``lag`` field). Fitting
    that slack is what keeps the map's imprecision from propagating into severity:
    without it, a one-year timing error in a region has to be absorbed by making
    the severity there wrong instead.
    """
    return (disease["onset"] + disease["lag0"]
            + jnp.dot(disease["lag_basis"].T, disease["w_lag"]))


def disease_k_fraction(disease, t_idx, K_base, onset_t=None, severity=None):
    """Fraction of K removed at ``t_idx``: severity * gate * (1 - recovery).

    Strictly in [0, 1), so ``K = K_base * (1 - fraction)`` can be driven toward
    zero only in the limit and never past it.

    ``recovery`` implements slowly increasing resilience after local arrival (from
    exposure, not necessarily evolution): the hit relaxes from its peak toward
    ``severity * (1 - rec)`` with an e-folding time of ``tau_rec`` years. Note
    ``age`` is clipped at 0, so a cell the front has not yet reached carries no
    recovery -- the gate is what suppresses it there, and the two must not
    double-count.
    """
    onset_t = disease_onset_timestep(disease) if onset_t is None else onset_t
    severity = disease_severity(disease, K_base) if severity is None else severity
    gate = jnn.sigmoid((t_idx - onset_t) / disease["tau"])
    age = jnp.maximum(t_idx - onset_t, 0.0)
    recovered = disease["rec"] * (-jnp.expm1(-age / disease["tau_rec"]))
    return severity * gate * (1.0 - recovered)


def project_and_scatter_age_structured(
    time, Ny, Nx,
    land_rows, land_cols,
    Z_gathered, Z_disp_gathered,
    disease_timestep, disease,
    beta_s,           # 1D feature weights for Survival Suitability (Shape: M)
    beta_r,           # 1D feature weights for Reproductive Suitability (Shape: M)
    beta_k,           # 1D feature weights for Carrying Capacity (Shape: M)
    beta_sj,          # 1D feature weights for JUVENILE Survival Suitability (Shape: M)
    k_trend_basis,    # (n_trend, time) time-centered cosines, continental
    w_k_trend,        # (n_trend,) weights on that trend
    alpha_a, gamma_a, # Adult survival intercept & slope
    alpha_j, gamma_j, # Juvenile survival intercept & slope
    alpha_f, gamma_f, # Max fecundity intercept & slope
    alpha_k, gamma_k  # Carrying capacity intercept (inside softplus) & slope
):
    """Project Z → (S_a, S_j, F_max, K, Q) for every year, on the land cells.

    Adult survival uses a sigmoid link on H_s, juvenile survival a sigmoid on its own
    H_sj, fecundity a softplus on H_r, and capacity a softplus on its own H_k. Q
    (in-cohort journey survival) reuses the juvenile intercept/slope on the
    land-conditioned neighborhood/path habitat H_sj_disp. Each returned array is
    (time, N_land[, K]). ``Kbase_flat`` is capacity BEFORE the disease effect --
    returned because severity is a function of it, so any exact diagnostic of the
    disease term needs the undepressed value rather than the depressed one.

    ``disease`` bundles the epizootic term's inputs and parameters (see
    ``disease_k_fraction``). Timesteps before ``disease_timestep`` get K_base
    unmodified -- exactly, not approximately -- which is what pins ``k_level`` on
    the 1966-1993 BBS record the disease term cannot reach.
    """
    # The onset field does not depend on t, so compute it once rather than 124
    # times inside the scan. Severity now depends on K_base, which varies by year
    # through the continental trend, so it is computed per year.
    onset_t = disease_onset_timestep(disease)

    # Checkpoint: don't store this function's large intermediates for the
    # backward pass; recompute them instead.
    @checkpoint
    def process_year(carry, t_idx):
        # 1. Pull slices from CPU RAM -> GPU VRAM
        z_t = jnp.take(Z_gathered, t_idx, axis=0)
        z_disp_t = jnp.take(Z_disp_gathered, t_idx, axis=0)

        # 2. Compute the 4 Correlated Habitat Manifolds (H_s, H_sj, H_r, H_k) --
        # purely covariate-driven (Z.beta), no spatiotemporal term mixed in.
        H_s_local = jnp.dot(z_t, beta_s)
        H_r_local = jnp.dot(z_t, beta_r)
        # Juvenile survival reads its OWN manifold. It used to reuse H_s_local with only a
        # scalar shift (alpha_j) and scale (gamma_j), so "where juveniles survive" could not
        # differ in PATTERN from "where adults survive" -- zero independent spatial degrees of
        # freedom. The two are still tightly coupled, but now through a PRIOR on their
        # correlation (rank-2 manifold prior, target 0.85) rather than by identity.
        H_sj_local = jnp.dot(z_t, beta_sj)
        # Capacity reads its OWN manifold. It used to reuse H_r, making K a strictly
        # monotone function of Fmax, so the disease term was the only way their
        # spatial patterns could differ -- see the module docstring.
        H_k_local = jnp.dot(z_t, beta_k)

        # 3. Path-Integrated Survival Suitability
        # z_disp_t is (N_land, K_kernels, M) -> dot with beta_s (M,) gives (N_land, K_kernels)
        # NOTE (Ružička contract): the "Z.Z^T ~= uncentered Ružička kernel + isotropic prior
        # => GP with the Ružička kernel" identity holds EXACTLY only for the LOCAL block
        # (H_s_local/H_r_local, raw Z). z_disp = A.Z is a land-normalized spatial convolution
        # of Z, so z_disp.z_disp^T ~= A.K_ružicka.A^T -- a spatially-SMOOTHED kernel, not
        # Ružička itself. Journey survival is tied to juvenile LOCAL survival (step 5 below),
        # so this uses beta_sj -- it previously used beta_s only because no juvenile manifold
        # existed. The exact GP-kernel interpretation remains only approximate on the dispersal
        # block. See kernel_contract note in model_inputs.py and the "dispersal-block prior"
        # future-work item.
        H_sj_disp = jnp.dot(z_disp_t, beta_sj)

        # 4. Map H_s and H_r to Demographic Rates using Intercepts and Slopes
        # Survival listens to H_s
        S_a_val = jnn.sigmoid(alpha_a + gamma_a * H_s_local)
        S_j_val = jnn.sigmoid(alpha_j + gamma_j * H_sj_local)

        # Reproduction listens to H_r
        F_max_val = jnn.softplus(alpha_f + gamma_f * H_r_local)

        # Continental, spatially uniform capacity drift. Time-centered basis, so
        # k_level still owns the level; this exists so "modern K below 1970s K" has
        # somewhere to go other than the disease term. Its prior is deliberately
        # near-zero -- it is a safety valve, not a mechanism.
        k_trend_t = jnp.dot(jnp.take(k_trend_basis, t_idx, axis=1), w_k_trend)

        # Baseline K with a SOFTPLUS link, applied in DENSITY space:
        #     K_base = softplus(alpha_k + gamma_k*H_k + trend)
        # alpha_k's location is solved so the post-transformation prior MEDIAN of capacity
        # is the measured 2.6183 route counts (age_priors._softplus_loc_for_median); see
        # config.population_model._link_comment for why the median and not the mean.
        #
        # One property of softplus is worth keeping in view: where its argument is
        # negative it behaves like exp(), so anything SUBTRACTED from the argument acts
        # multiplicatively and without a floor. Nothing is subtracted here any more --
        # the disease effect is applied to K_base as a bounded multiplier below, not to
        # this argument -- but that is why K collapsed regionally in early runs.
        K_base_val = jnn.softplus(alpha_k + gamma_k * H_k_local + k_trend_t)

        # 4b. Disease effect on K only: a bounded MULTIPLICATIVE rescale (see the
        # module docstring for why the earlier additive-inside-softplus form
        # annihilated eastern K instead of rescaling it). The severity and onset
        # fields are time-independent, so they are hoisted out of the scan.
        k_fraction = jnp.where(
            t_idx >= disease_timestep,
            disease_k_fraction(disease, t_idx, K_base_val, onset_t=onset_t),
            0.0,
        )
        K_val = K_base_val * (1.0 - k_fraction)

        # 5. Map path habitat to Journey Survival (Q) using the JUVENILE manifold and rules.
        # This links movement mortality to juvenile local survival mortality -- the same field
        # S_j reads, so Q and S_j can no longer be driven by different spatial patterns.
        Q_val = jnn.sigmoid(alpha_j + gamma_j * H_sj_disp)

        # K_base_val is returned as well: severity is a function of it, so exact
        # diagnostics (and the severity map) need the UNDEPRESSED capacity. Using
        # the depressed K would be circular. Costs one more (time, N_land) float32
        # array, ~8 MB at production size.
        return None, (S_a_val, S_j_val, F_max_val, K_val, Q_val, K_base_val)

    # We scan over the range of time indices
    t_indices = jnp.arange(time)
    _, (Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat, Kbase_flat) = lax.scan(
        process_year, None, t_indices)

    return Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat, Kbase_flat
