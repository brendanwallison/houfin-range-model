"""Priors and the NumPyro model for the age-structured range-expansion model.

:func:`build_model_2d` is the full probabilistic model: it samples parameters
(:func:`sample_priors`), maps the latent habitat manifold (Z) to spatial
demographic-rate fields (survival/fecundity/carrying capacity, via
``age_fields``), runs the age-structured dispersal forward simulation
(``age_forward``), and scores BBS counts under a negative-binomial (NB2)
observation model. ``prior_scale`` implements prior continuation: values below
one deliberately tighten scale priors during early optimization. See
docs/TEMPORAL.md for the invasion-timestep convention.
"""
import math

import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import numpyro
import numpyro.distributions as dist

from src.config_utils import load_age_model_config
from src.model.age_fields import disease_severity, project_and_scatter_age_structured
from src.model.age_forward import forward_sim_age_structured

# Prior settings read from config so they can be retuned without editing model
# code -- these are the knobs most likely to be revisited between runs. See
# sample_priors for the semantics of each.
_POP_SPEC = load_age_model_config()["population_model"]
_DISEASE_PRIOR = dict(_POP_SPEC["disease_prior"])
_MANIFOLD_PRIOR = dict(_POP_SPEC["manifold_prior"])
_K_TREND = dict(_POP_SPEC["k_trend"])
_CAPACITY_LEVEL = dict(_POP_SPEC["capacity_level_prior"])
_INVASION_PULSE = dict(_POP_SPEC["invasion_pulse_prior"])
_INITPOP_SEED = dict(_POP_SPEC["initpop_seed"])
_ALLEE_PRIOR = dict(_POP_SPEC["allee_prior"])
# THE GAUGE. Every absolute-scale prior is declared in expected BBS ROUTE COUNTS in
# config and divided by this at exactly one boundary, so changing the gauge cannot
# change any prior's meaning. (n50 already followed this convention; the capacity
# level, the Hill threshold, initpop and the invasion pulse now do too.) The old
# name said "birds", which it never was -- a BBS count is a 50-stop roadside index.
_POP_SCALAR = float(_POP_SPEC.get("population_scale_route_counts_per_relative_unit",
                                  _POP_SPEC.get("population_scale_birds_per_relative_unit")))


def _solve_softplus_loc(target_route_counts, scale, pop_scalar):
    """Prior location for a softplus-linked quantity, calibrated in ROUTE COUNTS.

    Returns ``loc`` such that ``E[softplus(Normal(loc, scale))] * pop_scalar`` equals
    ``target_route_counts``. Solved numerically (Gauss-Hermite + bisection) because
    softplus has no closed-form mean under a normal.

    This is how a softplus link keeps the "declare beliefs in route counts" rule:
    softplus does not commute with scaling, so unlike exp it cannot simply absorb the
    gauge as a shift. Stating the target in counts and solving for the raw location
    means changing pop_scalar re-derives the location instead of silently changing
    what the prior asserts.
    """
    from scipy.optimize import brentq
    z, w = np.polynomial.hermite_e.hermegauss(201)
    w = w / w.sum()
    sp = lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    target_density = float(target_route_counts) / float(pop_scalar)
    return float(brentq(lambda m: float(np.sum(w * sp(m + scale * z))) - target_density,
                        -30.0, 10.0, xtol=1e-13))


def counts_to_relative(route_counts):
    """Convert an expected-BBS-route-count quantity to model density units.

    The single boundary where the gauge is applied. See the _scale_comment in
    config/age_model_config.json: the likelihood is exactly invariant to the gauge,
    so any prior stated in relative units is silently gauge-dependent and will
    change meaning when the gauge changes. State priors in counts; convert here.
    """
    return route_counts / _POP_SCALAR


# alpha_k's prior location, solved so the level's MEAN matches the configured target
# in route counts. Computed at import (microseconds) rather than hardcoded, so it
# tracks pop_scalar and alpha_k_scale automatically.
_ALPHA_K_LOC = _solve_softplus_loc(
    _CAPACITY_LEVEL["target_level_mean_route_counts"],
    float(_CAPACITY_LEVEL["alpha_k_scale"]), _POP_SCALAR)


def validate_environment_kernel_contract(data):
    """Reject model inputs that would not represent the intended GP feature map."""
    contract = data.get("z_kernel_contract")
    if not contract:
        raise ValueError("model inputs lack z_kernel_contract; rerun scripts/ingest_model_data.py")
    if contract.get("kernel") != "ruzicka" or bool(contract.get("centered", True)):
        raise ValueError(f"age model requires uncentered Ružička Z features, got {contract}")
    if contract.get("feature_prior") != "isotropic":
        raise ValueError(f"age model GP recovery requires an isotropic feature prior, got {contract}")
    actual = int(data["Z_gathered"].shape[-1])
    if int(contract.get("latent_dim", -1)) != actual:
        raise ValueError(f"kernel contract latent_dim={contract.get('latent_dim')} != Z width {actual}")
    source = int(contract.get("source_latent_dim", actual))
    truncation = contract.get("truncation", "none")
    if source < actual or (source > actual and truncation != "top_eigenfeatures"):
        raise ValueError(f"invalid configured kernel truncation: {contract}")
    return contract


def age_structure_log_prior(rho_k, alpha=1.01, beta=1.01, effective_sites=100.0):
    """Resolution-invariant weak distributional prior for local age structure.

    ``mean(log p(rho))`` is the spatial/temporal integral for a uniformly chosen
    representative land cell-year. ``effective_sites`` is a fixed power-prior
    strength, not the number of raster cells, so changing grid resolution does
    not silently multiply the prior.
    """
    rho_safe = jnp.clip(rho_k, 1e-5, 1.0 - 1e-5)
    return float(effective_sites) * jnp.mean(
        dist.Beta(float(alpha), float(beta)).log_prob(rho_safe))


def equilibrium_age_quantities(Sa, Sj, Fmax, K, allee_gamma):
    """Return density brake, fecundity-at-K, growth rate, and juvenile fraction.

    Algebra matches :func:`reproduction_age_structured`: surviving adults
    reproduce, so the local projection matrix is ``[[Sa,Sj],[F*Sa,0]]``.
    """
    c = jnp.maximum((Fmax * Sa * Sj) / (1.0 - Sa + 1e-6) - 1.0, 0.0)
    F_at_K = Fmax / (1.0 + c) * (1.0 - jnp.exp(-allee_gamma * K))
    lam = (Sa + jnp.sqrt(Sa**2 + 4.0 * F_at_K * Sa * Sj)) / 2.0
    rho = (F_at_K * Sa) / (F_at_K * Sa + lam)
    return c, F_at_K, lam, rho


def sample_priors(prior_scale=1.0, M_features=None, time=None,
                  N_sev_basis=None, N_lag_basis=None):
    """Sample every model parameter and return them in a dict.

    Covers the correlated 4-manifold habitat weights (adult survival, reproduction,
    capacity, juvenile survival, via a rank-2 angular prior with deliberately ordered
    correlations -- two tightly-coupled pairs, weaker coupling across them), the
    continental time trend on K, the structured
    mycoplasmal-conjunctivitis effect on K, dispersal/demography rate parameters,
    and the Allee term. ``prior_scale`` multiplies scale parameters for
    continuation fitting; ``M_features``, ``time``, and the two disease basis
    sizes set the dimensions.
    """
    priors = {}
    
    # --- 1. CORRELATED 4-MANIFOLD HABITAT WEIGHTS ---
    # [survival_adult, reproduction, capacity, survival_juv]
    # H_s = Z.beta_s drives Sa, H_sj = Z.beta_sj drives Sj and Q, H_r drives Fmax,
    # H_k drives K.
    # K previously reused beta_r outright -- an implicit correlation of exactly 1.0 --
    # so K's spatial pattern could differ from Fmax's only via the disease term. That
    # forced every real disagreement between "where reproduction is good" and "where
    # birds are abundant" to be spelled "disease", and the disease term duly
    # saturated. Giving K its own weights makes that disagreement a covariate
    # statement, still prior-coupled to the others.
    #
    # ONE-FACTOR parameterization: beta_j = s_j * (h_j*f + sqrt(1-h_j^2)*eps_j), with
    # f and eps drawn IID ACROSS FEATURES. Then Var(beta_j) = s_j^2 and
    # Corr(beta_j, beta_k) = h_j*h_k, both EXACTLY, and the implied 3x3 covariance is
    # positive-definite for any parameter values -- no Cholesky positive-definiteness
    # guard whose clipping would corrupt gradients. (LKJCholesky is not usable here:
    # its concentration > 1 concentrates near the IDENTITY correlation, i.e. near
    # zero correlation, the opposite of the prior belief.) The iid-across-features
    # structure is what preserves the uncentered-Ružička GP contract -- see
    # validate_environment_kernel_contract -- exactly as the old 2-output version did.
    _m = _MANIFOLD_PRIOR
    # RANK-2 ANGULAR solve. Column order is
    #     [survival_adult, reproduction, capacity, survival_juv]
    # -- survival_juv is APPENDED, not inserted at 1. Eight viz/analysis modules index w_env
    # positionally (w_env[:,1] == "reproduction", w_env[:,2] == "capacity"); inserting would have
    # relabelled both with no error raised anywhere.
    #
    # Two groups {survival_adult, survival_juv} and {reproduction, capacity} sit at +/- phi/2.
    # With r_j the communality and th_j the angle, Corr(j,k) = r_j*r_k*cos(th_j - th_k), so
    #   within a group (th_j == th_k):  Corr = r_j*r_k          -> r = sqrt(rho_within)
    #   across groups:                  Corr = r^2*cos(phi)     -> cos(phi) = rho_cross/rho_within
    # A single angle_scale then makes within-pair correlation LESS prior-uncertain than
    # cross-group automatically, because cos is flat at zero separation -- which is exactly the
    # requested structure ("tight within pair, weaker but informative across") from one knob.
    _rho_within = 0.5 * (float(_m["target_corr_survival_adult_juv"])
                         + float(_m["target_corr_repro_capacity"]))
    _rho_cross = float(_m["target_corr_cross_group"])
    _r_med = math.sqrt(_rho_within)
    _phi = math.acos(min(max(_rho_cross / _rho_within, -1.0), 1.0))
    _logit_r = math.log(_r_med / (1.0 - _r_med))
    _th_med = jnp.array([-_phi / 2, _phi / 2, _phi / 2, -_phi / 2])  # Sa, F, K, Sj

    # Communality per field, logit-normal so r stays in (0,1) and no correlation can leave
    # [-1,1] and the idiosyncratic weight sqrt(1-r^2) stays real.
    r_load = numpyro.deterministic(
        "manifold_communality",
        jnn.sigmoid(numpyro.sample(
            "manifold_communality_raw",
            dist.Normal(_logit_r, float(_m["communality_logit_scale"]) * prior_scale)
            .expand([4]))),
    )
    th_load = numpyro.sample(
        "manifold_angle",
        dist.Normal(_th_med, float(_m["angle_scale"]) * prior_scale))
    # (4, 2) factor loadings. Emitted under the historical `manifold_loadings` name, now a
    # matrix rather than a vector -- map_diagnostics writes it to JSON as a flat list.
    L_load = numpyro.deterministic(
        "manifold_loadings",
        jnp.stack([r_load * jnp.cos(th_load), r_load * jnp.sin(th_load)], axis=-1))
    # GP AMPLITUDE, one per manifold, and now the SOLE amplitude: the per-rate slopes
    # are fixed at 1 (see the gamma block below), because w_scale and gamma multiplied
    # the same field and only their product entered the likelihood -- three exactly
    # flat ridge directions, harmless-ish under MAP but ruinous for HMC step-size
    # adaptation. Per-field locs/scales are calibrated to reproduce the amplitude prior
    # that the gamma*w_scale product used to imply, matched on the 5th/95th percentiles
    # (see the _amplitude_comment in config). They differ by field on purpose.
    #
    # softplus(Normal), NOT lognormal: lognormal's tails are ~2x heavier at the 99.9th
    # percentile here, and those tails are what destabilize this model under HMC.
    w_scale = numpyro.deterministic(
        "w_scale",
        jnn.softplus(numpyro.sample(
            "w_scale_raw",
            dist.Normal(jnp.asarray(_m["amplitude_loc"], dtype=float),
                        jnp.asarray(_m["amplitude_scale"], dtype=float) * prior_scale))),
    )
    # Now truthful for ALL FOUR fields: every per-rate slope is fixed at 1 (gamma_j included --
    # see the gamma block), so this IS the amplitude of each latent field.
    numpyro.deterministic("environment_kernel_variance", w_scale ** 2)

    # dim=-2 so the feature plate takes the ROW axis and leaves the rightmost axis for the
    # manifolds; the factors come out (M, 2) and the idiosyncratic draws (M, 4).
    #
    # IID ACROSS FEATURES is the load-bearing property, unchanged from the rank-1 version: with
    # f1, f2 and eps all iid over features, beta_j[m] is iid over m with variance w_scale_j^2
    # exactly (r^2 + (1-r^2) = 1), so H_j = Z.beta_j is still a GP with kernel
    # w_scale_j^2 * Z(x).Z(x') and validate_environment_kernel_contract's isotropic-feature
    # requirement still holds. A full covariance over FEATURES would have broken it; adding a
    # second FACTOR does not.
    with numpyro.plate("env_features", M_features, dim=-2):
        f_shared = numpyro.sample("manifold_factor", dist.Normal(0.0, 1.0).expand([2]))
        eps_idio = numpyro.sample("manifold_idio", dist.Normal(0.0, 1.0).expand([4]))

    w_env = numpyro.deterministic(
        "w_env",
        w_scale[None, :] * (f_shared @ L_load.T
                            + jnp.sqrt(1.0 - (L_load ** 2).sum(-1))[None, :] * eps_idio),
    )

    # Report the implied correlations (and a 4x4 Cholesky under the historical L_corr name) so
    # diagnostics can read what the fit concluded about how tightly the manifolds move together.
    corr = L_load @ L_load.T
    corr = corr + jnp.diag(1.0 - jnp.diag(corr))
    numpyro.deterministic("L_corr", jnp.linalg.cholesky(corr))
    numpyro.deterministic("rho", corr[0, 1])  # survival-reproduction, the old `rho`
    numpyro.deterministic("env_corr_repro_capacity", corr[1, 2])
    numpyro.deterministic("env_corr_survival_capacity", corr[0, 2])
    # The new one: adult vs juvenile survival, the pair this rank-2 extension exists to couple.
    numpyro.deterministic("env_corr_survival_adult_juv", corr[0, 3])

    priors['beta_s'] = w_env[:, 0]   # Adult survival suitability weights
    priors['beta_r'] = w_env[:, 1]   # Reproductive suitability weights
    priors['beta_k'] = w_env[:, 2]   # Carrying-capacity weights
    priors['beta_sj'] = w_env[:, 3]  # Juvenile survival -- its OWN manifold (drives Sj and Q)

    # --- 1a. CONTINENTAL TIME TREND ON K: A SAFETY VALVE, NOT A MECHANISM ---
    # This is not a modeled process and is not meant to carry signal. It exists only
    # because the model otherwise has NO temporal degree of freedom for capacity
    # besides the disease term, so "modern K below 1970s K" could only be expressed
    # as disease -- and the disease term saturated at 93% severity trying. The prior
    # is therefore very strongly concentrated on ZERO: the default budget keeps the
    # implied K multiplier within ~2% of 1.0 at the median and ~7% at the 95th
    # percentile.
    #
    # A fitted trend that escapes that range is a FAILURE SIGNAL: it means a real
    # temporal signal is not being explained by any mechanism in the model. The
    # correct response is to find the missing mechanism, NOT to loosen this budget.
    # scripts/viz/map_diagnostics.py reports k_trend.max_prior_sd_multiple so this
    # can be read off directly.
    #
    # Spatially UNIFORM by construction, with only a few time-centered cosines, so it
    # cannot compete with the disease term's spatial pattern or manufacture
    # year-to-year wiggle.
    k_trend_budget = float(_K_TREND["budget"]) * prior_scale
    n_k_trend = int(_K_TREND["n_basis"])
    priors['w_k_trend'] = numpyro.sample(
        "w_k_trend",
        dist.Normal(0.0, k_trend_budget / jnp.sqrt(n_k_trend)).expand([n_k_trend]),
    )

    # --- 1b. MYCOPLASMAL-CONJUNCTIVITIS EFFECT ON K ---
    # K = K_base * (1 - severity(x) * gate(x,t) * (1 - recovery(t - arrival))).
    # See age_fields.py's module docstring for the formulation and for why the
    # previous generic spatiotemporal field annihilated eastern K. Every parameter
    # below states a claim about the epizootic that someone could dispute; that is
    # the point of the structure. All scales carry prior_scale for continuation.
    _p = _DISEASE_PRIOR

    # SEVERITY: logit of the peak fraction of K removed once the front passes.
    # mu_loc=0 -> prior median 50% removal, matching the documented eastern
    # decline; mu_scale=0.5 -> 90% CI ~ [31%, 69%].
    priors['disease_mu_sev'] = numpyro.sample(
        "disease_mu_sev",
        dist.Normal(float(_p["mu_loc"]), float(_p["mu_scale"]) * prior_scale),
    )
    # EPIDEMIC-ATTENUATION: cells reached later in the epidemic's own history
    # (decades after disease_start_year=1993, NOT decades after that cell's own
    # local arrival -- see model_inputs' disease_onset_decades) were plausibly hit
    # less hard, because the documented pattern for mycoplasmal conjunctivitis is
    # pathogen/host co-evolution making it progressively milder since 1993 -- a
    # calendar-time trend, not a claim about any cell's own population genetics
    # (that is disease_b_native below). One coefficient per decade -- cheaper and
    # far more interpretable than asking the spatial field to rediscover a pattern
    # that is essentially the epidemic's own age.
    priors['disease_b_late'] = numpyro.sample(
        "disease_b_late",
        dist.Normal(float(_p["late_arrival_loc"]),
                    float(_p["late_arrival_scale"]) * prior_scale),
    )
    # NATIVE-LINEAGE resistance: the west coast population is genetically diverse
    # (the ancestral range), unlike the east's single-founder 1940 introduction, and
    # was documented to collapse far less under the epizootic. This is a DISTINCT
    # claim from b_late above -- population origin, not arrival timing -- carried by
    # its own coefficient on native_shape (the same core/margin map used to seed 1902
    # abundance: 1 in the native core, margin_fraction_of_core at the fringe, 0
    # elsewhere). SIGN-CONSTRAINED via -softplus(raw) <= 0: native lineage may only
    # SUPPRESS severity, never amplify it -- the documented asymmetry is one-
    # directional (west spared, not "east extra hit"), and constraining the sign is
    # what turns "the west did less badly" from a pattern the fit might or might not
    # find into a structural, falsifiable claim (see age_fields.py's module
    # docstring). raw_loc=0 puts the prior median suppression at softplus(0)=0.69
    # logit units in the native core, comparable in scale to the modifier's other
    # terms; raw_scale=0.7 lets the fit push that from near 0 to a strong effect.
    priors['disease_b_native'] = numpyro.deterministic(
        "disease_b_native",
        -jnn.softplus(numpyro.sample(
            "disease_b_native_raw",
            dist.Normal(float(_p["native_resistance_raw_loc"]),
                        float(_p["native_resistance_raw_scale"]) * prior_scale))),
    )
    # Regional severity deviation. Land-centered basis (see
    # model_inputs.generate_spatial_basis), so these can only redistribute around
    # disease_mu_sev, never shift its level -- which is what keeps the continental
    # severity reportable and stops it trading against alpha_k.
    sev_scale = float(_p["sev_field_budget"]) * prior_scale / jnp.sqrt(N_sev_basis)
    priors['disease_w_sev'] = numpyro.sample(
        "disease_w_sev", dist.Normal(0.0, sev_scale).expand([N_sev_basis])
    )

    # ONSET TIMING SLACK. The arrival surface is a smoothed reconstruction from the
    # documented spread history, coarse both continentally (its kernel smoother
    # shrinks the 1994 mid-Atlantic and 2006 California extremes inward) and
    # regionally. Fitting the slack keeps that imprecision from being absorbed as
    # wrong severity instead.
    priors['disease_lag0'] = numpyro.sample(
        "disease_lag0", dist.Normal(0.0, float(_p["lag0_scale"]) * prior_scale))
    lag_scale = float(_p["lag_field_budget"]) * prior_scale / jnp.sqrt(N_lag_basis)
    priors['disease_w_lag'] = numpyro.sample(
        "disease_w_lag", dist.Normal(0.0, lag_scale).expand([N_lag_basis])
    )
    # DENSITY DEPENDENCE of severity: the Hill threshold and steepness, both fitted,
    # so the shape is learned. k_half is a DENSITY, hence declared in route counts and
    # converted through the gauge like every other absolute-scale quantity.
    priors['disease_k_half'] = numpyro.deterministic(
        "disease_k_half",
        counts_to_relative(jnp.exp(numpyro.sample(
            "disease_log_k_half_counts",
            dist.Normal(jnp.log(float(_p["k_half_median_route_counts"])),
                        float(_p["k_half_log_sd"]) * prior_scale)))),
    )
    numpyro.deterministic("disease_k_half_route_counts",
                          priors['disease_k_half'] * _POP_SCALAR)
    # Steepness. n ~ 1 is smooth saturation (mass-action-ish); n ~ 3 a sharp
    # invasion threshold. BOUNDED, and the bound is DERIVED rather than guessed:
    # K = K_base*(1 - ceiling*hill(K_base)) must be monotone in K_base or two
    # habitat qualities map to one realized capacity. At the worst point
    # (K_base = k_half) the slope is 1 - ceiling*(1/2 + n/4), giving n < 4/ceiling - 2
    # (= 6.0 at ceiling 0.5). That closed form is slightly OPTIMISTIC because the
    # true worst point sits a little below k_half -- numerically n=5.9 already goes
    # non-monotone (slope -0.009) at ceiling 0.5 -- so an 0.85 margin is applied.
    # Enforcing this structurally means the ceiling and the steepness cannot drift
    # into an unidentifiable combination: raising the ceiling automatically tightens
    # the allowed steepness, which a LogNormal prior would happily ignore.
    _n_max = 0.85 * (4.0 / float(_p["severity_ceiling"]) - 2.0)
    _n_min = 0.5
    _n_med = float(_p["hill_n_median"])
    _n_raw_loc = math.log((_n_med - _n_min) / (_n_max - _n_med))
    priors['disease_hill_n'] = numpyro.deterministic(
        "disease_hill_n",
        _n_min + (_n_max - _n_min) * jnn.sigmoid(numpyro.sample(
            "disease_hill_n_raw",
            dist.Normal(_n_raw_loc, float(_p["hill_n_log_sd"]) * 2.0 * prior_scale))),
    )

    priors['disease_tau'] = numpyro.deterministic(
        "disease_tau",
        # Onset sharpness. Floored because a vanishing tau makes the gate a step
        # function, whose gradient w.r.t. the lag terms vanishes everywhere the
        # front is not exactly at t. Also absorbs per-cell timing scatter that the
        # smooth regional lag field cannot resolve, by widening the front.
        jnn.softplus(numpyro.sample("disease_tau_raw",
                                    dist.Normal(0.5, 0.5 * prior_scale))) + 0.25,
    )

    # RECOVERY: slowly increasing resilience after local arrival (exposure, not
    # necessarily evolution). The hit relaxes from its peak toward
    # severity*(1-rec) with an e-folding time tau_rec. rec's prior median is ~38%
    # recovered, spanning 0 to ~90%, so "no recovery at all" stays cheap.
    priors['disease_rec'] = numpyro.deterministic(
        "disease_rec",
        jnn.sigmoid(numpyro.sample(
            "disease_rec_raw",
            dist.Normal(float(_p["recovery_logit_loc"]),
                        float(_p["recovery_logit_scale"]) * prior_scale))),
    )
    priors['disease_tau_rec'] = numpyro.deterministic(
        "disease_tau_rec",
        # LogNormal keeps the timescale positive and multiplicatively symmetric:
        # "about 12 years, plausibly 5 to 30" rather than an additive window that
        # could stray nonpositive.
        jnp.exp(numpyro.sample(
            "disease_log_tau_rec",
            dist.Normal(jnp.log(float(_p["recovery_tau_median_years"])),
                        float(_p["recovery_tau_log_scale"]) * prior_scale))),
    )

    # --- 2. DEMOGRAPHIC INTERCEPTS (Alphas) ---
    # Adult survival baseline > Juvenile survival baseline
    priors['alpha_a'] = numpyro.sample("alpha_a", dist.Normal(0.5, 0.5 * prior_scale)) # ~60%
    priors['alpha_j'] = numpyro.sample("alpha_j", dist.Normal(-0.5, 0.5 * prior_scale)) # ~40%
    priors['alpha_f'] = numpyro.sample("alpha_f", dist.Normal(2.0, 0.5 * prior_scale))  # Fecundity
    # CAPACITY LEVEL, declared in expected BBS route counts (see counts_to_relative).
    # SOFTPLUS link (controlled test): K_counts = softplus(alpha_k + gamma_k*H_k +
    # trend). The previous run used exp with a LogNormal level; alpha_k_loc here was
    # solved so the POST-TRANSFORMATION prior mean of the capacity level is identical
    # -- E[softplus(N(2.814974, 0.8))] = 2.8920 counts = exp(log 2.1 + 0.8^2/2) -- with
    # the raw scale left at 0.8, so the prior variance falls to 7.4% of the LogNormal's.
    # That reduction is expected and deliberately uncorrected.
    #
    # Reminder of what this prior replaced and why it matters: alpha_k used to be
    # stated in relative DENSITY units, where it asserted a capacity of ~205 route
    # counts (~97% of the highest counts ever recorded) against a data-implied ~2, so
    # it sat 7 prior SDs from the fit and every term able to lower K -- disease
    # severity, the H_k deviation, the continental trend -- was recruited as a level
    # reducer and saturated. Route counts, not density units, is the load-bearing part.
    priors['alpha_k'] = numpyro.sample(
        "alpha_k", dist.Normal(_ALPHA_K_LOC,
                               float(_CAPACITY_LEVEL["alpha_k_scale"]) * prior_scale))
    # Capacity at H_k = 0: the analogue of the exp form's separable level, kept under
    # the same names so the diagnostics and response curves read unchanged. Under
    # softplus it is not separable from the covariate term, so this is "capacity where
    # the covariates are neutral" rather than a multiplicative level.
    priors['k_level'] = numpyro.deterministic(
        "k_level", jnn.softplus(priors['alpha_k']))
    numpyro.deterministic("k_level_route_counts", priors['k_level'] * _POP_SCALAR)
    
    # --- 3. DEMOGRAPHIC SENSITIVITIES (Gammas) ---
    # Positive by construction: better habitat = higher survival/fecundity.
    #
    # These are now DIMENSIONLESS and fixed at 1, because the amplitude they used to
    # carry lives in w_scale (see the amplitude block above). gamma_j * w_scale and
    # gamma_f * w_scale etc. only ever entered the likelihood as products, so keeping
    # both left three exactly flat ridge directions -- tolerable under MAP, ruinous for
    # HMC step-size adaptation.
    priors['gamma_a'] = 1.0
    # gamma_j is now fixed at 1 as well, and gamma_j_diff is GONE as a free parameter.
    #
    # It survived as long as it did because Sa/Sj/Q shared ONE manifold: with a single beta_s,
    # the juvenile/adult sensitivity CONTRAST was identified even though the overall scale was
    # not. Now that Sj has its own beta_sj carrying w_scale[3], the product
    # gamma_j * w_scale[3] multiplies one unit-variance latent and is exactly the flat ridge
    # described above -- the same pathology that fixed the other three at 1.
    #
    # "Juveniles are MORE environment-sensitive than adults" (rule 5) is preserved, moved to the
    # place where it IS identified: amplitude_loc[3] is calibrated so w_scale[3] reproduces
    # w_scale[0] * E[1 + HalfNormal(0.5)], i.e. the amplitude the old product implied.
    priors['gamma_j'] = 1.0

    # Fmax and K are each the SOLE user of their manifold, so gamma * w_scale was an
    # exactly flat ridge. Fixed at 1; the amplitude is w_scale[1] and w_scale[2].
    priors['gamma_f'] = 1.0
    priors['gamma_k'] = 1.0

    # The slopes are reported two ways. The friendly names are for new code; the
    # *_raw names are emitted as DETERMINISTIC constants equal to softplus^-1(1) =
    # log(e-1), so the several existing readers that compute softplus(gamma_f_raw) --
    # analysis/plots.py, _age_vis_common.py, visualize_age_model.py,
    # visualize_community_similarity.py -- keep returning the correct value (1.0)
    # without edits.
    #
    # gamma_j_diff makes the same SAMPLED -> DETERMINISTIC migration, emitted as 0.0. Seven
    # modules compute `gamma_j = gamma_a + gamma_j_diff`; with gamma_a = softplus(gamma_a_raw) = 1
    # they now get 1.0 + 0.0 = 1.0, which is exactly right, with no edits needed. Emitting it is
    # cheaper and safer than deleting it and breaking all seven with a KeyError.
    _SOFTPLUS_INV_ONE = math.log(math.e - 1.0)
    for _name in ("gamma_a_raw", "gamma_f_raw", "gamma_k_raw"):
        numpyro.deterministic(_name, jnp.asarray(_SOFTPLUS_INV_ONE))
    numpyro.deterministic("gamma_j_diff", jnp.asarray(0.0))
    for _name, _val in (("gamma_a", priors['gamma_a']), ("gamma_j", priors['gamma_j']),
                        ("gamma_f", priors['gamma_f']), ("gamma_k", priors['gamma_k'])):
        numpyro.deterministic(_name, jnp.asarray(_val, dtype=float))
    
    # n50 is on the BBS ROUTE-COUNT scale and needs no conversion: allee_gamma*N reduces
    # to ln2*C/n50, a function of the observed count alone, so pop_scalar cannot change
    # what this prior asserts.
    #
    # Intent: a route that consistently yields one bird is a colonized cell, and that bird
    # stands for a local community rather than a lone individual, so mate-finding should
    # not be limiting there. At loc -2.0 (median n50 0.127 counts) the retained fecundity
    # is 42% at 0.1 counts, 94% at 0.5, and 99.6% at one bird -- the brake acts only on
    # genuinely sparse cells. See the _allee_comment in config for the two things this
    # controls downstream (the Allee-dead area in figure 07, and barrier-crossing cost).
    n50_raw = numpyro.sample(
        "n50_raw", dist.Normal(float(_ALLEE_PRIOR["n50_raw_loc"]),
                               float(_ALLEE_PRIOR["n50_raw_scale"]) * prior_scale))
    n50 = jnn.softplus(n50_raw)

    # Derive the searching efficiency on the RAW count scale
    # gamma_raw = ln(2) / N50
    priors['gamma_raw'] = jnp.log(2.0) / (n50 + 1e-6)

    priors['dispersal_logit_intercept'] = numpyro.sample("dispersal_logit_intercept", dist.Normal(2.0, 1.0 * prior_scale))
    priors['dispersal_logit_slope'] = numpyro.sample("dispersal_logit_slope", dist.Normal(4.0, 1.0 * prior_scale))
    
    # Temporal Annual Noise (Maintained for dispersal probability fluctuations)
    priors['dispersal_random'] = numpyro.sample("dispersal_random", dist.Normal(0., 0.001 * prior_scale), sample_shape=(time,))
    
    return priors


def build_model_2d(data, prior_scale=1.0):
    """The NumPyro model: priors -> demographic fields -> forward sim -> NB2 likelihood.

    ``data`` bundles the model-ready arrays (grid dims, land indices, the gathered
    Z / dispersal-feature memmaps, spatiotemporal basis, BBS observations and
    their per-observation quality tier, and scaling constants). Samples priors,
    projects Z to per-cell/per-year survival, fecundity, and carrying-capacity
    fields, runs the age-structured forward simulation from the invasion year,
    and scores BBS counts with a negative-binomial (NB2) likelihood whose
    concentration is down-weighted for lower-quality (unscreened Mexico)
    observations. ``prior_scale`` controls tight-to-nominal prior continuation.
    """
    validate_environment_kernel_contract(data)
    Nx, Ny = data['Nx'], data['Ny']
    time = data['time']
    land_rows, land_cols = data['land_rows'], data['land_cols']
    M = data['Z_gathered'].shape[-1]
    
    # 1. Sample Parameters
    priors = sample_priors(prior_scale, M, time,
                           N_sev_basis=data['N_sev_basis'],
                           N_lag_basis=data['N_lag_basis'])

    # Native-lineage shape: the SAME dimensionless core/margin map used to seed 1902
    # abundance (see age_priors' initpop comment and bbs.generate_core_margin_init),
    # reused here as the covariate for disease_b_native. Computed once, before the
    # disease dict, and reused again below for the initpop seed so the two uses can
    # never drift onto different arrays.
    native_shape = data['initpop_latent'][land_rows, land_cols]

    # Bundle the disease term's data and parameters. Grouping them keeps
    # project_and_scatter_age_structured's signature from growing another eight
    # positional arguments, and lets the field helpers in age_fields.py be called
    # directly by diagnostics with the same structure.
    disease = {
        "sev_basis": data['disease_sev_basis'],
        "lag_basis": data['disease_lag_basis'],
        "onset": data['disease_onset'],
        "onset_decades": data['disease_onset_decades'],
        "native_shape": native_shape,
        "mu_sev": priors['disease_mu_sev'],
        "b_late": priors['disease_b_late'],
        "b_native": priors['disease_b_native'],
        "w_sev": priors['disease_w_sev'],
        "lag0": priors['disease_lag0'],
        "w_lag": priors['disease_w_lag'],
        "tau": priors['disease_tau'],
        "ceiling": float(_DISEASE_PRIOR["severity_ceiling"]),
        "k_half": priors['disease_k_half'],
        "hill_n": priors['disease_hill_n'],
        "rec": priors['disease_rec'],
        "tau_rec": priors['disease_tau_rec'],
    }

    # The 1940 release(s), as a fraction of a FIXED, EMPIRICALLY OBSERVED, K-
    # INDEPENDENT reference (global_q50_route_counts) -- deliberately NOT k_level or
    # local K_base, either of which ties a historical release event's size to a
    # fitted, moving quantity and (as a fraction of the CONTINENTAL level) badly
    # diluted the founder at the actual release site, which sits at ~90-95th
    # percentile local capacity. See config's _invasion_pulse_comment.
    #
    # One coefficient per (candidate site, year): data['inv_locations'] is
    # (n_sites, 2), so log_inv_pulse_fraction is (n_sites, inv_window) -- the fit can
    # allocate mass across sites and across years rather than assuming both. The
    # shared prior's SD shrinks as sqrt(n_sites*inv_window) grows (same pattern as
    # disease_prior's sev_field_budget/sqrt(N_sev_basis)), so adding candidate sites
    # cannot inflate the term's total flexibility -- see config's
    # _invasion_budget_comment.
    _n_inv_sites = data['inv_locations'].shape[0]
    _inv_log_sd = (float(_INVASION_PULSE["log_budget"])
                  / jnp.sqrt(_n_inv_sites * data['inv_window'])) * prior_scale
    _q50_density = float(_INVASION_PULSE["global_q50_route_counts"]) / _POP_SCALAR
    inv_pop = numpyro.deterministic("inv_pop_relative", _q50_density * jnp.exp(
        numpyro.sample("log_inv_pulse_fraction",
                       dist.Normal(jnp.log(float(_INVASION_PULSE["median_fraction_of_global_q50"])),
                                   _inv_log_sd),
                       sample_shape=(_n_inv_sites, data['inv_window']))))

    # Convert to the relative [0, 1] scale by multiplying by pop_scalar
    # Since N_relative = N_raw / pop_scalar, 
    # then gamma_relative = gamma_raw * pop_scalar
    allee_gamma_scaled = priors['gamma_raw'] * data['pop_scalar']
    priors['allee_gamma'] = numpyro.deterministic("allee_gamma", allee_gamma_scaled)

    # 1. Compute Biological Fields (2D Manifold -> Demographic Rates)
    # Notice we now pass beta_s and beta_r instead of a single beta_h
    Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat, Kbase_flat = project_and_scatter_age_structured(
        time, Ny, Nx, land_rows, land_cols,
        data['Z_gathered'], data['Z_disp_gathered'],
        data['disease_timestep'], disease,
        priors['beta_s'], priors['beta_r'], priors['beta_k'], priors['beta_sj'],
        data['k_trend_basis'], priors['w_k_trend'],
        priors['alpha_a'], priors['gamma_a'],
        priors['alpha_j'], priors['gamma_j'],
        priors['alpha_f'], priors['gamma_f'],
        priors['alpha_k'], priors['gamma_k']
    )
        
    # Save fields for viz
    numpyro.deterministic("Sa_flat", Sa_flat)
    numpyro.deterministic("Sj_flat", Sj_flat)
    numpyro.deterministic("Fmax_flat", Fmax_flat)
    numpyro.deterministic("K_flat", K_flat)
    numpyro.deterministic("K_base_flat", Kbase_flat)

    # NATIVE-RANGE SEED, as a fraction of LOCAL capacity in the first model year.
    # data['initpop_latent'] is a dimensionless SHAPE (core=1, margin=
    # initpop_seed.margin_fraction_of_core) marking WHERE the 1902 native range was --
    # the contrast is assumed, not measured, because an observed abundance ratio
    # double-counts the habitat gradient K_base already carries; the amplitude comes
    # from K_base itself, so the seed automatically tracks whatever capacity the
    # covariates imply for those cells.
    #
    # Why not an absolute value, and why not a fraction of the CONTINENTAL level:
    # both were tried today and both were wrong, in opposite directions, for the same
    # reason -- they compared the seed to the wrong K. An absolute 9.5 route counts is
    # ~33% of local capacity in the native core (observed 28.6 counts) and perfectly
    # sensible for a native range at equilibrium, but at INITIALIZATION AutoDelta sets
    # beta_k near its prior median so H_k ~ 0 and every cell has K = k_level = 2.1
    # counts: the seed is then 4.5x capacity, the density brake slams on, and step 0
    # of the fit scores badly. Rescaling to 10% of the CONTINENTAL level fixed step 0
    # but left the native core starting at <1% of its own local capacity. A fraction
    # of LOCAL K is correct at both ends: at step 0 local K = k_level so the seed is a
    # modest fraction of it, and at the optimum it scales with the fitted capacity of
    # those specific cells.
    #
    # The fraction is 1.0: a native range at equilibrium since long before 1902 starts
    # AT its capacity, not below it. Since dispersal_target_capacity_fraction is 0.8,
    # that puts the seed at N/K - 0.8 = +0.2 on the emigration logit, i.e. the native
    # range is a NET EXPORTER from year one. Deliberate: the thing to be explained is
    # that an exporting western population still did not cross the Great Plains for
    # eighty years, so the barrier has to be paid for by the pre-invasion pseudo-zeros
    # (bbs.py) and the covariates, not by a native range too weak to push.
    # Reuse native_shape (computed above, before the disease dict) rather than
    # re-slicing initpop_latent -- one array, two uses, so they cannot drift apart.
    _seed_flat = (native_shape * Kbase_flat[0]
                  * float(_INITPOP_SEED["core_fraction_of_local_capacity"]))
    initpop_seeded = numpyro.deterministic(
        "initpop_seeded",
        jnp.zeros((Ny, Nx)).at[land_rows, land_cols].set(_seed_flat))
    # Per-cell peak severity at MODERN capacity -- the model's falsifiable claim
    # about the epizootic. Density-dependent, so it must be evaluated against the
    # undepressed K_base (using the depressed K would be circular), and at a
    # specific year since K_base drifts with the continental trend.
    numpyro.deterministic("disease_severity_map",
                          disease_severity(disease, Kbase_flat[-1]))
    numpyro.deterministic("Q_flat", Q_flat)

    # --- POC IDENTIFIABILITY CONSTRAINT: SITE-LEVEL EQUILIBRIUM AT K ---
    
    # The forward census order is survival, then reproduction by surviving
    # adults. Its local linearized matrix is [[Sa, Sj], [F*Sa, 0]].
    # Thus fecundity at lambda=1 is (1-Sa)/(Sa*Sj).
    c_flat, F_at_K, lambda_K, rho_K = equilibrium_age_quantities(
        Sa_flat, Sj_flat, Fmax_flat, K_flat, priors["allee_gamma"]
    )

    # Weak belief about the distribution of LOCAL age structure. Average over
    # cell-years first, then apply a fixed effective-sample power; never let grid
    # resolution manufacture millions of independent prior observations.
    age_cfg = data.get("age_structure_prior") or {}
    numpyro.factor(
        "local_age_structure_regularizer",
        age_structure_log_prior(
            rho_K,
            alpha=age_cfg.get("alpha", 1.01),
            beta=age_cfg.get("beta", 1.01),
            effective_sites=age_cfg.get("effective_sites", 100.0),
        ),
    )

    densities, Na_grid, Nj_grid = forward_sim_age_structured(
        Sa_flat, Sj_flat, Fmax_flat, K_flat, c_flat, Q_flat,
        land_rows, land_cols,
        data['land_mask'],
        data['adult_fft_kernel'], data['juvenile_fft_kernel_stack'],
        data['adult_edge_correction'], data['juvenile_edge_correction_stack'],
        initpop_seeded, priors['dispersal_random'], inv_pop,
        time, data['inv_locations'], data['inv_timestep'],
        priors['dispersal_logit_intercept'], priors['dispersal_logit_slope'],
        priors['allee_gamma'],
        target_fraction=data["dispersal_target_fraction"],
    )

    numpyro.deterministic("simulated_density", densities)
    # Realized age-class pools (adults/juveniles separately, full grid, all
    # years) -- for the post-hoc "realized age structure" diagnostic. Not on
    # the likelihood path; DCE'd away during actual MAP/SVI optimization (see
    # forward_sim_age_structured's docstring), only materialized when
    # requested via Predictive(return_sites=[...]).
    numpyro.deterministic("Na_grid", Na_grid)
    numpyro.deterministic("Nj_grid", Nj_grid)

    # 4. Likelihood
    t_idx, rows, cols = data["obs_time_indices"], data["obs_rows"], data["obs_cols"]
    
    # densities output should be the sum of adult + juvenile (N_total)
    densities_obs = jnp.maximum(densities[t_idx, rows, cols] * data["pop_scalar"], 1e-6)
    
    numpyro.deterministic("expected_obs", densities_obs)
    # NB2 overdispersion: var = mean + mean^2 / concentration, so a LOWER
    # concentration = more overdispersion = a weaker likelihood constraint.
    concentration = numpyro.sample("concentration", dist.Exponential(1.0))

    # Observation-quality down-weighting. obs_quality is a per-observation tier
    # (0 = standard US/Canada + pseudo-zeros; 1 = Mexico unprocessed, which has
    # no RunType/RPID screening). Mexico obs get concentration * q_mult with
    # q_mult in (0,1), i.e. more overdispersion, so unprocessed data informs the
    # fit but is never treated as more reliable than screened data. The bound
    # (0,1) is the principled constraint; the data set the magnitude. This is a
    # no-op when only tier-0 observations are present (q_mult ** 0 == 1).
    obs_quality = data.get("obs_quality")
    if obs_quality is not None and int(jnp.max(obs_quality)) > int(jnp.min(obs_quality)):
        q_mult = numpyro.sample("quality_conc_mult", dist.Beta(2.0, 2.0))
        conc_obs = concentration * jnp.power(q_mult, obs_quality)
    else:
        conc_obs = concentration

    numpyro.sample(
        "obs",
        dist.NegativeBinomial2(mean=densities_obs, concentration=conc_obs),
        obs=data["observed_results"]
    )
