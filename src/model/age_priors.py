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
import jax.numpy as jnp
import jax.nn as jnn
import numpyro
import numpyro.distributions as dist

from src.model.age_fields import project_and_scatter_age_structured
from src.model.age_forward import forward_sim_age_structured


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


def sample_priors(prior_scale=1.0, M_features=None, N_basis=None, time=None):
    """Sample every model parameter and return them in a dict.

    Covers the correlated 2-D habitat-manifold weights (survival vs
    reproduction, with an explicit correlation ``rho``), the spatiotemporal
    basis weights, dispersal/demography rate parameters, and the Allee term.
    ``prior_scale`` multiplies scale parameters for continuation fitting;
    ``M_features``, ``N_basis``, ``time`` size the dimensions.
    """
    priors = {}
    
    # --- 1. CORRELATED 2D HABITAT MANIFOLD WEIGHTS ---
    
    # 1. Sample rho explicitly with a strong positive prior (e.g., centered at +0.7).
    # We bound it strictly between -0.99 and 0.99 to prevent NaN errors in the Cholesky math.
    rho = numpyro.sample(
        "rho",
        dist.TruncatedNormal(
            loc=0.7, scale=0.2 * prior_scale, low=-0.99, high=0.99
        ),
    )
    
    # 2. Manually construct the Cholesky factor of a 2x2 correlation matrix
    # The Cholesky decomposition of [[1, rho], [rho, 1]] is analytically:
    L_corr_matrix = jnp.array([
        [1.0, 0.0],
        [rho, jnp.sqrt(1.0 - rho**2)]
    ])
    
    # Save L_corr as a deterministic site so your visualization script doesn't break
    L_corr = numpyro.deterministic("L_corr", L_corr_matrix)
    
    # 3. Scale the response correlation matrix. Crucially, the same 2x2 prior is
    # repeated IID over feature dimensions by the plate below: conditional on
    # w_scale, Cov[Z(x)@beta_s, Z(x')@beta_s] = w_scale[0]^2 Z(x)@Z(x'), and
    # likewise for reproduction. This isotropy is what recovers the scaled
    # uncentered Ružička GP kernel represented by Z.
    w_scale = numpyro.sample(
        "w_scale", dist.HalfNormal(0.5 * prior_scale).expand([2])
    )
    numpyro.deterministic("environment_kernel_variance", w_scale ** 2)
    L_cov = w_scale[..., None] * L_corr
    
    # 4. Draw the correlated weights for all M features
    with numpyro.plate("env_features", M_features):
        w_env = numpyro.sample(
            "w_env", 
            dist.MultivariateNormal(loc=jnp.zeros(2), scale_tril=L_cov)
        )
        
    priors['beta_s'] = w_env[:, 0]  # Survival Suitability Weights
    priors['beta_r'] = w_env[:, 1]  # Reproductive Suitability Weights
    
    # 1D spectral weights for the K-only spatiotemporal correction (see
    # age_fields.py's _K_CORRECTION_OFFSET / project_and_scatter_age_structured).
    # This is NOT a smoothing term on Z/H_s/H_r -- it is a genuinely latent,
    # zero-mean multiplicative correction to carrying capacity, meant to soak
    # up dynamics (e.g. mycoplasmal conjunctivitis) this Z-driven covariate
    # structure has no way to see. A Normal (not Laplace) prior is used
    # deliberately: symmetric around zero, so "no correction" is the natural
    # center of the prior rather than a boundary the base K has to fight
    # against. (An earlier design bounded this to reduction-only via a
    # one-sided sigmoid link: by Jensen's inequality a concave link's
    # expectation under ANY zero-mean perturbation sits below its own
    # zero-perturbation value, and that shortfall is data-dependent -- larger
    # wherever the term is actually used -- so alpha_k would be pulled upward
    # to compensate, contaminating the very spatial/temporal pattern this term
    # is meant to isolate. softplus is convex everywhere, so it has the same
    # Jensen shortfall in the OTHER direction, but that shortfall depends only
    # on the prior scale, not on data/location/time, so alpha_k absorbs it as
    # a harmless flat constant instead.) The budget is distributed across
    # N_basis coefficients so total per-cell-year correction variance stays
    # roughly budget^2/2 regardless of how finely N_basis is set.
    global_k_correction_budget = 2.0 * prior_scale  # deliberately loose (was 0.001 under the old Laplace/Z-smoothing design)
    dynamic_scale = global_k_correction_budget / jnp.sqrt(N_basis)
    priors['st_weights'] = numpyro.sample(
        "st_weights",
        dist.Normal(0.0, dynamic_scale).expand([N_basis])
    )
    
    # --- 2. DEMOGRAPHIC INTERCEPTS (Alphas) ---
    # Adult survival baseline > Juvenile survival baseline
    priors['alpha_a'] = numpyro.sample("alpha_a", dist.Normal(0.5, 0.5 * prior_scale)) # ~60%
    priors['alpha_j'] = numpyro.sample("alpha_j", dist.Normal(-0.5, 0.5 * prior_scale)) # ~40%
    priors['alpha_f'] = numpyro.sample("alpha_f", dist.Normal(2.0, 0.5 * prior_scale))  # Fecundity
    priors['alpha_k'] = numpyro.sample("alpha_k", dist.Normal(0.5, 0.5 * prior_scale))  # Capacity
    
    # --- 3. DEMOGRAPHIC SLOPES (Gammas) ---
    # Enforce positive slopes: better habitat = higher survival/fecundity
    # Enforce Rule 5: Juvenile survival is more sensitive to environment than adult
    gamma_a_raw = numpyro.sample("gamma_a_raw", dist.Normal(0.0, 0.5 * prior_scale))
    gamma_j_diff = numpyro.sample("gamma_j_diff", dist.HalfNormal(0.5 * prior_scale))
    
    priors['gamma_a'] = jnn.softplus(gamma_a_raw)
    priors['gamma_j'] = priors['gamma_a'] + gamma_j_diff 
    
    priors['gamma_f'] = jnn.softplus(numpyro.sample("gamma_f_raw", dist.Normal(0.0, 1.0 * prior_scale)))
    priors['gamma_k'] = jnn.softplus(numpyro.sample("gamma_k_raw", dist.Normal(0.0, 1.0 * prior_scale)))
    
    # N50 is expressed on the BBS-route count scale. A single detected bird can
    # proxy for an established local population, so this deliberately places the
    # transition near the first observable presence rather than tens of detections.
    n50_raw = numpyro.sample("n50_raw", dist.Normal(-1.0, 1.0 * prior_scale))
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
    priors = sample_priors(prior_scale, M, data['N_basis'], time)
    
    inv_pop = jnn.softplus(numpyro.sample(
        "inv_eta", dist.Normal(-2.0, 1.0 * prior_scale), sample_shape=(data['inv_window'],)
    ))
    
    # Convert to the relative [0, 1] scale by multiplying by pop_scalar
    # Since N_relative = N_raw / pop_scalar, 
    # then gamma_relative = gamma_raw * pop_scalar
    allee_gamma_scaled = priors['gamma_raw'] * data['pop_scalar']
    priors['allee_gamma'] = numpyro.deterministic("allee_gamma", allee_gamma_scaled)

    # 1. Compute Biological Fields (2D Manifold -> Demographic Rates)
    # Notice we now pass beta_s and beta_r instead of a single beta_h
    Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat = project_and_scatter_age_structured(
        time, Ny, Nx, land_rows, land_cols,
        data['Z_gathered'], data['Z_disp_gathered'],
        data['st_basis'], priors['st_weights'], data['inv_timestep'],
        priors['beta_s'], priors['beta_r'],
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

    densities = forward_sim_age_structured(
        Sa_flat, Sj_flat, Fmax_flat, K_flat, c_flat, Q_flat, 
        land_rows, land_cols,           
        data['land_mask'],
        data['adult_fft_kernel'], data['juvenile_fft_kernel_stack'],
        data['adult_edge_correction'], data['juvenile_edge_correction_stack'],
        data['initpop_latent'], priors['dispersal_random'], inv_pop,
        time, data['inv_location'], data['inv_timestep'],
        priors['dispersal_logit_intercept'], priors['dispersal_logit_slope'],
        priors['allee_gamma'],
        target_fraction=data["dispersal_target_fraction"],
    )

    numpyro.deterministic("simulated_density", densities)

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
