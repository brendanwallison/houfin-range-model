"""Priors and the NumPyro model for the age-structured range-expansion model.

:func:`build_model_2d` is the full probabilistic model: it samples parameters
(:func:`sample_priors`), maps the latent habitat manifold (Z) to spatial
demographic-rate fields (survival/fecundity/carrying capacity, via
``age_fields``), runs the age-structured dispersal forward simulation
(``age_forward``), and scores BBS counts under a negative-binomial (NB2)
observation model. ``anneal`` scales prior widths for tempered
warm-up/optimization. See docs/TEMPORAL.md for the invasion-timestep convention.
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

def sample_priors(anneal=1.0, M_features=None, N_basis=None, time=None):
    """Sample every model parameter and return them in a dict.

    Covers the correlated 2-D habitat-manifold weights (survival vs
    reproduction, with an explicit correlation ``rho``), the spatiotemporal
    basis weights, dispersal/demography rate parameters, and the Allee term.
    ``anneal`` widens/narrows prior scales for tempered fitting; ``M_features``,
    ``N_basis``, ``time`` size the manifold/basis/temporal dimensions.
    """
    priors = {}
    
    # --- 1. CORRELATED 2D HABITAT MANIFOLD WEIGHTS ---
    
    # 1. Sample rho explicitly with a strong positive prior (e.g., centered at +0.7).
    # We bound it strictly between -0.99 and 0.99 to prevent NaN errors in the Cholesky math.
    rho = numpyro.sample("rho", dist.TruncatedNormal(loc=0.7, scale=0.2, low=-0.99, high=0.99))
    
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
    w_scale = numpyro.sample("w_scale", dist.HalfNormal(0.5).expand([2]))
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
    
    # 1D spectral weights (Spatio-temporal random effects)
    # 1. Define the global budget for spatial noise (e.g., 0.1 allows for moderate regional tweaks)
    global_spatial_budget = 0.001 * anneal
    
    # 2. Distribute that budget dynamically 
    dynamic_scale = global_spatial_budget / jnp.sqrt(N_basis)
    
    # 3. Apply the scaled L1 penalty
    priors['st_weights'] = numpyro.sample(
        "st_weights", 
        dist.Laplace(0.0, dynamic_scale).expand([N_basis])
    )
    
    # --- 2. DEMOGRAPHIC INTERCEPTS (Alphas) ---
    # Adult survival baseline > Juvenile survival baseline
    priors['alpha_a'] = numpyro.sample("alpha_a", dist.Normal(0.5, 0.5 * anneal)) # ~60%
    priors['alpha_j'] = numpyro.sample("alpha_j", dist.Normal(-0.5, 0.5 * anneal)) # ~40%
    priors['alpha_f'] = numpyro.sample("alpha_f", dist.Normal(2.0, 0.5 * anneal))  # Fecundity
    priors['alpha_k'] = numpyro.sample("alpha_k", dist.Normal(0.5, 0.5 * anneal))  # Capacity
    
    # --- 3. DEMOGRAPHIC SLOPES (Gammas) ---
    # Enforce positive slopes: better habitat = higher survival/fecundity
    # Enforce Rule 5: Juvenile survival is more sensitive to environment than adult
    gamma_a_raw = numpyro.sample("gamma_a_raw", dist.Normal(0.0, 0.5 * anneal))
    gamma_j_diff = numpyro.sample("gamma_j_diff", dist.HalfNormal(0.5 * anneal))
    
    priors['gamma_a'] = jnn.softplus(gamma_a_raw)
    priors['gamma_j'] = priors['gamma_a'] + gamma_j_diff 
    
    priors['gamma_f'] = jnn.softplus(numpyro.sample("gamma_f_raw", dist.Normal(0.0, 1.0 * anneal)))
    priors['gamma_k'] = jnn.softplus(numpyro.sample("gamma_k_raw", dist.Normal(0.0, 1.0 * anneal)))
    
    # Sample the physical threshold (N50: number of birds for 50% mate-finding prob)
    # Standard normal + softplus = mean of ~0.86
    n50_raw = numpyro.sample("n50_raw", dist.Normal(-1.0, 1.0 * anneal))
    n50 = jnn.softplus(n50_raw)

    # Derive the searching efficiency on the RAW count scale
    # gamma_raw = ln(2) / N50
    priors['gamma_raw'] = jnp.log(2.0) / (n50 + 1e-6)

    priors['dispersal_logit_intercept'] = numpyro.sample("dispersal_logit_intercept", dist.Normal(2.0, 1.0 * anneal))
    priors['dispersal_logit_slope'] = numpyro.sample("dispersal_logit_slope", dist.Normal(4.0, 1.0 * anneal))
    
    # Temporal Annual Noise (Maintained for dispersal probability fluctuations)
    priors['dispersal_random'] = numpyro.sample("dispersal_random", dist.Normal(0., 0.001 * anneal), sample_shape=(time,))
    
    return priors


def build_model_2d(data, anneal=1.0):
    """The NumPyro model: priors -> demographic fields -> forward sim -> NB2 likelihood.

    ``data`` bundles the model-ready arrays (grid dims, land indices, the gathered
    Z / dispersal-feature memmaps, spatiotemporal basis, BBS observations and
    their per-observation quality tier, and scaling constants). Samples priors,
    projects Z to per-cell/per-year survival, fecundity, and carrying-capacity
    fields, runs the age-structured forward simulation from the invasion year,
    and scores BBS counts with a negative-binomial (NB2) likelihood whose
    concentration is down-weighted for lower-quality (unscreened Mexico)
    observations. ``anneal`` tempers the priors.
    """
    validate_environment_kernel_contract(data)
    Nx, Ny = data['Nx'], data['Ny']
    time = data['time']
    land_rows, land_cols = data['land_rows'], data['land_cols']
    M = data['Z_gathered'].shape[-1]
    
    # 1. Sample Parameters
    priors = sample_priors(anneal, M, data['N_basis'], time)
    
    inv_pop = jnn.softplus(numpyro.sample(
        "inv_eta", dist.Normal(-2.0, 1.0 * anneal), sample_shape=(data['inv_window'],)
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
        data['st_basis'], priors['st_weights'], 
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
    
    # 1. Calculate dynamic 'c' exactly as in the forward sim
    # Using 1e-6 for the eps term to prevent division by zero
    c_dynamic = (Fmax_flat * Sj_flat) / (1.0 - Sa_flat + 1e-6) - 1.0
    c_dynamic = jnp.maximum(c_dynamic, 0.0)
    
    # 2. Evaluate Effective Fecundity at N = K (so N/K = 1.0)
    F_eff_K = Fmax_flat / (1.0 + c_dynamic)
    
    # 3. Evaluate the Allee Factor at N = K
    # priors['allee_gamma'] is already defined earlier in build_model_2d
    allee_factor_K = 1.0 - jnp.exp(-priors['allee_gamma'] * K_flat)
    
    # 4. Calculate actual realized fecundity at Carrying Capacity
    F_at_K = F_eff_K * allee_factor_K
    
    # 5. Calculate theoretical dominant eigenvalue of the Leslie matrix at K
    lambda_K = (Sa_flat + jnp.sqrt(Sa_flat**2 + 4.0 * F_at_K * Sj_flat)) / 2.0
    
    # 6. Calculate the theoretical juvenile fraction at K
    rho_K = F_at_K / (F_at_K + lambda_K)
    
    # # 7. Define the target mean and desired standard deviation (loosen via anneal if desired)
    # mu_target = 0.5
    # sigma_target = 0.2 * anneal  # Testing with a looser 0.2 baseline
    # variance_target = jnp.square(sigma_target)

    # # 8. Correctly convert to Beta shape parameters using mu_target explicitly
    # # This factor ensures the math scales even if you shift your target mean later
    # v_factor = (mu_target * (1.0 - mu_target) / variance_target) - 1.0
    
    # alpha_shape = jnp.maximum(mu_target * v_factor, 1.001)
    # beta_shape = jnp.maximum((1.0 - mu_target) * v_factor, 1.001)
    
    # 9. Apply the bounded Beta prior across the grid
    rho_K_safe = jnp.clip(rho_K, 1e-5, 1.0 - 1e-5)

    # 1. Compute c natively
    denominator_c = 1.0 - Sa_flat + 1e-6
    c_flat = jnp.maximum((Fmax_flat * Sj_flat) / denominator_c - 1.0, 0.0)

    # Existing age-structure target prior
    numpyro.factor(
        "poc_site_level_bounded_age_structure", 
        dist.Beta(1.01, 1.01).log_prob(rho_K_safe).sum()
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
        data['pseudo_zero']
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
