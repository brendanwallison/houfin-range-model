"""Age-structured forward population simulation (JAX / lax.scan).

One model year advances the two age classes — adults ``N_a`` and juveniles
``N_j`` — through invasion seeding, density-dependent dispersal (FFT convolution
with the precomputed kernels), and survival + Beverton-Holt / Allee
reproduction. The whole time loop is a gradient-checkpointed ``lax.scan`` so it
can be differentiated through under NumPyro without storing every year's
activations. Returns total density (adults + juveniles) per year on the grid.
"""
import jax.numpy as jnp
import jax.nn as jnn
from jax import lax
import jax


def clip_safe(x, min_val=-10.0, max_val=10.0):
    """Clip to a safe logit range so sigmoids don't saturate to 0/1 (NaN grads)."""
    return jnp.clip(x, min_val, max_val)


def rightpad(A, Lx, Ly, pad_value=1e-9):
    """Pad ``A`` to (Ly, Lx) on the bottom/right edges only (for linear FFT conv)."""
    pad_y = Ly - A.shape[0]
    pad_x = Lx - A.shape[1]
    return jnp.pad(A, ((0, pad_y), (0, pad_x)), constant_values=pad_value)


# population_model.numerics (set by age_priors at import; defaults = the legacy behaviour).
#   FFT_PAD_VALUE: what the linear-convolution padding is filled with. Legacy 1e-9 is a pseudo-zero:
#     the kernel spreads it back into the grid every step -- a continent-wide trickle of immigration.
#   FFT_FLOOR_ULPS: if set, convolution output below FFT_FLOOR_ULPS * eps(dtype) * max(output) is set
#     to 0 -- values the FFT cannot resolve. float32 round-off otherwise leaves a positive floor
#     (~1e-6..3e-5 per land cell) that acts as phantom immigration ahead of the front (E054c).
FFT_PAD_VALUE = 1e-9
FFT_FLOOR_ULPS = None
#   FFT_FLOOR_SMOOTH: replace the floor's hard cut-off with the C1 soft threshold x^3 / (x^2 + tol^2) (x > 0; 0 below):
#     ~x well above tol, ~x^3/tol^2 (effectively 0) well below. Same purpose (no float32 phantom floor), but U stays
#     continuous: the hard version made U jump (+.45 nats along -grad at the 66b stall) and the refine exploited it.
FFT_FLOOR_SMOOTH = False
#   FFT_FLOOR_FLOAT32_ONLY: apply the floor only to float32 convolutions (the MAP stage, where the round-off floor is
#     phantom immigration) and never in float64 (the refine), where exact extinction creates the Allee establishment
#     switch the optimiser exploits (E069) and the ~1e-16 round-off is suppressed by the Allee factor anyway.
FFT_FLOOR_FLOAT32_ONLY = False
#   SMOOTH_ALLOC: if set, the fitness ideal-free allocations (fitness_settle, subcell_reproduction) use a
#     compact C1 positive part with width = SMOOTH_ALLOC x the cell's (K/c) scale,
#     instead of max(n, 0) switched off at each cell's lambda0. The hard version puts a corner in the
#     allocation at every cell/slice threshold; L-BFGS stalled on such corners with |grad| ~1e6 along a
#     non-descent direction (E058b/c gradient anatomy).
SMOOTH_ALLOC = None


def _positive_part(n, scale):
    """max(n, 0), or compact C1 smoothing at SMOOTH_ALLOC x ``scale`` when enabled."""
    if SMOOTH_ALLOC is None:
        return jnp.maximum(n, 0.0)
    w = jnp.maximum(SMOOTH_ALLOC * scale, 1e-12)
    # Compact C1 smoothing: unlike softplus this is EXACTLY zero below -w, so an
    # empty/near-zero-capacity slice cannot acquire a continent-wide numerical
    # population tail.  Clip before evaluating the quadratic so unselected
    # branches cannot overflow and poison reverse-mode through 0 * inf.
    nm = jnp.clip(n, -w, w)
    transition = (nm + w) ** 2 / (4.0 * w)
    return jnp.where(n >= w, n, jnp.where(n <= -w, 0.0, transition))
# Occupancy (E063): below this psi a cell is treated as empty (m = 0). Together with the closure's own
# consistency psi >= 1 - exp(-b N) (enforced in occupied_density) m stays <= ~1/b birds wherever N > 0,
# even when float32 rounding breaks the identity or underflows psi.
_PSI_FLOOR = 1e-30


def occupied_density(N, psi, b):
    """(psi_c, m): psi made consistent with N (psi >= 1 - exp(-b N)) and the occupied-conditional density
    m = N / psi_c, 0 where the cell is effectively empty -- safe for values and gradients."""
    psi_c = jnp.clip(jnp.maximum(psi, -jnp.expm1(-b * jnp.maximum(N, 0.0))), 0.0, 1.0)
    ok = psi_c > _PSI_FLOOR
    m = jnp.where(ok, N / jnp.where(ok, psi_c, 1.0), 0.0)
    return psi_c, m


def pairing_fraction(birds_occupied, n_iter=12):
    """Fraction of birds that can form a pair in an OCCUPIED realisation holding on average
    ``birds_occupied`` (>= 1) birds -- sex-ratio stochasticity at the scale of individual birds.

    Birds are Poisson(lam) with random sexes, conditioned on >= 1 bird, so the occupied mean is
    lam / (1 - exp(-lam)); lam solves lam = x (1 - exp(-lam)) (x = 1 -> lam = 0), by Newton from
    lam0 = 2 (x - 1) near one bird (the root's expansion there) and lam0 = x above.
    Males and females are independent Poisson(lam/2); pairs = min(M, F), and
    E[2 min(M, F)] / E[M + F] = 1 - exp(-lam) (I0(lam) + I1(lam))  (Skellam). A lone bird -> 0; large
    groups -> 1 - sqrt(2 / (pi lam)) -> 1, so the mean-field model is recovered as b -> infinity.
    """
    from jax.scipy.special import i0e, i1e
    x = jnp.maximum(birds_occupied, 1.0)
    lam = jnp.where(x < 2.0, 2.0 * (x - 1.0), x)
    for _ in range(n_iter):
        f = lam + x * jnp.expm1(-lam)
        df = jnp.maximum(1.0 - x * jnp.exp(-lam), 1e-6)
        lam = jnp.maximum(lam - f / df, 0.0)
    lam = jnp.maximum(lam, 1e-12)
    return jnp.clip(1.0 - i0e(lam) - i1e(lam), 0.0, 1.0)


def rightpad_convolution(pop, dispersal_kernel_pad):
    """Convolve a population grid with a (pre-FFT'd, padded) dispersal kernel.

    Right-pads ``pop`` to the kernel's padded size so the circular FFT product
    is a *linear* convolution, then crops back to the original grid shape.
    """
    Ly, Lx = dispersal_kernel_pad.shape
    pop_pad = rightpad(pop, Lx, Ly, FFT_PAD_VALUE)
    conv = jnp.fft.ifft2(jnp.fft.fft2(pop_pad) * dispersal_kernel_pad)
    out = jnp.real(conv)[:pop.shape[0], :pop.shape[1]]
    if FFT_FLOOR_ULPS is not None and not (FFT_FLOOR_FLOAT32_ONLY and out.dtype != jnp.float32):
        tol = FFT_FLOOR_ULPS * jnp.finfo(out.dtype).eps * lax.stop_gradient(jnp.max(jnp.abs(out)))
        if FFT_FLOOR_SMOOTH:
            pos = jnp.maximum(out, 0.0)
            out = pos * pos * pos / (pos * pos + tol * tol + jnp.finfo(out.dtype).tiny)
        else:
            out = jnp.where(out > tol, out, 0.0)
    return out

N_DIRECTIONS = 4   # build_kernels.make_radial_directional_kernels: index = direction * n_bands + band


def juvenile_band_weights(juvenile_fft_kernel_stack, band_logw):
    """Per-kernel multipliers that reweight the juvenile kernel's RADIAL BANDS, mass-preserving.

    ``band_logw`` (n_bands - 1,) are log-weights of the outer bands relative to the innermost
    (fixed 0). Band b's weight is w_b = exp(theta_b) / sum_b M_b exp(theta_b), M_b its current
    mass (every kernel's mass is the zero-frequency term of its FFT), so the reweighted kernel
    still integrates to 1 and theta = 0 is the current kernel exactly. Raising an outer band's
    weight fattens the tail -- long jumps that seed satellites ahead of the front -- WITHOUT new
    path features: each band keeps its own precomputed journey survival Q.
    """
    K = juvenile_fft_kernel_stack.shape[0]
    n_bands = K // N_DIRECTIONS
    mass = jnp.real(juvenile_fft_kernel_stack[:, 0, 0]).reshape(N_DIRECTIONS, n_bands).sum(0)
    theta = jnp.concatenate([jnp.zeros((1,), band_logw.dtype), band_logw])
    w = jnp.exp(theta) / jnp.sum(mass * jnp.exp(theta))
    return jnp.tile(w, N_DIRECTIONS)                                   # (K,), direction-major


def juvenile_dispersal_vectorized(
    juvenile_dispersers: jnp.ndarray,       
    juvenile_fft_kernels: jnp.ndarray,      
    Q: jnp.ndarray,                         
    juvenile_edge_correction_stack: jnp.ndarray,
    eps: float = 1e-6,
    kernel_weights=None,
):
    """Disperse juveniles through the directional/radial kernel stack.

    Each kernel is applied to the (edge-corrected) disperser field, the arrivals
    are scaled by the per-kernel journey-survival ``Q``, and the surviving
    settlers are summed across kernels. The edge correction divides out the
    fraction of each kernel's footprint that lay over land, so probability mass
    isn't lost to ocean/off-grid cells.
    """
    def single_kernel_prop(kernel_fft, land_fraction_map):
        boosted_source = juvenile_dispersers / (land_fraction_map + eps)
        settled = rightpad_convolution(boosted_source, kernel_fft)
        return settled

    potential_settlers = jax.vmap(single_kernel_prop)(
        juvenile_fft_kernels, 
        juvenile_edge_correction_stack
    )
    
    successful_settlers = potential_settlers * Q
    if kernel_weights is not None:          # radial-band reweighting (juvenile_band_weights)
        successful_settlers = successful_settlers * kernel_weights[:, None, None]
    return jnp.sum(successful_settlers, axis=0)

def departure_logit(x, intercept, slope, target_fraction, crowding_beta=None, crowding_kappa=5.0):
    """Departure log-odds at relative density ``x = N/K``: a free-sign linear arm, plus the optional
    non-negative crowding arm that switches on smoothly above capacity (x > 1)."""
    z = intercept + slope * (x - target_fraction)
    if crowding_beta is not None:
        z = z + crowding_beta * jnn.softplus(crowding_kappa * (x - 1.0)) / crowding_kappa
    return z


def dispersal_step_age_structured(
    N_a, N_j, K, 
    dispersal_logit_intercept, dispersal_logit_slope, target_fraction,
    adult_edge_correction, juvenile_edge_correction_stack,
    adult_fft_kernel, juvenile_fft_kernel_stack,
    Q_grid,
    eps=1e-6,
    adult_logit_offset=0.0,
    crowding_beta=None,
    crowding_kappa=5.0,
    kernel_weights=None,
    return_adult_arrivals=False,
    veit_departure=False,
    departure_density=None,
):
    """One dispersal step for both age classes, with density-dependent departure.

    Departure probability rises logistically with local density relative to K
    (``dispersal_logit_*``, ``target_fraction``); ``adult_logit_offset`` shifts the adult
    curve only (breeding vs natal dispersal; 0 = shared). ``crowding_beta`` (>= 0, None = off)
    adds the above-capacity arm ``crowding_beta * softplus(kappa*(N/K - 1)) / kappa``, so a
    cell over K sends a pulse even when the low-density slope is negative (see
    departure_logit). Adults use the single isotropic
    kernel; juveniles use the directional/radial stack with journey survival.
    Returns (adults after dispersal, juvenile stayers, juvenile arrivals).
    """
    N_a = jnp.maximum(N_a, 0.0)
    N_j = jnp.maximum(N_j, 0.0)
    N_total = N_a + N_j
    # population_model.occupancy: birds decide on the density where they actually are (the
    # occupied-conditional m), while the emigrant COUNTS stay expected values (N * p).
    N_dep = N_total if departure_density is None else departure_density

    K_safe = jnp.maximum(K, eps)
    z_total = departure_logit(N_dep / K_safe, dispersal_logit_intercept, dispersal_logit_slope,
                              target_fraction, crowding_beta, crowding_kappa)
    z_total = clip_safe(z_total)
    p_total = jnn.sigmoid(z_total)
    p_adult = jnn.sigmoid(clip_safe(z_total + adult_logit_offset))
    if veit_departure:
        # Veit & Lewis (1996) eq. 18: p(N) = N / N* below capacity, 1 at or above it, both ages --
        # sparse populations stay together (mates), dense ones disperse. Smoothed (k = 10) for
        # gradients; no parameters.
        x = N_dep / K_safe
        p_v = jnp.clip(x - jnn.softplus(10.0 * (x - 1.0)) / 10.0, 1e-6, 1.0)
        p_total = p_adult = p_v

    adult_dispersers = N_a * p_adult
    adult_stayers = N_a * (1.0 - p_adult)
    adult_boosted = adult_dispersers / (adult_edge_correction + eps)
    adult_arriving = rightpad_convolution(adult_boosted, adult_fft_kernel)
    
    N_a_post = adult_stayers + adult_arriving
    
    juvenile_dispersers = N_j * p_total
    juvenile_stayers = N_j * (1.0 - p_total)
    juvenile_arriving = juvenile_dispersal_vectorized(
        juvenile_dispersers,
        juvenile_fft_kernel_stack,
        Q_grid,
        juvenile_edge_correction_stack,
        eps,
        kernel_weights=kernel_weights,
    )

    if return_adult_arrivals:       # settlement.apply_to = dispersers needs arrivals apart
        return N_a_post, juvenile_stayers, juvenile_arriving, adult_arriving
    return N_a_post, juvenile_stayers, juvenile_arriving

def forward_sim_age_structured(
    Sa_flat, Sj_flat, Fmax_flat, K_flat, c_flat, Q_flat,
    land_rows, land_cols,           
    land_mask,
    adult_fft_kernel, juvenile_fft_kernel_stack,
    adult_edge_correction, juvenile_edge_correction_stack,
    initpop_latent, dispersal_random, inv_pop,
    time, inv_locations, inv_timestep,
    dispersal_logit_intercept, dispersal_logit_slope,
    allee_gamma,
    target_fraction=0.8,
    allee_exempt_years=0,
    allee_relative=False,
    settle_block=0,
    dispersal_adult_offset=0.0,
    crowding_beta=None,
    crowding_kappa=5.0,
    settle_weight="capacity",
    settle_tau=0.0,
    strata_u=None,
    strata_a=None,
    subcell_beta_k=0.0,
    subcell_delta_f=0.0,
    subcell_sigma_s=None,
    subcell_survival_affects_allocation=True,
    disease_m_amp=None,
    disease_k_half=1.0,
    disease_hill_n=1.0,
    juvenile_kernel_weights=None,
    settle_dispersers_only=False,
    veit_departure=False,
    yearling_breeding=False,
    veit_reproduction=False,
    occupancy_b=None,
    process_noise=None,
):
    """Run the age-structured simulation for ``time`` years.

    Seeds the population from ``initpop_latent`` (split 50/50 adult/juvenile),
    injects the invasion pulse ``inv_pop`` (shape ``(n_sites, n_years)``) at each of
    ``inv_locations`` (shape ``(n_sites, 2)``, row/col pairs) starting at
    ``inv_timestep``, and scans year by year: invasion → scatter per-cell rates
    to the grid → dispersal → survival+reproduction → land-mask. The step is
    ``jax.checkpoint``-wrapped to bound memory when differentiating the scan.

    Multiple candidate sites (rather than one fixed release cell) exist because the
    historical record only documents ONE known sighting (Jones Beach, Long Island) --
    not that it was the only release -- so the model is given several nearby
    candidate cells and a per-(site, year) pulse, letting the fit allocate mass
    across sites and across years rather than assuming both in advance. A JAX
    scatter-add (``.at[rows, cols].add(...)``) naturally handles any number of sites
    in one vectorized op, including the ``n_sites=1`` case this replaces.

    ``allee_exempt_years`` switches the mate-finding Allee factor OFF (to 1) in the
    candidate release cells for that many years from ``inv_timestep``. The Allee term
    describes continental colonization: dispersers arriving thinly into a 27 km cell.
    The 1940 release was the opposite -- a small group of cage birds freed together,
    aggregated far below grid scale, for whom finding a mate was not the constraint.
    Applying the cell-averaged Allee factor to them made the fit manufacture extra
    founders to clear the threshold, and concentrating those founders in one
    (site, year) cell was the cheapest way to do that: the probe of MAP run_17 found 78
    of the 89 founder-reallocation directions curving downhill (a saddle at the even
    spread, with corner-shaped modes). 0 = legacy behaviour (no exemption); the
    no-release counterfactual in map_diagnostics relies on that default.

    Returns ``(total_densities, Na_densities, Nj_densities)``, each shape
    (time, Ny, Nx). The age-split outputs cost nothing extra during MAP/SVI
    optimization -- XLA's dead-code elimination strips them whenever the
    caller only uses the total (as ``build_model_2d``'s likelihood does); they
    only get materialized when explicitly requested via
    ``Predictive(..., return_sites=[...])`` for post-hoc age-structure
    diagnostics (see scripts/viz/map_diagnostics.py).

    ``process_noise`` (optional, (time, N_land)): log multipliers eps applied to both age
    classes at the end of each year, N <- N * exp(eps) (population_model.process_noise, E070).
    Multiplicative, so it can only rescale birds that are already there -- it cannot seed
    empty cells, and spread stays the dispersal kernel's job.
    """
    Ny, Nx = land_mask.shape
    inv_locations = jnp.asarray(inv_locations)
    inv_rows, inv_cols = inv_locations[:, 0], inv_locations[:, 1]

    init_N_a = initpop_latent * 0.5
    init_N_j = initpop_latent * 0.5
    # population_model.occupancy (E063): psi = P(cell holds >= 1 bird) across realisations, with
    # birds Poisson at occupancy_b birds per density unit; seeded from the initial densities.
    occ = occupancy_b is not None
    init_psi = -jnp.expm1(-occupancy_b * (init_N_a + init_N_j)) if occ else jnp.zeros_like(init_N_a)
    
    # Pre-allocate zero grid for scattering to avoid repeated memory allocations
    zero_grid = jnp.zeros((Ny, Nx))
    release_mask = zero_grid.at[inv_rows, inv_cols].set(1.0)
    
    # Pre-allocate the Q-grid to avoid creating it inside the scan loop
    K_kernels = Q_flat.shape[-1]
    zero_Q_grid = jnp.zeros((K_kernels, Ny, Nx))

    # Added c_t to the scatter function
    def scatter_t(Sa_t, Sj_t, Fmax_t, K_t, c_t, Q_t):
        Sa_g = zero_grid.at[land_rows, land_cols].set(Sa_t)
        Sj_g = zero_grid.at[land_rows, land_cols].set(Sj_t)
        Fmax_g = zero_grid.at[land_rows, land_cols].set(Fmax_t)
        K_g = zero_grid.at[land_rows, land_cols].set(K_t)
        c_g = zero_grid.at[land_rows, land_cols].set(c_t) # <-- Scattered to 2D
        
        # Scatter directly into the final (K_kernels, Ny, Nx) shape
        Q_g = zero_Q_grid.at[:, land_rows, land_cols].set(Q_t.T)
    
        return Sa_g, Sj_g, Fmax_g, K_g, c_g, Q_g

    # The step function inside lax.scan
    def step(pools, t):
        N_a, N_j, psi = pools
        
        # 1. Invasion: one pulse value per site, this year, scattered to every
        # candidate site cell at once (a no-op add of 0 at every site outside the
        # window/inv_pop's year range).
        k = t - inv_timestep
        is_invading = (k >= 0) & (k < inv_pop.shape[1])
        val = jnp.where(is_invading, inv_pop[:, jnp.minimum(jnp.maximum(0, k), inv_pop.shape[1]-1)], 0.0)
        N_a = N_a.at[inv_rows, inv_cols].add(val * 0.5)
        N_j = N_j.at[inv_rows, inv_cols].add(val * 0.5)
        if occ:     # released birds arrive as a Poisson group like any other arrivals
            psi = psi.at[inv_rows, inv_cols].set(
                1.0 - (1.0 - psi[inv_rows, inv_cols]) * jnp.exp(-occupancy_b * val))

        # 2. Scatter Parameters (Updated to include c_flat index)
        Sa_g, Sj_g, Fmax_g, K_g, c_g, Q_g = scatter_t(
            Sa_flat[t], Sj_flat[t], Fmax_flat[t], K_flat[t], c_flat[t], Q_flat[t]
        )

        # 3. Dispersal
        _disp = dispersal_step_age_structured(
            N_a, N_j, K_g, 
            dispersal_logit_intercept + dispersal_random[t],
            dispersal_logit_slope, target_fraction,
            adult_edge_correction, juvenile_edge_correction_stack,
            adult_fft_kernel, juvenile_fft_kernel_stack,
            Q_grid=Q_g, eps=1e-6, adult_logit_offset=dispersal_adult_offset,
            crowding_beta=crowding_beta, crowding_kappa=crowding_kappa,
            kernel_weights=juvenile_kernel_weights,
            return_adult_arrivals=settle_dispersers_only or occ,
            veit_departure=veit_departure,
            departure_density=occupied_density(N_a + N_j, psi, occupancy_b)[1] if occ else None,
        )
        if settle_dispersers_only or occ:
            N_a_post, juvenile_stayers, juvenile_arriving, adult_arriving = _disp
        else:
            N_a_post, juvenile_stayers, juvenile_arriving = _disp
        
        # 3b. Ideal-free settlement (settle_block > 0): within each block of cells, the
        # post-dispersal population is rearranged in proportion to capacity. See
        # ideal_free_settle.
        # settle_weight: "capacity" (shares in proportion to K), "uniform" (land cells
        # equally -- habitat-blind mixing, a control), "fitness" (equal realized lambda;
        # see fitness_settle), "capacity_fitness" (shares in proportion to K * lambda0^tau:
        # IMPERFECT sorting on empty-cell fitness; tau = 0 is capacity).
        if settle_block:
            # settle_dispersers_only (settlement.apply_to = dispersers): only the birds that
            # moved this year -- adult and juvenile ARRIVALS -- are allocated within the block;
            # residents keep their cells (site fidelity), so local vital rates accumulate in
            # each cell's own abundance instead of being re-averaged away every year.
            if settle_dispersers_only:
                adult_stay = N_a_post - adult_arriving
                pools = (adult_arriving, jnp.zeros_like(juvenile_stayers), juvenile_arriving)
            else:
                pools = (N_a_post, juvenile_stayers, juvenile_arriving)
            if settle_weight == "fitness":
                pools = fitness_settle(pools, K_g, Sa_g, Sj_g, Fmax_g, c_g, settle_block, t)
            else:
                if settle_weight == "capacity":
                    wgt = K_g
                elif settle_weight == "capacity_fitness":
                    lam0 = 0.5 * (Sa_g + jnp.sqrt(Sa_g ** 2 + 4.0 * Fmax_g * Sa_g * Sj_g))
                    wgt = K_g * jnp.exp(settle_tau * jnp.log(jnp.maximum(lam0, 1e-6)))
                elif settle_weight in ("capacity_r0", "r0"):
                    # R0 = Fmax * Sj / (1 - Sa): recruits per recruit at low density, the simplest
                    # closed-form breeding-quality signal a settler could follow.
                    r0 = jnp.maximum(Fmax_g * Sj_g / jnp.maximum(1.0 - Sa_g, 1e-6), 1e-6)
                    if settle_weight == "r0":
                        wgt = jnp.where(K_g > 1e-12, r0, 0.0)
                    else:
                        wgt = K_g * jnp.exp(settle_tau * jnp.log(r0))
                else:
                    wgt = jnp.where(K_g > 1e-12, 1.0, 0.0)
                pools = ideal_free_settle(pools, wgt, settle_block, t)
            if settle_dispersers_only:
                N_a_post = adult_stay + pools[0]
                juvenile_arriving = pools[2]
            else:
                N_a_post, juvenile_stayers, juvenile_arriving = pools

        # 3c. Occupancy merge (E063). Arrivals land in every realisation; an empty one becomes occupied
        # if >= 1 bird arrives (Poisson, mean b * arrivals). Expected densities are untouched; the
        # local dynamics below act on the occupied-conditional densities m = N / psi.
        if occ:
            arrivals = adult_arriving + juvenile_arriving
            psi = psi + (1.0 - psi) * (-jnp.expm1(-occupancy_b * jnp.maximum(arrivals, 0.0)))
            psi, _ = occupied_density(N_a_post + juvenile_stayers + juvenile_arriving, psi, occupancy_b)
            ok = psi > _PSI_FLOOR
            inv_psi = jnp.where(ok, 1.0 / jnp.where(ok, psi, 1.0), 0.0)
            N_a_post, juvenile_stayers, juvenile_arriving = (
                N_a_post * inv_psi, juvenile_stayers * inv_psi, juvenile_arriving * inv_psi)

        # 4. Survival & Reproduction. Release cells skip the Allee factor while
        # exempt (see allee_exempt_years in the docstring).
        allee_off = jnp.where((k >= 0) & (k < allee_exempt_years), release_mask, 0.0)
        # Epizootic ADDED MORTALITY (disease_prior.target = survival): m = amplitude(cell, t)
        # * hill(current local density). Applied to the survival of adults, juvenile stayers
        # and arrivers alike; c (computed upstream from the disease-free rates) is left
        # alone, so endemic mortality LOWERS the equilibrium below K instead of moving K.
        if disease_m_amp is not None:
            amp_g = zero_grid.at[land_rows, land_cols].set(disease_m_amp[t])
            Nloc = N_a_post + juvenile_stayers + juvenile_arriving
            hn = jnp.power(jnp.maximum(Nloc, 0.0) + 1e-12, disease_hill_n)
            m_g = amp_g * hn / (hn + disease_k_half ** disease_hill_n)
            Sa_g = Sa_g * (1.0 - m_g)
            Sj_g = Sj_g * (1.0 - m_g)
            juvenile_arriving = juvenile_arriving * (1.0 - m_g)
        if strata_u is None:
            N_a_new, N_j_new = reproduction_age_structured(
                N_a_post, juvenile_stayers, juvenile_arriving,
                Sa_g, Sj_g, Fmax_g, K_g, c_g, allee_gamma, allee_off=allee_off,
                allee_relative=allee_relative, yearling_breeding=yearling_breeding,
                veit=veit_reproduction,
                mate_factor=(pairing_fraction(occupancy_b * (N_a_post + juvenile_stayers + juvenile_arriving))
                             if occ else None),
            )
        else:
            # within-cell urbanisation slices (population_model.subcell_strata)
            u_g = jnp.zeros((strata_u.shape[-1], Ny, Nx), strata_u.dtype).at[
                :, land_rows, land_cols].set(strata_u[t].T)
            def _subcell_effect_grid(value):
                """Scatter a time/cell interaction; preserve scalar/legacy-time inputs."""
                value = jnp.asarray(value)
                if value.ndim == 0:
                    return value
                if value.ndim == 1 and value.shape[0] == time:
                    return value[t]
                cell_value = value if value.ndim == 1 else value[t]
                return jnp.zeros((Ny, Nx), cell_value.dtype).at[
                    land_rows, land_cols].set(cell_value)
            beta_k_g = _subcell_effect_grid(subcell_beta_k)
            sigma_s_g = (None if subcell_sigma_s is None else
                         _subcell_effect_grid(subcell_sigma_s))
            N_a_new, N_j_new = subcell_reproduction(
                (N_a_post, juvenile_stayers, juvenile_arriving), Sa_g, Sj_g, Fmax_g, K_g,
                allee_gamma, u_g, strata_a, beta_k_g, subcell_delta_f,
                allee_off=allee_off, allee_relative=allee_relative,
                yearling_breeding=yearling_breeding, veit=veit_reproduction,
                sigma_s=sigma_s_g,
                survival_affects_allocation=subcell_survival_affects_allocation)
            
        # 5. Mask & Final Clip
        N_a_new = jnp.maximum(N_a_new * land_mask, 0.0)
        N_j_new = jnp.maximum(N_j_new * land_mask, 0.0)
        if process_noise is not None:
            mult = jnp.ones((Ny, Nx), N_a_new.dtype).at[land_rows, land_cols].set(jnp.exp(process_noise[t]))
            N_a_new, N_j_new = N_a_new * mult, N_j_new * mult
        if occ:
            # back to expected densities, then chance extinction: an occupied realisation holding
            # Poisson(b * m) birds is empty with probability exp(-b * m). Empty realisations hold no
            # birds, so expected density is unchanged -- only psi falls and m rises.
            N_a_new, N_j_new = N_a_new * psi, N_j_new * psi
            psi, m_tot = occupied_density(N_a_new + N_j_new, psi, occupancy_b)
            psi = psi * (-jnp.expm1(-occupancy_b * m_tot)) * land_mask
        N_total_new = N_a_new + N_j_new

        return (N_a_new, N_j_new, psi), (N_total_new, N_a_new, N_j_new, psi)

    checkpointed_step = jax.checkpoint(step)

    _, (total_densities, Na_densities, Nj_densities, psi_t) = lax.scan(
        checkpointed_step, (init_N_a, init_N_j, init_psi), jnp.arange(time)
    )

    if occ:
        return total_densities, Na_densities, Nj_densities, psi_t
    return total_densities, Na_densities, Nj_densities

def ideal_free_settle(fields, K, b, t):
    """Ideal-free settlement after dispersal: each b x b block's population (each age component
    separately) is redistributed in proportion to capacity K -- an 'oracle' local settlement step
    with no parameters beyond the block size.

    Why: every refined fit so far needs lambda0 ~ 2 and a strong Allee effect. Arrivals that settle
    where capacity is locally highest fill cells and clear mate-finding without either (hypotheses
    H1/H2). Mass is conserved exactly within every block; blocks with no capacity are left as is.
    The block grid alternates between offset 0 and b//2 on even/odd years so no block edge is
    permanent.

    GPU form: the fields are stacked and placed on a FIXED-size zero canvas at a traced offset
    (dynamic_update_slice), block-summed by reshape, expanded back by broadcast+reshape (no
    gathers), and cut out again (dynamic_slice) -- one blocking per year, static shapes for lax.scan.
    """
    Ny, Nx = K.shape
    H = -(-(Ny + b // 2) // b) * b
    W = -(-(Nx + b // 2) // b) * b
    o = (t % 2) * (b // 2)
    stack = jnp.stack([K, *fields])                                    # (1 + n, Ny, Nx)
    canvas = lax.dynamic_update_slice(
        jnp.zeros((stack.shape[0], H, W), stack.dtype), stack, (0, o, o))
    blocks = canvas.reshape(-1, H // b, b, W // b, b).sum(axis=(2, 4))  # (1 + n, H/b, W/b)
    full = jnp.broadcast_to(blocks[:, :, None, :, None],
                            (blocks.shape[0], H // b, b, W // b, b)).reshape(-1, H, W)
    wsum, fsum = full[0], full[1:]
    w = canvas[0]
    ok = wsum > 1e-12
    share = jnp.where(ok, w / jnp.where(ok, wsum, 1.0), 0.0)
    moved = jnp.where(ok[None], share[None] * fsum, canvas[1:])
    out = lax.dynamic_slice(moved, (0, o, o), (moved.shape[0], Ny, Nx))
    return tuple(out[i] for i in range(len(fields)))


def _to_blocks(canvas, b):
    """(n, H, W) -> (n, H/b, W/b, b*b): each block's cells along the last axis (reshape only)."""
    n, H, W = canvas.shape
    return canvas.reshape(n, H // b, b, W // b, b).transpose(0, 1, 3, 2, 4).reshape(n, H // b, W // b, b * b)


def _from_blocks(blocks, b):
    n, bh, bw, _ = blocks.shape
    return blocks.reshape(n, bh, bw, b, b).transpose(0, 1, 3, 2, 4).reshape(n, bh * b, bw * b)


def fitness_settle(fields, K, Sa, Sj, Fmax, c, b, t, n_iter=40):
    """Ideal-free settlement on REALIZED FITNESS: within each b x b block, birds spread until local
    lambda at the settled density is equal across occupied cells; cells worse than that even when
    empty receive none.

    lambda_i(N) is the dominant eigenvalue of [[Sa, Sj], [F_i(N) Sa, 0]] with the model's own
    Beverton-Holt fecundity F_i(N) = Fmax / (1 + c N / K) -- the Allee factor is deliberately left
    out: with it lambda is humped and the IFD has multiple equilibria, and mate-finding is what
    settling birds avoid by choosing sites (the departure curve's low-density arm covers it).
    Its inverse is closed form: F* = w (w - Sa) / (Sa Sj), N_i(w) = (K/c)(Fmax/F* - 1)_+.
    The block level w* with sum_i N_i(w*) = block total is found by bisection (no gradient) and
    one Newton step from it (carries the implicit-function gradient). Because c depends only on
    vital rates, equal vital rates within a block give N_i proportional to K_i: this NESTS the
    capacity-weighted ideal_free_settle. At w = 1 each viable cell holds exactly K (c is solved
    so lambda = 1 at K). Mass is conserved exactly (shares of the block total); blocks with no
    capacity or no birds are left as is. Same fixed canvas / alternating offset as
    ideal_free_settle; all elementwise, no gathers.
    """
    Ny, Nx = K.shape
    H = -(-(Ny + b // 2) // b) * b
    W = -(-(Nx + b // 2) // b) * b
    o = (t % 2) * (b // 2)
    stack = jnp.stack([K, Sa, Sj, Fmax, c, *fields])
    canvas = lax.dynamic_update_slice(jnp.zeros((stack.shape[0], H, W), stack.dtype), stack, (0, o, o))
    blk = _to_blocks(canvas, b)
    Kb, Sab, Sjb, Fb, cb, fb = blk[0], blk[1], blk[2], blk[3], blk[4], blk[5:]
    tiny = 1e-12
    valid = (Kb > tiny) & (Sab > tiny) & (Sjb > tiny) & (Fb > tiny)
    Sa_s = jnp.where(valid, Sab, 0.5)
    Sj_s = jnp.where(valid, Sjb, 0.5)
    F_s = jnp.where(valid, Fb, 1.0)
    K_s = jnp.where(valid, Kb, 0.0)
    c_s = jnp.maximum(jnp.where(valid, cb, 1.0), 1e-6)
    lam0 = 0.5 * (Sa_s + jnp.sqrt(Sa_s ** 2 + 4.0 * F_s * Sa_s * Sj_s))
    M = fb.sum(axis=(0, -1))                                           # (bh, bw) block totals

    def density(w):                                                    # w (bh, bw) -> (bh, bw, b*b)
        ww = w[..., None]
        Fstar = jnp.maximum(ww * (ww - Sa_s) / (Sa_s * Sj_s), tiny)
        n = (K_s / c_s) * (F_s / Fstar - 1.0)
        if SMOOTH_ALLOC is not None:
            return jnp.where(valid & (ww > Sa_s), _positive_part(n, K_s / c_s), 0.0)
        return jnp.where(valid & (ww < lam0) & (ww > Sa_s), jnp.maximum(n, 0.0), 0.0)

    ok = valid.any(-1) & (M > tiny)
    lo0 = jnp.where(ok, jnp.max(jnp.where(valid, Sa_s, -jnp.inf), -1), 0.0)
    hi0 = jnp.where(ok, jnp.max(jnp.where(valid, lam0, -jnp.inf), -1), 1.0)
    Mg = lax.stop_gradient(M)

    def bisect(_, lh):
        lo, hi = lh
        mid = 0.5 * (lo + hi)
        over = density(mid).sum(-1) > Mg                               # too dense -> raise w
        return jnp.where(over, mid, lo), jnp.where(over, hi, mid)

    lo, hi = lax.fori_loop(0, n_iter, bisect, (lax.stop_gradient(lo0), lax.stop_gradient(hi0)))
    w0 = lax.stop_gradient(0.5 * (lo + hi))
    S0, dS = jax.jvp(lambda w: density(w).sum(-1), (w0,), (jnp.ones_like(w0),))
    step_ok = ok & (dS < -tiny)
    w = jnp.where(step_ok, w0 - (S0 - M) / jnp.where(step_ok, dS, -1.0), w0)
    n = density(w)
    tot = n.sum(-1, keepdims=True)
    good = ok[..., None] & (tot > tiny)
    share = jnp.where(good, n / jnp.where(good, tot, 1.0), 0.0)
    moved = jnp.where(good[None], share[None] * fb.sum(-1, keepdims=True), fb)
    out = lax.dynamic_slice(_from_blocks(moved, b), (0, o, o), (len(fields), Ny, Nx))
    return tuple(out[i] for i in range(len(fields)))


def subcell_reproduction(pools, Sa, Sj, Fmax, K, allee_gamma, u, a, beta_k, delta_f,
                         allee_off=None, allee_relative=False, n_iter=40,
                         yearling_breeding=False, veit=False, sigma_s=None,
                         survival_affects_allocation=True):
    """Survival + reproduction with the cell split into S within-cell urbanisation slices.

    The cell's birds are first redistributed among its slices by the fitness ideal-free rule
    (as fitness_settle, Allee excluded), then mate-finding and Beverton-Holt act on each slice's
    LOCAL density, and the slices are summed back to the cell. DESK's cell-level K and Fmax are
    preserved exactly: slice s (area a_s, standardised log built-up intensity u_s) gets the
    MEAN-PRESERVING tilts K_s = K e^{beta_k u_s} / sum_r a_r e^{beta_k u_r}, same for Fmax with
    delta_f, so sum_s a_s K_s = K. beta_k = delta_f = 0 (or u constant, e.g. cells without BUI
    coverage) gives every slice the cell's own rates, local density = cell density, and exactly
    the one-slice model. With delta_f > 0 the towns have the higher empty-cell lambda, so a
    sparse cell's birds concentrate there (high local density -> no Allee cost) and the
    countryside fills only once the towns crowd.

    ``sigma_s`` (scalar, this year's value; None = off): the TOWN SURVIVAL tilt -- adult and juvenile
    survival in slice s shift on the logit scale by sigma_s (u_s - sum_r a_r u_r), so towns can be where
    birds survive the winter (feeders) while the cell's area-mean survival stays ~DESK's.

    ``pools``: (N_a_post, juvenile_stayers, juvenile_arrivers) cell densities (Ny, Nx);
    ``u``: (S, Ny, Nx); ``a``: (S,). Returns (N_a_new, N_j_new) cell densities.
    """
    a = a[:, None, None]
    tiny = 1e-12

    def tilt(coef):
        e = jnp.exp(coef * u)
        return e / jnp.sum(a * e, axis=0, keepdims=True)

    K_s = K[None] * tilt(beta_k)
    F_s = Fmax[None] * tilt(delta_f)
    Sa_base = jnp.broadcast_to(Sa, K_s.shape)
    Sj_base = jnp.broadcast_to(Sj, K_s.shape)
    Sa_, Sj_ = Sa_base, Sj_base
    if sigma_s is not None:
        du = sigma_s * (u - jnp.sum(a * u, axis=0, keepdims=True))
        logit = lambda p: jnp.log(jnp.clip(p, 1e-6, 1 - 1e-6)) - jnp.log1p(-jnp.clip(p, 1e-6, 1 - 1e-6))
        Sa_ = jnn.sigmoid(logit(Sa_) + du)
        Sj_ = jnn.sigmoid(logit(Sj_) + du)
    c_s = jnp.maximum((F_s * Sa_ * Sj_) / (1.0 - Sa_ + 1e-6) - 1.0, 0.0)
    # Feeding/shelter can change survival without implying that birds know and
    # instantaneously ideal-free-sort on that survival difference.  Keeping the
    # allocation rates at their cell-level values isolates the physiological
    # mechanism and removes sigma_s from the implicit equal-fitness root.  E064's
    # survival-sorted version produced NaNs or gradients of 1e9--1e27; the no-sort
    # option is deliberately opt-in so historical models retain their semantics.
    Sa_alloc = Sa_ if survival_affects_allocation else Sa_base
    Sj_alloc = Sj_ if survival_affects_allocation else Sj_base
    c_alloc = jnp.maximum((F_s * Sa_alloc * Sj_alloc) /
                          (1.0 - Sa_alloc + 1e-6) - 1.0, 0.0)
    c_f = jnp.maximum(c_alloc, 1e-6)
    valid = (K_s > tiny) & (Sa_alloc > tiny) & (Sj_alloc > tiny) & (F_s > tiny)
    Sa_v = jnp.where(valid, Sa_alloc, 0.5)
    Sj_v = jnp.where(valid, Sj_alloc, 0.5)
    F_v = jnp.where(valid, F_s, 1.0)
    lam0 = 0.5 * (Sa_v + jnp.sqrt(Sa_v ** 2 + 4.0 * F_v * Sa_v * Sj_v))
    M = sum(pools)                                                     # (Ny, Nx) cell density

    def alloc(w):                                                      # cell-unit share per slice
        ww = w[None]
        Fstar = jnp.maximum(ww * (ww - Sa_v) / (Sa_v * Sj_v), tiny)
        n = a * (K_s / c_f) * (F_v / Fstar - 1.0)
        if SMOOTH_ALLOC is not None:
            return jnp.where(valid & (ww > Sa_v), _positive_part(n, a * K_s / c_f), 0.0)
        return jnp.where(valid & (ww < lam0) & (ww > Sa_v), jnp.maximum(n, 0.0), 0.0)

    ok = valid.any(0) & (M > tiny)
    lo0 = jnp.where(ok, jnp.max(jnp.where(valid, Sa_v, -jnp.inf), 0), 0.0)
    hi0 = jnp.where(ok, jnp.max(jnp.where(valid, lam0, -jnp.inf), 0), 1.0)
    Mg = lax.stop_gradient(M)

    def bisect(_, lh):
        lo, hi = lh
        mid = 0.5 * (lo + hi)
        over = alloc(mid).sum(0) > Mg
        return jnp.where(over, mid, lo), jnp.where(over, hi, mid)

    lo, hi = lax.fori_loop(0, n_iter, bisect, (lax.stop_gradient(lo0), lax.stop_gradient(hi0)))
    w0 = lax.stop_gradient(0.5 * (lo + hi))
    S0, dS = jax.jvp(lambda w: alloc(w).sum(0), (w0,), (jnp.ones_like(w0),))
    step_ok = ok & (dS < -tiny)
    w = jnp.where(step_ok, w0 - (S0 - M) / jnp.where(step_ok, dS, -1.0), w0)
    n = alloc(w)
    tot = n.sum(0, keepdims=True)
    good = ok[None] & (tot > tiny)
    # fallback (no birds / no capacity): uniform local density across slices
    share = jnp.where(good, n / jnp.where(good, tot, 1.0), a)
    local = [p[None] * share / a for p in pools]                       # slice LOCAL densities
    Na_s, Nj_s = reproduction_age_structured(
        local[0], local[1], local[2], Sa_, Sj_, F_s, K_s, c_s, allee_gamma,
        allee_off=allee_off, allee_relative=allee_relative,
        yearling_breeding=yearling_breeding, veit=veit)
    return jnp.sum(a * Na_s, axis=0), jnp.sum(a * Nj_s, axis=0)


# Below this capacity (density units; ~2e-7 route counts) N/K is not computed.
_RELATIVE_K_FLOOR = 1e-8
# exp(-gamma * ratio) is exactly 0 long before this for any plausible gamma.
_RELATIVE_RATIO_CAP = 1e4


def relative_density(N, K):
    """N/K for the relative Allee factor, finite in value AND gradient everywhere.

    Run_22's first MAP died inside the forward scan (TACC 3511184/3511695): FFT round-off
    left N ~ -1e-9 before the post-reproduction clamp, and N / max(K, 1e-12) turned it into
    a huge negative ratio, so exp(-gamma*ratio) overflowed. Cells where K underflows to ~0
    also give ratios whose gradient carries 1/K^2, and 0/0 in empty cells. The double-where
    keeps the division off those cells entirely: no capacity with birds present means
    mates are not limiting (ratio at the cap, factor 1); no birds means ratio 0. The cap
    leaves the factor unchanged (exactly 1) while bounding the argument.
    """
    ok = K > _RELATIVE_K_FLOOR
    ratio = N / jnp.where(ok, K, 1.0)
    ratio = jnp.where(ok, ratio, jnp.where(N > 0, _RELATIVE_RATIO_CAP, 0.0))
    # The FFT dispersal step leaves round-off densities ~ -1e-9 that are clamped only AFTER
    # reproduction. Under the absolute form exp(+gamma*1e-9) ~ 1 is harmless; divided by a
    # small K they became a large NEGATIVE ratio and exp(-gamma*ratio) overflowed to inf --
    # the exact failure JAX_DEBUG_INFS located in run_22 (TACC 3511695). Hence [0, cap].
    return jnp.clip(ratio, 0.0, _RELATIVE_RATIO_CAP)


def reproduction_age_structured(
    N_a_post, N_j_stayers, N_j_arrivers,
    S_a, S_j, F_max, K, c, allee_gamma, eps=1e-12, allee_off=None, allee_relative=False,
    yearling_breeding=False, veit=False, mate_factor=None,
    ):
    """Advance one year of survival + reproduction to next-year (N_a, N_j).

    Fecundity is Beverton-Holt density-dependent (``c`` precomputed from the
    local Leslie matrix) and multiplied by a mate-finding Allee factor
    ``1 - exp(-allee_gamma * N)``. The local linearized matrix is
    ``[[S_a, S_j], [F*S_a, 0]]``: adults are surviving adults plus surviving
    juveniles, and the surviving adults produce the next juvenile cohort.

    ``allee_off`` (optional, broadcastable to N, values in [0, 1]) removes that share
    of the Allee limitation: 1 = mate-finding never limits (the release cells during
    their exemption window), 0 = the normal Allee factor.

    ``allee_relative`` replaces N by N/K in the Allee factor (mate-finding relative to
    capacity; see age_priors' allee_prior.form).
    """
    N_total_post = N_a_post + N_j_stayers + N_j_arrivers
    K_safe = jnp.maximum(K, eps)

    F_eff = F_max / (1.0 + c * (N_total_post / K_safe))
    # allee_relative: mate-finding by N/K (age_priors' allee_prior.form = relative).
    allee_factor = 1.0 - jnp.exp(-allee_gamma * (relative_density(N_total_post, K)
                                                 if allee_relative else N_total_post))
    if allee_off is not None:
        allee_factor = allee_factor + allee_off * (1.0 - allee_factor)
    F_actual = F_eff * allee_factor
    if mate_factor is not None:      # occupancy: pairing stochasticity at the scale of birds (E063)
        F_actual = F_actual * mate_factor

    surviving_adults = N_a_post * S_a
    surviving_stayers = N_j_stayers * S_j      
    surviving_arrivers = N_j_arrivers          

    N_a_new = surviving_adults + surviving_stayers + surviving_arrivers
    if veit:
        # reproduction.form = veit (Veit & Lewis 1996 eqs. 5-8): ALL winter survivors breed
        # (yearlings included). Breeders B form pairs by random search, P = B^2/(r + 2B) with
        # r = 2*n50 (half of breeders mated at B = n50); pairs compete for nest sites,
        # H = P/(1 + P/delta) with delta = K/2 site-pairs (K is the habitat's site capacity in
        # breeder units, NOT the population equilibrium -- that is emergent); births = 2*Fmax*H,
        # so Fmax is per-breeder output when neither mates nor sites limit. The release
        # exemption sets the mated fraction to 1 (P = B/2).
        B = N_a_new
        r = 2.0 * jnp.log(2.0) / jnp.maximum(allee_gamma, eps)
        if allee_relative:          # n50 declared in N/K units
            r = r * K
        P = B * B / (r + 2.0 * B + eps)
        if allee_off is not None:
            P = P + allee_off * (0.5 * B - P)
        H = P / (1.0 + 2.0 * P / jnp.maximum(K, eps))
        N_j_new = 2.0 * F_max * H * (1.0 if mate_factor is None else mate_factor)
    elif yearling_breeding:
        # reproduction.yearling_breeding: yearlings breed in their first summer (House Finch),
        # so births come from ALL winter survivors; lambda = Sa + Sj*F (c re-solved upstream).
        N_j_new = N_a_new * F_actual
    else:
        N_j_new = surviving_adults * F_actual

    return N_a_new, N_j_new
