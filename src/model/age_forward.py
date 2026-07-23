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


def rightpad_convolution(pop, dispersal_kernel_pad):
    """Convolve a population grid with a (pre-FFT'd, padded) dispersal kernel.

    Right-pads ``pop`` to the kernel's padded size so the circular FFT product
    is a *linear* convolution, then crops back to the original grid shape.
    """
    Ly, Lx = dispersal_kernel_pad.shape
    pop_pad = rightpad(pop, Lx, Ly, 1e-9)
    conv = jnp.fft.ifft2(jnp.fft.fft2(pop_pad) * dispersal_kernel_pad)
    return jnp.real(conv)[:pop.shape[0], :pop.shape[1]]

def juvenile_dispersal_vectorized(
    juvenile_dispersers: jnp.ndarray,       
    juvenile_fft_kernels: jnp.ndarray,      
    Q: jnp.ndarray,                         
    juvenile_edge_correction_stack: jnp.ndarray,
    eps: float = 1e-6
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
    return jnp.sum(successful_settlers, axis=0)

def dispersal_step_age_structured(
    N_a, N_j, K, 
    dispersal_logit_intercept, dispersal_logit_slope, target_fraction,
    adult_edge_correction, juvenile_edge_correction_stack,
    adult_fft_kernel, juvenile_fft_kernel_stack,
    Q_grid,
    eps=1e-6
):
    """One dispersal step for both age classes, with density-dependent departure.

    Departure probability rises logistically with local density relative to K
    (``dispersal_logit_*``, ``target_fraction``). Adults use the single isotropic
    kernel; juveniles use the directional/radial stack with journey survival.
    Returns (adults after dispersal, juvenile stayers, juvenile arrivals).
    """
    N_a = jnp.maximum(N_a, 0.0)
    N_j = jnp.maximum(N_j, 0.0)
    N_total = N_a + N_j
    
    K_safe = jnp.maximum(K, eps)
    z_total = dispersal_logit_intercept + dispersal_logit_slope * (N_total / K_safe - target_fraction)
    z_total = clip_safe(z_total) 
    p_total = jnn.sigmoid(z_total)
    
    adult_dispersers = N_a * p_total
    adult_stayers = N_a * (1.0 - p_total)
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
        eps
    )

    return N_a_post, juvenile_stayers, juvenile_arriving

def forward_sim_age_structured(
    Sa_flat, Sj_flat, Fmax_flat, K_flat, c_flat, Q_flat,
    land_rows, land_cols,           
    land_mask,
    adult_fft_kernel, juvenile_fft_kernel_stack,
    adult_edge_correction, juvenile_edge_correction_stack,
    initpop_latent, dispersal_random, inv_pop,
    time, inv_location, inv_timestep,
    dispersal_logit_intercept, dispersal_logit_slope,
    allee_gamma,
    target_fraction=0.8
):
    """Run the age-structured simulation for ``time`` years; return total density.

    Seeds the population from ``initpop_latent`` (split 50/50 adult/juvenile),
    injects the invasion pulse ``inv_pop`` at ``inv_location`` starting at
    ``inv_timestep``, and scans year by year: invasion → scatter per-cell rates
    to the grid → dispersal → survival+reproduction → land-mask. The step is
    ``jax.checkpoint``-wrapped to bound memory when differentiating the scan.
    Output shape: (time, Ny, Nx).
    """
    Ny, Nx = land_mask.shape
    row, col = inv_location
    
    init_N_a = initpop_latent * 0.5
    init_N_j = initpop_latent * 0.5
    
    # Pre-allocate zero grid for scattering to avoid repeated memory allocations
    zero_grid = jnp.zeros((Ny, Nx))
    
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
        N_a, N_j = pools
        
        # 1. Invasion
        k = t - inv_timestep
        is_invading = (k >= 0) & (k < inv_pop.shape[0])
        val = jnp.where(is_invading, inv_pop[jnp.minimum(jnp.maximum(0, k), inv_pop.shape[0]-1)], 0.0)
        N_a = N_a.at[row, col].add(val * 0.5)
        N_j = N_j.at[row, col].add(val * 0.5)

        # 2. Scatter Parameters (Updated to include c_flat index)
        Sa_g, Sj_g, Fmax_g, K_g, c_g, Q_g = scatter_t(
            Sa_flat[t], Sj_flat[t], Fmax_flat[t], K_flat[t], c_flat[t], Q_flat[t]
        )

        # 3. Dispersal
        N_a_post, juvenile_stayers, juvenile_arriving = dispersal_step_age_structured(
            N_a, N_j, K_g, 
            dispersal_logit_intercept + dispersal_random[t],
            dispersal_logit_slope, target_fraction,
            adult_edge_correction, juvenile_edge_correction_stack,
            adult_fft_kernel, juvenile_fft_kernel_stack,
            Q_grid=Q_g, eps=1e-6
        )
        
        # 4. Survival & Reproduction (Updated reproduction call)
        N_a_new, N_j_new = reproduction_age_structured(
            N_a_post, juvenile_stayers, juvenile_arriving,
            Sa_g, Sj_g, Fmax_g, K_g, c_g, allee_gamma
        )
            
        # 5. Mask & Final Clip
        N_a_new = jnp.maximum(N_a_new * land_mask, 0.0)
        N_j_new = jnp.maximum(N_j_new * land_mask, 0.0)
        N_total_new = N_a_new + N_j_new
        
        return (N_a_new, N_j_new), N_total_new

    checkpointed_step = jax.checkpoint(step)

    _, total_densities = lax.scan(checkpointed_step, (init_N_a, init_N_j), jnp.arange(time))
    
    return total_densities

def reproduction_age_structured(
    N_a_post, N_j_stayers, N_j_arrivers,
    S_a, S_j, F_max, K, c, allee_gamma, eps=1e-12 # <-- Accept pre-computed c
    ):
    """Advance one year of survival + reproduction to next-year (N_a, N_j).

    Fecundity is Beverton-Holt density-dependent (``c`` precomputed from the
    local Leslie matrix) and multiplied by a mate-finding Allee factor
    ``1 - exp(-allee_gamma * N)``. The local linearized matrix is
    ``[[S_a, S_j], [F*S_a, 0]]``: adults are surviving adults plus surviving
    juveniles, and the surviving adults produce the next juvenile cohort.
    """
    N_total_post = N_a_post + N_j_stayers + N_j_arrivers
    K_safe = jnp.maximum(K, eps)

    # Completely removed the inline 'c = (F_max * S_j) / ...' logic block
    F_eff = F_max / (1.0 + c * (N_total_post / K_safe))
    allee_factor = 1.0 - jnp.exp(-allee_gamma * N_total_post)
    F_actual = F_eff * allee_factor

    surviving_adults = N_a_post * S_a
    surviving_stayers = N_j_stayers * S_j      
    surviving_arrivers = N_j_arrivers          

    N_a_new = surviving_adults + surviving_stayers + surviving_arrivers
    N_j_new = surviving_adults * F_actual

    return N_a_new, N_j_new
