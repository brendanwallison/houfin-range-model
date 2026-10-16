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
    
    K_safe = jnp.maximum(K, eps)
    z_total = departure_logit(N_total / K_safe, dispersal_logit_intercept, dispersal_logit_slope,
                              target_fraction, crowding_beta, crowding_kappa)
    z_total = clip_safe(z_total)
    p_total = jnn.sigmoid(z_total)
    p_adult = jnn.sigmoid(clip_safe(z_total + adult_logit_offset))

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
    """
    Ny, Nx = land_mask.shape
    inv_locations = jnp.asarray(inv_locations)
    inv_rows, inv_cols = inv_locations[:, 0], inv_locations[:, 1]

    init_N_a = initpop_latent * 0.5
    init_N_j = initpop_latent * 0.5
    
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
        N_a, N_j = pools
        
        # 1. Invasion: one pulse value per site, this year, scattered to every
        # candidate site cell at once (a no-op add of 0 at every site outside the
        # window/inv_pop's year range).
        k = t - inv_timestep
        is_invading = (k >= 0) & (k < inv_pop.shape[1])
        val = jnp.where(is_invading, inv_pop[:, jnp.minimum(jnp.maximum(0, k), inv_pop.shape[1]-1)], 0.0)
        N_a = N_a.at[inv_rows, inv_cols].add(val * 0.5)
        N_j = N_j.at[inv_rows, inv_cols].add(val * 0.5)

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
            Q_grid=Q_g, eps=1e-6, adult_logit_offset=dispersal_adult_offset,
            crowding_beta=crowding_beta, crowding_kappa=crowding_kappa,
        )
        
        # 3b. Ideal-free settlement (settle_block > 0): within each block of cells, the
        # post-dispersal population is rearranged in proportion to capacity. See
        # ideal_free_settle.
        # settle_weight: "capacity" (shares in proportion to K), "uniform" (land cells
        # equally -- habitat-blind mixing, a control), "fitness" (equal realized lambda;
        # see fitness_settle), "capacity_fitness" (shares in proportion to K * lambda0^tau:
        # IMPERFECT sorting on empty-cell fitness; tau = 0 is capacity).
        if settle_block:
            pools = (N_a_post, juvenile_stayers, juvenile_arriving)
            if settle_weight == "fitness":
                pools = fitness_settle(pools, K_g, Sa_g, Sj_g, Fmax_g, c_g, settle_block, t)
            else:
                if settle_weight == "capacity":
                    wgt = K_g
                elif settle_weight == "capacity_fitness":
                    lam0 = 0.5 * (Sa_g + jnp.sqrt(Sa_g ** 2 + 4.0 * Fmax_g * Sa_g * Sj_g))
                    wgt = K_g * jnp.exp(settle_tau * jnp.log(jnp.maximum(lam0, 1e-6)))
                else:
                    wgt = jnp.where(K_g > 1e-12, 1.0, 0.0)
                pools = ideal_free_settle(pools, wgt, settle_block, t)
            N_a_post, juvenile_stayers, juvenile_arriving = pools

        # 4. Survival & Reproduction. Release cells skip the Allee factor while
        # exempt (see allee_exempt_years in the docstring).
        allee_off = jnp.where((k >= 0) & (k < allee_exempt_years), release_mask, 0.0)
        if strata_u is None:
            N_a_new, N_j_new = reproduction_age_structured(
                N_a_post, juvenile_stayers, juvenile_arriving,
                Sa_g, Sj_g, Fmax_g, K_g, c_g, allee_gamma, allee_off=allee_off,
                allee_relative=allee_relative,
            )
        else:
            # within-cell urbanisation slices (population_model.subcell_strata)
            u_g = jnp.zeros((strata_u.shape[-1], Ny, Nx), strata_u.dtype).at[
                :, land_rows, land_cols].set(strata_u[t].T)
            N_a_new, N_j_new = subcell_reproduction(
                (N_a_post, juvenile_stayers, juvenile_arriving), Sa_g, Sj_g, Fmax_g, K_g,
                allee_gamma, u_g, strata_a, subcell_beta_k, subcell_delta_f,
                allee_off=allee_off, allee_relative=allee_relative)
            
        # 5. Mask & Final Clip
        N_a_new = jnp.maximum(N_a_new * land_mask, 0.0)
        N_j_new = jnp.maximum(N_j_new * land_mask, 0.0)
        N_total_new = N_a_new + N_j_new

        return (N_a_new, N_j_new), (N_total_new, N_a_new, N_j_new)

    checkpointed_step = jax.checkpoint(step)

    _, (total_densities, Na_densities, Nj_densities) = lax.scan(
        checkpointed_step, (init_N_a, init_N_j), jnp.arange(time)
    )

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
                         allee_off=None, allee_relative=False, n_iter=40):
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
    Sa_ = jnp.broadcast_to(Sa, K_s.shape)
    Sj_ = jnp.broadcast_to(Sj, K_s.shape)
    c_s = jnp.maximum((F_s * Sa_ * Sj_) / (1.0 - Sa_ + 1e-6) - 1.0, 0.0)
    c_f = jnp.maximum(c_s, 1e-6)
    valid = (K_s > tiny) & (Sa_ > tiny) & (Sj_ > tiny) & (F_s > tiny)
    Sa_v = jnp.where(valid, Sa_, 0.5)
    Sj_v = jnp.where(valid, Sj_, 0.5)
    F_v = jnp.where(valid, F_s, 1.0)
    lam0 = 0.5 * (Sa_v + jnp.sqrt(Sa_v ** 2 + 4.0 * F_v * Sa_v * Sj_v))
    M = sum(pools)                                                     # (Ny, Nx) cell density

    def alloc(w):                                                      # cell-unit share per slice
        ww = w[None]
        Fstar = jnp.maximum(ww * (ww - Sa_v) / (Sa_v * Sj_v), tiny)
        n = a * (K_s / c_f) * (F_v / Fstar - 1.0)
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
        allee_off=allee_off, allee_relative=allee_relative)
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
    S_a, S_j, F_max, K, c, allee_gamma, eps=1e-12, allee_off=None, allee_relative=False
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

    surviving_adults = N_a_post * S_a
    surviving_stayers = N_j_stayers * S_j      
    surviving_arrivers = N_j_arrivers          

    N_a_new = surviving_adults + surviving_stayers + surviving_arrivers
    N_j_new = surviving_adults * F_actual

    return N_a_new, N_j_new
