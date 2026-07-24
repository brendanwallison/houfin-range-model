"""Map the community-encoder latent Z to per-cell demographic rate fields.

Each year, the latent vector Z (and its path-integrated form Z_disp) is
projected through learned weights into two habitat manifolds — survival H_s
and reproduction H_r — then passed through link functions to per-cell
adult/juvenile survival (S_a, S_j), max fecundity (F_max), carrying capacity
(K), and journey survival (Q). Runs as a checkpointed ``lax.scan`` over years
to bound memory when differentiated.

K additionally receives a spatiotemporal multiplicative correction (see step 4b
in ``process_year`` below) meant to soak up latent, unmodeled dynamics this
Z-driven covariate structure can't see -- the motivating case is mycoplasmal
conjunctivitis, whose spread has no covariate of its own in this model. It is
NOT a smoothing term on Z/H_s/H_r (an earlier design added a shared spatial
random effect to both manifolds; that coupled the correction to survival and
reproduction alike and gave it no principled way to only capture something
disease-shaped). Restricting it to K alone, uncoupled from Sa/Sj/Fmax, keeps
the fundamental-niche quantities (see age_model_math.local_growth_lambda)
exactly covariate-driven.
"""
import math

import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, checkpoint

# softplus(x) = 1 <=> x = log(e - 1). Calibrating the K-correction's offset to
# this value gives a multiplier with EXACT median 1 (softplus is monotonic, so
# median is preserved under it) at k_smooth = 0 -- i.e. "no correction" is the
# natural center of the prior, not an edge case the base K has to compensate
# for. softplus is used (not exp) so that a loosened prior's tail draws scale
# K up only LINEARLY for large positive corrections (softplus(x) ~ x), rather
# than exploding exponentially; large negative draws still decay K toward zero
# either way (softplus(x) -> exp(x) as x -> -inf).
_K_CORRECTION_OFFSET = math.log(math.e - 1.0)


def project_and_scatter_age_structured(
    time, Ny, Nx,
    land_rows, land_cols,
    Z_gathered, Z_disp_gathered,
    st_basis, st_weights, inv_timestep,
    beta_s,           # 1D feature weights for Survival Suitability (Shape: M)
    beta_r,           # 1D feature weights for Reproductive Suitability (Shape: M)
    alpha_a, gamma_a, # Adult survival intercept & slope
    alpha_j, gamma_j, # Juvenile survival intercept & slope
    alpha_f, gamma_f, # Max fecundity intercept & slope
    alpha_k, gamma_k  # Carrying capacity intercept & slope
):
    """Project Z → (S_a, S_j, F_max, K, Q) for every year, on the land cells.

    Survival/journey rates use a sigmoid link on the survival manifold H_s;
    fecundity and capacity use softplus on the reproduction manifold H_r. Q (in-
    cohort survival) reuses the juvenile survival intercept/slope on the
    land-conditioned neighborhood/path habitat H_s_disp. Each returned array is
    (time, N_land[, K]).

    ``st_basis`` covers only the post-invasion window (its time axis has
    ``time - inv_timestep`` entries, not ``time`` -- see
    ``model_inputs.generate_spatiotemporal_basis``'s caller): there is nothing
    for a latent disease-shaped correction to explain before the species (and
    any disease it might carry) has arrived. Timesteps before ``inv_timestep``
    get K_val unmodified (multiplier exactly 1.0); this also avoids ever
    indexing ``st_basis`` out of bounds.
    """
    # Checkpoint: don't store this function's large intermediates for the
    # backward pass; recompute them instead.
    @checkpoint
    def process_year(carry, t_idx):
        # 1. Pull slices from CPU RAM -> GPU VRAM
        z_t = jnp.take(Z_gathered, t_idx, axis=0)
        z_disp_t = jnp.take(Z_disp_gathered, t_idx, axis=0)

        # 2. Compute the 2D Correlated Habitat Manifolds (H_s and H_r) --
        # purely covariate-driven (Z.beta), no spatiotemporal term mixed in.
        H_s_local = jnp.dot(z_t, beta_s)
        H_r_local = jnp.dot(z_t, beta_r)

        # 3. Path-Integrated Survival Suitability
        # z_disp_t is (N_land, K_kernels, M) -> dot with beta_s (M,) gives (N_land, K_kernels)
        # NOTE (Ružička contract): the "Z.Z^T ~= uncentered Ružička kernel + isotropic prior
        # => GP with the Ružička kernel" identity holds EXACTLY only for the LOCAL block
        # (H_s_local/H_r_local, raw Z). z_disp = A.Z is a land-normalized spatial convolution
        # of Z, so z_disp.z_disp^T ~= A.K_ružicka.A^T -- a spatially-SMOOTHED kernel, not
        # Ružička itself. Reusing beta_s here is deliberate (journey survival is tied to
        # juvenile LOCAL survival, step 5 below), but the exact GP-kernel interpretation is
        # only approximate on the dispersal block. See kernel_contract note in model_inputs.py
        # and the "dispersal-block prior" future-work item.
        H_s_disp = jnp.dot(z_disp_t, beta_s)

        # 4. Map H_s and H_r to Demographic Rates using Intercepts and Slopes
        # Survival listens to H_s
        S_a_val = jnn.sigmoid(alpha_a + gamma_a * H_s_local)
        S_j_val = jnn.sigmoid(alpha_j + gamma_j * H_s_local)

        # Reproduction listens to H_r
        F_max_val = jnn.softplus(alpha_f + gamma_f * H_r_local)
        K_base_val = jnn.softplus(alpha_k + gamma_k * H_r_local)

        # 4b. Latent spatiotemporal correction to K only (see module docstring
        # and _K_CORRECTION_OFFSET above). Clip the basis index so it's always
        # in-bounds; the clipped value is discarded via jnp.where whenever
        # t_idx < inv_timestep (pre-invasion: multiplier forced to 1.0).
        basis_len = st_basis.shape[1]
        basis_idx = jnp.clip(t_idx - inv_timestep, 0, basis_len - 1)
        st_basis_t = jnp.take(st_basis, basis_idx, axis=1)
        k_smooth = jnp.dot(st_basis_t.T, st_weights)
        k_multiplier = jnn.softplus(_K_CORRECTION_OFFSET + k_smooth)
        k_multiplier = jnp.where(t_idx >= inv_timestep, k_multiplier, 1.0)
        K_val = K_base_val * k_multiplier

        # 5. Map Path Habitat (H_s) to Journey Survival (Q) using juvenile rules
        # This perfectly links movement mortality to local survival mortality
        Q_val = jnn.sigmoid(alpha_j + gamma_j * H_s_disp)

        return None, (S_a_val, S_j_val, F_max_val, K_val, Q_val)

    # We scan over the range of time indices
    t_indices = jnp.arange(time)
    _, (Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat) = lax.scan(process_year, None, t_indices)

    return Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat
