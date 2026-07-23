"""Map the community-encoder latent Z to per-cell demographic rate fields.

Each year, the latent vector Z (and its path-integrated form Z_disp) is
projected through learned weights into two habitat manifolds — survival H_s and
reproduction H_r — plus a spatiotemporal random-effect term, then passed through
link functions to per-cell adult/juvenile survival (S_a, S_j), max fecundity
(F_max), carrying capacity (K), and journey survival (Q). Runs as a
checkpointed ``lax.scan`` over years to bound memory when differentiated.
"""
import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, checkpoint


def project_and_scatter_age_structured(
    time, Ny, Nx,
    land_rows, land_cols,
    Z_gathered, Z_disp_gathered,
    st_basis, st_weights,
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
    transit survival) reuses the juvenile survival intercept/slope on the path-
    integrated habitat H_s_disp. Each returned array is (time, N_land[, K]).
    """
    # Checkpoint: don't store this function's large intermediates for the
    # backward pass; recompute them instead.
    @checkpoint
    def process_year(carry, t_idx):
        # 1. Pull slices from CPU RAM -> GPU VRAM
        z_t = jnp.take(Z_gathered, t_idx, axis=0)
        z_disp_t = jnp.take(Z_disp_gathered, t_idx, axis=0)
        st_basis_t = jnp.take(st_basis, t_idx, axis=1) 
        
        # Spatio-temporal random effects (Baseline noise applied to both fields)
        z_smooth = jnp.dot(st_basis_t.T, st_weights)
        
        # 2. Compute the 2D Correlated Habitat Manifolds (H_s and H_r)
        H_s_local = jnp.dot(z_t, beta_s) + z_smooth
        H_r_local = jnp.dot(z_t, beta_r) + z_smooth
        
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
        K_val = jnn.softplus(alpha_k + gamma_k * H_r_local)
        
        # 5. Map Path Habitat (H_s) to Journey Survival (Q) using juvenile rules
        # This perfectly links movement mortality to local survival mortality
        Q_val = jnn.sigmoid(alpha_j + gamma_j * H_s_disp)
        
        return None, (S_a_val, S_j_val, F_max_val, K_val, Q_val)

    # We scan over the range of time indices
    t_indices = jnp.arange(time)
    _, (Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat) = lax.scan(process_year, None, t_indices)
    
    return Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat