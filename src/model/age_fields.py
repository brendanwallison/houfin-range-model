"""Map the community-encoder latent Z to per-cell demographic rate fields.

Each year, the latent vector Z (and its path-integrated form Z_disp) is
projected through learned weights into two habitat manifolds — survival H_s
and reproduction H_r — then passed through link functions to per-cell
adult/juvenile survival (S_a, S_j), max fecundity (F_max), carrying capacity
(K), and journey survival (Q). Runs as a checkpointed ``lax.scan`` over years
to bound memory when differentiated.

K additionally receives a spatiotemporal DEPRESSION representing mycoplasmal
conjunctivitis, whose 1994- continental epizootic has no covariate of its own in
this model (see step 4b in ``process_year`` below). It is NOT a smoothing term on
Z/H_s/H_r (an earlier design added a shared spatial random effect to both
manifolds; that coupled the correction to survival and reproduction alike and
gave it no principled way to only capture something disease-shaped). Restricting
it to K alone, uncoupled from Sa/Sj/Fmax, keeps the fundamental-niche quantities
(see age_model_math.local_growth_lambda) exactly covariate-driven.

Three properties distinguish the current form from the earlier free
multiplicative correction, which overfit as a general-purpose abundance modifier:

1. **Sign-constrained.** The term is subtracted INSIDE K's own softplus,
   ``K = softplus(alpha_k + gamma_k*H_r - d)`` with ``d >= 0``. A disease can
   only lower carrying capacity, so the term can no longer buy likelihood by
   raising K wherever the covariates undershoot.
2. **Onset-gated by an exogenous arrival map.** ``d`` is multiplied by a logistic
   gate on ``t - arrival_year(x)``, where the arrival surface is built
   independently from the documented spread history (see
   ``scripts/build_disease_arrival_map.py``). Where the disease had not yet
   arrived the gate is ~0 and K sits at its covariate-driven baseline by
   construction, not by prior encouragement. The basis therefore no longer has
   to DISCOVER a wavefront out of global cosines -- it only explains deviations
   in magnitude from a known front, which is what removes most of the
   flexibility that was being overfit.
3. **Windowed to the epizootic.** The basis's time axis covers
   ``disease_start_year..end_year`` (~33 years), not ``invasion_year..end_year``
   (~86), so the same coefficient budget buys finer spatial resolution at a
   fraction of the VRAM.

The gate's ``lag`` and ``tau`` are learned continental scalars, which is what
makes the coarseness of the hand-built arrival surface tolerable: a systematic
one- or two-year bias, or the shrinkage the map's kernel smoother induces at the
extremes, is absorbed rather than propagated.

On Jensen's inequality: an earlier revision moved AWAY from a one-sided link
because a concave link's expectation under a zero-mean perturbation sits below
its zero-perturbation value by a DATA-DEPENDENT amount -- larger wherever the
term was actually used -- which pulled alpha_k upward and contaminated the very
pattern the term was meant to isolate. That argument no longer binds. The
location- and time-dependence of the expected depression is now supplied
exogenously by the gate (a known function of the arrival map, not a latent one),
so what alpha_k must absorb is only ``E[softplus(mu_d + b.w)]``, a constant flat
in space and time.
"""
import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, checkpoint


def project_and_scatter_age_structured(
    time, Ny, Nx,
    land_rows, land_cols,
    Z_gathered, Z_disp_gathered,
    st_basis, st_weights, disease_timestep,
    disease_onset, disease_lag, disease_tau, disease_mu,
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

    ``st_basis`` covers only the epizootic window (its time axis has
    ``time - disease_timestep`` entries, not ``time`` -- see
    ``model_inputs.generate_spatiotemporal_basis``'s caller). Timesteps before
    ``disease_timestep`` get K_val unmodified (depression exactly 0.0); this also
    avoids ever indexing ``st_basis`` out of bounds.

    ``disease_onset`` is the per-land-cell arrival year expressed in MODEL
    TIMESTEP units (i.e. already converted from a calendar year), so the gate
    compares like with like against ``t_idx``.
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
        # K's own softplus is applied in step 4b, after the disease depression is
        # subtracted from its argument (the whole point of the sign constraint).

        # 4b. Disease depression of K only (see module docstring). Three pieces:
        #   gate      = logistic ramp in (t - arrival_year - lag) / tau, so a cell
        #               is undepressed until the front reaches it;
        #   magnitude = softplus(mu_d + basis.weights) >= 0, spatiotemporally
        #               flexible (including tapering back toward 0, which is how
        #               partial post-epizootic recovery is represented);
        #   depression = gate * magnitude, subtracted INSIDE K's softplus so K can
        #               only ever be reduced, never raised, by this term.
        # Clip the basis index to stay in-bounds; the clipped value is discarded
        # via jnp.where whenever t_idx < disease_timestep (pre-epizootic: exactly
        # zero depression, K_base_val recovered bit-for-bit).
        basis_len = st_basis.shape[1]
        basis_idx = jnp.clip(t_idx - disease_timestep, 0, basis_len - 1)
        st_basis_t = jnp.take(st_basis, basis_idx, axis=1)
        gate = jnn.sigmoid((t_idx - disease_onset - disease_lag) / disease_tau)
        magnitude = jnn.softplus(disease_mu + jnp.dot(st_basis_t.T, st_weights))
        depression = jnp.where(t_idx >= disease_timestep, gate * magnitude, 0.0)
        K_val = jnn.softplus(alpha_k + gamma_k * H_r_local - depression)

        # 5. Map Path Habitat (H_s) to Journey Survival (Q) using juvenile rules
        # This perfectly links movement mortality to local survival mortality
        Q_val = jnn.sigmoid(alpha_j + gamma_j * H_s_disp)

        return None, (S_a_val, S_j_val, F_max_val, K_val, Q_val)

    # We scan over the range of time indices
    t_indices = jnp.arange(time)
    _, (Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat) = lax.scan(process_year, None, t_indices)

    return Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat
