"""Map the community-encoder latent Z to per-cell demographic rate fields.

Each year, the latent vector Z (and its path-integrated form Z_disp) is
projected through learned weights into two habitat manifolds — survival H_s
and reproduction H_r — then passed through link functions to per-cell
adult/juvenile survival (S_a, S_j), max fecundity (F_max), carrying capacity
(K), and journey survival (Q). Runs as a checkpointed ``lax.scan`` over years
to bound memory when differentiated.

K additionally receives a mycoplasmal-conjunctivitis effect -- the 1994-
continental epizootic, which has no covariate of its own in this model (step 4b in
``process_year``). It applies to K alone, never to H_s/H_r, so the
fundamental-niche quantities (see age_model_math.local_growth_lambda) stay exactly
covariate-driven.

The effect is a STRUCTURED hypothesis, not a free field:

    K = K_base * (1 - severity(x) * gate(x,t) * (1 - recovery(t - arrival)))

    severity(x) = sigmoid(mu_sev + b_late*arrival_decades(x) + sev_basis(x).w_sev)
    gate(x,t)   = sigmoid((t - arrival(x) - lag0 - lag_basis(x).w_lag) / tau)
    recovery(a) = rec * (1 - exp(-a / tau_rec)),  a = years since local arrival

Each piece states a claim: once the front passes a cell, some FRACTION of local
carrying capacity is removed (prior median 50%); the arrival map's timing is
coarse, so the front's position is fitted with continental and regional slack; the
hit is not permanent, because exposure builds resilience, so it decays toward
``severity*(1-rec)`` over ``tau_rec`` years; and populations reached later were
plausibly hit less hard (more genetic diversity in the west), carried by the
single coefficient ``b_late`` on arrival year.

**Why this replaced a generic spatiotemporal basis.** The previous design
subtracted an unbounded ``d >= 0`` from K's pre-softplus argument, with ``d``
built from 967 free cosine coefficients over space x time. Two failures:

1. *Additive in an exponential regime.* Fitted K sits at pre-softplus arguments
   around -2 to -4, where ``softplus(x) ~ exp(x)``, so subtracting ``d``
   MULTIPLIED K by ``exp(-d)`` with no floor: ``d=3.5`` removed 97% of capacity.
   It annihilated carrying capacity across the entire eastern US instead of
   rescaling it. A fraction in (0,1) cannot do that at any operating point --
   that is the whole point of the multiplicative bounded form.
2. *Only spatial degree of freedom in the model.* There is no per-route effect,
   no detection model, no observation-level spatial term anywhere, and K is
   ~multiplicative on predicted abundance (only ``N/K`` enters the forward sim).
   A generic field in that position is the natural sink for every kind of spatial
   misfit, disease-shaped or not. Constraining its SHAPE is what makes it a
   disease term rather than a spatial escape hatch.

The remaining spatial freedom is two deliberately coarse, land-centered cosine
fields (~24 coefficients each, ~1000 km scale) that can only redistribute
severity and timing regionally -- the continental levels belong to ``mu_sev`` and
``lag0``, so both are directly reportable. Land-centering is what makes that split
exact; see ``model_inputs.generate_spatial_basis``.

On Jensen's inequality: an earlier revision rejected a one-sided link because a
concave link's expectation under a zero-mean perturbation sits below its
zero-perturbation value by a DATA-DEPENDENT amount, pulling alpha_k upward and
contaminating the pattern the term was meant to isolate. It does not bind here:
the gate is exogenous, so pre-arrival cells and all of 1902-1993 recover
``K_base`` exactly, which pins ``alpha_k`` on data the disease term cannot touch.
"""
import math

import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, checkpoint
import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, checkpoint


def disease_severity(disease):
    """Per-cell PEAK fraction of carrying capacity the epizootic removes, in (0,1).

    ``sigmoid`` is what bounds it: no draw of any parameter can remove more than
    all of K, and none can ADD capacity, so the term is sign-constrained and
    annihilation-proof by construction rather than by prior tuning.

    Time-independent, and deliberately coarse in space (~24 land-centered cosine
    coefficients). ``b_late`` on centered arrival year (in decades) carries "later
    front arrival implies a milder hit" -- the western-genetic-diversity
    hypothesis -- as one interpretable coefficient.
    """
    logit = (disease["mu_sev"]
             + disease["b_late"] * disease["onset_decades"]
             + jnp.dot(disease["sev_basis"].T, disease["w_sev"]))
    return jnn.sigmoid(logit)


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


def disease_k_fraction(disease, t_idx, onset_t=None, severity=None):
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
    severity = disease_severity(disease) if severity is None else severity
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

    ``disease`` bundles the epizootic term's inputs and parameters (see
    ``disease_k_fraction``). Timesteps before ``disease_timestep`` get K_base
    unmodified -- exactly, not approximately -- which is what pins ``alpha_k`` on
    the 1966-1993 BBS record the disease term cannot reach.
    """
    # The disease severity and onset fields do not depend on t, so compute them
    # once outside the scan rather than 124 times inside it.
    severity = disease_severity(disease)
    onset_t = disease_onset_timestep(disease)

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

        # 4b. Disease effect on K only: a bounded MULTIPLICATIVE rescale (see the
        # module docstring for why the earlier additive-inside-softplus form
        # annihilated eastern K instead of rescaling it). The severity and onset
        # fields are time-independent, so they are hoisted out of the scan.
        k_fraction = jnp.where(
            t_idx >= disease_timestep,
            disease_k_fraction(disease, t_idx, onset_t=onset_t, severity=severity),
            0.0,
        )
        K_val = K_base_val * (1.0 - k_fraction)

        # 5. Map Path Habitat (H_s) to Journey Survival (Q) using juvenile rules
        # This perfectly links movement mortality to local survival mortality
        Q_val = jnn.sigmoid(alpha_j + gamma_j * H_s_disp)

        return None, (S_a_val, S_j_val, F_max_val, K_val, Q_val)

    # We scan over the range of time indices
    t_indices = jnp.arange(time)
    _, (Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat) = lax.scan(process_year, None, t_indices)

    return Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat
