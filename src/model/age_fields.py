"""Map the community-encoder latent Z to per-cell demographic rate fields.

Each year, the latent vector Z (and its path-integrated form Z_disp) is
projected through learned weights into three correlated habitat manifolds --
survival H_s, reproduction H_r, and capacity H_k -- then passed through link
functions to per-cell adult/juvenile survival (S_a, S_j), max fecundity (F_max),
carrying capacity (K), and journey survival (Q). Runs as a checkpointed
``lax.scan`` over years to bound memory when differentiated.

K uses a LOG LINK:

    log K_base = log(k_level) + gamma_k*H_k + trend

``k_level`` is the continental capacity level, sampled in expected BBS route counts
and converted through the gauge (see ``age_priors.counts_to_relative``); the
covariate term is a log-fold deviation from it. Multiplicative because capacity
genuinely spans orders of magnitude -- observed per-cell mean counts run 0.1 to 226,
log sd 1.72.

The previous form was ``softplus(alpha_k + gamma_k*H_k)``, whose real defect was not
the link but ``alpha_k``'s prior: stated in relative density units, it asserted a
capacity of ~205 route counts (~97% of the highest counts ever recorded) when the
data want ~2, so it sat 7 prior SDs from the fit. Since K sits where softplus is
effectively ``exp()``, and every term able to lower K was being recruited to fight
that prior, K collapsed regionally -- through the disease term, then through this
covariate term, then wherever else was available. Stating the level correctly is
what fixed it; a tanh clamp on this term (briefly present) treated the symptom.

H_k is new. K previously reused H_r, making it a strictly monotone function of
F_max, so the disease term below was the ONLY way the two could differ spatially
-- and it duly saturated absorbing that disagreement. The three manifolds share a
one-factor prior whose correlations are ordered on purpose: Corr(F_max, K) ~ 0.85
sits above Corr(F_max, survival) ~ 0.70, since fecundity and capacity are both
productivity/resource axes. See ``age_priors.sample_priors``.

K also carries a continental, spatially uniform time trend (a few time-centered
cosines). Without it the model had no way at all to say "modern capacity is below
1970s capacity" -- which BBS demands, and which has non-disease causes -- except by
calling it disease.

K additionally receives a mycoplasmal-conjunctivitis effect -- the 1994-
continental epizootic, which has no covariate of its own in this model (step 4b in
``process_year``). It applies to K alone, never to H_s/H_r, so the
fundamental-niche quantities (see age_model_math.local_growth_lambda) stay exactly
covariate-driven.

The effect is a STRUCTURED hypothesis, not a free field:

    K = K_base * (1 - severity(x) * gate(x,t) * (1 - recovery(t - arrival)))

    severity(x) = ceiling * modifier(x) * K_base^n / (K_base^n + K_half^n)   # density-dependent
    modifier(x) = sigmoid(mu_sev + b_late*decades_since_epidemic_start(x)
                          + b_native*native_shape(x) + sev_basis(x).w_sev)
    gate(x,t)   = sigmoid((t - arrival(x) - lag0 - lag_basis(x).w_lag) / tau)
    recovery(a) = rec * (1 - exp(-a / tau_rec)),  a = years since local arrival

Severity is DENSITY-DEPENDENT (a Hill function of local capacity, both shape
parameters fitted) and capped at a configured ``ceiling`` (0.5). Epidemics need
hosts, so sparse populations are barely affected -- which also stops the Allee
effect from turning a small absolute capacity loss into a local extinction exactly
where that is least justified. The ceiling is load-bearing for identifiability: K
must stay monotone in K_base, which holds at 0.5 and fails by 0.7.

Each piece states a claim: once the front passes a cell, some FRACTION of local
carrying capacity is removed, scaled by how dense that population can get and
capped at the ceiling; the arrival map's timing is
coarse, so the front's position is fitted with continental and regional slack; the
hit is not permanent, because exposure builds resilience, so it decays toward
``severity*(1-rec)`` over ``tau_rec`` years; and TWO further, mechanistically
distinct claims each get their own coefficient rather than being folded into one:

* ``b_late`` on ``onset_decades`` -- decades since the epidemic's OWN historical
  start (``disease_start_year``, 1993), not since any individual cell's local
  arrival. Mycoplasmal conjunctivitis in House Finches is documented to have hit
  hardest in its first years and grown milder since, as pathogen and host
  co-evolved (attenuation/adaptation over the epidemic's own history): a cell
  reached by the front a decade after 1993 met an already-weaker epidemic than a
  cell reached in 1993 itself, for a reason that has nothing to do with THAT
  cell's population genetics. This is a CALENDAR-TIME trend, the same for every
  cell reached in a given decade regardless of where it sits.
* ``b_native`` on ``native_shape`` -- the SAME dimensionless core/margin map used
  to seed 1902 abundance (``bbs.generate_core_margin_initialization``): "was this
  cell part of the pre-1940 native range." The west coast lineage is genetically
  diverse (the ancestral range) rather than a single-founder 1940 introduction,
  and was documented to collapse far less than the east's -- a POPULATION-ORIGIN
  claim, independent of when the front happened to arrive. SIGN-CONSTRAINED to
  <= 0 (``-softplus(raw)``): native lineage may only suppress severity, never
  amplify it, since we have no comparable evidence that introduced lineage should
  be hit HARDER than a geography/timing/density-matched native cell, only that
  native lineage was hit less.

The two are collinear by construction (the west is both native AND late-arriving,
since the front happens to move west) -- that is expected, not a bug, for the
same reason ``b_late``/density collinearity below is expected: they are distinct
mechanisms that happen to share a geography, not the same mechanism twice. Do NOT
read ``b_late`` as a west/east proxy -- anchoring it at ``disease_start_year``
rather than the across-cell mean arrival time is what keeps it a calendar-time
claim instead of a second copy of ``native_shape`` under a different name.

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

Structure alone was not enough, though. The first structured fit still pinned
severity at 93% east / 73% west with zero recovery, every parameter 3-4.5 prior SDs
into its tail -- because the term was still the only channel for two things it
should not own: the spatial K-vs-Fmax difference (now H_k's job) and the temporal
capacity step (now the continental trend's job). Hence the ceiling, H_k, and the
trend together; each alone leaves the other outlets to saturate.

The remaining spatial freedom is two deliberately coarse, land-centered cosine
fields (~24 coefficients each, ~1000 km scale) that can only redistribute
severity and timing regionally -- the continental levels belong to ``mu_sev`` and
``lag0``, so both are directly reportable. Land-centering is what makes that split
exact; see ``model_inputs.generate_spatial_basis``.

On Jensen's inequality: an earlier revision rejected a one-sided link because a
concave link's expectation under a zero-mean perturbation sits below its
zero-perturbation value by a DATA-DEPENDENT amount, pulling the capacity level
upward and contaminating the pattern the term was meant to isolate. It does not bind
here: the gate is exogenous, so pre-arrival cells and all of 1902-1993 recover
``K_base`` exactly, which pins ``k_level`` on data the disease term cannot touch.
"""
import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, checkpoint


def disease_severity(disease, K_base):
    """Per-cell PEAK fraction of carrying capacity the epizootic removes.

    DENSITY-DEPENDENT, via a Hill function of local capacity::

        severity = ceiling * modifier(x) * K_base^n / (K_base^n + K_half^n)

    Epidemics need hosts. Mass-action transmission makes R0 scale with host
    density, and below a threshold density the pathogen cannot persist -- so
    severity goes to 0 as ``K_base -> 0`` and saturates at the ceiling in dense
    populations. This matters beyond realism: the model carries an Allee effect, so
    a small ABSOLUTE reduction in an already-small K can push a sparse population
    to local extinction. A density-independent severity therefore does its most
    violent damage exactly where it is least biologically justified.

    Both shape parameters are FITTED, so the shape is learned rather than
    asserted: ``k_half`` is the density at half-maximum severity and ``n`` the
    steepness (n~1 smooth saturation, n~3 a sharp invasion threshold).

    ``modifier(x)`` retains the smooth regional field, the epidemic-attenuation
    coefficient, and the native-lineage coefficient, as a multiplicative
    adjustment in (0, 1] around the density-driven value. ``b_late`` (on decades
    since ``disease_start_year``, NOT since a cell's own local arrival -- a
    calendar-time pathogen-attenuation/host-adaptation trend, see the module
    docstring) is now strongly collinear with the density term (the east is both
    early-arriving and dense) and is priored tightly for that reason; it is kept
    because epidemic-history timing and density are genuinely distinct
    mechanisms, so a strong fitted ``b_late`` ALONGSIDE the density term would be
    informative rather than redundant. ``b_native`` is collinear with BOTH --
    the west is native, late-arriving, and sparse -- but states a mechanistically
    distinct claim again (population origin/genetic diversity, not epidemic
    timing or density) and is the only one of the three constrained in SIGN:
    native lineage can only lower severity (see the module docstring).

    The ``ceiling`` is load-bearing for identifiability, not just plausibility:
    ``K = K_base * (1 - severity(K_base))`` must be monotone in ``K_base``, else
    two habitat qualities map to one realized capacity. At 0.5 that holds for every
    n tested; at 0.7 it fails.

    Severity keys off ``K_base``, the potential density, rather than realized N --
    partly because all demographic fields are precomputed before the forward
    simulation runs, so N does not exist yet, and partly because epidemic
    establishment depends on sustained density rather than a single year's count.
    """
    # Regional/arrival-order modifier, a plain sigmoid in (0, 1). NOT clipped: an
    # earlier version used min(1, 2*sigmoid(.)) so that a zero logit meant "no
    # modification", but that put a zero-gradient boundary exactly where mu_sev's
    # prior is centered, and the clip also broke the land-centered field's
    # guarantee that it cannot shift the continental level. With a plain sigmoid the
    # ceiling stays a strict upper bound (severity <= ceiling * density_term), the
    # field remains level-neutral in logit space, and mu_sev's prior is instead
    # centered on the modifier value we actually believe.
    modifier = jnn.sigmoid(
        disease["mu_sev"]
        + disease["b_late"] * disease["onset_decades"]
        + disease["b_native"] * disease["native_shape"]
        + jnp.dot(disease["sev_basis"].T, disease["w_sev"]))
    n = disease["hill_n"]
    density_term = K_base ** n / (K_base ** n + disease["k_half"] ** n)
    return disease["ceiling"] * modifier * density_term


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


def disease_k_fraction(disease, t_idx, K_base, onset_t=None, severity=None):
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
    severity = disease_severity(disease, K_base) if severity is None else severity
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
    beta_k,           # 1D feature weights for Carrying Capacity (Shape: M)
    k_trend_basis,    # (n_trend, time) time-centered cosines, continental
    w_k_trend,        # (n_trend,) weights on that trend
    alpha_a, gamma_a, # Adult survival intercept & slope
    alpha_j, gamma_j, # Juvenile survival intercept & slope
    alpha_f, gamma_f, # Max fecundity intercept & slope
    alpha_k, gamma_k  # Carrying capacity intercept (inside softplus) & slope
):
    """Project Z → (S_a, S_j, F_max, K, Q) for every year, on the land cells.

    Survival/journey rates use a sigmoid link on the survival manifold H_s;
    fecundity and capacity use softplus on the reproduction manifold H_r. Q (in-
    cohort survival) reuses the juvenile survival intercept/slope on the
    land-conditioned neighborhood/path habitat H_s_disp. Each returned array is
    (time, N_land[, K]). ``Kbase_flat`` is capacity BEFORE the disease effect --
    returned because severity is a function of it, so any exact diagnostic of the
    disease term needs the undepressed value rather than the depressed one.

    ``disease`` bundles the epizootic term's inputs and parameters (see
    ``disease_k_fraction``). Timesteps before ``disease_timestep`` get K_base
    unmodified -- exactly, not approximately -- which is what pins ``k_level`` on
    the 1966-1993 BBS record the disease term cannot reach.
    """
    # The onset field does not depend on t, so compute it once rather than 124
    # times inside the scan. Severity now depends on K_base, which varies by year
    # through the continental trend, so it is computed per year.
    onset_t = disease_onset_timestep(disease)

    # Checkpoint: don't store this function's large intermediates for the
    # backward pass; recompute them instead.
    @checkpoint
    def process_year(carry, t_idx):
        # 1. Pull slices from CPU RAM -> GPU VRAM
        z_t = jnp.take(Z_gathered, t_idx, axis=0)
        z_disp_t = jnp.take(Z_disp_gathered, t_idx, axis=0)

        # 2. Compute the 3 Correlated Habitat Manifolds (H_s, H_r, H_k) --
        # purely covariate-driven (Z.beta), no spatiotemporal term mixed in.
        H_s_local = jnp.dot(z_t, beta_s)
        H_r_local = jnp.dot(z_t, beta_r)
        # Capacity reads its OWN manifold. It used to reuse H_r, making K a strictly
        # monotone function of Fmax, so the disease term was the only way their
        # spatial patterns could differ -- see the module docstring.
        H_k_local = jnp.dot(z_t, beta_k)

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

        # Continental, spatially uniform capacity drift. Time-centered basis, so
        # k_level still owns the level; this exists so "modern K below 1970s K" has
        # somewhere to go other than the disease term. Its prior is deliberately
        # near-zero -- it is a safety valve, not a mechanism.
        k_trend_t = jnp.dot(jnp.take(k_trend_basis, t_idx, axis=1), w_k_trend)

        # Baseline K with a SOFTPLUS link, in route counts, converted by the gauge:
        #     K_counts = softplus(alpha_k + gamma_k*H_k + trend)
        # This is a CONTROLLED TEST against the exp form,
        #     K = k_level * exp(gamma_k*H_k + trend),
        # with alpha_k's prior location solved so the post-transformation prior MEAN of
        # capacity is identical (2.8920 route counts). Only the link changed.
        #
        # The two differ in more than tail weight: under softplus gamma_k*H_k is
        # ADDITIVE IN ROUTE COUNTS, so K's spatial spread is no longer multiplicative
        # and the covariate term cannot reproduce the log-normal shape of the observed
        # occupied-cell distribution (log sd 1.72, spanning 0.1 to 226 counts). Softplus
        # also has the property that made the ORIGINAL failure so hard to see: where its
        # argument is negative it behaves like exp(), so a term subtracted from it acts
        # multiplicatively and without a floor. That is not a problem here -- nothing is
        # subtracted from this argument any more -- but it is why K sat in the
        # exponential regime for the first several runs.
        K_base_val = jnn.softplus(alpha_k + gamma_k * H_k_local + k_trend_t)

        # 4b. Disease effect on K only: a bounded MULTIPLICATIVE rescale (see the
        # module docstring for why the earlier additive-inside-softplus form
        # annihilated eastern K instead of rescaling it). The severity and onset
        # fields are time-independent, so they are hoisted out of the scan.
        k_fraction = jnp.where(
            t_idx >= disease_timestep,
            disease_k_fraction(disease, t_idx, K_base_val, onset_t=onset_t),
            0.0,
        )
        K_val = K_base_val * (1.0 - k_fraction)

        # 5. Map Path Habitat (H_s) to Journey Survival (Q) using juvenile rules
        # This perfectly links movement mortality to local survival mortality
        Q_val = jnn.sigmoid(alpha_j + gamma_j * H_s_disp)

        # K_base_val is returned as well: severity is a function of it, so exact
        # diagnostics (and the severity map) need the UNDEPRESSED capacity. Using
        # the depressed K would be circular. Costs one more (time, N_land) float32
        # array, ~8 MB at production size.
        return None, (S_a_val, S_j_val, F_max_val, K_val, Q_val, K_base_val)

    # We scan over the range of time indices
    t_indices = jnp.arange(time)
    _, (Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat, Kbase_flat) = lax.scan(
        process_year, None, t_indices)

    return Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat, Kbase_flat
