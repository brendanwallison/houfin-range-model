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
import math

import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import numpyro
import numpyro.distributions as dist

from src.config_utils import load_age_model_config
from src.model.age_fields import (DISEASE_ONSET_SPEC, disease_mortality_amplitude, disease_severity,
                                  project_and_scatter_age_structured)
from src.model.age_forward import allee_mated_fraction, forward_sim_age_structured, juvenile_band_weights
from src.model.habitat_geometry import capacity_demography_coefficients, capacity_demography_loadings

# Prior settings read from config so they can be retuned without editing model
# code -- these are the knobs most likely to be revisited between runs. See
# sample_priors for the semantics of each.
_POP_SPEC = load_age_model_config()["population_model"]
_DISEASE_PRIOR = dict(_POP_SPEC["disease_prior"])
_MANIFOLD_PRIOR = dict(_POP_SPEC["manifold_prior"])
_K_TREND = dict(_POP_SPEC["k_trend"])
_CAPACITY_LEVEL = dict(_POP_SPEC["capacity_level_prior"])
_INVASION_PULSE = dict(_POP_SPEC["invasion_pulse_prior"])
_INITPOP_SEED = dict(_POP_SPEC["initpop_seed"])
_ALLEE_PRIOR = dict(_POP_SPEC["allee_prior"])
_RELEASE_ALLEE_EXEMPT_YEARS = int(_POP_SPEC.get("release_allee_exemption_years", 0))
# Structural switches. Each default reproduces the model before the switch existed, so a
# config without the key fits the same model it always did.
#   manifold_prior.form       "rank2" (two coupled groups) | "exchangeable" (one coupling)
#   capacity_level_prior.link "softplus" | "exp"
#   dispersal_random_enabled  per-year dispersal noise on/off
_MANIFOLD_FORM = str(_MANIFOLD_PRIOR.get("form", "rank2"))
_K_LINK = str(_CAPACITY_LEVEL.get("link", "softplus"))
#   capacity_level_prior.form "field" (K from its own habitat field) | "crowding"
_K_FORM = str(_CAPACITY_LEVEL.get("form", "field"))
_CROWDING = dict(_CAPACITY_LEVEL.get("crowding", {}))
if _K_FORM not in {"field", "crowding"}:
    raise ValueError(f"capacity_level_prior.form must be field or crowding, got {_K_FORM!r}")
_DISPERSAL_RANDOM = bool(_POP_SPEC.get("dispersal_random_enabled", True))
#   habitat_centering         "none" | "occupied" (fields centred at mean occupied-cell Z)
_HABITAT_CENTERING = str(_POP_SPEC.get("habitat_centering", "none"))
#   allee_prior.form          "absolute" (mates by density N) | "relative" (by N/K)
_ALLEE_FORM = str(_ALLEE_PRIOR.get("form", "absolute"))
if _ALLEE_FORM not in {"absolute", "relative"}:
    raise ValueError(f"allee_prior.form must be absolute or relative, got {_ALLEE_FORM!r}")
ALLEE_RELATIVE = _ALLEE_FORM == "relative"
#   allee_prior.aggregation_k   None (default) = Poisson-encounter mate finding; k > 0 = encounter averaged over
#                               negative-binomial within-cell aggregation (k = 1: hyperbolic). See age_forward.allee_mated_fraction.
ALLEE_AGG_K = None if _ALLEE_PRIOR.get("aggregation_k") is None else float(_ALLEE_PRIOR["aggregation_k"])
#   juvenile_survival.form    "independent" (own sigmoid) | "ratio" (Sj = Sa * sigmoid(...))
_JUV = dict(_POP_SPEC.get("juvenile_survival", {}))
_JUV_FORM = str(_JUV.get("form", "independent"))
if _JUV_FORM not in {"independent", "ratio"}:
    raise ValueError(f"juvenile_survival.form must be independent or ratio, got {_JUV_FORM!r}")
JUVENILE_RATIO = _JUV_FORM == "ratio"
#   pinned_sites              {site: value} held FIXED (numpyro condition) -- for profiling
#                             what the data resist. Values are on each site's raw scale
#                             (e.g. alpha_a is a logit; with habitat_centering, at typical
#                             occupied habitat). Pinned sites are not fitted at all.
#                             A list pins a vector site (e.g. w_scale_raw, one value per field); it is kept
#                             as a tuple so the dict stays JSON-serialisable.
PINNED = {k: (tuple(float(x) for x in v) if isinstance(v, (list, tuple)) else float(v))
          for k, v in dict(_POP_SPEC.get("pinned_sites", {})).items() if not k.startswith("_")}
#   settlement                {"form": "none" | "ideal_free", "block": cells}
_SETTLE = dict(_POP_SPEC.get("settlement", {}))
SETTLE_BLOCK = int(_SETTLE.get("block", 3)) if _SETTLE.get("form", "none") == "ideal_free" else 0
#                             "weight": "capacity" (default; shares in proportion to K) |
#                             "fitness" (equal realized lambda, Allee excluded; nests capacity
#                             when vital rates are equal within a block) | "uniform" (control)
SETTLE_WEIGHT = _SETTLE.get("weight", "capacity")
#                             "apply_to": "all" (default: every bird in the block is rearranged each
#                             year) | "dispersers" (only this year's arrivals; residents stay put)
SETTLE_DISPERSERS = _SETTLE.get("apply_to", "all") == "dispersers"
#   disease_prior.target      "capacity" (default: the epizootic depresses K) | "survival" (it is ADDED
#                             MORTALITY on adults and juveniles, Hill on CURRENT density, host
#                             resistance via rec/tau_rec; K disease-free; ceiling =
#                             disease_prior.mortality_ceiling, default 0.8)
_DISEASE_SPEC = dict(_POP_SPEC.get("disease_prior", {}))
DISEASE_TARGET = _DISEASE_SPEC.get("target", "capacity")
if DISEASE_TARGET not in {"capacity", "survival"}:
    raise ValueError(f"disease_prior.target must be capacity or survival, got {DISEASE_TARGET!r}")
_DISEASE_CEILING = float(_DISEASE_SPEC.get("mortality_ceiling", 0.8) if DISEASE_TARGET == "survival"
                         else _DISEASE_SPEC.get("severity_ceiling", 0.5))
#   subcell_strata            {"enabled": bool, "beta_k_scale", "delta_f_scale"}: split each cell
#                             into within-cell urbanisation slices (data: subcell_strata.npz from
#                             src/data/preprocess/subcell_strata.py), fitness ideal-free among them,
#                             Allee + Beverton-Holt per slice; mean-preserving tilts of K
#                             (subcell_beta_k) and Fmax (subcell_delta_f), both 0 = the one-slice model.
#   dispersal_kernel          {"band_weights": bool, "band_logw_scale", "time_knots", "time_scale"}
#                             (time_knots, e.g. [1940, 1960, 1980, 2000, 2020]: era-varying band log-weights,
#                             band_logw + linear-hat-interpolated deviations ~ N(0, time_scale) at each knot): learn the juvenile kernel's
#                             radial-band weights (mass-preserving; 0 = current kernel) -- tail
#                             weight without new path features (age_forward.juvenile_band_weights)
_KERNEL_SPEC = dict(_POP_SPEC.get("dispersal_kernel", {}))
BAND_WEIGHTS = bool(_KERNEL_SPEC.get("band_weights", False))
#   year_effects              {"enabled": bool, "sd_scale"}: continental per-year multipliers on expected
#                             counts (survey-wide fluctuation: weather, observer turnover), sd learned
#                             ~ HalfNormal(sd_scale), centred over observed years -- so a deterministic
#                             model need not chase annual swings with fast dynamics.
#   observation               {"form": "post_breeding" (default: count ~ adults + ALL of the year's new
#                             juveniles) | "breeding" (count ~ the BREEDING population of year t -- the
#                             previous year's end-of-year total, all of breeding age -- plus a learned
#                             fraction phi ~ Beta(a, b) of year t's fledglings)}. BBS runs in June and
#                             counts mostly breeding adults; under post_breeding, juveniles outnumber
#                             adults ~F*Sa:1 at low density, so modelled counts in a colonising cell grow
#                             SLOWER than the breeders and the fit needs a higher lambda0.
_OBS_SPEC = dict(_POP_SPEC.get("observation", {}))
OBS_BREEDING = _OBS_SPEC.get("form", "post_breeding") == "breeding"
#   observation.route_occupancy {"enabled", "m_loc", "m_scale"} (E072): mean-preserving route-level zero inflation.
#   A route samples a patch of its 27 km cell; with cell mean mu (route counts) the route is EMPTY with probability
#   pi = exp(-mu / m) (Poisson patches of mean size m counts), and an occupied route counts NB with mean mu / (1 - pi),
#   so E[y] = mu is unchanged: sparse cells may show mostly empty routes without their capacity collapsing.
#   log m ~ Normal(log m_loc, m_scale).
_ROUTE_OCC = dict(_OBS_SPEC.get("route_occupancy") or {})
ROUTE_OCCUPANCY = bool(_ROUTE_OCC.get("enabled", False))
#   observation.observer_effect {"enabled" (default false), "sigma_u", "sigma_mode": "fixed" | "prior",
#   "sigma_prior_scale", "eta_prior_scale"}: BBS observer x route random effect + first-year effect on real
#   counts (bbsBayes2). See observer_log_offset. Read at CALL time from this dict, so tests can patch it.
_OBSERVER_SPEC = dict(_OBS_SPEC.get("observer_effect") or {})


def _hat_basis(years, knots):
    """(time, n_knots) linear-hat interpolation weights over the knot years, constant beyond the end knots."""
    y = np.clip(np.asarray(years, float), knots[0], knots[-1]); k = np.asarray(knots, float)
    B = np.zeros((len(y), len(k)))
    j = np.clip(np.searchsorted(k, y, side="right") - 1, 0, len(k) - 2)
    f = (y - k[j]) / (k[j + 1] - k[j])
    B[np.arange(len(y)), j] = 1.0 - f
    B[np.arange(len(y)), j + 1] += f
    return B


def observer_effect_spec(spec=None):
    """Validated ``population_model.observation.observer_effect`` (defaults: OFF, fixed sigma)."""
    s = dict(_OBSERVER_SPEC if spec is None else spec)
    out = {
        "enabled": bool(s.get("enabled", False)),
        "sigma_u": float(s.get("sigma_u", 0.0)),
        "sigma_mode": str(s.get("sigma_mode", "fixed")),
        "sigma_prior_scale": float(s.get("sigma_prior_scale", 1.0)),
        "eta_prior_scale": float(s.get("eta_prior_scale", 1.0)),
    }
    if out["sigma_mode"] not in {"fixed", "prior"}:
        raise ValueError(f"observer_effect.sigma_mode must be fixed or prior, got {out['sigma_mode']!r}")
    if out["sigma_u"] < 0 or out["sigma_prior_scale"] <= 0 or out["eta_prior_scale"] <= 0:
        raise ValueError(f"observer_effect scales must be positive: {out}")
    return out


def observer_log_offset(data, spec):
    """Per-observation log multiplier ``eta * first_year + u[obsroute]``; exactly 0 for pseudo-zeros.

    The observer submodel of Sauer & Link (2011) / bbsBayes2, as the age model needs it: one global
    first-year coefficient ``eta`` and one effect per observer x ROUTE (bbsBayes2's ``obs_route``;
    Mexico has no observer id -> one effect per Mexico route). Ids: data["obs_obsroute_idx"] (-1 =
    pseudo-zero), data["obs_first_year"] (0/1), static size data["n_obsroute"] (data_loading).

        u = sigma_u * eps,  eps ~ ZeroSumNormal(1)      (non-centred, as bbsBayes2's sdobs * obs_raw
                                                         with sum_to_zero_vector obs_raw)

    * ZERO-SUM rather than a -sigma^2/2 lognormal correction. The capacity level is anchored on a
      MEDIAN in route counts (docs/PRIORS.md#capacity-level). Zero-sum makes the geometric mean of
      the observer-route multipliers exactly 1 and, the effects being symmetric, the median ~1, so
      density * pop_scalar keeps meaning "the count of a typical observer-route". A -sigma^2/2 offset
      preserves the ARITHMETIC mean instead and moves the median observer-route down by
      exp(-sigma^2/2) (-34% at sigma 0.9). Zero-sum also removes the common-shift direction of u that
      would otherwise trade against the capacity level.
    * sigma FIXED by default. At a JOINT mode (AutoDelta MAP, then float64 L-BFGS, age_refine_map) a
      hierarchical scale is degenerate: centred u ~ N(0, sigma) drives sigma -> 0; non-centred
      rewards inflating sigma (for fixed u the eps penalty u^2 / (2 sigma^2) falls as sigma grows)
      until sigma's prior stops it -- observation.route_effects' free obs_route_sd went to 7.5 in
      run_41 for exactly this reason (map_run59a pinned it). sigma_u is estimated beforehand with the
      effects integrated out (research/age_model/exp/obsroute_sigma.py). sigma_mode "prior" samples
      sigma ~ HalfNormal(sigma_prior_scale): for HMC ONLY, never for a mode-finding fit.
    * eta is an ordinary fixed effect (bbsBayes2: eta ~ std_normal), so the joint-mode problem does
      not arise; ``density`` then refers to an experienced observer. pinned_sites may pin
      observer_first_year_eta / observer_route_sigma like any other site.
    * sigma_u == 0 in fixed mode creates no eps site at all.
    """
    for key in ("obs_obsroute_idx", "obs_first_year"):
        if data.get(key) is None:
            raise ValueError(f"observation.observer_effect needs data[{key!r}]: ingest a BBS npz with "
                             "observer ids, or build obs_observer_ids.npz "
                             "(research/age_model/exp/build_observer_ids.py)")
    idx = data["obs_obsroute_idx"]
    real = idx >= 0
    fy = jnp.where(real, data["obs_first_year"], 0)
    if "observer_first_year_eta" in PINNED:
        eta = numpyro.deterministic("observer_first_year_eta", jnp.asarray(PINNED["observer_first_year_eta"]))
    else:
        eta = numpyro.sample("observer_first_year_eta", dist.Normal(0.0, spec["eta_prior_scale"]))
    offset = eta * fy
    if spec["sigma_mode"] == "prior":
        if "observer_route_sigma" in PINNED:
            sigma = numpyro.deterministic("observer_route_sigma", jnp.asarray(PINNED["observer_route_sigma"]))
        else:
            sigma = numpyro.sample("observer_route_sigma", dist.HalfNormal(spec["sigma_prior_scale"]))
    else:
        sigma = spec["sigma_u"]
    if spec["sigma_mode"] == "prior" or sigma > 0:
        n = data.get("n_obsroute")            # a plain int recorded at load time: it sets a shape
        if n is None:
            n = int(np.asarray(idx).max()) + 1
        eps = numpyro.sample("observer_route_eps", dist.ZeroSumNormal(1.0, event_shape=(int(n),)))
        offset = offset + jnp.where(real, sigma * eps[jnp.maximum(idx, 0)], 0.0)
    return offset


def route_occupancy_parts(mu, m):
    """(gate, occupied-route mean, gate logits) for observation.route_occupancy; finite in value and gradient as
    mu -> 0. The logits are exact (gate is within 1e-11 of 1 in empty cells, where 1 - gate loses its digits)."""
    x = jnp.maximum(mu, 1e-12) / m
    occ = -jnp.expm1(-x)                                   # 1 - pi, accurate for small x
    return 1.0 - occ, m * x / occ, -x - jnp.log(occ)       # mu / (1 - pi) -> m as mu -> 0
_YEAR_FX = dict(_POP_SPEC.get("year_effects", {}))
YEAR_EFFECTS = bool(_YEAR_FX.get("enabled", False))
#   habitat_change            {"shrink": "learned", "eta_a", "eta_b"}: Z_eff(t) = Z_mean + eta*(Z(t) - Z_mean),
#                             eta ~ Beta(a, b) learned (1 = raw covariates, 0 = static habitat); applied
#                             to local and path features alike. Default off.
_HABCHG = dict(_POP_SPEC.get("habitat_change", {}))
HABITAT_SHRINK = _HABCHG.get("shrink") == "learned"
_SUBCELL = dict(_POP_SPEC.get("subcell_strata", {}))
SUBCELL = bool(_SUBCELL.get("enabled", False))
SUBCELL_SURVIVAL_AFFECTS_ALLOCATION = bool(
    _SUBCELL.get("survival_affects_allocation", True))
#                             | "capacity_fitness" (K * lambda0^tau, tau >= 0 learned with prior
#                             HalfNormal(settlement.tau_scale, default 2); tau = 0 is capacity)
#                             | "capacity_r0" (K * R0^tau, R0 = Fmax*Sj/(1-Sa), tau learned) | "r0" (R0)
if SETTLE_WEIGHT not in {"capacity", "fitness", "uniform", "capacity_fitness", "capacity_r0", "r0"}:
    raise ValueError("settlement.weight must be capacity, fitness, uniform, capacity_fitness, "
                     f"capacity_r0 or r0, got {SETTLE_WEIGHT!r}")
#   dispersal_departure       {"form": "shared" | "age", "adult_offset_loc", "adult_offset_scale",
#                              "crowding_arm": bool, "crowding_scale", "crowding_sharpness"}
#                             "age" adds a logit offset to the ADULT departure probability
#                             (breeding dispersal) on top of the shared density-dependent curve,
#                             so adults can be site-faithful without making juveniles so.
#                             crowding_arm adds + beta_hi * softplus(kappa (N/K - 1)) / kappa,
#                             beta_hi >= 0: departure pulses from cells ABOVE capacity, while the
#                             existing free-sign slope stays the low-density arm. beta_hi = 0 is
#                             exactly the one-slope curve. kappa (sharpness) is fixed.
_DEPART = dict(_POP_SPEC.get("dispersal_departure", {}))
_DEPART_FORM = _DEPART.get("form", "shared")
#                             | "veit" (Veit & Lewis 1996 eq. 18: p = min(N/K, 1) for both ages, no
#                             parameters; the logit intercept/slope sites are then unused)
if _DEPART_FORM not in {"shared", "age", "veit"}:
    raise ValueError(f"dispersal_departure.form must be shared, age or veit, got {_DEPART_FORM!r}")
VEIT_DEPARTURE = _DEPART_FORM == "veit"
#   reproduction              {"form": "standard" | "veit", "yearling_breeding": bool}. standard: Beverton-
#                             Holt on fecundity with K the no-dispersal equilibrium; yearling_breeding lets
#                             yearlings breed (House Finch; the default lets only adults breed, i.e. first
#                             breeding at ~2 y). veit: Veit & Lewis 1996 pair formation + nest-site limitation
#                             (K = site capacity, equilibrium emergent; yearlings breed).
#   covariate_drift           {"form": "intercept", "fields": [...], "n_basis", "budget", "window_start"}: a
#                             weak, smooth, time-varying shift of each listed rate's linear predictor
#                             (fields from adult_survival, fecundity, capacity, juvenile_survival). P0 of
#                             the E057 proxy-drift ladder; REPLACES k_trend when capacity is listed.
_DRIFT = dict(_POP_SPEC.get("covariate_drift", {}))
DRIFT_ON = bool(_DRIFT.get("form"))
if DRIFT_ON and _DRIFT["form"] not in ("intercept", "covariate"):
    raise NotImplementedError(f"covariate_drift.form {_DRIFT['form']!r}: 'intercept' (P0) or 'covariate' (P2)")
DRIFT_FIELDS = ("adult_survival", "fecundity", "capacity", "juvenile_survival")
# "fields" entries may be lists: a list is a TIED group sharing one set of weights (e.g. both survivals).
_DRIFT_GROUPS = tuple(tuple(g) if isinstance(g, (list, tuple)) else (g,)
                      for g in (_DRIFT.get("fields", DRIFT_FIELDS) if DRIFT_ON else ()))
_DRIFT_ON_FIELDS = tuple(f for g in _DRIFT_GROUPS for f in g)
if not set(_DRIFT_ON_FIELDS) <= set(DRIFT_FIELDS):
    raise ValueError(f"covariate_drift.fields must be a subset of {DRIFT_FIELDS}")
#   numerics                  {"fft_pad", "fft_floor_ulps", "smooth_allocation"}: zero-pad the dispersal FFT
#                             (legacy pads 1e-9, a pseudo-zero immigration) and drop convolution output
#                             below the FFT's resolution (float32 phantom floor, E054c). Absent = legacy.
_NUMERICS = dict(_POP_SPEC.get("numerics", {}))
if _NUMERICS:
    import src.model.age_forward as _af
    _af.FFT_PAD_VALUE = float(_NUMERICS.get("fft_pad", 1e-9))
    _af.FFT_FLOOR_ULPS = (None if _NUMERICS.get("fft_floor_ulps") is None
                          else float(_NUMERICS["fft_floor_ulps"]))
    _af.FFT_FLOOR_SMOOTH = bool(_NUMERICS.get("fft_floor_smooth", False))
    _af.FFT_FLOOR_FLOAT32_ONLY = bool(_NUMERICS.get("fft_floor_float32_only", False))
    _af.SMOOTH_ALLOC = (None if _NUMERICS.get("smooth_allocation") is None
                        else float(_NUMERICS["smooth_allocation"]))
import src.model.age_forward as _af_allee                    # allee_prior.aggregation_k (always set: no stale value)
_af_allee.ALLEE_AGG_K = ALLEE_AGG_K
#   occupancy                 {"enabled": bool, "birds_per_route_count": float}: E063 occupancy closure --
#                             each cell carries psi = P(>= 1 bird) across realisations; departure, disease and
#                             reproduction act on the occupied-conditional density m = N / psi; arrivals
#                             establish by Poisson chance. birds_per_route_count: birds in a 27-km cell per
#                             unit of expected route count (default 58 = 729/25 km2 sampled / detection .5).
#   readout                   {"split": bool, "ref_window": [y0, y1], "rank": int}: SPLIT readout -- the cell's
#                             long-run DESK level z-bar and its within-cell deviation z(t) - z-bar get separate
#                             manifold weights and amplitudes (DESK E059e/E012: one amplitude makes the temporal
#                             coefficients the spatial ones). rank: leading DESK features used (data_loading).
_READOUT = dict(_POP_SPEC.get("readout", {}))
SPLIT_READOUT = bool(_READOUT.get("split", False))
#   st_correction             {"enabled", "scale", "time": "linear"|"cosine", "n_time", "window_start"}: E067 smooth
#                             spatiotemporal habitat correction h(cell,t) = sum_b S_b(cell) w_b(t), shared by all four
#                             fields, carried into path habitat through the path operator; held before window_start.
_ST = dict(_POP_SPEC.get("st_correction", {}))
ST_CORRECTION = bool(_ST.get("enabled", False))
#   process_noise             {"enabled", "sigma", "n_freq", "window_start", "window_end", "detrend_degree",
#                             "active_years"}: E070 process error on
#                             abundance, N <- N * exp(eps_t), eps_t = sum_b S_b u_{b,t}, S a land-centred cosine basis
#                             scaled to unit land-RMS (data_loading), u iid N(0, sigma^2) per year inside the window
#                             (zero outside, and in held-out forecast years). sigma is PINNED (point estimates collapse or
#                             explode a learned noise scale); sampled non-centred as u = sigma * u_raw.
#   native_departure          {"enabled", "beta_a", "beta_b"} (E077): departure probability x (1 - r w(x)), w = the
#                             native core/margin map scaled to [0, 1], r ~ Beta(a, b). Western House Finches are largely
#                             sedentary; the continental kernel (Veit & Lewis) comes from the eastern invasion, whose
#                             population became partly migratory. r = 0 is the current model.
#   settlement_attraction     {"enabled", "a_scale" (.25), "eps" (.1), "adults" (true)}: conspecific attraction in
#                             settlement -- dispersers settle among the destinations their kernel reaches in proportion
#                             to kernel x (eps + N/K)^a (pre-dispersal N/K, capped at 1), mass-preserving per origin and
#                             kernel; journey survival unchanged. a ~ HalfNormal(a_scale), pinnable; a = 0 = current.
#   directional_choice        {"enabled", "kappa_scale" (.3)}: departing juveniles split across the 4 direction kernels
#                             of each band by softmax(kappa A) (mass-preserving per band; age_forward.
#                             directional_choice_weights), A = the juvenile-survival habitat logit AHEAD of the origin
#                             = the opposite direction's path feature: gamma_j Z_disp.beta_sj (+ gamma_a Z_disp.beta_s
#                             under juvenile_survival.form ratio). kappa ~ HalfNormal(kappa_scale) per logit unit,
#                             pinnable; kappa = 0 = current. Journey survival stays coupled.
#   path_compounding          {"enabled", "n_seg" (4), "reference" ("endpoint" | "mean"), "lambda_scale" (.25),
#                             "length_km" (100), "cell_km" (54)}: compounding path survival (DISPERSAL_EXPRESSIVENESS_PLAN
#                             design note): log Q += lambda sum_seg (L_k / (n_seg l)) min(0, log Qc(H_seg) - ref), Qc the
#                             current coupled survival, ref = the better END segment (interior segments costed) or the
#                             path mean (all segments). lambda ~ HalfNormal(lambda_scale), pinnable; 0 = current.
#   nest_likelihood           {"enabled", "path", "year_range", "max_precision", "broods": {"form": "pinned" | "lognormal",
#                             "values": [East, Plains, West], "log_sd"}, "cycle_days", "covariates", "mean_scale",
#                             "surv_scale", "dsr_logit_loc", "dsr_logit_scale", "log_conc_loc", "log_conc_scale",
#                             "floor_sharpness"}: a joint NestWatch nest-record block (FECUNDITY_CONSTRAINT_SCOPING_2026-10-31
#                             sec. 5). Mean fledged per attempt m = 2 F_real(cell, year) / B_region * exp(X delta_m), F_real the
#                             REALISED per-breeder fecundity of the forward simulation (density-dependent, Allee, at the
#                             simulated N), B = attempts per female per season (pinned per region from the literature, or
#                             lognormal about it). Hurdle: daily nest survival s = sigmoid(a_s + X delta_s) by logistic
#                             exposure (s^e_pre, failures x (1 - s^d_last)); full-cycle success pi = s^cycle_days; successful
#                             nests fledge 1 + NB2(mu1, phi) with mu1 = softplus_k(m / pi - 1), so pi (1 + mu1) = m. Nuisance
#                             delta_m ~ N(0, mean_scale) (tight), delta_s ~ N(0, surv_scale). Off = the current model exactly.
_NEST = dict(_POP_SPEC.get("nest_likelihood", {}))
NEST_LIKELIHOOD = bool(_NEST.get("enabled", False))
_COMP = dict(_POP_SPEC.get("path_compounding", {}))
PATH_COMPOUNDING = bool(_COMP.get("enabled", False))
if PATH_COMPOUNDING and _COMP.get("reference", "endpoint") not in ("endpoint", "mean"):
    raise ValueError("path_compounding.reference must be endpoint or mean")
_DIRCH = dict(_POP_SPEC.get("directional_choice", {}))
DIRECTIONAL_CHOICE = bool(_DIRCH.get("enabled", False))
_ATTR = dict(_POP_SPEC.get("settlement_attraction", {}))
ATTRACTION = bool(_ATTR.get("enabled", False))
_NATDEP = dict(_POP_SPEC.get("native_departure", {}))
NATIVE_DEPARTURE = bool(_NATDEP.get("enabled", False))
_PN = dict(_POP_SPEC.get("process_noise", {}))
PROCESS_NOISE = bool(_PN.get("enabled", False))
#   process_noise.target      "abundance" (default: N <- N e^eps) | "fecundity" (Fmax <- Fmax e^eps inside the forward
#                             simulation only: a transient deviation of the growth rate that compounds through
#                             reproduction; Fmax_flat, c and every lambda readout stay the skeleton's)
#                             | "low_density" (N <- N e^{eps h/(h + N/K)}, h = low_density_h, default .05: the nudge
#                             acts on colonising / front cells only -- an establishment correction)
#   process_noise.conserve    "none" (default) | "total" (continental bird count preserved by every nudge/reset) |
#                             "regional" (preserved within square blocks of conserve_block cells): the correction can
#                             MOVE birds but not create or remove them, so it cannot stand in for growth.
#   process_noise.skeleton_tau  (optional float): skeleton-consistency penalty. The forward simulation is also run
#                             WITHOUT the correction; every cell-year from the window start to the last fitted year
#                             adds -0.5 w (d / tau)^2, d = log(N_full + n0) - log(N_skeleton + n0), w = N_full / (N_full + n0),
#                             n0 = skeleton_n0 (default 1e-3 relative density): takeover is costed in the objective.
PN_TARGET = str(_PN.get("target", "abundance"))
if PN_TARGET not in {"abundance", "fecundity", "low_density"}:
    raise ValueError(f"process_noise.target must be abundance, fecundity or low_density, got {PN_TARGET!r}")
#   obs_region_effects        {"enabled", "sigma", "n_freq", "detrend_degree"}: year x region multipliers on EXPECTED
#                             COUNTS only (dynamics untouched): eff = sigma * u_t S^T, S land cosines, u iid N(0, 1)
#                             per year 1966 to the last fitted year (zero in held-out years); detrended per coefficient
#                             (default degree 1) so it carries transient regional count swings, not trends.
#   state_warp                {"enabled", "sigma_cells", "n_freq", "active_years"}: displacement reset -- in the listed
#                             years the state is moved by a smooth field d(x) = sigma_cells * (S u_y, S u_x), S land
#                             cosines (unit land-RMS), u ~ N(0, 1); total birds preserved (age_forward state_warp).
#   state_anchor              {"enabled", "year", "obs_years", "alpha_a", "alpha_b"}: observation-anchored reset -- at the
#                             end of `year` the state is blended toward the abundance the TRAINING counts of `obs_years`
#                             show (cell mean / pop_scalar; surveyed cells only), weight alpha ~ Beta(a, b).
_ANCHOR = dict(_POP_SPEC.get("state_anchor", {}))
STATE_ANCHOR = bool(_ANCHOR.get("enabled", False))
#   journey                   {"decouple": bool, "band_cost": bool, "offset_scale", "log_ratio_scale", "cost_scale"}:
#                             journey survival Q = sigmoid(alpha_j + a_off + gamma_j e^{g} H_path) * exp(-c_band). decouple
#                             frees a_off ~ N(0, offset_scale), g ~ N(0, log_ratio_scale) (0, 0 = Q tied to juvenile
#                             survival, the current model); band_cost adds c_band ~ HalfNormal(cost_scale) per radial band
#                             (journey mortality that grows with distance class; kernel k's band = k % n_bands).
_JOURNEY = dict(_POP_SPEC.get("journey", {}))
#                             distance_hazard (DISPERSAL_EXPRESSIVENESS_PLAN J1): ONE parameter h ~ HalfNormal(hazard_scale,
#                             default .05) per 100 km; band b's journey survival x exp(-h dbar_b / 100), dbar_b = band_mean_km
#                             (the kernel's mean distance within each radial band; required). h = 0 is the current model.
#                             water_hazard (J2): ONE parameter hw ~ HalfNormal(water_scale, default 1); kernel k's journey
#                             survival at destination p x exp(-hw W[p, k]), W = water share along the cohort's paths
#                             (mask-only, src/data/preprocess/journey_water_share.py; data['journey_water_W']).
#                             hw = 0 is the current model.
JOURNEY = bool(_JOURNEY.get("decouple", False) or _JOURNEY.get("band_cost", False)
               or _JOURNEY.get("distance_hazard", False) or _JOURNEY.get("water_hazard", False))
_WARP = dict(_POP_SPEC.get("state_warp", {}))
STATE_WARP = bool(_WARP.get("enabled", False))
_ORE = dict(_POP_SPEC.get("obs_region_effects", {}))
OBS_REGION_EFFECTS = bool(_ORE.get("enabled", False))
_OCC = dict(_POP_SPEC.get("occupancy", {}))
OCCUPANCY = bool(_OCC.get("enabled", False))
OCC_BIRDS_PER_COUNT = float(_OCC.get("birds_per_route_count", 58.0))
_REPRO = dict(_POP_SPEC.get("reproduction", {}))
VEIT_REPRO = _REPRO.get("form", "standard") == "veit"
YEARLING = bool(_REPRO.get("yearling_breeding", False)) or VEIT_REPRO
DEPART_BY_AGE = _DEPART_FORM == "age"
CROWDING_ARM = bool(_DEPART.get("crowding_arm", False))
CROWDING_KAPPA = float(_DEPART.get("crowding_sharpness", 5.0))
if _HABITAT_CENTERING not in {"none", "occupied"}:
    raise ValueError(f"habitat_centering must be none or occupied, got {_HABITAT_CENTERING!r}")
if _MANIFOLD_FORM not in {"rank2", "exchangeable", "capacity_demography"}:
    raise ValueError(f"unsupported manifold_prior.form: {_MANIFOLD_FORM!r}")
if _K_LINK not in {"softplus", "exp"}:
    raise ValueError(f"capacity_level_prior.link must be softplus or exp, got {_K_LINK!r}")
# THE GAUGE. Every absolute-scale prior is declared in expected BBS ROUTE COUNTS in
# config and divided by this at exactly one boundary, so changing the gauge cannot
# change any prior's meaning. (n50 already followed this convention; the capacity
# level, the Hill threshold, initpop and the invasion pulse now do too.) The old
# name said "birds", which it never was -- a BBS count is a 50-stop roadside index.
_POP_SCALAR = float(_POP_SPEC.get("population_scale_route_counts_per_relative_unit",
                                  _POP_SPEC.get("population_scale_birds_per_relative_unit")))


def _softplus_loc_for_median(target_route_counts, pop_scalar):
    """Prior location for a softplus-linked quantity, calibrated on its MEDIAN in ROUTE COUNTS.

    softplus is monotone, so median(softplus(X)) == softplus(median(X)) exactly for any scale.
    The location is therefore closed-form -- softplus^-1(target / pop_scalar) -- with no
    quadrature and no dependence on alpha_k_scale.

    Median, not mean: the empirical anchor is the median occupied cell (see
    scripts/diagnostics/bbs_abundance_quantiles.py), and matching a mean instead required
    converting through a lognormal ratio that softplus only approximates -- which landed the
    realized median 8.6% above the measured value. Matching the median directly is exact.

    Stating the target in counts and solving for the raw location is what keeps the "declare
    beliefs in route counts" rule: softplus does not commute with scaling, so unlike exp it
    cannot absorb the gauge as a shift, and changing pop_scalar re-derives the location rather
    than silently changing what the prior asserts.
    """
    density = float(target_route_counts) / float(pop_scalar)
    if density <= 0:
        raise ValueError(f"target must be positive; got {target_route_counts}")
    return float(np.log(np.expm1(density)))


def counts_to_relative(route_counts):
    """Convert an expected-BBS-route-count quantity to model density units.

    The single boundary where the gauge is applied. See the _scale_comment in
    config/age_model_config.json: the likelihood is exactly invariant to the gauge,
    so any prior stated in relative units is silently gauge-dependent and will
    change meaning when the gauge changes. State priors in counts; convert here.
    """
    return route_counts / _POP_SCALAR


# alpha_k's prior location, solved so the level's MEDIAN matches the configured target in
# route counts. Closed form, evaluated at import, so it tracks pop_scalar automatically.
# Under the exp link the median of exp(alpha_k) is exp(loc), so loc = log(target density).
# Under the crowding form alpha_k is log(kappa0), capacity per unit of surplus growth, so
# the target is met at the reference R0: K = kappa0 * (R0_ref - 1) = target.
_TARGET_DENSITY = float(_CAPACITY_LEVEL["target_level_median_route_counts"]) / _POP_SCALAR
if _K_FORM == "crowding":
    _ALPHA_K_LOC = float(np.log(_TARGET_DENSITY / (float(_CROWDING["reference_R0"]) - 1.0)))
elif _K_LINK == "exp":
    _ALPHA_K_LOC = float(np.log(_TARGET_DENSITY))
else:
    _ALPHA_K_LOC = _softplus_loc_for_median(
        _CAPACITY_LEVEL["target_level_median_route_counts"], _POP_SCALAR)


def k_link(x, link=None):
    """Capacity link: K = link(alpha_k + gamma_k*H_k + trend), in density space."""
    return jnp.exp(x) if (link or _K_LINK) == "exp" else jnn.softplus(x)


def crowding_capacity(alpha_k, H, trend, Sa, Sj, Fmax, bound=None, softness=None):
    """Growth-derived capacity: K = kappa(x) * exp(trend) * (R0 - 1)_+.

    R0 = Fmax*Sa*Sj/(1 - Sa) is the cell's mate-unlimited net reproductive number --
    the same quantity reproduction_age_structured's Beverton-Holt term uses as c + 1.
    With this K that term's c/K reduces to 1/kappa: classic Beverton-Holt crowding,
    F_eff = Fmax / (1 + N/kappa), equilibrium N* = kappa * (R0 - 1). So a cell can only
    hold birds by being a source; absence comes from R0 <= 1 (a sink), never from an
    independent capacity field driven to zero (the run_18/19 'K switch': 40% of
    lambda-suitable cells Allee-dead).

    kappa(x) = exp(alpha_k + B*tanh(H/B)): the habitat field H (K's manifold column) can
    shift abundance by at most a factor e^B around what growth implies, and cannot create
    absence. B = 0 drops it (constant crowding). (x)_+ is smoothed at scale ``softness`` so
    the potential stays differentiable at R0 = 1.
    """
    bound = float(_CROWDING.get("modifier_bound", 0.0)) if bound is None else bound
    softness = float(_CROWDING.get("softness", 0.02)) if softness is None else softness
    R0 = Fmax * Sa * Sj / (1.0 - Sa + 1e-6)
    surplus = softness * jnn.softplus((R0 - 1.0) / softness)
    modifier = bound * jnp.tanh(H / bound) if bound > 0 else 0.0
    return jnp.exp(alpha_k + modifier + trend) * surplus


def validate_environment_kernel_contract(data):
    """Reject model inputs that would not represent the intended GP feature map."""
    from src.community_encoder.cube_contract import validate_model_feature_contract
    contract = data.get("z_kernel_contract")
    if not contract:
        raise ValueError("model inputs lack z_kernel_contract; rerun scripts/ingest_model_data.py")
    actual = int(data["Z_gathered"].shape[-1])
    return validate_model_feature_contract(contract, actual)


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


if PATH_COMPOUNDING and (SPLIT_READOUT or HABITAT_SHRINK or _JOURNEY.get("decouple", False)):
    raise NotImplementedError("path_compounding is tied to the coupled survival on the plain readout "
                              "(not readout.split, habitat_change or journey.decouple)")
if DIRECTIONAL_CHOICE and (ATTRACTION or SPLIT_READOUT or HABITAT_SHRINK or ST_CORRECTION):
    raise NotImplementedError("directional_choice is wired for the plain readout without attraction / split / "
                              "habitat_change / st_correction")
if ATTRACTION and (SETTLE_BLOCK or OCCUPANCY):
    raise NotImplementedError("settlement_attraction is not combined with ideal-free settlement or occupancy")
if OCCUPANCY and (SETTLE_BLOCK or SUBCELL):
    raise NotImplementedError("occupancy is not wired into ideal-free settlement or subcell_strata")

if NEST_LIKELIHOOD and (VEIT_REPRO or SUBCELL):
    raise NotImplementedError("nest_likelihood reads F_actual of the standard reproduction form (not veit / subcell_strata)")
if NEST_LIKELIHOOD and str(dict(_NEST.get("broods") or {}).get("form", "pinned")) not in ("pinned", "lognormal"):
    raise ValueError("nest_likelihood.broods.form must be pinned or lognormal")


def nest_log_likelihood(F_i, B_i, X, delta_m, a_s, delta_s, log_conc, outcome, fledged, e_pre, d_last, w_exp, w_cnt,
                        cycle_days=32.0, sharpness=10.0):
    """Per-attempt nest log-likelihood terms (exposure, count) and (m, pi, mu1). Pure: tests call it directly.

    m = 2 F / B exp(X delta_m) fledged per attempt; log s = log_sigmoid(a_s + X delta_s) daily nest survival;
    exposure: w_exp [e_pre log s + 1[failed] log(1 - s^d_last)]; pi = s^cycle_days; count (successful nests with a
    fledged count): w_cnt log NB2(fledged - 1 | mu1, phi), mu1 = softplus(k (m/pi - 1)) / k.
    """
    m = 2.0 * F_i / B_i * jnp.exp(X @ delta_m)
    logs = jax.nn.log_sigmoid(a_s + X @ delta_s)
    failed = outcome == 0
    d_safe = jnp.where(failed & (w_exp > 0), d_last, 1.0)
    ll_exp = w_exp * (e_pre * logs + jnp.where(failed, jnp.log(-jnp.expm1(d_safe * logs)), 0.0))
    pi = jnp.exp(cycle_days * logs)
    mu1 = jax.nn.softplus(sharpness * (m / pi - 1.0)) / sharpness
    y1 = jnp.where(w_cnt > 0, fledged - 1, 0)
    ll_cnt = w_cnt * dist.NegativeBinomial2(mean=jnp.maximum(mu1, 1e-12), concentration=jnp.exp(log_conc)).log_prob(y1)
    return ll_exp, ll_cnt, m, pi, mu1


def _nest_site(name, d):
    if name in PINNED:
        return numpyro.deterministic(name, jnp.asarray(PINNED[name]))
    return numpyro.sample(name, d)


# Paths whose growth algebra still assumes adult-only breeding (lambda0 from the 2x2 delayed-
# maturity matrix): refuse them rather than mix life cycles silently.
if YEARLING and (SETTLE_WEIGHT not in ("capacity", "uniform") or SUBCELL or _K_FORM == "crowding"):
    raise NotImplementedError(
        "reproduction.yearling_breeding / form=veit is not wired into fitness/r0 settlement, "
        "subcell_strata or crowding capacity (they assume adult-only breeding)")


def equilibrium_age_quantities(Sa, Sj, Fmax, K, allee_gamma):
    """Return density brake, fecundity-at-K, growth rate, and juvenile fraction.

    Algebra matches :func:`reproduction_age_structured`: surviving adults
    reproduce, so the local projection matrix is ``[[Sa,Sj],[F*Sa,0]]``.
    """
    # Relative Allee: mate-finding depends on N/K, which is 1 at K.
    allee_at_K = allee_mated_fraction(allee_gamma * (1.0 if ALLEE_RELATIVE else K))   # default: 1 - exp(-x), as before
    if VEIT_REPRO:
        # pairs + nest sites: per-breeder births at B = K (K is site capacity, not N*)
        r = 2.0 * jnp.log(2.0) / allee_gamma * (K if ALLEE_RELATIVE else 1.0)
        P = K * K / (r + 2.0 * K + 1e-12)
        F_at_K = 2.0 * Fmax * P / (1.0 + 2.0 * P / jnp.maximum(K, 1e-12)) / jnp.maximum(K, 1e-12)
        c = jnp.maximum((Fmax * Sj) / (1.0 - Sa + 1e-6) - 1.0, 0.0)
        lam = Sa + Sj * F_at_K
        return c, F_at_K, lam, F_at_K / (1.0 + F_at_K)
    if YEARLING:
        # yearlings breed: lambda = Sa + Sj*F, replacement F = (1 - Sa)/Sj. (Under veit the
        # brake c is unused by the forward model; reported for the age-structure prior only.)
        c = jnp.maximum((Fmax * Sj) / (1.0 - Sa + 1e-6) - 1.0, 0.0)
        F_at_K = Fmax / (1.0 + c) * allee_at_K
        lam = Sa + Sj * F_at_K
        rho = F_at_K / (1.0 + F_at_K)
        return c, F_at_K, lam, rho
    c = jnp.maximum((Fmax * Sa * Sj) / (1.0 - Sa + 1e-6) - 1.0, 0.0)
    F_at_K = Fmax / (1.0 + c) * allee_at_K
    lam = (Sa + jnp.sqrt(Sa**2 + 4.0 * F_at_K * Sa * Sj)) / 2.0
    rho = (F_at_K * Sa) / (F_at_K * Sa + lam)
    return c, F_at_K, lam, rho


def _exchangeable_w_env(M_features, w_scale, prior_scale):
    """EXCHANGEABLE manifold prior: one coupling rho shared by all four fields.

        beta_j = w_scale_j * (sqrt(rho) * f + sqrt(1 - rho) * eps_j),  f, eps iid over features

    so Var(beta_j) = w_scale_j^2 and Corr(j, k) = rho for EVERY pair, exactly. The
    uncentered-Ruzicka GP contract holds unchanged: f and eps are iid across features.

    Why it replaced rank-2 for run_19: under rank-2 each field has its own communality and
    angle, so the fit could decouple ONE field -- the run_18 HMC descent took reproduction's
    communality from 0.76 to 0.27 and juvenile survival's to 0.43 while adult survival and
    capacity stayed coupled, which is what let lambda and K come apart (K turned into an
    absence switch). With a single rho, loosening one field's coupling loosens all of them.
    Loses: within-group vs cross-group structure (all pairs share rho) and trade-offs
    (rho > 0). Also drops the angle/rotation directions that made up part of the
    manifold block's negative curvature at the run_17/18 MAP.
    """
    _m = _MANIFOLD_PRIOR
    fixed = _m.get("fixed_coupling")
    if fixed is not None:
        # SINGLE HABITAT FIELD at fixed_coupling = 1: beta_j = w_scale_j * f, so every rate
        # reads the same field H = Z.f with its own positive sensitivity w_scale_j. No
        # private parts are sampled (they would carry zero weight).
        rho_c = numpyro.deterministic("manifold_coupling", jnp.asarray(float(fixed)))
    else:
        target = float(_m["coupling_target"])
        logit_target = math.log(target / (1.0 - target))
        rho_c = numpyro.deterministic("manifold_coupling", jnn.sigmoid(numpyro.sample(
            "manifold_coupling_logit",
            dist.Normal(logit_target, float(_m["coupling_logit_scale"]) * prior_scale))))
    with numpyro.plate("env_features", M_features, dim=-2):
        f_shared = numpyro.sample("manifold_factor", dist.Normal(0.0, 1.0))   # (M, 1)
        if fixed is None or float(fixed) < 1.0:
            with numpyro.plate("manifolds", 4, dim=-1):
                eps_idio = numpyro.sample("manifold_idio", dist.Normal(0.0, 1.0))  # (M, 4)
        else:
            eps_idio = jnp.zeros((M_features, 4))
    w_env = numpyro.deterministic(
        "w_env", w_scale[None, :] * (jnp.sqrt(rho_c) * f_shared + jnp.sqrt(1.0 - rho_c) * eps_idio))
    # The rank-2 reporting names, filled with their exchangeable equivalents (every field at
    # communality sqrt(rho) on one factor, angle 0), so diagnostics read unchanged.
    r = jnp.sqrt(rho_c) * jnp.ones(4)
    numpyro.deterministic("manifold_communality", r)
    L_load = numpyro.deterministic("manifold_loadings", jnp.stack([r, jnp.zeros(4)], axis=-1))
    return w_env, L_load @ L_load.T


def _capacity_demography_w_env(M_features, w_scale, prior_scale):
    """One shared demographic direction and a distinct capacity direction.

    Unlike exchangeable, only K can depart from the demographic direction.
    Unlike rank2, Fmax stays with survival. Existing fixed_coupling settings do
    not configure this form: its separate correlation must be explicit.
    """
    spec = _MANIFOLD_PRIOR
    fixed = spec.get("fixed_capacity_correlation")
    if fixed is not None:
        if not 0.0 <= float(fixed) <= 1.0:
            raise ValueError("fixed_capacity_correlation must be in [0, 1]")
        r = jnp.asarray(float(fixed))
    else:
        target = float(spec["capacity_correlation_target"])
        if not 0.0 < target < 1.0:
            raise ValueError("capacity_correlation_target must be in (0, 1)")
        scale = float(spec["capacity_correlation_logit_scale"])
        if scale <= 0.0:
            raise ValueError("capacity_correlation_logit_scale must be positive")
        r = jnn.sigmoid(numpyro.sample(
            "capacity_correlation_logit",
            dist.Normal(math.log(target / (1.0 - target)), scale * prior_scale)))
    r = numpyro.deterministic("capacity_demography_correlation", r)
    with numpyro.plate("env_features", M_features, dim=-2):
        shared = numpyro.sample("manifold_factor", dist.Normal(0.0, 1.0))
        # At r=1 the old single-field model is recovered with identical sites.
        # Do not leave an unused capacity factor in the optimizer vector.
        capacity = (numpyro.sample("capacity_factor", dist.Normal(0.0, 1.0))
                    if fixed is None or float(fixed) < 1.0 else jnp.zeros_like(shared))
    loadings = numpyro.deterministic("manifold_loadings", capacity_demography_loadings(r))
    numpyro.deterministic("manifold_communality", jnp.ones(4))
    weights = numpyro.deterministic(
        "w_env", capacity_demography_coefficients(shared, capacity, w_scale, r))
    return weights, loadings @ loadings.T


def sample_priors(prior_scale=1.0, M_features=None, time=None,
                  N_sev_basis=None, N_lag_basis=None):
    """Sample every model parameter and return them in a dict.

    Covers the correlated 4-manifold habitat weights (adult survival, reproduction,
    capacity, juvenile survival, via a rank-2 angular prior with deliberately ordered
    correlations -- two tightly-coupled pairs, weaker coupling across them), the
    continental time trend on K, the structured
    mycoplasmal-conjunctivitis effect on K, dispersal/demography rate parameters,
    and the Allee term. ``prior_scale`` multiplies scale parameters for
    continuation fitting; ``M_features``, ``time``, and the two disease basis
    sizes set the dimensions.
    """
    priors = {}
    
    # --- 1. CORRELATED 4-MANIFOLD HABITAT WEIGHTS ---
    # [survival_adult, reproduction, capacity, survival_juv]
    # H_s = Z.beta_s drives Sa, H_sj = Z.beta_sj drives Sj and Q, H_r drives Fmax,
    # H_k drives K.
    # K previously reused beta_r outright -- an implicit correlation of exactly 1.0 --
    # so K's spatial pattern could differ from Fmax's only via the disease term. That
    # forced every real disagreement between "where reproduction is good" and "where
    # birds are abundant" to be spelled "disease", and the disease term duly
    # saturated. Giving K its own weights makes that disagreement a covariate
    # statement, still prior-coupled to the others.
    #
    # Two properties the construction must have, and does: the implied covariance is
    # positive-definite for ANY parameter values, so there is no Cholesky guard whose
    # clipping would corrupt gradients (LKJCholesky is not usable here -- its
    # concentration > 1 concentrates near the IDENTITY correlation, i.e. near zero
    # correlation, the opposite of the prior belief); and the factors are drawn IID
    # ACROSS FEATURES, which is what preserves the uncentered-Ružička GP contract. See
    # validate_environment_kernel_contract.
    _m = _MANIFOLD_PRIOR
    # RANK-2 ANGULAR solve. Column order is
    #     [survival_adult, reproduction, capacity, survival_juv]
    # -- survival_juv is APPENDED, not inserted at 1. Eight viz/analysis modules index w_env
    # positionally (w_env[:,1] == "reproduction", w_env[:,2] == "capacity"); inserting would have
    # relabelled both with no error raised anywhere.
    #
    # Two groups {survival_adult, survival_juv} and {reproduction, capacity} sit at +/- phi/2.
    # With r_j the communality and th_j the angle, Corr(j,k) = r_j*r_k*cos(th_j - th_k), so
    #   within a group (th_j == th_k):  Corr = r_j*r_k          -> r = sqrt(rho_within)
    #   across groups:                  Corr = r^2*cos(phi)     -> cos(phi) = rho_cross/rho_within
    # A single angle_scale then makes within-pair correlation LESS prior-uncertain than
    # cross-group automatically, because cos is flat at zero separation -- which is exactly the
    # requested structure ("tight within pair, weaker but informative across") from one knob.
    _rho_within = 0.5 * (float(_m["target_corr_survival_adult_juv"])
                         + float(_m["target_corr_repro_capacity"]))
    _rho_cross = float(_m["target_corr_cross_group"])
    _r_med = math.sqrt(_rho_within)
    _phi = math.acos(min(max(_rho_cross / _rho_within, -1.0), 1.0))
    _logit_r = math.log(_r_med / (1.0 - _r_med))
    _th_med = jnp.array([-_phi / 2, _phi / 2, _phi / 2, -_phi / 2])  # Sa, F, K, Sj

    if _MANIFOLD_FORM == "rank2":
        # Communality per field, logit-normal so r stays in (0,1) and no correlation can leave
        # [-1,1] and the idiosyncratic weight sqrt(1-r^2) stays real.
        r_load = numpyro.deterministic(
            "manifold_communality",
            jnn.sigmoid(numpyro.sample(
                "manifold_communality_raw",
                dist.Normal(_logit_r, float(_m["communality_logit_scale"]) * prior_scale)
                .expand([4]))),
        )
        th_load = numpyro.sample(
            "manifold_angle",
            dist.Normal(_th_med, float(_m["angle_scale"]) * prior_scale))
        # (4, 2) factor loadings. Emitted under the historical `manifold_loadings` name, now a
        # matrix rather than a vector -- map_diagnostics writes it to JSON as a flat list.
        L_load = numpyro.deterministic(
            "manifold_loadings",
            jnp.stack([r_load * jnp.cos(th_load), r_load * jnp.sin(th_load)], axis=-1))

    # GP AMPLITUDE, one per manifold, and now the SOLE amplitude: the per-rate slopes
    # are fixed at 1 (see the gamma block below), because w_scale and gamma multiplied
    # the same field and only their product entered the likelihood -- three exactly
    # flat ridge directions, harmless-ish under MAP but ruinous for HMC step-size
    # adaptation. Per-field locs/scales are calibrated to reproduce the amplitude prior
    # that the gamma*w_scale product used to imply, matched on the 5th/95th percentiles
    # (see the _amplitude_comment in config). They differ by field on purpose.
    #
    # softplus(Normal), NOT lognormal: lognormal's tails are ~2x heavier at the 99.9th
    # percentile here, and those tails are what destabilize this model under HMC.
    w_scale = numpyro.deterministic(
        "w_scale",
        jnn.softplus(numpyro.sample(
            "w_scale_raw",
            dist.Normal(jnp.asarray(_m["amplitude_loc"], dtype=float),
                        jnp.asarray(_m["amplitude_scale"], dtype=float) * prior_scale))),
    )
    # Now truthful for ALL FOUR fields: every per-rate slope is fixed at 1 (gamma_j included --
    # see the gamma block), so this IS the amplitude of each latent field.
    numpyro.deterministic("environment_kernel_variance", w_scale ** 2)

    # dim=-2 so the feature plate takes the ROW axis and leaves the rightmost axis for the
    # manifolds; the factors come out (M, 2) and the idiosyncratic draws (M, 4).
    #
    # IID ACROSS FEATURES is the load-bearing property, unchanged from the rank-1 version: with
    # f1, f2 and eps all iid over features, beta_j[m] is iid over m with variance w_scale_j^2
    # exactly (r^2 + (1-r^2) = 1), so H_j = Z.beta_j is still a GP with kernel
    # w_scale_j^2 * Z(x).Z(x') and validate_environment_kernel_contract's isotropic-feature
    # requirement still holds. A full covariance over FEATURES would have broken it; adding a
    # second FACTOR does not.
    # The rightmost axis is plated too (factors / manifolds): numpyro warned "Missing a
    # plate statement for batch dimension -1" on every run when it was a bare .expand().
    # Same shapes, same log density.
    if _MANIFOLD_FORM == "rank2":
        with numpyro.plate("env_features", M_features, dim=-2):
            with numpyro.plate("manifold_factors", 2, dim=-1):
                f_shared = numpyro.sample("manifold_factor", dist.Normal(0.0, 1.0))
            with numpyro.plate("manifolds", 4, dim=-1):
                eps_idio = numpyro.sample("manifold_idio", dist.Normal(0.0, 1.0))
        w_env = numpyro.deterministic(
            "w_env",
            w_scale[None, :] * (f_shared @ L_load.T
                                + jnp.sqrt(1.0 - (L_load ** 2).sum(-1))[None, :] * eps_idio),
        )
        corr = L_load @ L_load.T
    elif _MANIFOLD_FORM == "capacity_demography":
        w_env, corr = _capacity_demography_w_env(M_features, w_scale, prior_scale)
    else:
        w_env, corr = _exchangeable_w_env(M_features, w_scale, prior_scale)

    # Report the implied correlations (and a 4x4 Cholesky under the historical L_corr name) so
    # diagnostics can read what the fit concluded about how tightly the manifolds move together.
    corr = corr + jnp.diag(1.0 - jnp.diag(corr))
    # Reporting only. With fixed_coupling = 1 every pair correlates exactly 1, the matrix is
    # singular and the bare Cholesky is NaN -- harmless to the fit, but it is the first NaN
    # JAX_DEBUG_NANS reports, which hid run_22's real failure. A 1e-6 jitter is invisible
    # in the reported values.
    numpyro.deterministic("L_corr", jnp.linalg.cholesky(corr + 1e-6 * jnp.eye(4)))
    numpyro.deterministic("rho", corr[0, 1])  # survival-reproduction, the old `rho`
    numpyro.deterministic("env_corr_repro_capacity", corr[1, 2])
    numpyro.deterministic("env_corr_survival_capacity", corr[0, 2])
    # The new one: adult vs juvenile survival, the pair this rank-2 extension exists to couple.
    numpyro.deterministic("env_corr_survival_adult_juv", corr[0, 3])

    if SPLIT_READOUT:
        # Within-cell deviation block: an independent exchangeable draw (same coupling rho) with its
        # OWN amplitudes, same prior as the level block -- the data decide how much the deviation counts.
        if _MANIFOLD_FORM != "exchangeable":
            raise NotImplementedError("readout.split is wired for manifold_prior.form = exchangeable")
        if _READOUT.get("within_tied"):
            # TIED deviation readout (E080g): beta_dev = c * beta_level, c = within_ratio_fixed -- a scaled-down
            # space-for-time substitution. A cell's change in DESK features over time acts on habitat exactly as the
            # same feature difference does across space, at strength c; nothing about the temporal relationship is fitted.
            if _READOUT.get("within_ratio_fixed") is None:
                raise ValueError("readout.within_tied needs readout.within_ratio_fixed (the scaling c)")
            w_env_w = numpyro.deterministic("w_env_within", float(_READOUT["within_ratio_fixed"]) * w_env)
            numpyro.deterministic("w_scale_within", w_scale * float(_READOUT["within_ratio_fixed"]))
            priors['beta_s_w'], priors['beta_r_w'] = w_env_w[:, 0], w_env_w[:, 1]
            priors['beta_k_w'], priors['beta_sj_w'] = w_env_w[:, 2], w_env_w[:, 3]
        else:
            _wr = _READOUT.get("within_ratio_prior")
            if _READOUT.get("within_ratio_fixed") is not None:
                # Deviation amplitude FIXED at ratio x level (E080): 0 = STATIC habitat (each cell keeps its long-run DESK
                # level; no temporal habitat change through the readout); .14 = DESK's own measured between/within split.
                w_scale_w = numpyro.deterministic(
                    "w_scale_within", w_scale * float(_READOUT["within_ratio_fixed"]) * jnp.ones(4))
            elif _wr:
                # Deviation amplitude = level amplitude x ratio, ratio ~ LogNormal centred on DESK's measured
                # between/within split (E059e: kernel variances ~50x apart -> sd ratio ~ sqrt(1/50) ~ .14). DESK's change is
                # poorly aimed, so the deviation block should count for much less than the level unless the data insist.
                ratio = numpyro.deterministic("within_ratio", jnp.exp(numpyro.sample(
                    "within_ratio_log", dist.Normal(float(_wr.get("loc_log", math.log(0.14))),
                                                    float(_wr.get("scale_log", 0.7)) * prior_scale).expand([4]))))
                w_scale_w = numpyro.deterministic("w_scale_within", w_scale * ratio)
            else:
                w_scale_w = numpyro.deterministic("w_scale_within", jnn.softplus(numpyro.sample(
                    "w_scale_within_raw",
                    dist.Normal(jnp.asarray(_m["amplitude_loc"], dtype=float),
                                jnp.asarray(_m["amplitude_scale"], dtype=float) * prior_scale))))
            rho_w = corr[0, 1]                 # the level block's coupling rho (exchangeable: every off-diagonal)
            with numpyro.plate("env_features_within", M_features, dim=-2):
                f_w = numpyro.sample("manifold_factor_within", dist.Normal(0.0, 1.0))
                with numpyro.plate("manifolds_within", 4, dim=-1):
                    e_w = numpyro.sample("manifold_idio_within", dist.Normal(0.0, 1.0))
            u_w = jnp.sqrt(rho_w) * f_w + jnp.sqrt(1.0 - rho_w) * e_w           # unit-scale own directions
            _wc = _READOUT.get("within_corr")
            if _wc:
                # CORRELATED deviation readout (E080h): the temporal coefficients lean on the spatial ones,
                # u_dev = r u_level + sqrt(1 - r^2) u_own, r ~ Beta(a, b) per field (r = 1 tied, r = 0 independent).
                r_w = numpyro.sample("within_level_corr", dist.Beta(float(_wc.get("a", 8.0)),
                                                                     float(_wc.get("b", 2.0))).expand([4]))
                u_level = w_env / jnp.maximum(w_scale[None, :], 1e-12)
                u_w = r_w[None, :] * u_level + jnp.sqrt(1.0 - r_w[None, :] ** 2) * u_w
            w_env_w = numpyro.deterministic("w_env_within", w_scale_w[None, :] * u_w)
            priors['beta_s_w'], priors['beta_r_w'] = w_env_w[:, 0], w_env_w[:, 1]
            priors['beta_k_w'], priors['beta_sj_w'] = w_env_w[:, 2], w_env_w[:, 3]
    priors['beta_s'] = w_env[:, 0]   # Adult survival suitability weights
    priors['beta_r'] = w_env[:, 1]   # Reproductive suitability weights
    priors['beta_k'] = w_env[:, 2]   # Carrying-capacity weights
    priors['beta_sj'] = w_env[:, 3]  # Juvenile survival -- its OWN manifold (drives Sj and Q)

    # --- 1a. CONTINENTAL TIME TREND ON K: A SAFETY VALVE, NOT A MECHANISM ---
    # This is not a modeled process and is not meant to carry signal. It exists only
    # because the model otherwise has NO temporal degree of freedom for capacity
    # besides the disease term, so "modern K below 1970s K" could only be expressed
    # as disease -- and the disease term saturated at 93% severity trying. The prior
    # is therefore very strongly concentrated on ZERO: the default budget keeps the
    # implied K multiplier within ~2% of 1.0 at the median and ~7% at the 95th
    # percentile.
    #
    # A fitted trend that escapes that range is a FAILURE SIGNAL: it means a real
    # temporal signal is not being explained by any mechanism in the model. The
    # correct response is to find the missing mechanism, NOT to loosen this budget.
    # scripts/viz/map_diagnostics.py reports k_trend.max_prior_sd_multiple so this
    # can be read off directly.
    #
    # Spatially UNIFORM by construction, with only a few time-centered cosines, so it
    # cannot compete with the disease term's spatial pattern or manufacture
    # year-to-year wiggle.
    k_trend_budget = float(_K_TREND["budget"]) * prior_scale
    n_k_trend = int(_K_TREND["n_basis"])
    if ("capacity" in _DRIFT_ON_FIELDS and _DRIFT.get("form") == "intercept") or _K_TREND.get("enabled", True) is False:
        # covariate_drift owns K's temporal intercept; never both (two names for one direction).
        # k_trend.enabled = false (E080 static habitat): no continental K trend at all.
        priors['w_k_trend'] = numpyro.deterministic("w_k_trend", jnp.zeros(n_k_trend))
    else:
        priors['w_k_trend'] = numpyro.sample(
            "w_k_trend",
            dist.Normal(0.0, k_trend_budget / jnp.sqrt(n_k_trend)).expand([n_k_trend]),
        )

    # --- 1b. MYCOPLASMAL-CONJUNCTIVITIS EFFECT ON K ---
    # K = K_base * (1 - severity(x) * gate(x,t) * (1 - recovery(t - arrival))).
    # See age_fields.py's module docstring for the formulation and for why the
    # previous generic spatiotemporal field annihilated eastern K. Every parameter
    # below states a claim about the epizootic that someone could dispute; that is
    # the point of the structure. All scales carry prior_scale for continuation.
    _p = _DISEASE_PRIOR

    # SEVERITY: logit of the MODIFIER on density-dependent severity, not of the removed
    # fraction itself -- severity = severity_ceiling * sigmoid(mu + ...) * hill(K_base).
    # mu_loc = logit(0.8) = 1.386 puts the prior median modifier at 0.8, i.e. a median
    # severity of 0.8 * 0.5 = 40% in the densest populations; mu_scale=0.5 gives a 90% CI
    # on the modifier of ~[0.62, 0.91] -> severity [31%, 46%]. See the config's
    # disease_prior._mu_comment for why that brackets the documented eastern decline from
    # slightly below, leaving the ceiling a strict bound rather than the expected value.
    priors['disease_mu_sev'] = numpyro.sample(
        "disease_mu_sev",
        dist.Normal(float(_p["mu_loc"]), float(_p["mu_scale"]) * prior_scale),
    )
    # EPIDEMIC-ATTENUATION: cells reached later in the epidemic's own history
    # (decades after disease_start_year=1993, NOT decades after that cell's own
    # local arrival -- see model_inputs' disease_onset_decades) were plausibly hit
    # less hard, because the documented pattern for mycoplasmal conjunctivitis is
    # pathogen/host co-evolution making it progressively milder since 1993 -- a
    # calendar-time trend, not a claim about any cell's own population genetics
    # (that is disease_b_native below). One coefficient per decade -- cheaper and
    # far more interpretable than asking the spatial field to rediscover a pattern
    # that is essentially the epidemic's own age.
    priors['disease_b_late'] = numpyro.sample(
        "disease_b_late",
        dist.Normal(float(_p["late_arrival_loc"]),
                    float(_p["late_arrival_scale"]) * prior_scale),
    )
    # NATIVE-LINEAGE resistance: the west coast population is genetically diverse
    # (the ancestral range), unlike the east's single-founder 1940 introduction, and
    # was documented to collapse far less under the epizootic. This is a DISTINCT
    # claim from b_late above -- population origin, not arrival timing -- carried by
    # its own coefficient on native_shape (the same core/margin map used to seed 1902
    # abundance: 1 in the native core, margin_fraction_of_core at the fringe, 0
    # elsewhere). SIGN-CONSTRAINED via -softplus(raw) <= 0: native lineage may only
    # SUPPRESS severity, never amplify it -- the documented asymmetry is one-
    # directional (west spared, not "east extra hit"), and constraining the sign is
    # what turns "the west did less badly" from a pattern the fit might or might not
    # find into a structural, falsifiable claim (see age_fields.py's module
    # docstring). raw_loc=0 puts the prior median suppression at softplus(0)=0.69
    # logit units in the native core, comparable in scale to the modifier's other
    # terms; raw_scale=0.7 lets the fit push that from near 0 to a strong effect.
    priors['disease_b_native'] = numpyro.deterministic(
        "disease_b_native",
        -jnn.softplus(numpyro.sample(
            "disease_b_native_raw",
            dist.Normal(float(_p["native_resistance_raw_loc"]),
                        float(_p["native_resistance_raw_scale"]) * prior_scale))),
    )
    # Regional severity deviation. Land-centered basis (see
    # model_inputs.generate_spatial_basis), so these can only redistribute around
    # disease_mu_sev, never shift its level -- which is what keeps the continental
    # severity reportable and stops it trading against alpha_k.
    sev_scale = float(_p["sev_field_budget"]) * prior_scale / jnp.sqrt(N_sev_basis)
    priors['disease_w_sev'] = numpyro.sample(
        "disease_w_sev", dist.Normal(0.0, sev_scale).expand([N_sev_basis])
    )

    # ONSET TIMING SLACK. The arrival surface is a smoothed reconstruction from the
    # documented spread history, coarse both continentally (its kernel smoother
    # shrinks the 1994 mid-Atlantic and 2006 California extremes inward) and
    # regionally. Fitting the slack keeps that imprecision from being absorbed as
    # wrong severity instead.
    priors['disease_lag0'] = numpyro.sample(
        "disease_lag0", dist.Normal(0.0, float(_p["lag0_scale"]) * prior_scale))
    lag_scale = float(_p["lag_field_budget"]) * prior_scale / jnp.sqrt(N_lag_basis)
    priors['disease_w_lag'] = numpyro.sample(
        "disease_w_lag", dist.Normal(0.0, lag_scale).expand([N_lag_basis])
    )
    # DENSITY DEPENDENCE of severity: the Hill threshold and steepness, both fitted,
    # so the shape is learned. k_half is a DENSITY, hence declared in route counts and
    # converted through the gauge like every other absolute-scale quantity.
    priors['disease_k_half'] = numpyro.deterministic(
        "disease_k_half",
        counts_to_relative(jnp.exp(numpyro.sample(
            "disease_log_k_half_counts",
            dist.Normal(jnp.log(float(_p["k_half_median_route_counts"])),
                        float(_p["k_half_log_sd"]) * prior_scale)))),
    )
    numpyro.deterministic("disease_k_half_route_counts",
                          priors['disease_k_half'] * _POP_SCALAR)
    # Steepness. n ~ 1 is smooth saturation (mass-action-ish); n ~ 3 a sharp
    # invasion threshold. BOUNDED, and the bound is DERIVED rather than guessed:
    # K = K_base*(1 - ceiling*hill(K_base)) must be monotone in K_base or two
    # habitat qualities map to one realized capacity. At the worst point
    # (K_base = k_half) the slope is 1 - ceiling*(1/2 + n/4), giving n < 4/ceiling - 2
    # (= 6.0 at ceiling 0.5). That closed form is slightly OPTIMISTIC because the
    # true worst point sits a little below k_half -- numerically n=5.9 already goes
    # non-monotone (slope -0.009) at ceiling 0.5 -- so an 0.85 margin is applied.
    # Enforcing this structurally means the ceiling and the steepness cannot drift
    # into an unidentifiable combination: raising the ceiling automatically tightens
    # the allowed steepness, which a LogNormal prior would happily ignore.
    # (disease_prior.target = survival: the same monotonicity argument applies to survivors
    # S*N*(1 - m(N)) in N, with the mortality ceiling in place of the severity ceiling.)
    _n_max = 0.85 * (4.0 / _DISEASE_CEILING - 2.0)
    _n_min = 0.5
    _n_med = float(_p["hill_n_median"])
    _n_raw_loc = math.log((_n_med - _n_min) / (_n_max - _n_med))
    priors['disease_hill_n'] = numpyro.deterministic(
        "disease_hill_n",
        _n_min + (_n_max - _n_min) * jnn.sigmoid(numpyro.sample(
            "disease_hill_n_raw",
            dist.Normal(_n_raw_loc, float(_p["hill_n_log_sd"]) * 2.0 * prior_scale))),
    )

    priors['disease_tau'] = numpyro.deterministic(
        "disease_tau",
        # Onset sharpness. Floored because a vanishing tau makes the gate a step
        # function, whose gradient w.r.t. the lag terms vanishes everywhere the
        # front is not exactly at t. Also absorbs per-cell timing scatter that the
        # smooth regional lag field cannot resolve, by widening the front.
        (jnn.softplus(numpyro.sample("disease_tau_raw",
                                    dist.Normal(0.5, 0.5 * prior_scale))) + 0.25
         if DISEASE_ONSET_SPEC == {"form": "sigmoid"}
         else jnp.asarray(DISEASE_ONSET_SPEC.get("tau_years", 1.0))),
    )

    # RECOVERY: slowly increasing resilience after local arrival (exposure, not
    # necessarily evolution). The hit relaxes from its peak toward
    # severity*(1-rec) with an e-folding time tau_rec. rec's prior median is ~38%
    # recovered, spanning 0 to ~90%, so "no recovery at all" stays cheap.
    priors['disease_rec'] = numpyro.deterministic(
        "disease_rec",
        jnn.sigmoid(numpyro.sample(
            "disease_rec_raw",
            dist.Normal(float(_p["recovery_logit_loc"]),
                        float(_p["recovery_logit_scale"]) * prior_scale))),
    )
    priors['disease_tau_rec'] = numpyro.deterministic(
        "disease_tau_rec",
        # LogNormal keeps the timescale positive and multiplicatively symmetric:
        # "about 12 years, plausibly 5 to 30" rather than an additive window that
        # could stray nonpositive.
        jnp.exp(numpyro.sample(
            "disease_log_tau_rec",
            dist.Normal(jnp.log(float(_p["recovery_tau_median_years"])),
                        float(_p["recovery_tau_log_scale"]) * prior_scale))),
    )

    # --- 2. DEMOGRAPHIC INTERCEPTS (Alphas) ---
    # Adult survival baseline > Juvenile survival baseline
    # Adult survival at typical habitat (logit). Default N(0.5, 0.5) (median ~0.62);
    # population_model.adult_survival_prior {"loc", "scale"} overrides (E037: with the fecundity
    # prior firmed, the loose Sa prior absorbed the growth demand -- Sa drifted to ~0.84).
    _ap = dict(_POP_SPEC.get("adult_survival_prior", {}))
    priors['alpha_a'] = numpyro.sample("alpha_a", dist.Normal(
        float(_ap.get("loc", 0.5)), float(_ap.get("scale", 0.5)) * prior_scale))
    if JUVENILE_RATIO:
        # alpha_j is the LOGIT OF THE JUVENILE/ADULT SURVIVAL RATIO at typical habitat:
        # Sj = Sa * sigmoid(alpha_j + gamma_j*H), so Sj < Sa everywhere by construction.
        priors['alpha_j'] = numpyro.sample("alpha_j", dist.Normal(
            float(_JUV.get("ratio_logit_loc", 0.0)), float(_JUV.get("ratio_logit_scale", 0.5)) * prior_scale))
    else:
        priors['alpha_j'] = numpyro.sample("alpha_j", dist.Normal(-0.5, 0.5 * prior_scale)) # ~40%
    # Fecundity at typical habitat, softplus scale. Default N(2.0, 0.5) (median F ~2.1);
    # population_model.fecundity_prior {"loc", "scale"} overrides (E027: the data are flat along the
    # fecundity/survival ridge at lambda0 ~1.6, so this prior is what sets the split).
    _fp = dict(_POP_SPEC.get("fecundity_prior", {}))
    priors['alpha_f'] = numpyro.sample("alpha_f", dist.Normal(
        float(_fp.get("loc", 2.0)), float(_fp.get("scale", 0.5)) * prior_scale))  # Fecundity
    # CAPACITY LEVEL, declared in expected BBS route counts (see counts_to_relative).
    # SOFTPLUS link: K_counts = softplus(alpha_k + gamma_k*H_k + trend). alpha_k_loc is
    # solved by _softplus_loc_for_median so the POST-TRANSFORMATION prior MEDIAN of the
    # capacity level equals the measured 2.6183 route counts. The median, not the mean:
    # a mean target landed the realized median 8.6% high (config _link_comment).
    #
    # Reminder of what this prior replaced and why it matters: alpha_k used to be
    # stated in relative DENSITY units, where it asserted a capacity of ~205 route
    # counts (~97% of the highest counts ever recorded) against a data-implied ~2, so
    # it sat 7 prior SDs from the fit and every term able to lower K -- disease
    # severity, the H_k deviation, the continental trend -- was recruited as a level
    # reducer and saturated. Route counts, not density units, is the load-bearing part.
    priors['alpha_k'] = numpyro.sample(
        "alpha_k", dist.Normal(_ALPHA_K_LOC,
                               float(_CAPACITY_LEVEL["alpha_k_scale"]) * prior_scale))
    # Capacity at H_k = 0: the analogue of the exp form's separable level, kept under
    # the same names so the diagnostics and response curves read unchanged. Under
    # softplus it is not separable from the covariate term, so this is "capacity where
    # the covariates are neutral" rather than a multiplicative level.
    # k_level: capacity where the habitat field is neutral -- under the crowding form, at
    # the reference R0 (where the prior median puts it at the measured target).
    priors['k_level'] = numpyro.deterministic(
        "k_level",
        jnp.exp(priors['alpha_k']) * (float(_CROWDING["reference_R0"]) - 1.0)
        if _K_FORM == "crowding" else k_link(priors['alpha_k']))
    numpyro.deterministic("k_level_route_counts", priors['k_level'] * _POP_SCALAR)
    
    # --- 3. DEMOGRAPHIC SENSITIVITIES (Gammas) ---
    # Positive by construction: better habitat = higher survival/fecundity.
    #
    # These are now DIMENSIONLESS and fixed at 1, because the amplitude they used to
    # carry lives in w_scale (see the amplitude block above). gamma_j * w_scale and
    # gamma_f * w_scale etc. only ever entered the likelihood as products, so keeping
    # both left three exactly flat ridge directions -- tolerable under MAP, ruinous for
    # HMC step-size adaptation.
    priors['gamma_a'] = 1.0
    # gamma_j is now fixed at 1 as well, and gamma_j_diff is GONE as a free parameter.
    #
    # It survived as long as it did because Sa/Sj/Q shared ONE manifold: with a single beta_s,
    # the juvenile/adult sensitivity CONTRAST was identified even though the overall scale was
    # not. Now that Sj has its own beta_sj carrying w_scale[3], the product
    # gamma_j * w_scale[3] multiplies one unit-variance latent and is exactly the flat ridge
    # described above -- the same pathology that fixed the other three at 1.
    #
    # "Juveniles are MORE environment-sensitive than adults" (rule 5) is preserved, moved to the
    # place where it IS identified: amplitude_loc[3] is calibrated so w_scale[3] reproduces
    # w_scale[0] * E[1 + HalfNormal(0.5)], i.e. the amplitude the old product implied.
    priors['gamma_j'] = 1.0

    # Fmax and K are each the SOLE user of their manifold, so gamma * w_scale was an
    # exactly flat ridge. Fixed at 1; the amplitude is w_scale[1] and w_scale[2].
    priors['gamma_f'] = 1.0
    priors['gamma_k'] = 1.0

    # The slopes are reported two ways. The friendly names are for new code; the *_raw
    # names are emitted as DETERMINISTIC constants equal to softplus^-1(1) = log(e-1), and
    # gamma_j_diff as 0.0, so a reader that computes softplus(gamma_f_raw) or
    # gamma_a + gamma_j_diff still gets exactly 1.0.
    #
    # DO NOT DELETE THESE. The live consumer is vis/age_model_math._gamma_slope, reached
    # through the hardcoded fold-in list in scripts/viz/map_diagnostics.py:251-253 --
    # auto_delta_params_to_latents returns SAMPLED sites only, so a deterministic that is
    # not in that list vanishes from the latents dict. Removing these broke diagnostics
    # twice in production (KeyError after figure 04, before metrics.json), which is why
    # tests/test_kernel_contract.py and tests/test_map_diagnostics.py now pin them.
    _SOFTPLUS_INV_ONE = math.log(math.e - 1.0)
    for _name in ("gamma_a_raw", "gamma_f_raw", "gamma_k_raw"):
        numpyro.deterministic(_name, jnp.asarray(_SOFTPLUS_INV_ONE))
    numpyro.deterministic("gamma_j_diff", jnp.asarray(0.0))
    for _name, _val in (("gamma_a", priors['gamma_a']), ("gamma_j", priors['gamma_j']),
                        ("gamma_f", priors['gamma_f']), ("gamma_k", priors['gamma_k'])):
        numpyro.deterministic(_name, jnp.asarray(_val, dtype=float))
    
    # n50 is on the BBS ROUTE-COUNT scale and needs no conversion: allee_gamma*N reduces
    # to ln2*C/n50, a function of the observed count alone, so pop_scalar cannot change
    # what this prior asserts.
    #
    # Intent: a route that consistently yields one bird is a colonized cell, and that bird
    # stands for a local community rather than a lone individual, so mate-finding should
    # not be limiting there. At loc -2.0 (median n50 0.127 counts) the retained fecundity
    # is 42% at 0.1 counts, 94% at 0.5, and 99.6% at one bird -- the brake acts only on
    # genuinely sparse cells. See the _allee_comment in config for the two things this
    # controls downstream (the Allee-dead area in figure 07, and barrier-crossing cost).
    n50_raw = numpyro.sample(
        "n50_raw", dist.Normal(float(_ALLEE_PRIOR["n50_raw_loc"]),
                               float(_ALLEE_PRIOR["n50_raw_scale"]) * prior_scale))
    n50 = jnn.softplus(n50_raw)

    # Derive the searching efficiency on the RAW count scale
    # gamma_raw = ln(2) / N50
    priors['gamma_raw'] = jnp.log(2.0) / (n50 + 1e-6)

    priors['dispersal_logit_intercept'] = numpyro.sample("dispersal_logit_intercept", dist.Normal(2.0, 1.0 * prior_scale))
    priors['dispersal_logit_slope'] = numpyro.sample("dispersal_logit_slope", dist.Normal(4.0, 1.0 * prior_scale))
    # Breeding dispersal is far rarer than natal dispersal in passerines; the default loc -2
    # puts adult departure at ~1/7 the juvenile odds, with a loose scale.
    if DEPART_BY_AGE:
        priors['dispersal_logit_adult_offset'] = numpyro.sample(
            "dispersal_logit_adult_offset",
            dist.Normal(float(_DEPART.get("adult_offset_loc", -2.0)),
                        float(_DEPART.get("adult_offset_scale", 1.5)) * prior_scale))
    else:
        priors['dispersal_logit_adult_offset'] = 0.0
    # Above-capacity departure pulse. Its logit gain per unit of N/K above 1 is ~beta_hi;
    # loose half-normal (scale 4 -> a cell at 2K could send most of its birds).
    # Within-cell urbanisation tilts (per sd of log built-up intensity). Free sign: the data say
    # whether towns hold more capacity / higher fecundity than the countryside.
    if SUBCELL:
        priors['subcell_beta_k'] = numpyro.sample(
            "subcell_beta_k", dist.Normal(0.0, float(_SUBCELL.get("beta_k_scale", 1.0)) * prior_scale))
        priors['subcell_delta_f'] = numpyro.sample(
            "subcell_delta_f", dist.Normal(0.0, float(_SUBCELL.get("delta_f_scale", 0.5)) * prior_scale))
        beta_mod_scale = float((_SUBCELL.get("capacity_moderator") or {}).get("coefficient_scale", 0.0))
        priors['subcell_beta_k_mod'] = (numpyro.sample(
            "subcell_beta_k_mod", dist.Normal(0.0, beta_mod_scale * prior_scale))
            if beta_mod_scale > 0 else 0.0)
        # Town SURVIVAL tilt (feeders act mainly on overwinter survival); optional linear drift in time.
        sigma_scale = float(_SUBCELL.get("sigma_s_scale", 0.0))
        sigma_mod_scale = float((_SUBCELL.get("survival_moderator") or {}).get("coefficient_scale", 0.0))
        if sigma_scale > 0 or sigma_mod_scale > 0:
            s0 = (numpyro.sample("subcell_sigma_s", dist.Normal(0.0, sigma_scale * prior_scale))
                  if sigma_scale > 0 else 0.0)
            s1 = (numpyro.sample("subcell_sigma_s_drift", dist.Normal(
                0.0, float(_SUBCELL["sigma_s_drift_scale"]) * prior_scale))
                if float(_SUBCELL.get("sigma_s_drift_scale", 0.0)) > 0 else 0.0)
            priors['subcell_sigma_s'] = (s0, s1)          # -> per-year tilt in build_model_2d
            priors['subcell_sigma_s_mod'] = (numpyro.sample(
                "subcell_sigma_s_mod", dist.Normal(0.0, sigma_mod_scale * prior_scale))
                if sigma_mod_scale > 0 else 0.0)
        else:
            priors['subcell_sigma_s'] = None
            priors['subcell_sigma_s_mod'] = 0.0
    else:
        priors['subcell_beta_k'] = priors['subcell_delta_f'] = 0.0
        priors['subcell_beta_k_mod'] = 0.0
        priors['subcell_sigma_s'] = None
        priors['subcell_sigma_s_mod'] = 0.0
    # Settlement selectivity: how strongly birds follow empty-cell fitness when settling
    # (E014: perfect fitness sorting was too strong; K-only ignores vital rates entirely).
    priors['settlement_tau'] = (numpyro.sample(
        "settlement_tau", dist.HalfNormal(float(_SETTLE.get("tau_scale", 2.0)) * prior_scale))
        if SETTLE_WEIGHT in ("capacity_fitness", "capacity_r0") else 0.0)
    priors['dispersal_crowding_beta'] = (numpyro.sample(
        "dispersal_crowding_beta",
        dist.HalfNormal(float(_DEPART.get("crowding_scale", 4.0)) * prior_scale))
        if CROWDING_ARM else None)
    
    # Temporal Annual Noise (Maintained for dispersal probability fluctuations)
    # At sd 0.001 on the logit this moves dispersal by ~0.025%: the probe of run_18 found its
    # 124 directions exactly at prior curvature (1e6) and L-BFGS moved it ~3e-5. It only
    # added sampling dimensions, so it can be switched off (dispersal_random_enabled).
    if _DISPERSAL_RANDOM:
        priors['dispersal_random'] = numpyro.sample(
            "dispersal_random", dist.Normal(0., 0.001 * prior_scale), sample_shape=(time,))
    else:
        priors['dispersal_random'] = jnp.zeros(time)
    
    return priors


def build_model_2d(data, prior_scale=1.0):
    """The NumPyro model: priors -> demographic fields -> forward sim -> NB2 likelihood.

    ``data`` bundles the model-ready arrays (grid dims, land indices, the gathered
    Z / dispersal-feature memmaps, spatiotemporal basis, BBS observations and
    their per-observation quality tier, and scaling constants). Samples priors,
    projects Z to per-cell/per-year survival, fecundity, and carrying-capacity
    fields, runs the age-structured forward simulation from the invasion year,
    and scores BBS counts with a negative-binomial (NB2) likelihood. Its concentration
    is down-weighted for lower-quality (unscreened Mexico) observations, but only when
    more than one quality tier is actually present -- with US/Canada alone the term is
    identically off. ``prior_scale`` controls tight-to-nominal prior continuation.
    """
    validate_environment_kernel_contract(data)
    Nx, Ny = data['Nx'], data['Ny']
    time = data['time']
    land_rows, land_cols = data['land_rows'], data['land_cols']
    M = data['Z_gathered'].shape[-1]
    
    # 1. Sample Parameters
    _sample = (numpyro.handlers.condition(sample_priors, data={k: jnp.asarray(v) for k, v in PINNED.items()})
               if PINNED else sample_priors)
    priors = _sample(prior_scale, M, time,
                     N_sev_basis=data['N_sev_basis'],
                     N_lag_basis=data['N_lag_basis'])

    # Native-lineage shape: the SAME dimensionless core/margin map used to seed 1902
    # abundance (see age_priors' initpop comment and bbs.generate_core_margin_init),
    # reused here as the covariate for disease_b_native. Computed once, before the
    # disease dict, and reused again below for the initpop seed so the two uses can
    # never drift onto different arrays.
    native_shape = data['initpop_latent'][land_rows, land_cols]

    # Bundle the disease term's data and parameters. Grouping them keeps
    # project_and_scatter_age_structured's signature from growing another eight
    # positional arguments, and lets the field helpers in age_fields.py be called
    # directly by diagnostics with the same structure.
    disease = {
        "sev_basis": data['disease_sev_basis'],
        "lag_basis": data['disease_lag_basis'],
        "onset": data['disease_onset'],
        "onset_decades": data['disease_onset_decades'],
        "native_shape": native_shape,
        "mu_sev": priors['disease_mu_sev'],
        "b_late": priors['disease_b_late'],
        "b_native": priors['disease_b_native'],
        "w_sev": priors['disease_w_sev'],
        "lag0": priors['disease_lag0'],
        "w_lag": priors['disease_w_lag'],
        "tau": priors['disease_tau'],
        "ceiling": _DISEASE_CEILING,
        "k_half": priors['disease_k_half'],
        "hill_n": priors['disease_hill_n'],
        "rec": priors['disease_rec'],
        "tau_rec": priors['disease_tau_rec'],
    }

    # The 1940 release(s), as a fraction of a FIXED, EMPIRICALLY OBSERVED, K-
    # INDEPENDENT reference (global_q50_route_counts) -- deliberately NOT k_level or
    # local K_base, either of which ties a historical release event's size to a
    # fitted, moving quantity and (as a fraction of the CONTINENTAL level) badly
    # diluted the founder at the actual release site, which sits at ~90-95th
    # percentile local capacity. See config's _invasion_pulse_comment.
    #
    # One coefficient per (candidate site, year): data['inv_locations'] is
    # (n_sites, 2), so log_inv_pulse_fraction is (n_sites, inv_window) -- the fit can
    # allocate mass across sites and across years rather than assuming both. The
    # shared prior's SD shrinks as sqrt(n_sites*inv_window) grows (same pattern as
    # disease_prior's sev_field_budget/sqrt(N_sev_basis)), so adding candidate sites
    # cannot inflate the term's total flexibility -- see config's
    # _invasion_budget_comment.
    _q50_density = float(_INVASION_PULSE["global_q50_route_counts"]) / _POP_SCALAR
    if _INVASION_PULSE.get("form", "grid") == "single":
        # form = single (user, 2026-10-16): the record describes ONE release event, so 9 sites x
        # 10 years = 90 coefficients was an order of magnitude too many. One parameter, the
        # TOTAL founder number (in fractions of the q50 reference; 1 unit ~ 40 birds by a rough
        # route-sampling conversion), released at the n_sites nearest candidate cells (nearest-
        # first from Jones Beach; default 1) in equal shares over the first n_years years of the
        # window (default 1 = 1940). The Allee exemption then covers only those cells.
        _k = int(_INVASION_PULSE.get("n_sites", 1))
        _ny = int(_INVASION_PULSE.get("n_years", 1))
        inv_locs = data['inv_locations'][:_k]
        # Sampled here, outside sample_priors(), so the pinned_sites condition does not reach it:
        # pin explicitly (GRID54 step 8b found the pin silently ignored).
        if "log_inv_total" in PINNED:
            log_total = numpyro.deterministic("log_inv_total", jnp.asarray(PINNED["log_inv_total"]))
        else:
            log_total = numpyro.sample(
                "log_inv_total",
                dist.Normal(jnp.log(float(_INVASION_PULSE.get("total_median", 1.0))),
                            float(_INVASION_PULSE.get("total_log_sd", 1.0)) * prior_scale))
        shape = jnp.zeros((_k, data['inv_window'])).at[:, :_ny].set(1.0 / (_k * _ny))
        inv_pop = numpyro.deterministic("inv_pop_relative", _q50_density * jnp.exp(log_total) * shape)
    else:
        inv_locs = data['inv_locations']
        _n_inv_sites = inv_locs.shape[0]
        _inv_log_sd = (float(_INVASION_PULSE["log_budget"])
                      / jnp.sqrt(_n_inv_sites * data['inv_window'])) * prior_scale
        inv_pop = numpyro.deterministic("inv_pop_relative", _q50_density * jnp.exp(
            numpyro.sample("log_inv_pulse_fraction",
                           dist.Normal(jnp.log(float(_INVASION_PULSE["median_fraction_of_global_q50"])),
                                       _inv_log_sd),
                           sample_shape=(_n_inv_sites, data['inv_window']))))

    # Convert to the relative [0, 1] scale by multiplying by pop_scalar
    # Since N_relative = N_raw / pop_scalar, 
    # then gamma_relative = gamma_raw * pop_scalar
    # RELATIVE form (allee_prior.form = relative): n50 is a FRACTION OF LOCAL CAPACITY, so
    # gamma = ln2/n50 multiplies N/K, is dimensionless, and needs no gauge conversion. Then
    # whether a cell can hold a population depends only on its growth rate and this one
    # global gamma -- never on K -- so capacity cannot carve out absence ("Allee-dead"
    # cells, 20-40% of suitable habitat in run_19..21). Less mechanistic than absolute
    # density (mates are found by density), tested for fit and biological plausibility.
    allee_gamma_scaled = priors['gamma_raw'] * (1.0 if ALLEE_RELATIVE else data['pop_scalar'])
    priors['allee_gamma'] = numpyro.deterministic("allee_gamma", allee_gamma_scaled)

    # Habitat centring: intercepts become "the rate at typical occupied habitat", so a prior
    # on alpha_k is a prior on K's level where birds live. Without it the uncentered
    # Ruzicka features' near-constant leading direction gives every field a second, nearly
    # free intercept, which is how run_19's K level drifted to 0.03 route counts (anchor
    # 2.6) while alpha_k sat 5.6 prior sds out. Offsets reported for the visualizations.
    z_center = None
    if _HABITAT_CENTERING == "occupied":
        z_center = data["habitat_center_occupied"]
        numpyro.deterministic("habitat_center_offsets", jnp.stack([
            jnp.dot(z_center, priors['beta_s']), jnp.dot(z_center, priors['beta_r']),
            jnp.dot(z_center, priors['beta_k']), jnp.dot(z_center, priors['beta_sj'])]))

    st_in = None
    if ST_CORRECTION:
        nb, nt = data['st_S'].shape[-1], data['st_g'].shape[0]
        st_w = numpyro.sample("st_correction_w", dist.Normal(
            0.0, float(_ST.get("scale", 0.1)) * prior_scale).expand([nb, nt]))
        Gw = numpyro.deterministic("st_correction_gw", (st_w @ data['st_g']).T)          # (time, nb)
        st_in = (data['st_S'], data['st_S_disp'], Gw)
    state_anchor = None
    if STATE_ANCHOR:
        _alpha = numpyro.sample("state_anchor_alpha", dist.Beta(float(_ANCHOR.get("alpha_a", 2.0)),
                                                                 float(_ANCHOR.get("alpha_b", 2.0))))
        state_anchor = (int(data['anchor_t']), data['anchor_A'], data['anchor_s'], _alpha)
    state_warp = None
    if STATE_WARP:
        _wu = numpyro.sample("state_warp_u", dist.Normal(0.0, 1.0).expand(
            [2, data['warp_active'].shape[0], data['warp_S'].shape[-1]])) * data['warp_active'][None, :, None]
        _d = float(_WARP.get("sigma_cells", 2.0)) * (_wu @ data['warp_S'].T)                # (2, time, N_land)
        numpyro.deterministic("state_warp_rms_cells", jnp.sqrt(jnp.mean(jnp.sum(_d ** 2, 0), axis=1)))
        state_warp = (_d[0], _d[1])
    process_noise = None
    if PROCESS_NOISE:
        u_raw = numpyro.sample("process_noise_u", dist.Normal(0.0, 1.0).expand(
            [data['pn_active'].shape[0], data['pn_S'].shape[-1]]))                          # (time, nb)
        u_act = u_raw * data['pn_active'][:, None]
        if "pn_P" in data:      # detrend_degree: no sustained regional trend in eps (stage 2: that is lambda(t)'s job)
            u_act = data['pn_P'] @ u_act
        process_noise = (float(_PN.get("sigma", 0.05)) * u_act) @ data['pn_S'].T
        numpyro.deterministic("process_noise_rms_t", jnp.sqrt(jnp.mean(process_noise ** 2, axis=1)))
    field_drift = None
    covariate_drift_in = None
    if DRIFT_ON:
        # Prior sd per weight = budget/sqrt(n_basis) on the LINK scale (logit for survival, softplus
        # argument for fecundity and capacity), as k_trend. Weak by construction: the default 0.2
        # moves a survival logit ~0.2 (a few % of survival) over the window.
        nb = data['drift_basis'].shape[0]
        sd = float(_DRIFT.get("budget", 0.2)) / jnp.sqrt(nb) * prior_scale
        if _DRIFT["form"] == "intercept":
            w_on = numpyro.sample("covariate_drift_w", dist.Normal(0.0, sd).expand([len(_DRIFT_GROUPS), nb]))
            W = jnp.zeros((4, nb), dtype=w_on.dtype)
            for i, grp in enumerate(_DRIFT_GROUPS):
                for f in grp:
                    W = W.at[DRIFT_FIELDS.index(f)].set(w_on[i])
            field_drift = numpyro.deterministic("covariate_drift", (W @ data['drift_basis']).T)   # (time, 4)
        else:
            # P2: weights per (covariate, field group, basis fn), per sd of the covariate. prior "normal"
            # (sd = budget/sqrt(nb)) or "cauchy" (scale = cauchy_scale): heavy-tailed, strongly peaked at 0, so
            # weak effects shrink to ~0 while strong signals escape -- a MAP/L-BFGS-friendly stand-in for a
            # horseshoe (whose hierarchical funnel collapses under point optimisation).
            C = data['drift_x'].shape[-1]
            shape = [C, len(_DRIFT_GROUPS), nb]
            if _DRIFT.get("prior", "normal") == "cauchy":
                w_on = numpyro.sample("covariate_drift_w", dist.Cauchy(
                    0.0, float(_DRIFT.get("cauchy_scale", 0.02)) * prior_scale).expand(shape))
            else:
                w_on = numpyro.sample("covariate_drift_w", dist.Normal(0.0, sd).expand(shape))
            W = jnp.zeros((C, 4, nb), dtype=w_on.dtype)
            for i, grp in enumerate(_DRIFT_GROUPS):
                for f in grp:
                    W = W.at[:, DRIFT_FIELDS.index(f)].set(w_on[:, i])
            G = numpyro.deterministic("covariate_drift", jnp.einsum("cfb,bt->tcf", W, data['drift_basis']))
            covariate_drift_in = (data['drift_x'], G)                                # G (time, C, 4)

    # 1. Compute Biological Fields (2D Manifold -> Demographic Rates)
    # Notice we now pass beta_s and beta_r instead of a single beta_h
    compounding_in = None
    if PATH_COMPOUNDING:
        if "path_compounding_lambda" in PINNED:
            _lam_c = numpyro.deterministic("path_compounding_lambda", jnp.asarray(PINNED["path_compounding_lambda"]))
        else:
            _lam_c = numpyro.sample("path_compounding_lambda",
                                    dist.HalfNormal(float(_COMP.get("lambda_scale", 0.25)) * prior_scale))
        compounding_in = (_lam_c, data['kernel_mean_km'] / float(_COMP.get("length_km", 100.0)),
                          data['Z_disp_seg_gathered'], str(_COMP.get("reference", "endpoint")))
    journey_in = None
    if JOURNEY:
        _Kk = data['juvenile_fft_kernel_stack'].shape[0]
        _nbj = _Kk // 4
        if _JOURNEY.get("decouple", False):
            _j_off = numpyro.sample("journey_alpha_offset", dist.Normal(0.0, float(_JOURNEY.get("offset_scale", 1.0)) * prior_scale))
            _j_lr = numpyro.sample("journey_log_slope_ratio", dist.Normal(0.0, float(_JOURNEY.get("log_ratio_scale", 0.5)) * prior_scale))
        else:
            _j_off, _j_lr = jnp.asarray(0.0), jnp.asarray(0.0)
        if _JOURNEY.get("band_cost", False):
            _c_band = numpyro.sample("journey_band_cost", dist.HalfNormal(float(_JOURNEY.get("cost_scale", 1.0)) * prior_scale).expand([_nbj]))
        else:
            _c_band = jnp.zeros((_nbj,))
        if _JOURNEY.get("distance_hazard", False):
            _dbar = jnp.asarray([float(x) for x in _JOURNEY["band_mean_km"]])
            if _dbar.shape[0] != _nbj:
                raise ValueError(f"journey.band_mean_km has {_dbar.shape[0]} bands, kernel stack has {_nbj}")
            if "journey_distance_hazard" in PINNED:
                _h = numpyro.deterministic("journey_distance_hazard", jnp.asarray(PINNED["journey_distance_hazard"]))
            else:
                _h = numpyro.sample("journey_distance_hazard", dist.HalfNormal(float(_JOURNEY.get("hazard_scale", 0.05)) * prior_scale))
            _c_band = _c_band + _h * _dbar / 100.0
        numpyro.deterministic("journey_band_survival", jnp.exp(-_c_band))
        _cost_k = _c_band[jnp.arange(_Kk) % _nbj]
        if _JOURNEY.get("water_hazard", False):
            _W = data['journey_water_W']                                       # (N_land, K), built in data_loading
            if "journey_water_hazard" in PINNED:
                _hw = numpyro.deterministic("journey_water_hazard", jnp.asarray(PINNED["journey_water_hazard"]))
            else:
                _hw = numpyro.sample("journey_water_hazard", dist.HalfNormal(float(_JOURNEY.get("water_scale", 1.0)) * prior_scale))
            _cost_k = _cost_k[None, :] + _hw * _W                              # (N_land, K)
        journey_in = (_j_off, _j_lr, _cost_k)
    Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat, Kbase_flat = project_and_scatter_age_structured(
        time, Ny, Nx, land_rows, land_cols,
        data['Z_gathered'], data['Z_disp_gathered'],
        data['disease_timestep'], disease,
        priors['beta_s'], priors['beta_r'], priors['beta_k'], priors['beta_sj'],
        data['k_trend_basis'], priors['w_k_trend'],
        priors['alpha_a'], priors['gamma_a'],
        priors['alpha_j'], priors['gamma_j'],
        priors['alpha_f'], priors['gamma_f'],
        priors['alpha_k'], priors['gamma_k'],
        k_link=k_link,
        capacity=crowding_capacity if _K_FORM == "crowding" else None,
        z_center=z_center,
        juvenile_ratio=JUVENILE_RATIO,
        disease_target=DISEASE_TARGET,
        z_shrink=(numpyro.sample("habitat_change_eta", dist.Beta(
            float(_HABCHG.get("eta_a", 2.0)), float(_HABCHG.get("eta_b", 2.0))))
            if HABITAT_SHRINK else None),
        Z_mean=data.get("Z_mean"), Z_disp_mean=data.get("Z_disp_mean"),
        field_drift=field_drift,
        covariate_drift=covariate_drift_in,
        st_correction=st_in,
        split_readout=((data['Z_ref'], data['Z_disp_ref'],
                        (priors['beta_s_w'], priors['beta_r_w'], priors['beta_k_w'], priors['beta_sj_w']))
                       if SPLIT_READOUT else None),
        journey=journey_in,
        compounding=compounding_in,
    )
    if BAND_WEIGHTS:
        _nb = data['juvenile_fft_kernel_stack'].shape[0] // 4
        band_logw = numpyro.sample("dispersal_band_logw", dist.Normal(
            0.0, float(_KERNEL_SPEC.get("band_logw_scale", 1.0)) * prior_scale).expand([_nb - 1]))
        if _KERNEL_SPEC.get("time_knots"):
            _B = jnp.asarray(data['band_time_B'], band_logw.dtype)   # built in data_loading (traced under refine)
            _dlt = numpyro.sample("dispersal_band_logw_time", dist.Normal(
                0.0, float(_KERNEL_SPEC.get("time_scale", 0.5)) * prior_scale).expand([_B.shape[1], _nb - 1]))
            _logw_t = numpyro.deterministic("dispersal_band_logw_t", band_logw[None, :] + _B @ _dlt)   # (time, nb-1)
            juv_w = jax.vmap(lambda lw: juvenile_band_weights(data['juvenile_fft_kernel_stack'], lw))(_logw_t)
            _mass = jnp.real(data['juvenile_fft_kernel_stack'][:, 0, 0])[None] * juv_w
            numpyro.deterministic("dispersal_band_mass_t", _mass.reshape(-1, 4, _nb).sum(1))
        else:
            juv_w = juvenile_band_weights(data['juvenile_fft_kernel_stack'], band_logw)
            _mass = jnp.real(data['juvenile_fft_kernel_stack'][:, 0, 0]) * juv_w
            numpyro.deterministic("dispersal_band_mass", _mass.reshape(4, _nb).sum(0))
    else:
        juv_w = None
    if DISEASE_TARGET == "survival":
        _tt = jnp.arange(time, dtype=Sa_flat.dtype)[:, None]
        disease_m_amp = numpyro.deterministic("disease_mortality_amp", jnp.where(
            _tt >= data['disease_timestep'],
            disease_mortality_amplitude(disease, _tt), 0.0))
    else:
        disease_m_amp = None
        
    # Save fields for viz
    numpyro.deterministic("Sa_flat", Sa_flat)
    numpyro.deterministic("Sj_flat", Sj_flat)
    numpyro.deterministic("Fmax_flat", Fmax_flat)
    numpyro.deterministic("K_flat", K_flat)
    numpyro.deterministic("K_base_flat", Kbase_flat)

    # NATIVE-RANGE SEED, as a fraction of LOCAL capacity in the first model year.
    # data['initpop_latent'] is a dimensionless SHAPE (core=1, margin=
    # initpop_seed.margin_fraction_of_core) marking WHERE the 1902 native range was --
    # the contrast is assumed, not measured, because an observed abundance ratio
    # double-counts the habitat gradient K_base already carries; the amplitude comes
    # from K_base itself, so the seed automatically tracks whatever capacity the
    # covariates imply for those cells.
    #
    # Why not an absolute value, and why not a fraction of the CONTINENTAL level:
    # both were tried today and both were wrong, in opposite directions, for the same
    # reason -- they compared the seed to the wrong K. An absolute 9.5 route counts is
    # ~33% of local capacity in the native core (observed 28.6 counts) and perfectly
    # sensible for a native range at equilibrium, but at INITIALIZATION AutoDelta sets
    # beta_k near its prior median so H_k ~ 0 and every cell has K = k_level = 2.1
    # counts: the seed is then 4.5x capacity, the density brake slams on, and step 0
    # of the fit scores badly. Rescaling to 10% of the CONTINENTAL level fixed step 0
    # but left the native core starting at <1% of its own local capacity. A fraction
    # of LOCAL K is correct at both ends: at step 0 local K = k_level so the seed is a
    # modest fraction of it, and at the optimum it scales with the fitted capacity of
    # those specific cells.
    #
    # The fraction is 1.0: a native range at equilibrium since long before 1902 starts
    # AT its capacity, not below it. Since dispersal_target_capacity_fraction is 0.8,
    # that puts the seed at N/K - 0.8 = +0.2 on the emigration logit, i.e. the native
    # range is a NET EXPORTER from year one. Deliberate: the thing to be explained is
    # that an exporting western population still did not cross the Great Plains for
    # eighty years, so the barrier has to be paid for by the pre-invasion pseudo-zeros
    # (bbs.py) and the covariates, not by a native range too weak to push.
    # Reuse native_shape (computed above, before the disease dict) rather than
    # re-slicing initpop_latent -- one array, two uses, so they cannot drift apart.
    _seed_flat = (native_shape * Kbase_flat[0]
                  * float(_INITPOP_SEED["core_fraction_of_local_capacity"]))
    initpop_seeded = numpyro.deterministic(
        "initpop_seeded",
        jnp.zeros((Ny, Nx)).at[land_rows, land_cols].set(_seed_flat))
    # Per-cell peak severity at MODERN capacity -- the model's falsifiable claim
    # about the epizootic. Density-dependent, so it must be evaluated against the
    # undepressed K_base (using the depressed K would be circular), and at a
    # specific year since K_base drifts with the continental trend.
    numpyro.deterministic("disease_severity_map",
                          disease_severity(disease, Kbase_flat[-1]))
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

    subcell_beta_k_t = priors['subcell_beta_k']
    subcell_sigma_s_t = None
    if SUBCELL:
        subcell_beta_k_t = jnp.broadcast_to(
            priors['subcell_beta_k'], (time, data['land_rows'].shape[0]))
        if "subcell_capacity_moderator" in data:
            subcell_beta_k_t = (subcell_beta_k_t + priors['subcell_beta_k_mod'] *
                                data["subcell_capacity_moderator"])
        subcell_beta_k_t = numpyro.deterministic("subcell_beta_k_t", subcell_beta_k_t)
        if priors['subcell_sigma_s'] is not None:
            sigma_base_t = (priors['subcell_sigma_s'][0] + priors['subcell_sigma_s'][1] *
                            (data["subcell_drift_g"] if "subcell_drift_g" in data else jnp.zeros(time)))
            subcell_sigma_s_t = jnp.broadcast_to(
                sigma_base_t[:, None], (time, data['land_rows'].shape[0]))
            if "subcell_survival_moderator" in data:
                subcell_sigma_s_t = (subcell_sigma_s_t + priors['subcell_sigma_s_mod'] *
                                     data["subcell_survival_moderator"])
            subcell_sigma_s_t = numpyro.deterministic("subcell_sigma_s_t", subcell_sigma_s_t)

    departure_scale = None
    if NATIVE_DEPARTURE:
        r_nat = numpyro.sample("native_departure_r", dist.Beta(float(_NATDEP.get("beta_a", 1.5)),
                                                               float(_NATDEP.get("beta_b", 1.5))))
        w_nat = native_shape / jnp.maximum(jnp.max(native_shape), 1e-12)
        departure_scale = 1.0 - r_nat * w_nat
    attraction_in = None
    if ATTRACTION:
        if "settlement_attraction_a" in PINNED:
            a_attr = numpyro.deterministic("settlement_attraction_a", jnp.asarray(PINNED["settlement_attraction_a"]))
        else:
            a_attr = numpyro.sample("settlement_attraction_a",
                                    dist.HalfNormal(float(_ATTR.get("a_scale", 0.25)) * prior_scale))
        eps_attr = float(_ATTR.get("eps", 0.1))
        attraction_in = (a_attr, eps_attr, bool(_ATTR.get("adults", True)))
        # readout: attractiveness of a front cell (N/K = .01, .05) relative to a full one, and mate-finding success there
        # under the fitted Allee (relative form): the two clumping assumptions read together (plan, E079 note)
        _xf = jnp.asarray([0.01, 0.05])
        numpyro.deterministic("attraction_front_weight", jnp.exp(a_attr * (jnp.log(eps_attr + _xf) - jnp.log(eps_attr + 1.0))))
        numpyro.deterministic("attraction_front_mate_success", _af_allee.allee_mated_fraction(
            priors['allee_gamma'] * (_xf if ALLEE_RELATIVE else _xf * jnp.mean(Kbase_flat))))
    directional_in = None
    if DIRECTIONAL_CHOICE:
        if "directional_choice_kappa" in PINNED:
            kappa = numpyro.deterministic("directional_choice_kappa", jnp.asarray(PINNED["directional_choice_kappa"]))
        else:
            kappa = numpyro.sample("directional_choice_kappa",
                                   dist.HalfNormal(float(_DIRCH.get("kappa_scale", 0.3)) * prior_scale))
        _Zd = data['Z_disp_gathered']
        _bj = (priors['gamma_j'] * priors['beta_sj']).astype(_Zd.dtype)
        _ba = (priors['gamma_a'] * priors['beta_s']).astype(_Zd.dtype)

        def _ahead(t):     # per-year slices, as project_and_scatter reads Z_disp (time, N_land, K)
            zt = jnp.take(_Zd, t, axis=0)
            return jnp.dot(zt, _bj) + (jnp.dot(zt, _ba) if JUVENILE_RATIO else 0.0)
        _sig = jax.lax.map(_ahead, jnp.arange(time)).astype(Fmax_flat.dtype)
        _mass = jnp.real(data['juvenile_fft_kernel_stack'][:, 0, 0]).astype(Fmax_flat.dtype)
        directional_in = _af_allee.directional_choice_weights(_sig, _mass, kappa)
        numpyro.deterministic("directional_choice_mean_abs_dev", jnp.mean(jnp.abs(directional_in - 1.0)))
    _F_sim = Fmax_flat * jnp.exp(process_noise) if (process_noise is not None and PN_TARGET == "fecundity") else Fmax_flat
    _pn_groups = None
    if process_noise is not None and str(_PN.get("conserve", "none")) != "none":
        _pn_groups = (data['pn_groups'], int(data['pn_n_groups']))

    def _run_forward(_pn_arg, _F_arg, _warp_arg=state_warp, _anchor_arg=state_anchor, _nest_arg=False):
      return forward_sim_age_structured(
        Sa_flat, Sj_flat, _F_arg, K_flat, c_flat, Q_flat,
        land_rows, land_cols,
        data['land_mask'],
        data['adult_fft_kernel'], data['juvenile_fft_kernel_stack'],
        data['adult_edge_correction'], data['juvenile_edge_correction_stack'],
        initpop_seeded, priors['dispersal_random'], inv_pop,
        time, inv_locs, data['inv_timestep'],
        priors['dispersal_logit_intercept'], priors['dispersal_logit_slope'],
        priors['allee_gamma'],
        target_fraction=data["dispersal_target_fraction"],
        allee_exempt_years=_RELEASE_ALLEE_EXEMPT_YEARS,
        allee_relative=ALLEE_RELATIVE,
        settle_block=SETTLE_BLOCK,
        settle_weight=SETTLE_WEIGHT,
        settle_dispersers_only=SETTLE_DISPERSERS,
        veit_departure=VEIT_DEPARTURE,
        yearling_breeding=YEARLING and not VEIT_REPRO, veit_reproduction=VEIT_REPRO,
        settle_tau=priors['settlement_tau'],
        strata_u=data["strata_u"] if SUBCELL else None,
        strata_a=data["strata_a"] if SUBCELL else None,
        subcell_beta_k=subcell_beta_k_t, subcell_delta_f=priors['subcell_delta_f'],
        subcell_sigma_s=subcell_sigma_s_t,
        subcell_survival_affects_allocation=SUBCELL_SURVIVAL_AFFECTS_ALLOCATION,
        dispersal_adult_offset=priors['dispersal_logit_adult_offset'],
        crowding_beta=priors['dispersal_crowding_beta'], crowding_kappa=CROWDING_KAPPA,
        disease_m_amp=disease_m_amp, disease_k_half=priors['disease_k_half'],
        disease_hill_n=priors['disease_hill_n'],
        juvenile_kernel_weights=juv_w,
        occupancy_b=(data['pop_scalar'] * OCC_BIRDS_PER_COUNT) if OCCUPANCY else None,
        process_noise=_pn_arg if PN_TARGET in ("abundance", "low_density") else None,
        departure_scale=departure_scale,
        process_noise_low_density=(float(_PN.get("low_density_h", 0.05)) if PN_TARGET == "low_density" else None),
        process_noise_groups=_pn_groups,
        state_warp=_warp_arg,
        state_anchor=_anchor_arg,
        attraction=attraction_in,
        directional_weights=directional_in,
        nest_cells=((data['nest_rows'], data['nest_cols']) if (NEST_LIKELIHOOD and _nest_arg) else None),
      )
    _sim = _run_forward(process_noise, _F_sim, _nest_arg=True)
    densities, Na_grid, Nj_grid = _sim[:3]
    _skel_spec = _WARP if STATE_WARP else _ANCHOR
    if (STATE_WARP or STATE_ANCHOR) and _skel_spec.get("skeleton_tau"):
        _skel = _run_forward(None, Fmax_flat, None, None)[0]
        _n0 = float(_skel_spec.get("skeleton_n0", 1e-3))
        _full_l, _skel_l = densities[:, land_rows, land_cols], _skel[:, land_rows, land_cols]
        _d = jnp.log(_full_l + _n0) - jnp.log(_skel_l + _n0)
        _w = (_full_l / (_full_l + _n0)) * data['skel_penalty_years'][:, None]
        numpyro.deterministic("skeleton_logdiff_rms", jnp.sqrt(jnp.sum(_w * _d ** 2) / jnp.maximum(jnp.sum(_w), 1e-12)))
        numpyro.factor("skeleton_consistency", -0.5 * jnp.sum(_w * (_d / float(_skel_spec["skeleton_tau"])) ** 2))
    if process_noise is not None and _PN.get("skeleton_tau"):
        _skel = _run_forward(None, Fmax_flat, None, None)[0]                             # (time, Ny, Nx), no correction
        _n0 = float(_PN.get("skeleton_n0", 1e-3))
        _full_l, _skel_l = densities[:, land_rows, land_cols], _skel[:, land_rows, land_cols]
        _d = jnp.log(_full_l + _n0) - jnp.log(_skel_l + _n0)
        _w = (_full_l / (_full_l + _n0)) * data['pn_penalty_years'][:, None]
        numpyro.deterministic("skeleton_logdiff_rms", jnp.sqrt(jnp.sum(_w * _d ** 2) / jnp.maximum(jnp.sum(_w), 1e-12)))
        numpyro.factor("skeleton_consistency", -0.5 * jnp.sum(_w * (_d / float(_PN["skeleton_tau"])) ** 2))
    if OCCUPANCY:
        # psi per cell and year (post-step): occupied-conditional density is simulated_density / psi.
        numpyro.deterministic("occupancy_grid", _sim[3])

    if NEST_LIKELIHOOD:
        # population_model.nest_likelihood: the realised fecundity at each attempt's cell and year
        _F_i = numpyro.deterministic("nest_F_real", _sim[-1][data['nest_t'], data['nest_ci']])
        _br = dict(_NEST.get("broods") or {})
        _bv = jnp.asarray([float(x) for x in _br.get("values", [2.5, 2.5, 2.5])])
        if str(_br.get("form", "pinned")) == "lognormal":
            _B = jnp.exp(_nest_site("nest_broods_log", dist.Normal(
                jnp.log(_bv), float(_br.get("log_sd", 0.3)) * prior_scale).to_event(1)))
        else:
            _B = _bv
        numpyro.deterministic("nest_broods", _B)
        _p = data['nest_X'].shape[-1]
        if _p:
            _dm = _nest_site("nest_mean_delta", dist.Normal(0.0, float(_NEST.get("mean_scale", 0.1)) * prior_scale)
                             .expand([_p]).to_event(1))
            _ds = _nest_site("nest_surv_delta", dist.Normal(0.0, float(_NEST.get("surv_scale", 0.5)) * prior_scale)
                             .expand([_p]).to_event(1))
        else:
            _dm = _ds = jnp.zeros((0,), _F_i.dtype)
        _as = _nest_site("nest_dsr_logit", dist.Normal(float(_NEST.get("dsr_logit_loc", 3.5)),
                                                        float(_NEST.get("dsr_logit_scale", 1.0)) * prior_scale))
        _lc = _nest_site("nest_log_conc", dist.Normal(float(_NEST.get("log_conc_loc", 2.3)),
                                                       float(_NEST.get("log_conc_scale", 1.0)) * prior_scale))
        _lle, _llc, _m, _pi, _ = nest_log_likelihood(
            _F_i, _B[data['nest_region']], data['nest_X'], _dm, _as, _ds, _lc, data['nest_outcome'],
            data['nest_fledged'], data['nest_e_pre'], data['nest_d_last'], data['nest_w_exp'], data['nest_w_cnt'],
            cycle_days=float(_NEST.get("cycle_days", 32.0)), sharpness=float(_NEST.get("floor_sharpness", 10.0)))
        numpyro.deterministic("nest_mean_fledged", _m)
        numpyro.deterministic("nest_cycle_success", _pi)
        numpyro.deterministic("nest_loglik_exposure", jnp.sum(_lle))
        numpyro.deterministic("nest_loglik_count", jnp.sum(_llc))
        numpyro.factor("nest_loglik", jnp.sum(_lle) + jnp.sum(_llc))

    numpyro.deterministic("simulated_density", densities)
    # Realized age-class pools (adults/juveniles separately, full grid, all
    # years) -- for the post-hoc "realized age structure" diagnostic. Not on
    # the likelihood path; DCE'd away during actual MAP/SVI optimization (see
    # forward_sim_age_structured's docstring), only materialized when
    # requested via Predictive(return_sites=[...]).
    numpyro.deterministic("Na_grid", Na_grid)
    numpyro.deterministic("Nj_grid", Nj_grid)

    # 4. Likelihood
    t_idx, rows, cols = data["obs_time_indices"], data["obs_rows"], data["obs_cols"]
    
    # densities output should be the sum of adult + juvenile (N_total)
    if OBS_BREEDING:
        phi = numpyro.sample("obs_fledgling_weight", dist.Beta(
            float(_OBS_SPEC.get("fledgling_weight_a", 1.0)), float(_OBS_SPEC.get("fledgling_weight_b", 4.0))))
        breeders = jnp.where(t_idx > 0, densities[jnp.maximum(t_idx - 1, 0), rows, cols],
                             initpop_seeded[rows, cols])
        densities_obs = jnp.maximum((breeders + phi * Nj_grid[t_idx, rows, cols]) * data["pop_scalar"], 1e-6)
    else:
        densities_obs = jnp.maximum(densities[t_idx, rows, cols] * data["pop_scalar"], 1e-6)
    # observation.route_effects: PERSISTENT per-route multipliers on expected counts (a route
    # through a town with finches stays high every year) -- patchiness as a fixed route offset
    # learned from repeated years, instead of extra per-observation noise (kappa), so each
    # observation keeps its information about the cell mean. Pseudo-zeros (route -1) get none.
    if _OBS_SPEC.get("route_effects"):
        # Most PINNED sites are sampled by sample_priors(), which is conditioned above.
        # This likelihood-level site lives outside that handler, so condition it here as
        # well.  E059/E064 route arms exposed the omission: their resolved configs said
        # obs_route_sd=0.7 was fixed, but AutoDelta/L-BFGS were actually free to fit it.
        # Record a deterministic at the fixed value so Predictive diagnostics can still
        # request the named quantity without adding an optimizable coordinate.
        if "obs_route_sd" in PINNED:
            _sr = numpyro.deterministic("obs_route_sd", jnp.asarray(PINNED["obs_route_sd"]))
        else:
            _sr = numpyro.sample("obs_route_sd", dist.HalfNormal(
                float(_OBS_SPEC.get("route_sd_scale", 0.5)) * prior_scale))
        _zr = numpyro.sample("obs_route_raw", dist.Normal(0.0, 1.0).expand([int(data["n_routes"])]))
        _ri = data["obs_route"]
        route_fx = jnp.where(_ri >= 0, _sr * _zr[jnp.maximum(_ri, 0)], 0.0)
        densities_obs = densities_obs * jnp.exp(route_fx)
    # observation.observer_effect (default off): observer x route effect + first-year effect.
    _obs_fx = observer_effect_spec()
    if _obs_fx["enabled"]:
        numpyro.deterministic("expected_obs_before_observer", densities_obs)
        densities_obs = densities_obs * jnp.exp(observer_log_offset(data, _obs_fx))
    if OBS_REGION_EFFECTS:
        _u = numpyro.sample("obs_region_u", dist.Normal(0.0, 1.0).expand(
            [data['ore_active'].shape[0], data['ore_S'].shape[-1]])) * data['ore_active'][:, None]
        if "ore_P" in data:
            _u = data['ore_P'] @ _u
        _eff = numpyro.deterministic("obs_region_effect", float(_ORE.get("sigma", 0.2)) * (_u @ data['ore_S'].T))
        densities_obs = densities_obs * jnp.exp(_eff[t_idx, data['obs_land_idx']])
    if YEAR_EFFECTS:
        _sd = numpyro.sample("obs_year_sd",
                             dist.HalfNormal(float(_YEAR_FX.get("sd_scale", 0.3)) * prior_scale))
        _raw = numpyro.sample("obs_year_raw", dist.Normal(0.0, 1.0).expand([time]))
        _w = jnp.bincount(t_idx, length=time).astype(_raw.dtype)
        _raw_c = _raw - jnp.sum(_w * _raw) / jnp.sum(_w)          # centred over OBSERVED years
        year_fx = numpyro.deterministic("obs_year_effect", _sd * _raw_c)
        densities_obs = densities_obs * jnp.exp(year_fx[t_idx])
    
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
    # Whether the site EXISTS must be a plain Python bool. data_loading records the tier
    # count as an int; the numpy fallback works on any concrete array. Never jnp here:
    # under jit (NUTS, the posterior probe, VI's scan) jnp.max of even constant data is
    # an abstract tracer and int() of it raises (TACC job 3480304).
    n_tiers = data.get("n_obs_quality_tiers")
    if n_tiers is None and obs_quality is not None:
        n_tiers = int(np.unique(np.asarray(obs_quality)).size)
    if obs_quality is not None and n_tiers > 1:
        q_mult = numpyro.sample("quality_conc_mult", dist.Beta(2.0, 2.0))
        conc_obs = concentration * jnp.power(q_mult, obs_quality)
    else:
        conc_obs = concentration
    # observation.conc_density_power: NB2 concentration falls at low expected count,
    # phi(mu) = phi0 * mu^kappa, kappa ~ HalfNormal(scale) >= 0 (0 = one global phi). Sparse
    # populations are PATCHY (mostly zero routes, locally dense at a few); with one global phi
    # the cheapest fit is thin, widespread presence (plains: predicted P(0) .85 vs observed .93).
    if _OBS_SPEC.get("conc_density_power"):
        kappa = numpyro.sample("obs_conc_kappa", dist.HalfNormal(
            float(_OBS_SPEC.get("conc_density_power_scale", 0.5)) * prior_scale))
        conc_obs = conc_obs * jnp.power(jnp.maximum(densities_obs, 1e-3), kappa)

    # Held-out scoring (observation_mask holdouts, E070) needs the per-observation concentration.
    numpyro.deterministic("obs_concentration", jnp.broadcast_to(conc_obs, densities_obs.shape))
    # population_model.observation_mask (diagnostic): data_loading sets obs_weight 0 on the
    # dropped observations; they stay in expected_obs, so their fit can still be read.
    # observation_weighting (diagnostic): positive weights SCALE each observation's log-likelihood.
    if ROUTE_OCCUPANCY:
        log_m = numpyro.sample("route_occupancy_log_m", dist.Normal(
            jnp.log(float(_ROUTE_OCC.get("m_loc", 0.3))), float(_ROUTE_OCC.get("m_scale", 1.0)) * prior_scale))
        gate, mu_occ, gate_logits = route_occupancy_parts(densities_obs, jnp.exp(log_m))
        numpyro.deterministic("obs_zero_gate", gate)
        obs_dist = dist.ZeroInflatedNegativeBinomial2(mean=mu_occ, concentration=conc_obs, gate_logits=gate_logits)
    else:
        obs_dist = dist.NegativeBinomial2(mean=densities_obs, concentration=conc_obs)
    obs_weight = data.get("obs_weight")
    if obs_weight is None:
        numpyro.sample("obs", obs_dist, obs=data["observed_results"])
    else:
        with numpyro.handlers.mask(mask=obs_weight > 0.0), \
                numpyro.handlers.scale(scale=jnp.where(obs_weight > 0.0, obs_weight, 1.0)):
            numpyro.sample("obs", obs_dist, obs=data["observed_results"])

    # Source-specific winter count likelihood. Seasonal operator has no demographic feedback.
    _cbc = dict(_POP_SPEC.get("cbc_likelihood") or {})
    if _cbc.get("enabled"):
        from src.model.cbc_joint import cbc_observe
        cbc_observe(data, Na_grid, Nj_grid, data["pop_scalar"], _cbc)
