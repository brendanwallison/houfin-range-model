# E057 -- weak time-dependent covariate effects ("proxy drift") -- DESIGN, not built (2026-10-18)

## Idea (user, 2026-10-18)
The fundamental niche is constant in the TRUE niche variables v(x,t); our inputs are proxies. Each field is
H_j(x,t) = z(x,t)' beta_j with z = DESK features of the covariates c(x,t). If the relationship between proxy and true
variable changes (feeding practices change what "built-up land" means to a House Finch), the effective readout drifts:
beta_j(t) = beta_j + Delta_j(t), Delta tiny and smooth. A fitted drift is interpretable as either a change in the
niche or a change in the proxy -> niche-variable relationship, and is a safety valve for species-specific structure the
community-trained DESK cannot carry.

## Evidence it is needed
The continental K trend (a spatially uniform drift of K's intercept -- the zero-order special case of this idea) is pushed
11-13 prior sds at every E050-E053 MAP (K x0.3-1.5 over the century). Refinement pulls it back to 2.5-3.4 sds WHILE the
demography contorts -- consistent with the refine trading an unrepresentable temporal signal for vital-rate distortion.
E047: the contortion buys eastern positives from 1980 and modern plains positives -- both temporal.

## Parametrisations (most parsimonious first)
- P0: intercept drift on each field (generalises k_trend to the four fields). 4 x n_basis params. No spatial content.
- P2 (hypothesis-specific): drift on a FEW interpretable raw covariates added beside DESK,
  H_j += x_c(x,t) * g_cj(t), c in {built-up / human density} first; one covariate, K (and F) only, g(t) on 2-3 cosines.
  Directly answers "did the finch-human relationship change?". x_c residualised on z if a STATIC term is also added
  (the species-specific safety valve), so the static part means "what DESK missed"; the drift needs no residualisation.
- P1 (agnostic): rank-1 drift in DESK space, Delta_j(t) = a_j(t) u, u a unit M-vector, a_j(t) smooth. M + 4 n_basis params.
  Interpretation: one niche axis changed meaning; u is read back onto raw covariates through the encoder (which
  covariates load on the drifted direction).
- Corroboration design: fit P1 and P2 separately. If P1's data-chosen direction aligns with the human covariates P2
  assumed, the feeder hypothesis is supported by two independent routes. Full beta(t) (M x 4 x n_basis ~ 300) is out.

## Constraints that make it "extremely weak" and identifiable
1. Magnitude in interpretable units: prior on the century-scale RMS change of H over occupied cells as a fraction of
   H's cross-sectional sd (budget ~0.1-0.2), plus the k_trend-style prior-sd-multiple diagnostic (a drift far outside
   its prior is a failure signal, not a finding).
2. Smooth: <= 2-3 cosines over the observed window; cannot compete with the disease term's sharp 1994+ onset or make
   decadal wiggles.
3. Window: drift defined on the BBS era only (1966-2025) and held at its 1966 value earlier. Otherwise the unobserved
   1902-1965 extrapolation is free and can manufacture the invasion.
4. Anchor: time-centred (weighted by observation density), so beta keeps meaning "the average relationship" and the
   drift is the departure. Replaces k_trend (P0 subsumes it) -- never both.

## Confounds and guards
- Invasion front: a moving front is not of the form feature(x) x time, but DESK features are spatially smooth and
  longitude-correlated, so a front can be approximated. GUARD (mandatory, before any real fit): null calibration --
  simulate from a MAP with zero drift (invasion, disease, real dispersal), refit with drift on, require drift ~ 0;
  then simulate with a known drift and require recovery. Mirror the existing synthetic-recovery guards
  ([[inventory-existing-guards-first]]).
- Disease: check the drift / disease-amplitude posterior correlation; smoothness should keep them apart.
- Covariate time-series error (HYDE decadal steps, interpolation): a drift can absorb measurement error in the
  covariate trajectories themselves -- a third interpretation besides niche change and proxy change.

## Niche change vs proxy change -- what can and cannot be separated
Not identifiable from these data alone. Partial levers:
- Direction: drift loading on human covariates -> proxy relationship (feeders); on climate -> adaptation or climate proxy.
- Lineage: one extra parameter letting the amplitude differ native west vs introduced east. Continent-wide drift fits a
  cultural/proxy change; east-only fits rapid adaptation of the bottlenecked eastern population -- but regional feeding
  culture also differs, so this is suggestive, not decisive.
- External timeline: feeder-use history (e.g. FeederWatch-era surveys) as an out-of-model check on g(t)'s shape.

## Interactions (user: parsimony first, then informed combinations)
- Run on the pinned-Allee base (E055), never with free n50: drift x Allee x low K is exactly the absence-switch loop.
- Complementary to sub-cell urban structure, not a substitute: drift raises urban-cell quality at 27 km but cannot
  create within-cell town sources in a plains sink. Relative Allee (55e) is the cheap closed form of clustering.
- Order: E056 (loosened continental trend) -> P2 human drift on K -> P1 agnostic -> informed pairs (drift x relative
  Allee; drift x bounded route effect).
- Prediction if drift is the missing mechanism: refined demography stays near its MAP values, the plains trough
  survives refinement, the drift is modest and loads on human covariates. Failure: drift outside its prior (absorbing
  dynamics) or demography still contorting.

## P0 BUILT + pre-registered (2026-10-18)
`population_model.covariate_drift = {form: intercept, fields, n_basis 3, budget .2, window_start 1966}`: per-field
link-scale shift d_j(t) = W_j . B(t); B = cos(m pi s) over 1966-last year, held at its 1966 value before, centred by
the observation-weighted mean over the window (data_loading.covariate_drift_basis). Replaces k_trend when capacity is
drifted. Diagnostics: metrics.json covariate_drift (per-field start/end/range on the link scale, prior-sd multiple).
Null/recovery machinery: exp/simulate_observations.py draws NB2 counts on the real design at a fitted point with --set
values for latents the point lacks; HOUFIN_SYNTHETIC_OBS=<npz> swaps them in at load. Rehearsed on CPU toy inputs.
Truth point = 55a refined with its k_trend dropped (57's structure has none) -- recorded in the npz.
- 57n NULL: drift_w = 0. Pass: every field's fitted drift range < ~2 prior sds' worth and no field's start-end change
  exceeds the recovery injection's 1/3.
- 57r RECOVERY: capacity w = [-0.3, 0, 0] (K argument +0.6 over 1966-2023). Pass: capacity range recovered within
  ~30% with the right sign; other fields stay null-like.
- 57a REAL: 55a + P0 on the real counts. Read only if 57n passes. Compare to 55a and 56a (loosened k_trend): is the
  temporal signal per-field (survival vs fecundity vs capacity), and does the refine contortion shrink?

## P2 BUILT + pre-registered (2026-10-18; REDESIGNED after user review -- Plummer is a reference arm only)
form "covariate": per-cell link-scale drift sum_c a_c,field * x_c(cell,t) * g(t). Covariates standardised (sd units):
"builtup" (HISDAC-US, CONUS) or the raw panel (src/data/preprocess/drift_covariates.py -> drift_covariates.npz):
log_popd, log_urban_pop (HYDE 3.5), cropland, pasture, rangeland, urban_land, forest (LUH3), tmin_q10, ppt_q50,
snow_q50 (climate). Default shape LINEAR in time over 1966-2023 (one slope per covariate x field); "cosine" and the
Plummer "ramp" (1979.5-2005, hinge fit to the digitised products) are alternatives. Held before 1966, obs-centred.
Prior: Normal(0, budget) for single-covariate arms; Cauchy(0, .02) per slope for the panel -- heavy-tailed and
peaked at 0 so weak effects shrink to ~0 and strong ones escape (a MAP/L-BFGS-friendly stand-in for a horseshoe,
whose hierarchical funnel collapses under point optimisation; horseshoe for HMC later).
Base 59a (V&L dispersal, n50 pinned, route effects sd .7).
- 57p popd x capacity; 57q popd x survival (tied); 57b built-up x capacity -- linear, the specific hypothesis.
- 57g popd x capacity, 2 cosines (shape check). 57z built-up x capacity, Plummer ramp (REFERENCE only).
- 57x PANEL: 11 covariates x {survival (tied), fecundity, capacity}, linear, Cauchy.
- 57xn NULL (truth 59a refined, all 33 slopes 0) and 57xr RECOVERY (only log_popd x capacity = 0.3) on 57x's
  structure. Read the panel only if no 57xn slope leaves the Cauchy core (|a| < ~5 scales) and 57xr recovers .3
  within ~30% with the other 32 near 0. Single-covariate arms are read against the same null logic.
Later: STATIC safety-valve arm (time-constant panel effects, residualised on Z) -- separate, so time-varying and
static signals are not confounded in the first test.

## 57a RESULT (P0 intercept drift on 55a; real data; READ PROVISIONALLY until 57n/57r report)
MAP 199,179 (f64 199,214, clean); MAP drift small (<= 1.4 prior scales). Refined 186,294 (8000 it, unconverged; 55a
186,101): the drift is used hard -- link-scale change 1966 -> 2023: fecundity -.50 -> +.75, capacity -.77 -> +.36,
juvenile survival -1.03 -> +.13, adult survival ~0; 5.3 prior sds out. Every rate but adult survival RISES over the
record. But the contortion is unchanged (F 15.1, lam0 2.14, plains K .0076, Allee-dead .15/.15/.02): a continental
intercept drift absorbs a temporal signal without relieving the refine. Whether the drift is real or absorbs
dynamics is exactly 57n's question.
