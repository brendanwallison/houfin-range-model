# E070 -- process noise on abundance, and lambda(t) corrections: competing formulations (DESIGN, 2026-10-20)

## Why
Every formulation so far is a deterministic skeleton from 1902 + observation error. Any structural error compounds
along a century-long trajectory and can only be absorbed by parameters: the refine distorts vital rates (F 10-16,
Sj -> Sa) to fix TIMING (Wood 2010: deterministic fits of nonlinear ecological dynamics give ill-conditioned,
multimodal likelihood surfaces). The realistic float32 MAPs are unconverged points (early stopping acting as an implicit
prior toward init), not better estimates; the refine's ~15,000-nat gain says the data want something realistic
demography cannot give them inside this structure. 67c2 shows the risk of offering only a PERSISTENT correction: the
refine used habitat to fake transient timing (native west a sink in 1966 -> lam ~2 now).

## Terms (each with one job)
- eps (process noise, the TRANSIENT term): N_{t+1}(x) = f(N_t)(x) * exp(eps_t(x)), both age classes, after
  reproduction. Multiplicative: cannot create birds where there are none (no phantom immigration; spread stays
  dispersal's job). eps_t = sum_b S_b(x) u_{b,t}, S = land-centred cosines (n_freq 3 -> 15; variant n_freq 5), u iid
  N(0, sigma^2) per year, sigma PINNED (MAP collapses or explodes a learned sigma). Excluded from lambda and source/sink
  readouts (those use the deterministic skeleton).
- lambda(t) correction (the PERSISTENT term, space-for-time error), two candidate forms:
  - S: split readout (E066) -- spatial level and temporal deviation of z get separate weights; deviation/level ratio
    prior centred on DESK's measured .14 (route 3). Tied to observed environmental change.
  - H: free smooth habitat correction (E067) extended to 1902-2023, time basis linear or 2-3 low-frequency terms,
    prior scale calibrated from DESK's out-of-community temporal error (captures ~15-18% of available change;
    route 1), not hand-set.
- Separation when eps is combined with S/H: eps is projected orthogonal to the correction's time basis per spatial
  basis function (eps carries no regional trend; any sustained regional change must come from habitat or dynamics).

## Arms (base B1 no floor = 69a; B3 only for the stage-3 winner)
Stage 1 -- eps alone (does process error release the contortion, and at what sigma?):
- 70a/70b/70c: eps from 1902, sigma .02/.05/.10.   70d: eps from 1966, sigma .05.
- Guards: 70n NULL (truth = 69a MAP, eps = 0; fit sigma .05 from 1902); 70r RECOVERY (inject a regional eps pulse,
  e.g. +.3 on one basis function over 1950-1960, the eastern-invasion window).
Stage 2 -- lambda(t) term with eps at the stage-1 sigma* (and ablations without eps):
- 70s eps*+S; 70h eps*+H; 70sh eps*+S+H.   Ablations: 70s0 S only, 70h0 H only (no-floor reruns).
- Cross-recovery guards (both terms fitted): 70xh inject H only -> eps must stay ~0; 70xe inject eps only -> H/S must
  stay ~0; the stage-1 null covers "neither".
Stage 3 -- winner on B3; sensitivity sigma*/2, 2 sigma*; window variant if stage 1 found 1902 vs 1966 to matter.

## Held-out data (same mask for every arm; production refit on all data afterwards)
In-sample U cannot rank arms (more freedom always fits better). Two holdouts via population_model.observation_mask:
- TEMPORAL: BBS 2019, 2021-2023 (2020 cancelled) withheld; eps is ZERO in the forecast years, so an arm that only fits
  through eps forecasts badly, while a correct skeleton/lambda(t) forecasts well.
- SPATIAL: ~10% of routes in spatial blocks withheld over all years.
(To build: observation_mask today only supports the transient mask; add holdout_years and spatial_blocks modes.)
Score: held-out NB log predictive density, plus log1p RMSE / correlation, each vs 69a.

## Criteria (pre-registered; applied to the REFINED point)
GATES -- an arm failing any gate is not viable, whatever its fit:
- G1 its guards pass (null: terms <= 2 prior sds and demography within the truth's band; recovery >= 70% of the
  injection with the right sign, other terms at the null level; cross-recovery: the un-injected term at null level).
- G2 plausible demography: Sa .50-.70, Sj .25-.45 with Sj <= Sa - .05, F 3-7, lam0 <= 1.6.
- G3 plains trough: plains median lam_f below west AND east; Allee-dead < 5% of suitable habitat.
- G4 corrections within budget: max |term| <= 3 prior sds (eps: realised RMS u <= 1.5 sigma).
- G5 MAP -> refine stability: Sa, Sj within .08, F within 30%.
RANKING among viable arms, in order:
- R1 held-out predictive density (temporal first, then spatial).
- R2 process-noise need: realised RMS eps / sigma (less noise needed = better skeleton).
- R3 lambda(t) plausibility: native west lam_f >= 1 throughout (the historic range persisted); regional dlam within
  the calibrated budget; sign of western/eastern trends vs published BBS regional trends (external, not fitted).
- R4 robustness: demography and lambda(t) conclusions unchanged across sigma*/2 .. 2 sigma*.
- R5 independence: the lambda(t) term moves < 1 prior sd when eps is toggled.
U / dU reported for information only.

## Readouts per arm (MAP and refined, as always)
Standard viz + west-sink audit; eps map by era (where/when the skeleton is wrong); lambda(t) by region (1902-2023);
held-out scores; gate table. One summary table across arms.

## Open
If no arm passes G2-G3 even with eps, the remaining structural suspects are the observation process (route effects
and BBS detection) and the age structure; record which gates fail per arm so the failure pattern points to it.

## Stage 1 BUILT (2026-10-20)
- population_model.process_noise {enabled, sigma, n_freq, window_start, window_end}: u_raw (time, nb) ~ N(0, 1),
  eps = sigma * (u_raw * active) @ S^T, S scaled to unit land mean-square (sigma = land-RMS nudge per year); applied in
  forward_sim_age_structured after the end-of-year clip. Off from the first held-out year onward.
- observation_mask.holdout_years / holdout_blocks (data_loading.holdout_observation_mask): data run to 2025, so the
  temporal holdout is 2021-2025 (the contiguous tail after the cancelled 2020 season; ~12k obs); spatial = 10% of
  8-cell (~216 km) blocks, seed 70, all years. 70z = 69a refitted with the mask: the baseline every arm is ranked against.
- Diagnostics: metrics.json process_noise (rms/max u_raw for G4, era means) + 18_process_noise.png; holdout
  (mean NB log density and log1p RMSE on temporal / spatial / fitted sets).
- Change from the design: the recovery pulse is +2.5 u_raw on basis 0 (cos pi x) over 1975-1985, not 1950-1960:
  observations start in 1966, so a 1950s pulse is visible only through its after-effects. Pass: fitted mean u_raw over
  the pulse >= 1.75 (70%), other coefficients at the null's level.
Queue: 70z, 70n, 70r, 70b, then 70a, 70c, 70d (tasks/e070_task.sh).

## 70n NULL (read 2026-10-21): PASS
Truth 69a MAP (.57/.31/5.22; plains .87 / west 1.01 / east 1.07; no noise); synthetic mean 2.69, zeros .746. Fitted
with 70b's structure (sigma .05, 1902-2020 active; holdout 34,752 obs = 18.1%). Refined 148,842 (1,409 it):
.53/.26/5.78, lam0 1.20; plains .85 / west .99 / east 1.07; no Allee-dead. Noise: RMS u_raw .17 (MAP .32), max |u|
.98, land-RMS eps <= .025/yr; era RMS u .05/.10/.21/.26 (grows where data are dense, still far below the prior's 1).
G1 null: noise stays small, demography within the truth's band (F +11%, Sj -.05). Holdout lpd (refined): temporal
-1.747, spatial -1.189, fitted -.934 -- the reference scale for a correctly specified fit.
