# SDM baseline plan -- step 0 (design table + scorer) and B0 naive forecasts (2026-10-30)

Parent plan: research/experiments/SDM_BASELINE_PLAN_2026-10-30.md (sec. 3 D1/D2, sec. 4 B0). Reference fit: GRID54
step 3 refined (`refined_float64__age_map_float32_grid54_step3_obsroute_quick90`; frozen code
/mnt/e/Datasets/houfin/grid54/runs/step3_obsroute/code; scores in runs/step3_obsroute/out/observer_holdout_scores.txt:
T plug-in -1.84357, T marginal -1.84390, T conditional -1.73114, S plug-in -1.49253, S marginal = conditional -1.21940).

Code (repo copies; run copies staged in /mnt/e/Datasets/houfin/grid54/baselines/b0/ with b0env.sh):
- research/exp/sdm_b0_design.py -- D1: design table from `load_data` under the step-3 config, plus step 3's own
  per-observation predictive (expected counts with / without the observer effect, NB concentration, sigma_u, eta).
- research/exp/sdm_b0_score.py -- D2: the one scorer (NB2 log density, plug-in / marginal / conditional, strata).
- research/exp/sdm_b0_naive.py -- B0 baselines.
Run environment: WSL, /home/breallis/houfin/venv/bin/python, JAX_PLATFORMS=cpu, OMP/OPENBLAS/MKL = 2, XLA intra-op 2,
taskset -c 0-3, nice 19 (a GPU fit is running on this machine). No TACC, no pueue changes.

## PRE-REGISTRATION (written before any B0 prediction was scored)

### Gate (D2 parity)
The scorer, given step 3's own per-observation expected counts and concentration, must reproduce the five stored
step-3 numbers above to < .001 nats/obs. If it cannot, stop and report why; nothing else is scored.

### Sets and conventions (identical for every model)
- TEMPORAL = every observation in 2021-25 (n 12,204). SPATIAL = the stored held-out mask minus the temporal years
  (pinned 31 x 216 km blocks, all other years; n 22,846). Both exactly as map_diagnostics._holdout_metrics.
- The SPATIAL set contains pre-invasion pseudo-zeros (1902-39 synthetic zeros; step-3 record: 36% of spatial held-out
  rows are not real observer-routes). They are not observations; any model that knows "no birds before the invasion
  outside the native halo" scores ~0 nats on them. So every spatial score is reported twice: S-all (the ladder's
  definition, for the gate and continuity) and S-real (real BBS rows only; the comparison that means something).
  The age model is re-scored on S-real by the same scorer.
- Likelihood: NB2, one concentration phi per baseline (constant across rows, like the age model's obs_concentration
  unless it varies -- reported), log density per observation, mean over the set.
- Training rows for every baseline: real BBS observations (not pseudo-zeros) outside BOTH held-out sets, weight > 0.
  Exception: S-IDW also uses training pseudo-zeros (the age model trains on them), so it can predict 1902-39 blocks.
- Dispersion and tuning are fitted on TRAINING rows only, by an inner validation that mimics each test:
  temporal baselines are re-run as if the forecast origin were 2015 and scored on training rows 2016-19 (2020 had no
  BBS); k, the shrinkage weight a, the trend switch and phi are chosen there by maximum NB likelihood. Spatial
  baselines use leave-one-block-out on training cells (same 216 km block lattice, own block's data removed); the
  bandwidth h, year half-window w, and phi are chosen there. Then everything is frozen and run once on the real sets.

### Baselines
Observer information is matched to the age model's score it is compared with:
- P-obsroute (vs T CONDITIONAL): mean count of the same observer-route over its last k surveyed training years,
  shrunk toward the cell-level prediction with weight a: mu = (n m + a mu_cell) / (n + a). Fallback for unseen
  observer-routes: route (any observer), then cell, then kernel neighbourhood.
- P-route (between the two; reported, not ranked): same, keyed by route regardless of observer.
- P-cell (vs T MARGINAL): mean count of all training routes in the 54 km cell over its last k surveyed years; fallback
  kernel neighbourhood (Gaussian, h = 100 km, fixed), then 324 km tile, then continental mean. No route or observer
  memory, like the age model's marginal score.
- +trend variants: each of the above multiplied by exp(b (year - ybar)), b = the 2011-20 (inner: 2006-15) log-linear
  slope of counts in its 324 km tile (6 x 6 cells), Poisson with route fixed effects, |b| capped at .15/yr. This is the
  "regional trend" placebo (E016, E039i). Per M6, a regional trend can be habitat change; it is a forecast floor, not
  a claim about dynamics.
- S-IDW (vs S MARGINAL): Gaussian-kernel average of training counts at other cells within +-w years of the target
  year, mu = sum K y / sum K, with a weak pull to the tile mean; no habitat, no dynamics. Observer-free, so compared
  with the marginal score (all spatial held-out routes are unseen, so marginal = conditional).
- S-tile (coarser reference): tile x year mean of training counts (+-w years).

### Readouts
For each baseline and the age model on the same rows: mean log density on T (and its seen / unseen observer-route
split), S-all, S-real; log1p RMSE and correlation; region (west < -105, plains -105..-95, midwest -95..-85, east
>= -85) for T and S-real; era (1966-79, 80-93, 94-07, 08-20) x region for S-real; and the paired per-observation
difference (age minus baseline) with a block bootstrap over the 31 held-out blocks (S) or over 216 km blocks of the
held-out temporal rows (T), 1,000 draws, as a 90% interval.

### Predictions (mine, before scoring)
- P1 T: P-obsroute beats the age model's conditional score (likely by .05-.15 nats/obs): a 5-year count forecast is
  dominated by route-and-observer memory. LIKELY.
- P2 T: the best route-free baseline (P-cell or P-cell+trend) is within +-.05 of the age model's marginal score.
  UNKNOWN direction.
- P3 S-real: S-IDW is within +-.05 of the age model (216 km blocks are mostly interpolable from neighbours; DESK's own
  216 km test lost to no-change, config comment in esk_desk_config.json). UNKNOWN direction.
- P4 strata: the age model's advantage, if any, is largest in the east and plains during the expansion eras on S-real
  (dynamics know where the front is; IDW smooths across it). LIKELY.

### Rules -- how B0 informs the ladder (implications reported, not decided)
- R1 If the best route-free naive is >= the age model's T marginal - .01: T marginal does not show skill beyond
  persistence/trend at this horizon; ladder steps that differ by ~.01-.03 on T (steps 3 vs 4: .017; seeds pending)
  cannot be read as better dynamics from T alone; rank on S-real, front timing (M-F) and source/sink plausibility.
- R2 If P-obsroute beats T conditional by > .01: the conditional score is route-memory dominated and should not drive
  selection (it rewards observer bookkeeping, not dynamics).
- R3 If S-IDW >= the age model's S-real - .01: the 216 km blocks are interpolable without habitat; S is a weak test of
  habitat and process; stratified S-real (east/plains, expansion eras) becomes the relevant read.
- R4 If the age model beats every naive by > .05 on T marginal and S-real with block-bootstrap intervals excluding 0:
  held-out count scores do carry process/habitat skill, and the current ladder rules stand.
- Pseudo-zero note: if S-all and S-real rank models differently, the ladder's S (S-all) must be replaced by S-real in
  later rules.

## RESULTS (2026-10-30; outputs /mnt/e/Datasets/houfin/grid54/baselines/b0/design_step3/: design.npz, z_cells.npy,
step3_pred.npz, step3_density.npy, design_manifest.json, parity_gate.json, b0_results.json; logs ../design_step3.log,
../parity.log, ../b0.log)

### Design table (D1)
192,123 observations (68,932 pseudo-zeros 1902-39, route id < 0; no real row before 1966; every real row has an
observer-route). TEMPORAL 12,204 (no pseudo-zeros). SPATIAL 22,846 = 14,524 real + 8,322 pseudo-zeros (36%). Z: 124
years x 4,318 land cells x 64 dims. sigma_u .97 (pinned), eta .0153, no route-occupancy gate. Design hash
ff0bad03...e036c6. Note: the frozen step-3 tree's reconstruct_map does not request expected_obs_before_observer (fixed
later in 30238e0); the export asks the model for it directly (same model code, same point).

### Gate: PASS
The scorer reproduces all five stored step-3 numbers to < 1e-13 nats/obs (T plug-in -1.84357, marginal -1.84390,
conditional -1.73114; S plug-in -1.49253, marginal -1.21940).

### Tuning (inner, training rows only; frozen before the real sets were scored)
| baseline | chosen | phi (NB2, ML on inner predictions) | inner mean lpd |
|---|---|---:|---:|
| P-obsroute | k 8, a 8 | .887 | -1.575 |
| P-route | k 8, a 4 | .844 | -1.589 |
| P-cell | k 8, a 2 | .694 | -1.623 |
| +trend variants | k 5-8 | .61-.76 | .025-.03 worse than flat in every case |
| S-IDW | h 100 km, w +-4 yr | .316 | -1.559 (leave-one-block-out) |
| S-tile | w +-4 yr | .207 | -1.734 |
Caveat: k = 8 and a = 8 are the edges of the pre-registered grid, so persistence is, if anything, slightly
under-tuned (favours the age model; does not affect R1, see below).

### Scores on the held-out sets (mean NB log density per observation; higher is better)
| model | T | S-all | S-real | T log1p RMSE / corr | S-real log1p RMSE / corr |
|---|---:|---:|---:|---|---|
| age step 3, plug-in | -1.844 | -1.493 | -2.348 | .707 / .778 | -- |
| age step 3, MARGINAL | -1.844 | -1.219 | -1.918 | | .806 / .803 |
| age step 3, CONDITIONAL | -1.731 | (= marginal) | (= marginal) | | |
| P-obsroute (observer + route memory) | -1.795 | | | .716 / .779 | |
| P-route (route memory) | -1.810 | | | .716 / .777 | |
| P-cell (cell only; vs marginal) | -1.836 | | | .768 / .743 | |
| P-cell + tile trend | -1.866 | | | .795 / .727 | |
| S-IDW | | -1.303 | -2.050 | | 1.016 / .691 |
| S-tile | | -1.379 | -2.170 | | 1.055 / .661 |

Paired differences, age minus baseline, 90% block-bootstrap intervals (216 km blocks; 262 for T, 30-31 for S):
| comparison (matched information) | all | west | plains | midwest | east |
|---|---|---|---|---|---|
| T conditional vs P-obsroute | +.063 (.032, .103) | +.112 (.028, .218) | +.059 (.011, .116) | +.039 (.009, .070) | +.041 (-.010, .103) |
| T marginal vs P-cell | -.008 (-.039, .030) | +.028 (-.043, .120) | -.003 (-.064, .058) | -.039 (-.085, .006) | -.019 (-.069, .039) |
| T marginal vs P-route (more info than marginal; reported only) | -.034 (-.072, .011) | -.025 | +.021 | -.060 (-.108, -.010) | -.052 |
| S-real marginal vs S-IDW | +.132 (.038, .209) | +.053 (-.097, .200) | +.044 (-.007, .103) | +.064 (.005, .108) | +.252 (.135, .359) |
| S-all marginal vs S-IDW | +.084 (.022, .140) | | | | |
| S-real marginal vs S-tile | +.252 (.125, .355) | +.233 | +.085 | +.142 | +.382 |
| S-real PLUG-IN vs S-IDW | -.298 (-.659, -.077) | -1.081 (-2.01, -.31) | -.026 | -.073 | +.024 |

### Predictions vs outcome
- P1 FAILED: persistence with observer-route memory does NOT beat the age model's conditional score; the age model
  (dynamics + fitted observer effects) is better by +.063 (interval excludes 0), in every region (east interval touches
  0). ESTABLISHED for this fit.
- P2 HELD (tie): the route-free persistence P-cell ties the age model's marginal score (-.008, interval -.04 to +.03);
  the age model is slightly behind in the midwest and east, ahead in the west, none clearly. ESTABLISHED.
- P3 FAILED: S-IDW does not tie; the age model beats it by +.132 on S-real (interval excludes 0). ESTABLISHED.
- P4 PARTLY: the spatial advantage is largest in the EAST (+.25, the invasion region), positive in the midwest, not
  clearly in the plains or west. Era x region strata for the age model AND the baselines are in b0_results.json
  (summary.S_real.by_region_era). Correction: I first wrote that the baseline strata were missing; they were written,
  only not printed.
- Regional trend extrapolation is worse than flat persistence at a 5-year horizon in every variant, on the inner
  validation and on the held-out set (by .03-.05). ESTABLISHED for this horizon.

### Rules applied (implications reported, not decided)
- R1 TRIGGERED: the best route-free naive (P-cell, -1.836) is >= the age model's T marginal (-1.844) - .01. At the
  5-year horizon, the marginal temporal score shows no skill beyond "each cell keeps its last ~8 years' mean count".
  Implication: ladder differences on T marginal of the size seen between steps (step 3 vs step 4 amp1: .017; amp1 vs
  amp2: .00) cannot be read as better or worse dynamics. GRID54 step 4's selection rule ("best held-out MARGINAL score,
  T and S") leans on a score that does not separate the age model from persistence. S-real (stratified) and front
  timing / source-sink plausibility carry the discrimination.
- R2 NOT triggered: the conditional temporal score is not mere route memory; it beats observer-route persistence.
  Mechanism (LIKELY, not tested): the fitted observer x route effects plus a smooth dynamic expectation pool
  information across routes that a single route's last 8 years cannot (P-obsroute had to shrink hard, a = 8).
- R3 NOT triggered: 216 km blocks are not interpolable to the age model's level; S-real carries habitat/process skill,
  concentrated in the east. Mechanism (LIKELY): the dynamics place the invasion front and its density gradient, which
  a 100 km kernel smooths across; in the native west and the plains, where abundance is closer to stationary and
  dominated by route-level patchiness (E076 west smoothing), the habitat field adds little over neighbours.
- R4 NOT met (T marginal is a tie), so held-out count scores do NOT uniformly carry process skill; S-real does,
  T marginal does not.
- Pseudo-zero note: S-all and S-real rank every model the same way, but the 36% pseudo-zero rows add ~+.7 nats/obs
  to S-all for the age model (and +.75 for IDW). S-all hides the scale; S-real should be reported next to S-all in
  ladder tables.
- Side finding: the age model's PLUG-IN spatial score loses to IDW (-.298), driven by the west (-1.08): with sigma_u
  .97, putting unseen observer-routes at the median observer is badly overconfident. This confirms the ladder's switch
  to the integrated (marginal) score from step 3 on; any older plug-in S number with observer effects is not
  comparable.

### Why T marginal ties persistence (mechanism candidates; none tested)
- The 5-year horizon is short relative to population change in most cells: persistence is a strong forecaster when
  the state changes slowly (ESTABLISHED in general; LIKELY here).
- E076 found the age model over-predicts the modern east (2010-25 mean +26% in 70z) -- the continued eastern decline
  is under-represented. The T marginal deficit vs P-cell sits in the midwest and east, consistent with that (LIKELY).
- The marginal score integrates over observers; P-cell's single NB concentration (.69) absorbs observer spread as
  overdispersion, which may be as good as a lognormal integration at sigma .97 (UNKNOWN).

### Next: the ladder rescore on S-real (research/experiments/GRID54_SREAL_RESCORE_2026-10-30.md, tool
research/exp/rescore_run.sh) and B1 (research/experiments/SDM_BASELINE_B1_2026-10-30.md) use this design table and
scorer unchanged.
