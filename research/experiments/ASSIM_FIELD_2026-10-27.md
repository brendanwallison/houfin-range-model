# Observation-anchored community field: is there a step change over covariate-only Z?

Pre-registered 2026-10-27 ~09:30 CDT, before any evaluation-panel score of the new arms exists. The only
evaluation-panel number seen is the control's own (reproduced exactly by the new scorer: known all 0.223,
regional 0.318, local 0.097, teacher 0.310; known sedentary common all 0.332, regional 0.458).
considerations: G1, G3, G11, M2, M3, M4, M6, M11, M14, M17, M23, M28; memory: sedentary common primary.

## Why

Every covariate-only route (DESK, linear and change-fitted maps E045/E073, the 2026-10-27 CPU change ridge,
reweighted losses E043) lands near the same centred change correlation. The observed teacher passed through the
same modern readout reaches 0.76, so Z can carry the change. The strongest earlier predictors used observed
trajectories of a place and its neighbours (position x time placebo, spacetime GP), which DESK cannot use at
prediction time. Production has summer observations 1966-2025 at most surveyed cells plus sparse winter CBC
before that, so the backcast is closer to reanalysis than to covariate extrapolation.

## Model (research/exp/z_assim.py)

Z(s,t) = Dbar(s) + u(s) + A.d(s,t) + R(s,t), in the fresh teacher spaces (dims 0-63 summer, 64-127 winter; the
masked control cube is verified to share them, corr 0.89 / 0.84 with observed teacher coordinates).
- u: per-cell level offset on the control's 2006-2025 mean map, GMRF (4-neighbour) smoothness + ridge: the fine map.
- A.d: the control DESK's own covariate-driven change, recalibrated by a 64x64 matrix per season (ridge to I).
- R: shared latent scores a(s,t) = S(s) C T(t) (128 Gaussian RBF at 200 km x cubic B-spline, 10-year knots,
  centred on 2006-2025, P-spline penalty), rank 16, with SEASON-SPECIFIC loadings: winter reaches summer only
  through couplings estimated where both seasons are observed.
Fitted by Adam on permitted sources only: summer training points (1996+, nested ABBA training half at known
places, nothing at spatial holdouts or buffer) and training-circle winter coordinates. No evaluation records,
sealed species or House Finch.

## Arms (one code path, switches)

| arm | D | R | winter in R | time basis | role |
|---|---|---|---|---|---|
| control_s1 | DESK | - | - | - | covariate-only reference |
| fine_d | yes | - | - | - | fine map + recalibrated DESK change |
| trend_summer | - | yes | no | linear | position x linear time from summer: placebo analog (M11) |
| field_summer | - | yes | no | spline | regional field from summer only |
| assim | - | yes | yes | spline | summer + winter field, no covariates in change |
| assim_d | yes | yes | yes | spline | full candidate |
| assim_d_sparse10 | yes | yes | 10% circles | spline | production-like winter density |
| assim_d_no_target | yes | yes | winter without 1966-85 | spline | is any gain from target-era winter? |

Hyperparameters: chosen on an inner backcast inside training data only (summer 1996-2005 hidden at training
cells; change 1996-2005 -> 2016-2025; INNER*.json). The best assim_d configuration's overrides are applied to
every R arm.

## Yardstick (research/exp/z_assim_evaluate.py)

The agent's estimands, unchanged: one modern spatial ridge readout per species reused over time; reserved
evaluation halves disjoint from training (no modern-end memorization, E055h); centred cross-half covariance;
6x6-block bootstrap; regional = 6x6-block means (>= 2 cells), local = departures. Strata: sedentary common
(AVONET migration 1, training-row presence >= 0.08) PRIMARY, sedentary, all. Plus summer64 change vs the
observed teacher projection, and winter composition at held-out circles. All arms paired against control_s1.

## Decision rule (fixed now)

Primary cell: known places, k = 2 survey years per half, sedentary common, REGIONAL change correlation.
STEP CHANGE for assim_d (or assim) if all hold:
1. regional corr >= 0.50 AND >= the best of {control, trend_summer, field_summer} + 0.10, with the paired
   difference vs control CI above 0;
2. all-change MSE gain over no change > 0 (sedentary common, known);
3. historical-endpoint MSE <= control's;
4. modern-endpoint MSE no worse than control by more than 2% (relative).
HOLDS AT PRODUCTION DENSITY if assim_d_sparse10 meets 1 with +0.05 in place of +0.10.
WINTER IS THE ROUTE if assim - field_summer regional >= +0.05 and assim_d_no_target falls back by >= half of it.
Otherwise: report which arm is best and by how much; no step change claimed.
Withheld places, local components, k = 4 and the other strata are reported, not decisive.

## What it cannot show

- Early summer (1902-1939) is never observed; the emulation hides summer before 1996 and tests 1966-85.
- R carries community-shared regional change whatever its cause (M6, M17): a gain is not evidence that the
  change is habitat rather than dynamics, nor the reverse. Community-level change is the habitat signature, and
  House Finch stays sealed.
- Repeatable route/observer effects shared by both halves remain in the truth (M2, M3).
- Reused development records under new masks: exploratory, not fresh confirmation.

## Amendment after the inner grid (2026-10-27 ~10:00, before any evaluation-panel score of new arms)

Inner backcast (ranking only; the fitted arms' offsets saw the 2016-25 rows that form the inner truth's modern
end, the control did not, so margins over the control are inflated; comparisons among fitted arms are fair):
field_summer 0.326 / regional 0.496 (slope 0.91), trend_summer 0.326 / 0.496, fine_d 0.312 / 0.483, control
0.273 / 0.427; every shared-latent winter arm is worse: assim 0.211 / 0.316, assim_d 0.264 / 0.371 (best
variant rank 8: 0.285 / 0.426); freer fields (lam_C 1e-3) are worse still. Winter-driven latent change mapped
into summer by overlap-era loadings points the wrong way, as with the rank-2 residual (791) and E066c.
Added arm field_d = fine map + DESK change + summer-only spline field (D yes, R yes, winter no). It enters the
decision rule's candidate set alongside assim_d and assim; the baseline set is unchanged. The winter arms keep
the best inner assim_d configuration.

Second amendment (same time, before any evaluation-panel score): the winter arms use winter_weight 0.3 (best inner
assim_d entry, 0.323 / 0.490). The sparse10 and no_target winter sources come with their OWN winter teachers, so
their coordinates do not share the control's winter dims (full teacher). In those two arms the winter block is
offset + field in the variant's own space with no control winter mean or change; the summer block is unchanged.
Caveat: the control DESK was trained with full winter, so even these arms' summer D carries full-winter
information; the package has no sparse-winter control.

## Results (2026-10-27 13:20; assim/backcast40/ASSIM_SCORES.json, 500 block bootstraps, paired vs control_s1)

Primary cell: k2, known places, sedentary common (21 species), regional change corr; paired difference vs control
with 95% CI; MSE ratios relative to control.

| arm | regional | paired vs control | all-change MSE gain | historical MSE | modern MSE | withheld places, paired |
|---|---:|---|---:|---:|---:|---|
| control_s1 | 0.458 | - | 0.056 | 1.000 | 1.000 | - |
| continue100_tacc | 0.455 | -0.002 [-0.004, -0.001] | 0.054 | 1.005 | 0.992 | -0.003 |
| trend_summer | 0.444 | -0.014 [-0.043, +0.015] | 0.080 | 1.004 | 0.772 | -0.038 [-0.132, +0.054] |
| field_summer | 0.497 | +0.039 [+0.014, +0.070] | 0.095 | 0.956 | 0.772 | +0.002 [-0.078, +0.089] |
| fine_d | 0.483 | +0.026 [+0.008, +0.044] | 0.081 | 0.994 | 0.776 | -0.014 [-0.049, +0.032] |
| field_d | 0.495 | +0.037 [+0.017, +0.059] | 0.082 | 0.984 | 0.774 | -0.010 [-0.052, +0.044] |
| assim | 0.308 | -0.150 [-0.200, -0.104] | -0.018 | 1.209 | 0.776 | -0.181 |
| assim_d | 0.455 | -0.003 [-0.019, +0.012] | 0.058 | 1.054 | 0.778 | -0.015 |
| assim_d_sparse10 | 0.469 | +0.011 [-0.019, +0.037] | 0.052 | 1.061 | 0.774 | -0.049 [-0.108, -0.001] |
| assim_d_no_target | 0.504 | +0.046 [+0.019, +0.077] | 0.069 | 1.021 | 0.772 | -0.011 [-0.059, +0.053] |

All species, known, regional (paired): field_d +0.074 [+0.058, +0.088], field_summer +0.067, fine_d +0.044,
assim_d +0.031, assim_d_no_target +0.022. Teacher-space change (all species, known): control 0.310 / regional
0.429; field_d 0.372 / 0.492 (best).

Decision rule:
- STEP CHANGE: NOT MET. assim_d 0.455 and assim 0.308 fail rule 1. Under the amendment field_d (0.495) fails too:
  it is < 0.50 and it ties the best baseline, field_summer (0.497), rather than exceeding it by 0.10.
  assim_d_no_target is the only arm >= 0.50 (0.504). It is +0.007 over field_summer, not +0.10, and its
  historical MSE is 2% above control's, failing rule 3.
- HOLDS AT PRODUCTION DENSITY: moot. sparse10 is +0.011 over control, with a CI spanning 0.
- WINTER IS THE ROUTE: NO. assim - field_summer = -0.189. Removing target-era winter IMPROVES the shared-latent arm
  (no_target 0.504 vs full 0.455). Target-era winter routed through overlap-era couplings points summer the wrong way.
  This matches the inner grid, the rank-2 residual (791) and E066c.

Interpretation:
- Established: anchoring Z to observations (a fine level map plus a regional spline field fitted to modern summer)
  adds +0.04 regional at known places in the primary stratum and +0.07 across all species.
- Established: at withheld places, for the SEDENTARY strata, no arm differs from control (every CI spans 0).
  For ALL species at withheld places, field_d gains +0.092 regional [+0.058, +0.128] and +0.075 all change
  (k4: +0.096 regional), and field_summer gains +0.081. So the all-species gain carries to unseen places, and it
  comes from the non-sedentary species. Likely: the summer field interpolates smooth regional trends of migrants
  from neighbouring training cells. Local change is unmoved everywhere (about 0.1).
- CORRECTION (14:00): "winter is not the route" holds only for THIS construction (shared latent, couplings
  learned 1996-2025). It does not show that winter is uninformative.
  - E086u found target-era CBC Ruzicka supervision inside DESK improves sedentary-common change: +0.080 with
    target-era winters vs +0.017 without, at sparse target data, all 4 seeds.
  - In this study the non-oracle winter map added +0.03 regional to field_d (0.49 -> 0.52, teacher space).
  - "Removing target-era winter helps" is CONFOUNDED: assim_d_no_target also uses a different winter teacher
    (its own), and its summer D is unchanged.
  - Resident-specific links were measured before. E066c: the season link for residents is about 0.52 temporal /
    0.83 spatial; migrants -0.06. E086r-b: own-species CBC conditioning added nothing for sedentary species on
    55-70 evaluation cells. E069-E081: CBC presence supervision redirected sedentary breeding change and harmed it.
- Reporting note (added 13:40 at the user's request): the primary cell (sedentary common, 21 species, regional) is
  the most favourable slice. The control is 0.458 there against 0.223 for all species, all change. Every
  headline from here on reports both.
- Established: the modern-endpoint MSE drops about 23% for every fitted arm; that is the anchoring.
- Established: routing winter change into summer through couplings learned 1996-2025 hurts.
- Likely: target-era winter carries the information. The oracle winter map reaches 0.68 regional
  (WINTER_TRANSFER_field_d_s1.json), but the winter -> summer map is not stationary across eras: the non-oracle
  map fitted on 1996-2025 contrasts gets 0.41 alone and 0.52 added to field_d (teacher space, all species).
- Unknown: whether a map constrained to be era-stable (e.g. per-species readout of winter directly, G11) recovers it.
  field_d_sep (third amendment, exploratory) tests the simplest version and is pending.
- TACC continuation of the change-loss model (continue100) is indistinguishable from control (-0.002).

Third amendment (2026-10-27 ~12:10, after the summer-only and shared-winter arms were scored, before this arm
exists): field_d_sep = field_d plus an INDEPENDENT winter field (own latent scores, no shared scores, no
winter -> summer coupling); the modern species readout reads both seasonal blocks. Post hoc and exploratory: it
is reported, not entered into the pre-registered step-change rule.

## Endpoint swap for field_d (pre-registered 2026-10-27 ~14:30, before any score; method of E055g/h)

Hybrid cubes, symlinked by year:
- swap_fdhist: field_d for 1902-1995 + control for 1996-2025 (field_d's HISTORICAL end only);
- swap_fdmod: control for 1902-1995 + field_d for 1996-2025 (field_d's MODERN end only).
Same scorer and panels; paired vs control.
Reading (primary k2 known sedentary common regional, plus k2 known all-species regional):
- HISTORICAL RECONSTRUCTION if swap_fdhist keeps >= half of field_d's paired gain (+0.037 / +0.074);
- MODERN-END ANCHORING if swap_fdmod carries more than half and swap_fdhist less than half.
Also reported: withheld places. Known-place modern evaluation halves are disjoint from training, so pure memorization
is excluded by design, but anchoring the modern end to nearby training rows of the same cells is not.

## In-era interpolation test (pre-registered 2026-10-27 ~14:45, before any run)

Question: when summer is observed through 1966-2025 at training places (the production situation for that era),
how well does an observation-anchored field reproduce 1966-85 -> modern change at WITHHELD places?
Nearest prior: E016 skeptic round 3. A position x linear-time placebo (24 RFF fields, 300 km, old r24 teacher, all
change, no regional split) scored 0.436 at held-out places at zero reach; DESK base 0.417 (E019 0.38-0.42).
Differences here: flexible spline time + 128 RBF + fine map, fresh summer64 teacher, regional/local split, residency
strata, species readout.
Design (research/exp/z_assim_interp_prep.py, z_assim.py --summer interp): add every valid summer row before 1996
outside the spatial holdout and its buffer to the summer sources (assert: none at a withheld panel cell). Arms
field_d, field_summer and trend_summer with the default cfg; export *_interp_s1. Score with the same evaluator;
ONLY withheld-place rows are valid (known-place historical evaluation rows were added to training).
Reading (withheld, k2, regional; all species and sedentary common):
- report each arm vs control and vs its own backcast version (field_d_s1 etc.);
- interpolation "largely solves" the production era if field_d_interp regional >= 0.6 for all species;
- below 0.5: in-era interpolation of change is itself limited at this panel's distances; report by distance to the
  nearest training cell if needed.
The control DESK never saw pre-1996 summer, so "vs control" mixes data and model; the main comparison is
interp vs backcast of the same arm.

### Endpoint swap results (14:58; ASSIM_SWAP.json; paired vs control, k2 regional [95% CI])

| panel | field_d | historical end only (swap_fdhist) | modern end only (swap_fdmod) |
|---|---|---|---|
| known, sedentary common | +0.037 [+0.017, +0.059] | -0.010 [-0.030, +0.010] | -0.058 [-0.083, -0.036] |
| known, all species | +0.074 [+0.058, +0.088] | +0.033 [+0.019, +0.047] | -0.103 [-0.122, -0.086] |
| withheld, sedentary common | -0.010 [-0.052, +0.044] | +0.017 [-0.026, +0.063] | -0.099 [-0.140, -0.058] |
| withheld, all species | +0.092 [+0.058, +0.128] | +0.085 [+0.055, +0.120] | -0.104 [-0.140, -0.072] |

Pre-registered reading:
- Primary cell (known, sedentary common): NEITHER criterion is met. The historical end alone carries none of the gain,
  and the modern end alone hurts. The gain needs both ends from the same model.
- Mechanism (likely): field_d's per-cell level offset u cancels inside its own change but appears as a spurious level
  jump in either hybrid. That is the coordinate-mismatch caveat of E055h. So the swap cannot attribute the
  known-place sedentary gain. It is NOT evidence of memorization: the modern evaluation halves are disjoint from
  training.
- Withheld places, all species: the historical end alone reproduces the gain (+0.085 of +0.092). There u is only
  smoothed, not fitted to the cell, so the mismatch is small. The all-species withheld-place gain is historical
  reconstruction (established for this panel).
- Known places, all species: the historical end carries 45% (+0.033 of +0.074), just under the pre-registered half.

### field_d_sep results (third amendment; exploratory, post hoc; ASSIM_SCORES_SEP.json, ASSIM_SEP_VS_FD.json)

k2 regional, paired vs field_d [95% CI]:
- known, sedentary common: 0.521, +0.026 [+0.002, +0.049] (vs control +0.063 [+0.039, +0.087]);
- withheld, sedentary common: 0.485, +0.053 [-0.010, +0.114];
- known, all species: 0.366, -0.026 [-0.038, -0.014];
- withheld, all species: 0.322, -0.047 [-0.076, -0.015].
Winter conditional composition vs field_d: worse in every era. 1902-1919 -0.166, 1920-1939 -0.329, 1940-1952
-0.208, 1953-1965 -0.106, 1966-1985 -0.026.
Reading:
- A winter field read directly by the species readout trades: residents gain, migrants lose. This is the direction
  of E066c's season link (residents 0.52, migrants -0.06) (likely).
- Its early-winter failure is likely the separate winter spline extrapolating where pre-1966 circles are sparse
  (unknown until a field confined to the winter data's support is tested).
- Not a candidate as built: the all-species and winter costs outweigh the resident gain.

### In-era interpolation results (ASSIM_INTERP.json; WITHHELD places only, k2; all / regional / local)

| arm | sedentary common | sedentary | all species | regional vs control, all species |
|---|---|---|---|---|
| control (DESK, no pre-1996 summer) | .313 / .443 / .105 | .276 / .403 / .092 | .186 / .277 / .074 | - |
| field_d (backcast) | .332 / .433 / .134 | .298 / .397 / .118 | .261 / .369 / .082 | +0.092 |
| field_d_interp | .491 / .664 / .157 | .432 / .601 / .137 | .387 / .547 / .119 | +0.270 [+0.221, +0.314] |
| field_summer_interp | .498 / .638 / .224 | .435 / .576 / .179 | .379 / .521 / .124 | +0.244 |
| trend_summer_interp (linear time) | .501 / .646 / .205 | .437 / .583 / .162 | .388 / .538 / .111 | +0.261 |

k4 all species, regional: field_d_interp 0.530, trend_summer_interp 0.523.
Reading against the pre-registered thresholds:
- All-species regional 0.547 falls between the "below 0.5" and ">= 0.6" readings. With 1966-1995 summer observed at
  training places, regional change at unseen places rises from 0.37 to 0.55 for all species, and to 0.66 for
  sedentary common species.
- A linear position x time trend does as well as the flexible field (0.538). Once the era is observed, a smooth
  neighbour trend captures what is interpolable; extra flexibility adds nothing here. This agrees with E016 (placebo
  0.436 at zero reach, a different estimand).
- Local change stays near 0.1-0.2 even with in-era data.
- Production implication (not a decision): for 1966-2025, observation-anchored Z at surveyed cells plus neighbour
  interpolation reaches about 0.55-0.66 regional at unsurveyed places. The covariate-only route gives 0.28-0.44. The
  gap between backcast (0.37) and interpolation (0.55) is the value of observing the era. Before 1966 that value is
  unavailable.

## Pre-registration: in-era interpolation DESKs (2026-10-27, before any training)

Question (user, 2026-10-27): does a DESK retrained from scratch with the same 1966-1995 summer rows, plus random
Fourier features of space-time coordinates, match field_d_interp? And how much comes from the data alone?

Nearest prior: no DESK with calendar-time or position inputs has been trained (DESK_EXPLICIT_SPACETIME_AUDIT_2026-10-26).
Position x time evidence so far is from predictors outside the encoder (E016, E033c, E039i). The 790 pilot (time-spline
modulation, no position, no pre-1996 summer) is still running.

Arms (research/exp/z_coord_run.py, frozen snapshot coord_code; masked-control recipe, seed 1, fixed 500 epochs, no
epoch selection on historical outcomes, CBC winter head unchanged):
- desk_interp: covariate-only DESK; summer rows = the control's 24,600 + the 31,693 pre-1996 training-place rows used
  by field_d_interp (asserted identical row indices; none at a withheld panel cell).
- desk_interp_coord: the same, plus 256 random Fourier features of (x/200 km, y/200 km, clamp(year, 1966, 2025)/10 yr).
  They enter DESK's pointwise hidden layer through one zero-initialised linear map, before the mixer; at initialisation
  the model is desk_interp. The clamp makes the coordinate term constant before 1966 rather than periodic. Scales match
  the field (200 km centres, 10-yr knots).

Readout and estimand as for the field interp arms: withheld places only (known-place historical evaluation rows are
training rows here); k2 headline, k4 reported; all / regional / local; sedentary common, sedentary, all species.

Decision rules (all-species regional, withheld, paired bootstrap):
- data alone: desk_interp - control > +0.05 with CI above 0 -> pre-1996 summer reaches DESK's change without
  coordinates.
- coordinates: desk_interp_coord - desk_interp > +0.05 with CI above 0 -> the coordinates carry the interpolable trend.
- "essentially the same effect": |desk_interp_coord - field_d_interp| <= 0.03 on all-species regional and sedentary-
  common regional. If desk_interp_coord is lower by more than 0.03, the field's separation (an explicit, penalised
  regional term) matters.
Same reading for sedentary common (primary per the residency rule), reported alongside all species.

Interpretation constraint (CONSIDERATIONS; houfin-regional-trend-not-dynamics): a position x time term predicts
regional change without identifying cause (habitat, dynamics or observation). A coordinate gain shows only that the
change is interpolable in space. Inside DESK, that term is no longer separable from the environmental geometry of Z.

### Result: desk_interp (data only; assim/backcast40/ASSIM_DESK_INTERP_interp.{md,json}; WITHHELD places, k2)

| arm | sedentary common (all / regional / local) | sedentary | all species |
|---|---|---|---|
| control | .313 / .443 / .105 | .276 / .403 / .092 | .186 / .277 / .074 |
| desk_interp | .396 / .554 / .138 | .340 / .496 / .111 | .278 / .422 / .082 |
| field_d_interp | .491 / .664 / .157 | .432 / .601 / .137 | .387 / .547 / .119 |
| trend_summer_interp | .501 / .646 / .205 | .437 / .583 / .162 | .388 / .538 / .111 |

Regional vs control: all species +0.145 [+0.099, +0.187] (k4 +0.145 [+0.100, +0.196]); sedentary common +0.112
[+0.038, +0.185]; sedentary +0.094 [+0.028, +0.160]. Local: +0.008 [-0.011, +0.029] (all species).
- Data-alone rule MET (+0.145 > 0.05, CI above 0): DESK uses pre-1996 summer rows without coordinates.
- It recovers about half of the field's gain (all species regional 0.422 vs 0.547; sedentary common 0.554 vs 0.664).
  Point difference vs field_d_interp: -0.125 all species, -0.110 sedentary common (difference of paired gains; direct
  paired CI not computed).
- Reading (likely): a covariate-only encoder can represent regional change only where covariate change tracks it. The
  added rows teach it the historical covariate -> community relation at training places, which partly transfers.
  The part of the regional trend not carried by covariate change is out of reach without a position x time term.
  desk_interp_coord tests that directly.

### Result: desk_interp_coord (ASSIM_DESK_INTERP_interp_coord.json; paired vs desk_interp: ASSIM_DESK_INTERP_coord_vs_interp.json)

Trained 500 epochs on TACC (task 13, 8.6 s/epoch, VRAM 24.5 GB), the same recipe as desk_interp plus the 256 RFF
coordinate features. WITHHELD places, k2, all / regional / local:

| arm | sedentary common | sedentary | all species |
|---|---|---|---|
| desk_interp | .396 / .554 / .138 | .340 / .496 / .111 | .278 / .422 / .082 |
| desk_interp_coord | .411 / .567 / .134 | .357 / .508 / .113 | .294 / .431 / .094 |
| field_d_interp | .491 / .664 / .157 | .432 / .601 / .137 | .387 / .547 / .119 |

Paired vs desk_interp, regional [95% CI], withheld: sedentary common +0.013 [-0.030, +0.055]; sedentary +0.012
[-0.027, +0.051]; all species +0.009 [-0.018, +0.037]. Training places (known panel): all species +0.021 [+0.012, +0.030];
sedentary common +0.014 [+0.002, +0.026].
field_d_interp vs desk_interp, regional, withheld: sedentary common +0.110 [+0.051, +0.177]; all species +0.125
[+0.085, +0.164] (this is the direct paired CI that was missing above).
- Coordinate rule NOT MET (+0.01, CI spans 0).
- "Same effect" NOT MET: the field is ahead by 0.116 (all species) and 0.097 (sedentary common) regional.
- The coordinates were used: coord_proj grew from zero to Frobenius norm 24.1, compared with 17.2 for mixer[0]. They
  give a small gain at training places that does not transfer to withheld places.
- Reading (likely, not tested): the field fits the change in Z directly, with an explicit smooth 200 km x 10-yr
  regional term under its own penalty. DESK sees the coordinates only through a pointwise nonlinear encoder whose loss
  is mostly covariate reconstruction and metric heads. What it learns from them fits the training places rather than
  forming a smooth surface that carries to the gaps. A second possible contributor (unknown): 256 plane waves at a
  200 km / 10-yr scale cover a continent x 60 years only coarsely, whereas the field's 128 localized RBFs x splines
  are rank-limited but local. Answer to the comparison question: with the same data, the simpler design does NOT
  reproduce the field. The explicit, separately penalised regional term is what carries the remaining ~0.1.

#### Correction to the desk_interp_coord reading (2026-10-28)
Two parts of the reading above were wrong:
1. DESK's loss is NOT "mostly covariate reconstruction". Weighted shares at epoch 500 of interp_coord_s1 are: winter-half
   coordinate head ~36%, summer teacher fit (stab) ~27%, presence ~18%, reconstruction ~8%, metric ~3%.
2. The coordinates did NOT "fit the training places". The known-panel gain over desk_interp is only +0.021 regional
   (all species), against +0.124 for field_d_interp. The coordinate path UNDERFITS regional change even where it is
   observed, so the gap is not a transfer failure.
Candidate causes (unknown, untested):
- the 256 global plane waves cannot represent a localized 200 km x 10-yr surface;
- the term enters before a nonlinear mixer, under dropout 0.1 and weight decay 1e-4;
- in DESK the same capacity must also carry the static spatial structure, whereas field_d fits on top of D-bar;
- the target is the causal output EMA (learned half-life 10.9 y), which lags a step in z_raw.

## Pre-registration: comparable-conditions DESKs (2026-10-28, before any training)

Question (user): can a simpler DESK match field_d_interp under comparable conditions? desk_interp_coord was not
comparable on two counts:
1. Optimization budget: DESK takes ONE full-batch step per epoch (500 updates, peak lr 1.5e-3, cosine to 5%;
   sum of lr about 0.39), while the field takes 2000 Adam steps at 0.02. The zero-initialized coordinate map barely moved.
2. Basis: it used 256 global random Fourier features rather than the field's local basis.

All four arms use desk_interp's data (control rows + 31,693 pre-1996 summer rows at training places). Code:
research/exp/z_field_run.py and z_field_models.py; snapshot field_code. New modules train in their own AdamW group at
lr 0.02 (the field's) with no weight decay, under the same epoch schedule. Field penalties (lam_C 1e-2, lam_curv 1.0,
loading scale) are multiplied by the stabilizing weight (64), so their ratio to the summer-teacher MSE matches z_assim.
Basis = z_assim's R exactly: [1, 128 Gaussian RBF at 200 km, centres by farthest-point selection over field_d_interp's
observation cells] x cubic B-spline (10-yr knots, 1902-2025, centred on 2006-2025), rank 16.
- A fieldout: Z_summer += S C T L, added after the spatial residual and the decoder, trained jointly.
- B fieldin: the same S (x) T features (129 x 16) into the pointwise hidden layer, through a zero-initialized map.
- C fieldout_change: A + ChangeVectorLoss. Contrasts span six decades, 1966-2025: adjacent pairs and pairs two apart,
  7,988 contrasts in 1,847 cells. The weight is FIXED at change100's calibrated 279.907, not recalibrated.
- D rff_fast: desk_interp_coord unchanged except that coord_proj is in the fast group (lr 0.02, no decay). This tests
  the budget explanation on the original design.

Scoring: z_assim_evaluate, WITHHELD places, k2 headline; sedentary common first, then sedentary and all species;
regional / all / local. Paired vs desk_interp_s1 (same draws).
Decision rules (regional, withheld, sedentary common AND all species):
- "Matches the field": |arm - field_d_interp| <= 0.03 (field: 0.664 sedentary common, 0.547 all species).
- "Helps": arm - desk_interp > +0.05 with CI above 0.
- A vs B (placement given basis); D vs desk_interp_coord (budget given the RFF design): differences > 0.05 count.
- C vs A: change loss on top of the field term; > +0.03 with CI above 0 counts.
Training-place (known-panel) gains are reported, to separate underfitting from failure to transfer.
Interpretation constraint (CONSIDERATIONS; houfin-regional-trend-not-dynamics): a position x time term predicts
regional change without identifying cause. A, B and D test interpolability and optimization, not ecology. C's change
targets are teacher-coordinate contrasts at training cells only.

### Result: arm A fieldout (ASSIM_FIELDDESK_fieldout.json; paired vs desk_interp; WITHHELD, k2)
- Regional: sedentary common .540 (-0.015 [-0.058, +0.029]); sedentary .477 (-0.019 [-0.055, +0.017]); all species .421
  (-0.001 [-0.029, +0.024]).
- Known panel: all-species regional -0.030 [-0.037, -0.022].
- Rules: "helps" NOT MET; "matches field" NOT MET.
- The term was learned. Its regional change 1970 -> 2020 has RMS 0.033 over cells, against 0.015 for field_d_interp's R
  (C norm 2.20 vs 1.92). The null is therefore not an optimization-budget failure.

## Pre-registration: score DESK cubes as their trainer's supervised quantity (2026-10-28, before scoring)
Finding (established, from code): DESK supervises z_ema = causal output EMA of z_raw (learned half-life about 11 y).
build_spacetime_cube exports z_raw (desk_training.apply_output_ema's docstring: "every downstream export is raw, so
anything grading DESK against an observed community has to reconstruct z_ema"). z_assim_evaluate applies no EMA.
Every DESK-vs-field comparison in this file therefore scored DESK's raw output, while the field was scored on the very
quantity it was fitted to. The observation audits (z_repair_ema_audit) did apply the EMA; these species scores did not.
Rescore (research/exp/z_ema_cube.py): <cube>_ema = apply_output_ema over 1902-2025 at each model's TRAINED half-life,
all 128 dims, as in z_repair_ema_audit. Arms: control_s1, desk_interp_s1, desk_interp_coord_s1, desk_fieldout_s1,
change100 and continue100. Field arms are unchanged (fitted raw).
Same estimands and tiers. Paired comparisons: vs desk_interp_s1_ema (desk arms) and vs continue100_ema (change100); each
_ema arm also vs its own raw cube. The field-match rule (|diff| <= 0.03) is applied to the _ema scores.
Applying the EMA is not a free choice: it is the quantity DESK was trained to produce. It may raise or lower
correlations; both are reported.

### Result: DESK scored as its supervised EMA (ASSIM_EMA_DESK.json, ASSIM_EMA_CHANGE.json; WITHHELD, k2, regional)
| arm | sedentary common | sedentary | all species |
|---|---|---|---|
| control_s1 raw / ema | .443 / .355 | .403 / .325 | .277 / .212 |
| desk_interp raw / ema | .554 / .634 | .496 / .562 | .422 / .503 |
| desk_interp_coord ema | .646 | .574 | .508 |
| desk_fieldout (A) ema | .638 | .563 | .506 |
| field_d_interp (raw = its fitted quantity) | .664 | .601 | .547 |
| continue100 raw / ema | .442 / .353 | .402 / .324 | .277 / .213 |
| change100 ema | .401 | .367 | .248 |

- desk_interp: ema - raw = +0.079 [+0.032, +0.132] sedentary common, +0.081 [+0.049, +0.113] all species.
- field_d_interp - desk_interp_ema: sedentary common +0.0305 [-0.0065, +0.080]; sedentary +0.039 [+0.000, +0.085];
  all species +0.044 [+0.020, +0.065].
- "Matches the field" (|diff| <= 0.03): NOT MET, narrowly. Sedentary common misses by 0.0005 with the CI spanning 0;
  all species misses by 0.014 with the CI above 0. Most of the earlier gap (0.110 / 0.125) was the raw-vs-EMA scoring
  mismatch, not the model.
- On the EMA scale the additions are null. Coordinates: +0.012 [-0.023, +0.050] sedentary common, +0.005 all species;
  at known places +0.031 [+0.026, +0.036]. Field term A: +0.005 / +0.003; at known places 0.000.
- Change loss on the EMA scale: change100_ema - continue100_ema +0.048 [+0.020, +0.078] sedentary common, +0.035
  [+0.017, +0.050] all species. But both EMA levels sit below their raw cubes (continue100 raw .442 / .277).
- Models trained WITHOUT pre-1996 summer rows score WORSE as EMA (control: -0.088 sedentary common, -0.065 all species).
  Models trained WITH them score better (+0.08).
- Reading (likely): the EMA is supervised only in years with teacher rows. Without pre-1996 rows, z_ema before 1996 is an
  unsupervised lagged mix running from 1902, and the raw output tracks change better. With them, z_ema is fitted in
  1966-1995 and is the right quantity. So raw-vs-EMA is not a free choice. The supervised quantity is EMA, but which of
  the two scores better depends on which years were supervised.
- Consequence for earlier records: desk-vs-control conclusions that used raw cubes for both are unaffected, since raw is
  control's better scale. Comparisons of DESK with field arms (fitted raw) understated DESK whenever pre-1996 rows were
  supervised.
- Arms B, C and D (TACC) will be scored on both scales.

### Result: arms B fieldin and C fieldout_change (2026-10-29; ASSIM_FIELDDESK_fieldin.json, ASSIM_FIELDDESK_fieldout_change.json)
Withheld places, k2, regional. Paired vs desk_interp_s1_ema (same draws). EMA scale = each model's trained quantity.

| arm (EMA) | sedentary common | sedentary | all species |
|---|---|---|---|
| desk_interp | .634 | .562 | .503 |
| A fieldout | .638 | .563 | .506 |
| B fieldin | .627 (-0.007 [-0.048, +0.038]) | .560 (-0.003 [-0.039, +0.036]) | .506 (+0.003 [-0.019, +0.025]) |
| C fieldout_change | .573 (-0.061 [-0.114, -0.010]) | .515 (-0.047 [-0.090, -0.002]) | .477 (-0.026 [-0.054, -0.003]) |
| field_d_interp | .664 | .601 | .547 |

Raw scale (regional, sedentary common / all species): B .518 / .406; C .408 / .369; desk_interp raw .554 / .422.
Both new arms are below desk_interp on the raw scale too.

Rules:
- B: "helps" NOT MET. "Matches field" NOT MET (-0.037 sedentary common, -0.041 all species). B vs A: -0.011 / 0.000, so
  placement of the field term (input vs output) makes no difference.
- C: C vs A = -0.065 sedentary common, -0.029 all species. The rule (> +0.03) is NOT MET; the change loss HURTS at
  withheld places. "Helps" and "matches" are NOT MET.

Known panel:
- B: regional +0.015 [+0.006, +0.025] sedentary common, +0.026 [+0.021, +0.031] all species.
- C: regional +0.031 [+0.015, +0.044] / +0.057 [+0.049, +0.065]; local +0.19 / +0.17.

Reading:
- Established: C fits training-place change much better (local +0.19) and transfers worse (withheld regional -0.06
  sedentary common). That is overfitting of place-specific change, not underfitting.
- Likely: the change targets are teacher-coordinate contrasts at training cells only. They include observer turnover
  (~30% of change; E028, E034b), and the fast field term plus shared weights absorb place-specific contrasts that the
  covariates cannot carry to new places.
- Unknown: why this differs from change100, which gained on the EMA scale (+0.048 vs continue100). change100's base had
  no pre-1996 rows, so its change loss supplied historical information. Here desk_interp already supervises those
  years, so the loss mainly re-weights training-place change. Untested.
- Comparable-conditions conclusion so far (A, B, C): putting the field's own basis inside DESK, at either placement and
  with the field's optimizer budget, does not close the 0.03-0.04 gap to field_d_interp at withheld places. Arm D
  (budget on the RFF design) is still running.

### Result: arm D rff_fast (2026-10-29; ASSIM_FIELDDESK_rff_fast.json)
Withheld places, k2, regional. EMA scale; paired vs desk_interp_s1_ema.
- Sedentary common .568 (-0.065 [-0.131, +0.002]); sedentary .504 (-0.059 [-0.116, -0.002]); all species .440
  (-0.063 [-0.096, -0.028]).
- Raw scale: .473 / .331 (desk_interp raw .554 / .422).
- Known panel: regional +0.049 [+0.036, +0.063] sedentary common, +0.063 [+0.055, +0.070] all species; local +0.10.

Rules:
- D vs desk_interp_coord (.646 / .508): -0.078 sedentary common, -0.068 all species. The difference exceeds 0.05, in
  the HARMFUL direction. More optimization budget for the RFF coordinate term overfits.
- "Helps" and "matches field" are NOT MET.

Reading:
- Established: the same pattern as C. Training-place regional change fits better (+0.05 to +0.06) and transfer to
  withheld places is worse (-0.06). The budget explanation for desk_interp_coord's null is refuted in the direction
  that matters: with budget, the coordinate path learns training-place change that does not carry to new places.
- Contrast with A, the same fast budget on the field's local RBF basis: null on both panels (known 0.000, withheld
  +0.005). So A's term was learned (RMS larger than field_d's R) but neither fit nor hurt.
- Likely: 256 global plane waves at high frequency can represent cell-scale patterns between training places, while
  200 km RBFs cannot.

Comparable-conditions conclusion (A-D):
- No variant of a space x time term inside DESK closes the 0.03-0.04 regional gap to field_d_interp at withheld places.
  The variants were: input or output placement, the field's own basis and budget, change supervision, and an RFF term
  with budget.
- Terms flexible enough to fit training places' change (C, D) overfit. The local term (A, B) does nothing.
- Likely: field_d's edge comes from being fitted as a separate stage on top of the frozen DESK mean, where its
  penalties are the only regulariser. It does not come from capacity or optimizer budget. Untested directly.

## Pre-registered 2026-10-30: is the FIELD output better as an EMA? (user: should production Z be EMA'd?)
Plan (FIELD54_NATIVE_PLAN, user-accepted 2026-10-29): the field output is not EMA'd (fitted to the teacher coordinates
directly); EMA applies to DESK's D input, decided by the twin. Untested: the field output itself as an EMA.
Arm: field_d_interp_s1_ema = apply_output_ema(field_d_interp_s1) at interp_s1's trained half-life (z_ema_cube.py;
nothing selected on evaluation data). Scored with z_assim_evaluate (WITHHELD, k2, regional; sedentary common first,
then sedentary, all species) next to field_d_interp_s1 raw; out ASSIM_FIELD_EMA. pueue 933 / 934.
Rule: EMA is adopted for production only if it beats raw by >= .02 on sedentary common AND all species with the paired
CI above 0; otherwise raw stays (EMA adds ~a decade of lag to habitat change, which the age model reads as timing).
Prediction: EMA scores lower (as control_s1: -0.088), since the field was not fitted as an EMA.

### Result (2026-10-30, pueue 934; evaluation log ASSIM_FIELD_EMA): the field output is WORSE as an EMA
Half-life 11.03 y (interp_s1's trained value). Regional change correlation, k2:
| | withheld sed. common | withheld sedentary | withheld all | known sed. common | known all |
|---|---:|---:|---:|---:|---:|
| field_d_interp raw | .664 | .601 | .547 | .710 | .611 |
| field_d_interp EMA | .549 (-.115) | .493 (-.108) | .466 (-.081) | .667 (-.043) | .565 (-.046) |
Raw reproduces the recorded .664 / .547 exactly. Rule (EMA adopted only if >= +.02 on both strata, CI > 0): NOT MET --
EMA is lower everywhere, by more than twice the typical paired CI half-width of these comparisons (no paired CI
computed for this pair; the sign and size are not in doubt). Prediction (EMA lower, as control -0.088) held.
Reading (likely): the field is fitted to the teacher coordinates year by year with a 10-year spline already in its
time basis; a further 11-year causal EMA only lags its change (~a decade), which the change estimand penalises.
Decision: production Z (native 54 km build) is the RAW field output, as the plan says. The DESK-input question (D raw
vs EMA for the field fit) is unaffected and is still decided by the twin.
