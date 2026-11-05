# DESK representation and downstream-model synthesis

2026-10-26. This distinguishes completed evidence, fixed-point diagnostics,
historical results with limitations, and work that has not been completed.
Community structure is to be learned at the community level; House Finch should
consume a frozen Z with a shared space-for-time readout.

## 1. What the clean seasonal comparisons establish

Final v1 supplies 64 summer coordinates. Final v2 supplies 64 summer plus 64
winter coordinates, with the sum of seasonal kernels. Full128 is therefore a
change in information, feature count and prior covariance energy, not a pure
rank experiment. Equal coefficient variances do not make its total prior field
variance equal to a single-season64 model. That is a property of the declared
sum kernel, not evidence of a normalization bug.

Export contracts and seasonal coordinate selection were corrected and tested.
The CBC completeness calculation had used excluded membership; clean encoders
were retrained using training-only completeness. Clean comparisons use matched
two-seed ensembles, not two seeds versus the older four-seed means. Most summer
scores changed little after this correction; the original contamination remains
relevant to provenance even when the score intervention is small.

With full modern calibration, clean full128 improves heldout summer species-change
correlation over clean v1: +0.026 [0.008, 0.043] for a temporally recalibrated
readout, and +0.051 [0.028, 0.071] for a spatially fitted readout applied to change.
The latter is closer to the desired space-for-time transfer, although neither is
the mechanistic population model. At 3% calibration data, that spatial-readout gain
persists (+0.027/+0.035 in two thinning realizations), while the temporal-readout
comparison becomes negative (−0.020/−0.027). More dimensions are not uniformly
helpful under sparse temporal calibration.

Historical winter results differ. Relative to each model's own modern-feature
placebo, full128's temporal gain is lower than v1 by −0.195 [−0.331, −0.057] in
1902–1919 and −0.155 [−0.214, −0.104] in 1920–1939. Winter64 alone is worse;
v2 summer64 is near v1 in those early comparisons. This locates the early
disadvantage in the winter/full combination more specifically than "128 overfits."
It does not explain its cause. Both v1 and full128 still improve over their own
modern-feature placebo in the first era: approximately +0.538 and +0.342.

These are conditional recorded-list composition scores, not latent abundance
truth. The earliest era has 74 lists at 35 circles; the same-circle sensitivity
has only ten lists at three circles. Readouts are trained modernly. There is no
early-summer observational validation. Spatial holdouts are reused, encoders
receive other locations over the timeline, and these are exploratory comparisons.

Sources: FINAL_CLEAN_RESULTS.md, FINAL_CLEAN_SEASONAL_RESULTS.md and their JSONs.

## 2. Why a single "ceiling" is not an established explanation

The expensive experiments measure different things: community-coordinate
direction, distances between changes, species-specific change, continental mean
change, or observation-aware population prediction. Their scores cannot be pooled
as one variance budget or treated as independent methods repeatedly hitting an
identical limit.

Several experiments isolate real bottlenecks:

- E086c: much apparent benefit from multiple linear heads disappeared when the
  baseline could read all64 coordinates. Some methods were alternative access to
  the same span, not genuinely independent information sources.
- E086m: in its archived older representation, spatial kernel fidelity was .910
  but change-distance fidelity .025. The latter concerns ranking the SIZE of
  change, not direction. High spatial fidelity did not certify temporal geometry.
- E086x/y/y2: even observed community geometry plus the tested linear readout
  missed important target changes; changing reference species and seasonal
  information improved the observed-kernel results. Thus neither the NN alone nor
  irreducible noise explains everything.
- E066: the96-species kernel's effective contribution count was approximately27;
  an expanded201-species pool had approximately39. Many named species do not
  guarantee equally weighted independent ecological signals.
- E086u: target-era winter observations did help in a sparse-data ablation.
  Removing those winters reduced a gain from +0.080 to +0.017. Historical data
  can contain useful information; the transmission route matters.
- E070/E086v-b: poor historical counts distort count similarities; correcting a
  presence observation operator removed both an apparent harm and an apparent
  gain. This identifies an observation-to-representation bottleneck. It does not
  establish that the current clean128 early objective is inert; that remains
  unmeasured.
- E055g/h: known-place change-score improvements could arise from improving the
  modern endpoint. The archived endpoint-swap results were reproduced exactly.
  Historical reconstruction must be evaluated separately from modern memorization.

The E086 reliability headline was also too loosely described. In the reproduced
analysis, median split-half64D change-vector cosine was about .82, not near-perfect
alignment. The .8–.9 figures included per-coordinate reliability calculations.
Repeatability is substantial, but can include persistent route/observer effects.
It rules out dismissing the entire prediction gap as independent sampling noise;
it does not certify all repeatable change as ecological truth.

## 3. Regional structure becomes more prominent over longer spans

E014's .558 nearby value was a cross-route change covariance divided by average
same-route energy for one approximately40-year contrast. It was not Pearson
correlation, a fitted decay rate, or a universal predictability ceiling. A fresh
reproduction matches its original numbers to floating-point precision.

The new local extension keeps survey effort, routes and pairs constant across
5/10/20/30-year contrasts. At30–60km, C/V rises from .069 to .325 on the
2016–2019 endpoint panel (211 routes,70 pairs), and .007 to .380 on the
2006–2009 endpoint panel (199 routes,62 pairs). Paired30-minus5-year differences
are +.255 [.104,.376] and +.374 [.126,.496]. More distant bins also increase.
The anchor checks partly reuse records and are not independent replications.

Shared covariance itself grows, not just the ratio. But V−C also grows: these
results show increasing relative regional sharing, not local deviations vanishing
or a demonstrated restoring mechanism. Matched5-versus40/50-year panels support
increases but have changing support; a single strict5–50-year panel is too small.
Nearest-neighbor and long-span unchanged-observer comparisons remain weakly
supported. Some increase persists with unchanged observers through20years.

This supports the premise that multidecadal regional reconstruction is worth
testing. It does not establish that a fitted method can recover it out of time,
or extend the breeding result to1902 winter and unobserved early summer.

Source: CHANGE_COHERENCE_BY_SPAN_2026-10-26.md and its result artifacts.

## 4. The current lambda pathology, measured on the exact displayed fit

The downloaded figures show pre-clean final-v1 four-seed mean features, read by
the old correlated split readout. The accepted restart passes its configured
plateau gate but ends with raw gradient norm65379. It is a retained diagnostic
point, not a securely converged final ecological model.

A new local NumPy replay reconstructs the exact coefficient vectors, inherited
occupied-habitat centering, links and fundamental eigenvalue. It matches the
refined figure's mean lambdas to floating-point precision and MAP to <1e−8.
No Z transform, rescaling, prior change, refit or population simulation is used.

Fundamental lambda is the post-establishment, density-independent local
eigenvalue. It excludes dispersal, disease mortality, density limitation and the
Allee factor. The plotted change is therefore not directly a plotting artifact
of Allee suppression, although those processes influence which fields the
population likelihood rewards.

| Geographic support | MAP early → modern lambda | Refined early → modern lambda |
|---|---:|---:|
| All model land | .737 → .825 | .543 → .992 |
|95–80°W,25–49°N | .860 →1.035 | .459 →1.407 |
|East of80°W,25–49°N | .789 → .872 | .585 →1.045 |

The regional masks are geographic bands, not exact political CONUS boundaries.
The same Z supplies both stages. Central-east change increases from .175 to .948
and all-land change from .088 to .449. Refinement therefore amplifies the effect
by roughly fivefold without changing Z.

What changes in the readout is concrete:

| Quantity | MAP | Refined |
|---|---:|---:|
|Adult-survival amplitude parameter |1.107|6.693|
|Juvenile-ratio amplitude parameter |1.783|10.415|
|Adult temporal coefficient norm |1.225|10.270|
|Juvenile temporal coefficient norm |1.825|15.319|
|Adult spatial/temporal coefficient cosine |.560|.032|
|Juvenile spatial/temporal coefficient cosine |.551|.735|

The adult temporal vector becomes almost orthogonal to its spatial vector. The
fixed .14 ratio is a ratio of amplitude parameters, not a bound on realized
change. Raising the spatial amplitude raises the temporal amplitude with it;
the independently fitted directions and latent coefficient magnitudes still
matter. The fecundity intercept also rises from4.961 to13.500 on its softplus
argument scale, while fecundity's temporal field becomes smaller.

Fixed-point interventions sharpen the distinction. Keep refined spatial fields
and intercepts, but substitute MAP temporal vectors: all-land lambda change falls
from .449 to .091; central-east change from .948 to .194. Keep refined temporal
directions but reduce their norms to MAP norms: the changes are .067 and .151.
Holding adult survival's temporal departure at zero leaves all-land change .157;
holding the juvenile ratio's departure at zero leaves .364; holding fecundity's
departure at zero leaves .449. These interventions do not add because links and
the eigenvalue interact. They are not refits and make no predictive-skill claim.

## 5. What this does and does not say about the new basis's contribution

On the same land cells and1902–1915 to2010–2025 contrast, final-v1 has spatial
trace .445 versus .470 in the earlier CBC1902 r64 representation; century change
energy is .128 versus .170. Change RMS / spatial SD is .537 versus .602. In the
central-east band it is .910 versus1.050, and in the eastern band .937 versus .975.

Thus an increase in aggregate raw feature movement is not observed in this
comparison. **That does not exonerate the new basis.** Covariance orientation,
where movement occurs, its alignment with fitted survival vectors, and the cost
of producing a particular field under the prior can change even if total energy
falls. The earlier and current bases also represent different community
information. These summaries do not identify how much the new basis contributes
to the fitted solution.

The remaining causal comparison requires adequately optimized, matched old/new
representations under the same shared readout, with population and observation
settings held fixed, and more than one start if basins differ. Invalid or
memory-failed fits cannot answer it. The fixed-Z replay identifies amplification
within the current fit; it is not that across-basis experiment.

No new input-Z normalization was introduced. The inherited centering anchors
intercepts but does not rescale coordinates. The separate modern calibration
pilot normalizes candidate kernel traces for comparison; that exploratory
operation was not applied to exported Z or population inputs.

Artifact: lambda_audit_20261026/basis_readout_audit.json; implementation:
research/exp/final_lambda_basis_audit.py. Original checkpoints and source notes
are retained beside the artifact; original input metadata/tensor are cached
under .research_runtime/lambda_audit_20261026.

## 6. What earlier downstream experiments already established

These are age-model experiment identifiers, distinct from similarly numbered
community experiments. Copies of the relevant source notes are in
lambda_audit_20261026/notes/.

E047's frozen-block tests located contortion in habitat reshaping, but its base
had a subsequently rejected dispersal clamp; it is conditional evidence, not a
clean current causal result. E050 varied literature dispersal and still saw
refinement contortion, but several runs were unconverged or precision-confounded.
E054 separated those cases: some gains repaired float32/float64 invasion timing,
whereas agreeing runs still gained from eastern positives, western temporal
decline and plains zeros. There is more than one data pressure.

E055 showed pinning absolute Allee strength could relocate an absence switch
into low K; relative Allee removed that category by construction without fixing
demographic contortion. E056's larger K-trend budget could manufacture enormous
century-scale growth in capacity. These are evidence of substitutions among
mechanisms, not reasons to adopt a relative Allee law or loosen another trend.

E063's corrected knife-edge analysis found steep timing sensitivity under both
mean-field and occupancy formulations: roughly20years per5% change in F near the
tested threshold. Its early "occupancy cures precision" claim was withdrawn.
Those tests do not transfer an exact sensitivity number to the current fit, and
some historical refits carried a later-discovered numerical floor.

E057 and E067 already tested intercept drift and shared regional time-varying
habitat corrections, including recorded null/recovery checks. They sometimes
helped particular spatial patterns, but did not remove demographic contortion;
several real fits remained unconverged or demanded large corrections. These
were downstream adjustments, not a validated repair of community Z.

E075's no-floor audit found the broad contortion across numerical regimes.
E078/E079 located sharp establishment sensitivity even without a hard floor.
E080 showed that suppressing temporal habitat could remove the artificial
corridor without fixing fecundity demand; however its static comparisons hit
iteration caps, and its .14 calibration belongs to an older representation.
They do not validate that ratio for current encoders, nor validate static habitat
as the final answer.

The user's proposed tradeoff is therefore a serious mechanism to test: low early
eastern suitability can suppress historic occupancy, later high suitability can
accelerate expansion, and K/Allee/dispersal can trade against those fields. It is
not yet established that this alone causes the current rise. A current
observation-level gain decomposition and frozen-field/trajectory tests are still
needed; modern aggregate holdout scores cannot identify historical vital rates.

## 7. Numerical problems and their present status

The current shared-readout models have no hard FFT floor, set fft_pad=0 and
absolute aggregated mate finding k=3, and include a10-year release-cell Allee
exemption. Old floor failures cannot simply be assigned to them. A local
reproduction-only precision audit nevertheless demonstrates extremely steep,
smooth establishment transitions; it does not locate a full-model defect.

Separately, all three shared-readout refinements failed allocating large buffers
in habitat projection. This was an implementation failure. The corrected source
adds optimization barriers to the two per-year data slices before projection;
it changes compiler scheduling opportunities, not field equations or precision.
Small-reproduction value, gradient and Hessian checks agree. The actual winter64
MAP point now passes the complete GPU gradient evaluation and sampled finite
differences (<.2% discrepancy), and local L-BFGS has passed the former allocation
failure and begun reducing U. That is not yet a converged refinement.

The previous diagnostic job also failed because the uploaded script did not
accept --steps. It was not a model failure. Earlier statements implying these
corrections had already been resubmitted were inaccurate; they were resubmitted
during this investigation:

- 3535014: corrected clean-full128 stalled-point audit, two-hour TACC dev, pending
 at the last queue check.
- 3535018: corrected shared-full128 refinement, two-hour TACC dev, pending.
- Local784: corrected winter64 shared refinement, running and past iteration320
  (U144766→138466, raw gradient norm approximately537 at that iteration).
- Local785: corrected summer64 shared refinement, queued behind784.

All restart from completed MAPs, preserve the failed outputs, retain the original
convergence guards and keep one shared coefficient vector per demographic field.
No .14 ratio is active in these shared-readout fits. Diagnostics are produced
only after a refinement passes its existing validity gate. Their results remain
pending; the old split-readout figure must not stand in for them.

## 8. Actual status of Z repair and priorities

Completed: clean seasonal evaluations, feature geometry diagnostics, original
experiment/root-cause audits, the multi-span coherence extension, and one modern
seasonal metric-reweighting pilot. The pilot uses1548 cells,152 calibration and
121 validation development species, spatial folds, and1996–2005 versus2016–2025
ABBA targets. It selected summer-only geometry in its limited family and improved
held-species proxy losses. Its target is a log-abundance Gram proxy rather than
the original Ruzicka kernel, and its conditional bootstrap does not include full
fitting uncertainty.

Not completed: a historical winter calibration objective, direction-dependent
community metric calibration, demonstration that the current early winter
objective actually constrains early trajectories, or a validated replacement Z.
No repaired representation has been exported or promoted. Seasonal reweighting
on modern breeding data is not a substitute for these tasks. The .14 ratio has
not been re-estimated and should not be treated as current calibration.
The encoders themselves do include early winter supervision; it is the separate
metric-calibration pilot that omitted historical winter targets.

The next decisions require separating three questions: whether observed regional
historical signal is recoverable with the available observation operator;
whether a frozen community representation captures it; and whether the
population model consumes that representation without explaining invasion via
unsupported habitat trends. Historical winter validation must use the existing
effort/composition logic and independent endpoint controls. Early summer remains
an inference without direct observations. Any community-level correction must
be fitted without House Finch feedback; focal split temporal coefficients cannot
serve as a validated repair of Z.

The existing evidence supports continuing this investigation, but does not yet
justify a final encoder choice, a new calibrated temporal amplitude, an
unrestricted space-time network, or a claim that downstream misspecification
alone accounts for the problem.

## 9. Restored core plan: community Z repair and controlled space-time DESK

The preceding synthesis omitted the architecture and reconstruction program that
motivated the morning's discussion. It remains the core work, not a discarded
idea or something contingent on completing the House Finch refits. The concrete
plan is [COMMUNITY_Z_REPAIR_PLAN_2026-10-26.md](COMMUNITY_Z_REPAIR_PLAN_2026-10-26.md).

The first explicit-space-time candidate is a low-rank regional community residual
added to a covariate encoder and frozen before downstream inference. The second
is a low-rank time modulation of a small environmental-response block inside
DESK. An existing-feature PSD metric correction is a parallel cheap control; it
cannot create absent information and is not a gate on the two encoder routes.
There is no initial unrestricted latitude/longitude/year concatenation, no
species-specific temporal repair and no inherited .14 amplitude.

The decisive validation emulates the sparse historical problem: hide all summer
supervision for a multidecadal target era, retain controlled sparse winter
community observations, then score unseen summer observations and independent
modern repeats. Refit both control and candidates under the mask, including
teacher/coordinate targets, derived trends and completeness. Also test actual
1902–1939 winter reconstruction with held circles/regions and proper observation
operators. Early summer remains unobserved. Existing GP/position-time baselines,
no-target-era-winter ablations, endpoint controls and correction null/recovery
checks are required to decide whether extra expressiveness earns its place.

Execution update: the validation/mask package, training-only sources, fresh
summer/winter ESK teachers and winter observation auxiliaries are complete.
Candidate modules and isolated time-conditioned training/export are implemented
and tested. The matched control, time-conditioned pilot, residual fit and
automatic paired evaluations are queued. Historical direction-dependent metric
calibration remains unimplemented. See
[the execution record](COMMUNITY_Z_REPAIR_EXECUTION_2026-10-26.md) for concrete
task IDs, resources, paths and outstanding checks. The plan maps this work to
already tested E073/E086e, GP, seasonal and downstream drift experiments.

## 9. Shared-readout refinements: results (recorded 2026-10-29)

Local tasks 784 (winter64) and 785 (summer64) finished 2026-10-26 16:13; first reported in final_shared_round1_20261025/evening_20261026/STATUS.md (figures in research/validation_figures/shared_clean_{s64,w64}_refined/). This section adds the MAP-stage comparison. Both stopped
on the tol_du rule (U fell < 0.05 over 20 iterations); final |grad| 4970 (w64, iter 1810) and 742 (s64, iter 4038), so
the plateau rule is met but the gradient is not small. Source: /home/breallis/houfin/work/houfin/research/
FINAL_SHARED_ROUND1_20261025/{w64,s64}/viz_map/metrics.json and projectionfix1/{w64,s64}/viz_ref/metrics.json.
All-land mean fundamental lambda, early = 1902-15, modern = 2010-25. Temporal holdout = years 2021-25 (a 5-year
forecast); spatial holdout = 10% of 8-cell blocks. Mean log density, higher is better. No paired intervals.

| Fit | lambda early -> modern | change | suitable early -> modern | temporal lpd | spatial lpd |
|---|---|---:|---|---:|---:|
| final-v1 split readout, cold MAP (sec. 4) | .737 -> .825 | +.088 | - | -1.941 | -1.316 |
| final-v1 split readout, refined (sec. 4) | .543 -> .992 | +.449 | - | -1.800 | -1.211 |
| shared w64, cold MAP (400 steps) | .592 -> .852 | +.260 | 12.6% -> 35.6% | -1.912 | -1.267 |
| shared w64, refined | .644 -> 1.103 | +.459 | 5.2% -> 35.3% | -1.830 | -1.228 |
| shared s64, cold MAP (400 steps) | .626 -> .820 | +.194 | 17.6% -> 35.3% | -1.895 | -1.255 |
| shared s64, refined | .656 -> .956 | +.300 | 10.5% -> 34.5% | -1.804 | -1.210 |

Reading: shared s64 refined matches the old refined fit's held-out scores (-1.804 vs -1.800; -1.210 vs -1.211) with
two-thirds of its century lambda change (+.30 vs +.45). Refinement still raises change 1.5x (s64) / 1.8x (w64) over the
cold MAP. The 2021-25 holdout cannot say whether century change is over-fitted; it tests a 5-year forecast. Regional
(central-east band) lambda for these fits has not been computed (final_lambda_basis_audit.py would do it).

TACC outcomes (checked 2026-10-29):
- 3535018 shared full128 refinement: FAILED by the validity gate after 1 h 05 m. Stalled at iter 4580, U 133948, recent
  median |grad| 2.27e6 > plateau_grad_max 1e4 (gradient alternates ~1e3 / ~2.3e6 between iterations). Outputs retained:
  final_shared_round1_20261025/projectionfix1/r128. Its cold MAP (r128/viz_map): lambda .530 -> .826 (+.296), suitable
  13.5% -> 37.2%, temporal lpd -1.875, spatial lpd -1.237.
- 3535014 stalled-point audit of the split-readout clean full128 refined point: COMPLETED. |grad| 5.06e4 is genuine:
  autodiff agrees with central differences to 0.1% (gradient direction, step 1e-8) and 0.05% (alpha_a). Gradient is
  concentrated in the intercepts (alpha_a 4.18e4, alpha_j 1.10e4, alpha_f, alpha_k ~6e3) and the manifold factor
  (2.26e4). A move of 1e-7 along the gradient changes U by only ~0.005: a real, steep, narrow direction, so the stall is
  likely ill-conditioning between intercepts and field amplitudes, not a gradient bug.
  Source: final_age_handoff_20261025/dev_diagnostics_clean2_fix1/stalled_gradient_audit.json.

E080 / E081 temporal-habitat ladder, results tabulated 2026-10-29 from TACC research/age_model/{E080,E081}/*/viz_ref/
metrics.json (the written E080 record stops before 80g/80h; E081 has no results section). Refined points; lambda =
all-land mean 1902-15 -> 2010-25; T / S = temporal (2021-25) / spatial held-out mean log density. One fit each, no CIs.
| arm | temporal habitat | lambda change | T | S |
|---|---|---:|---:|---:|
| 80d (prod DESK) | static | 0 | -1.818 | -1.192 |
| 80g | tied, 0.14 x spatial coefficients | +.011 | -1.816 | -1.197 |
| 80h | correlated direction, amplitude 0.14 | +.279 | -1.814 | -1.191 |
| 79b (E080 text) | free, K trend on | - | -1.820 | -1.175 |
| 81c (cbc1902 DESK) | static | 0 | -1.836 | -1.214 |
| 81b | tied 0.14 | +.034 | -1.830 | -1.215 |
| 81a | free | +.323 | -1.820 | -1.209 |
| 81f (ctl1902 DESK) | static | 0 | -1.845 | -1.204 |
| 81e | tied 0.14 | +.024 | -1.830 | -1.218 |
| 81d | free | +.254 | -1.824 | -1.227 |
Reading: on every DESK, tied/static habitat forecasts 2021-25 within ~.01 nats/obs of free habitat with 1/10 the
century lambda change. Free directions buy the change, not the forecast. Static does not fix the fecundity demand (E080).
