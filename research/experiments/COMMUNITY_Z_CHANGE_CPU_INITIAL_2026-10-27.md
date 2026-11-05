# Change-learning CPU control and observation-contract audit

The first cheap control has completed. It adapts E073's transformed covariate
histories and physical-cover grouping to the current masks and summer64 teacher.
It is an observation-space diagnostic, not a production instantaneous Z export.

## Observation contract correction

The earlier control/residual reports scored the instantaneous population cube
against bird observations. DESK's training instead supervises the learned
output EMA. The original scores remain reproducible instantaneous-feature
diagnostics, but were insufficient to characterize the trained observation
predictor. A separate audit now applies the control's already learned
10.8396-year half-life, starting1902, with the existing `apply_output_ema` helper.
No lag is selected from heldout observations and population exports are unchanged.

| Control observation target | Known-place change correlation | Withheld-place change correlation |
|---|---:|---:|
|Original instantaneous-feature readout|0.2233|0.1862|
|Training-EMA feature readout|0.2237|0.1561|

Summer64 coordinate agreement with the observed teacher changes from0.3096 to
0.3261 at known places and0.2367 to0.2180 at withheld places. Correcting this
contract does not close the directional gap. It changes magnitude/endpoint
assessments substantially and must be carried into future comparisons. The
existing raw residual and time-candidate comparisons do not replace an
observation-aligned assessment; raw residual fitting itself did not propagate
the training EMA through its correction, which needs a separate corrected arm.

## CPU community-change comparison

Targets are differences between nonoverlapping1996–2005,2006–2015 and2016–2025
community-coordinate epochs, each with four permitted survey years and ABBA
halves. Modern training records are disjoint from both modern evaluation halves.
The map has zero predicted change for zero covariate change. PCs and scales are
fitted inside each spatial/contrast validation split; ridge strength is selected
using forward, reverse and long-span training-era contrast checks.

Winter predictors are a shared rank4 coordinate-change map, not per-species
temporal regressions. They use only permitted winter sources. Validation fields
remove winter sources in the held spatial blocks, and every source cell receives
one aggregate weight within the temporal window. Unsupported locations retain a
fixed modern field. Smoother settings are pilot conventions, not fitted ecological
correlation lengths. The ESK teacher is fixed from outer training sources across
inner folds; it is not refitted within each inner fold. These are exploratory
selection diagnostics, not a completely nested confirmation experiment.

The repaired observation features hold a modern mean field fixed, replace its
summer64 dynamics with learned community change and retain the control's winter64
observation features. Species readouts remain modern spatial maps reused in time.

| Primary2year-half panel | Known correlation | Withheld-place correlation | Known change-MSE gain over no change | Withheld-place gain |
|---|---:|---:|---:|---:|
|Control with trained observation EMA|0.2237|0.1561|−0.02377|−0.02535|
|Covariates,10year history|0.1348|0.1121|+0.00436|+0.00359|
|Covariates,25year history|0.2029|0.1511|+0.00689|+0.00504|
|Covariates25year + winter rank4|0.2218|0.1694|+0.00748|+0.00570|
|Winter rank4 only|0.2053|0.1496|+0.00547|+0.00348|

The combined arm's paired correlation difference from the EMA control is−0.0018
[−0.0173,+0.0149] at known places and+0.0132 [−0.0284,+0.0555] at withheld places.
Its historical MSE differences are−0.02594 [−0.02928,−0.02292] and−0.03048
[−0.04093,−0.01932], respectively. Modern MSE rises about0.00042 at known places
and changes−0.00028 at withheld places. These are improvement in size/endpoint
calibration, not a demonstrated direction breakthrough. No0.5 result is claimed.
Separate regional/local scoring remains to be attached to this CPU comparison.

## Runtime and verification

Local805 fitted all four maps and stopped at an exact target-equality guard:
float32 subtraction differed from the established float64 subtraction by at
most1.19e-7. The guard was retained; evaluation now uses the identical precision.
806 completed the independent EMA audit.807 reused the frozen fitted maps,
without reselecting strengths on evaluation data, and completed the corrected
comparison against the observation-EMA baseline.12 relevant checks pass across
the new utilities/models and existing EMA tests.

Runtime: `.research_runtime/community_z_repair_20261026/change_cpu/RESULTS.json`
and `FIT_DONE.json`, with the source snapshots `change_cpu_code/` and
`change_cpu_code_fix1/`. EMA audit: `ema_audit/DONE.json` and
`ema_audit/evaluation/backcast40/control/SUMMER_DONE.json`.
Small JSON summaries are copied to `research/experiments/community_z_repair_20261026/`.
