# Why the current change correlations are low

The current result is a shortfall, not an established0.3 ceiling. This note adds
two matched diagnostics to the initial masked control/residual comparison.
Neither fits a correction on historical evaluation outcomes or promotes a Z.

Observation-contract amendment: the numbers below originally used instantaneous
population features. The completed training-EMA audit gives summer teacher
agreement0.326/0.218 and species-change correlation0.224/0.156 at known/withheld
places. Thus the directional shortfall remains, but amplitude and endpoint
conclusions must use the proper observation contract. See
`COMMUNITY_Z_CHANGE_CPU_INITIAL_2026-10-27.md`; the original raw-feature
diagnostics are retained rather than silently relabeled as EMA predictions.

## Observation repeatability on the actual current panels

Using the independent reserved halves, center each species across cells and
sum its cross-half covariance. Let C be this shared change signal and H the
average within-half change variance. The parallel-half estimate for reliability
of their average is2C/(H+C). This is an aggregate variance-based repeatability
statistic, not per-cell angular agreement. Its reliability interpretation
assumes comparable halves and independent measurement errors; persistent route,
observer and reporting effects can remain shared across halves.

| Known-place panel | Parallel-half full-estimate repeatability,95% CI | Model noise-corrected change correlation,95% CI |
|---|---|---|
|All changes,2years per half|0.721 [0.711,0.729]|0.223 [0.206,0.238]|
|Regional means,2years per half|0.849 [0.838,0.859]|0.318 [0.293,0.341]|
|Local departures,2years per half|0.614 [0.600,0.626]|0.097 [0.078,0.116]|
|All changes,4years per half|0.839 [0.833,0.845]|0.232 [0.212,0.249]|
|Regional means,4years per half|0.918 [0.910,0.924]|0.312 [0.288,0.337]|

These columns are different statistics, not quantities to subtract to obtain
an ecological variance budget. The model score already estimates target signal
using cross-half covariance. More precise observation halves do not rescue
the model score. Independent sampling noise alone does not explain the deficit.
The4year panel has a smaller cell set, so its score difference is not a paired
effect of adding years.

## Matched breeding-teacher capacity diagnostic

Project the actual heldout survey counts into the fresh fixed64-dimensional
summer ESK teacher, using the original log1p transform and species order.
Average annual projections on exactly the primary panel's reserved survey rows.
Fit one spatial species readout on permitted modern training observations,
then reuse it across time. The evaluation prediction halves remain separate;
cross-half prediction/target covariance estimates their shared agreement.

Also compare DESK's summer64 change on those same survey years directly with
the observed teacher change. This isolates a coordinate-learning shortfall
without fitting species-specific temporal coefficients.

| Diagnostic | Known places,1,262cells | Withheld places,284cells |
|---|---|---|
|Observed summer teacher → species change oracle|0.760 [0.751,0.769]|0.758 [0.737,0.776]|
|DESK summer64 → observed teacher change|0.310 [0.290,0.327]|0.237 [0.186,0.288]|
|Full learned128 → species change, initial evaluation|0.223 [0.206,0.238]|0.186 [0.135,0.232]|

The oracle includes the scored species in its observed-community inputs. It is
not a species-cross-fitted prediction, an attainable historical predictor, or
a demonstration that environmental covariates contain the missing information.
It is a best-information capacity diagnostic: the teacher and modern spatial
readout can transmit much more observed temporal signal when supplied actual
community observations. Its noise correction also assumes independent halves.
The summer64 coordinate comparison and full128 species comparison are different
estimands; their difference is not an additive loss attribution.

Thus a large bottleneck occurs between environmental inputs/training and
recovered historical community coordinates. Teacher compression and the tested
spatial readout do not force this particular low score. Whether missing
environmental information, training-objective emphasis, generalization, or the
historical observation bridge dominates that bottleneck remains unidentified.

## What prior experiments establish, and what they do not

The prior root-cause audit documents E086m's high spatial kernel fidelity0.910
but change-size fidelity0.025 for an older DESK candidate. Change-size fidelity
is not direction correlation. It supports a mismatch between preserving
dominant spatial geometry and learning temporal differences; it is not a
numerical decomposition of the present candidate's error.

E086c showed that several apparent alternative methods were readouts of the
same feature span, rather than independent information sources. E086y/y2
changed observed community information and obtained changes in performance.
Therefore agreement among some baselines does not establish a universal ceiling.
E086u showed target-era winter observations could improve summer change in a
sparse-data regime, while E086v/v-b showed an early-quality observation route
could fail to transmit that benefit. E066c's winter/summer temporal bridge was
weaker than its spatial bridge. These support auditing the information path,
not assuming capacity alone or all unexplained change is ecological noise.

The current residual likewise gains some winter composition skill while losing
summer direction and historical endpoint accuracy. That is evidence against
this coupling/prior/objective combination, not against every shared space-time
community model. Its winter gains do not yet beat the fixed-modern placebo.

There is also a separate amplitude problem: the current control's primary
known-place centered truth-on-prediction slope is0.352 and its change-MSE gain
over no change is negative. A diagnostic scalar shrink can improve magnitude
calibration, but leaves correlation unchanged. These heldout slopes are not
a proposed Z normalization, and do not justify reinstating0.14.

## Provenance

Repeatability and amplitude outputs:
`research/experiments/community_z_repair_20261026/CURRENT_SCORE_BOTTLENECK.json`.
Teacher diagnostic summary:
`research/experiments/community_z_repair_20261026/TEACHER_BOTTLENECK.json`.
Diagnostic code: `research/exp/z_repair_teacher_bottleneck.py`; focused estimator
check: `tests/test_z_repair_teacher_bottleneck.py`.
The initial raw/log input guard stopped task803 before any fit; corrected
log1p-input task804 completed. Original logs and projections remain in the
ignored runtime `teacher_bottleneck/` directory. Current training/evaluation
snapshots and their queued jobs were not changed by these diagnostics.

Interpretation of the prior work and original source references:
`DESK_ROOT_CAUSE_AUDIT_2026-10-26.md` and `E086_RELIABILITY_AUDIT_2026-10-26.md`.
Neither the statistical repeatability nor the oracle validates1902–1940 summer,
where observations remain absent.
