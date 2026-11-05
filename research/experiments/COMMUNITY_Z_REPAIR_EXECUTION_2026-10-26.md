# Community Z repair: implemented and dispatched

This is the execution record for COMMUNITY_Z_REPAIR_PLAN_2026-10-26.md.
The first control/residual development results are now available in
`COMMUNITY_Z_REPAIR_INITIAL_RESULTS_2026-10-27.md`. No repaired-Z adoption is claimed.

## Data and exclusion controls

Two reconstruction tasks are prepared from nonsealed community/development
species. No House Finch observations or sealed response matrices were used.

| Task | Hidden breeding target | Breeding training begins | Training cell-years | Known-place evaluation cells | Withheld-place evaluation cells |
|---|---|---:|---:|---:|---:|
|backcast40|1966–1985|1996|24,600|1,262|284|
|backcast20|1986–2005|2006|16,857|1,855|410|

For eligible known places, the existing ABBA helper partitions modern records
into training records and two independent evaluation halves. Every historical
summer record is excluded from encoder training. The outer spatial masks and
buffer are held fixed. Evaluation further matches effort at exactly two survey
years per half in each epoch; four-year halves are a prespecified sensitivity on
their own eligible panel. Those choices were fixed before candidate scores.

The point sources physically exclude evaluation records before ESK construction,
window-mean augmentation, coordinate targets and metric pair generation. The
original all-year basis and final checkpoints are not used to initialize these
masked controls. Observation masks and source hashes are saved in the package.

Winter reconstruction sources include full coverage, a fixed10% circle sample,
and removal of target-era winter records. A relocated circle cannot occur in
training under one cell and evaluation under another. Effort references and
benchmark species identities are refitted on permitted training lists only.
The full case has87,088 training lists, including4,191 in1902–1939, and6,353
evaluation lists across eras. The sparse case has9,820 training lists, including
447 in1902–1939. Target-era exclusion is applied before rebuilding winter counts
and completeness benchmarks.

The first auxiliary pipeline deliberately reuses the existing E086x winter
observation operator and scalar relative-completeness nuisance. It is not a new
claim that this operator perfectly removes historical observation bias, and does
not use leave-one-species-out completeness. Benchmark sufficient statistics are
retained for a later explicit operator ablation. Conditional list scores and
same-circle/low-counter sensitivities remain essential.

## Completed preparation

- 786: fresh summer ESK teacher, completed on local CPU.
- 787: fresh full-winter ESK teacher, completed on local CPU.
- 788: masked winter observation operator and seasonal auxiliaries, completed.
- 796–798: second-reach summer, sparse-winter and no-target-era winter teachers,
  completed successfully on local CPU.
- 799: sparse-winter observation operator and auxiliaries, completed successfully.
- 801: second-reach auxiliaries, completed successfully. The winter teacher and
  observation operator were reused only after verifying identical permitted
  winter inputs and projected coordinates. The summer teacher and cross-season
  calibration were rebuilt for the second reach.
- 800: no-target-era winter observation operator, completed successfully at
  12:48 CDT. All currently dispatched teacher/auxiliary preparation has succeeded.

Both teachers use64 coordinates and the existing16,000-landmark budget, with
training-only window augmentation. The winter operator uses73,302 permitted
modern lists; its internal blocked validation selected lambda1e−6. Its recorded
heldout list NLL is4.0925; this is an internal operator diagnostic, not the new
historical reconstruction evaluation.

## Candidate implementations and queued workflow

| Local task | Work | Dependency |
|---|---|---|
|789|Covariate-only seasonal DESK control, then128-coordinate cube|788|
|790|Rank-two time-conditioned DESK, then cube|788|
|791|Rank-two regional residual fitted to the frozen control|789|
|792|Control summer/winter evaluation|789|
|793|Time-conditioned summer/winter evaluation|790|
|794|Regional-residual summer/winter evaluation|791|
|795|Paired comparison and endpoint attribution|792,793,794|
|796|Second-reach summer teacher preparation|none|
|797|Sparse-winter teacher preparation|none|
|798|No-target-era-winter teacher preparation|none|
|799|Sparse-winter observation operator and auxiliaries|797|
|800|No-target-era-winter observation operator and auxiliaries|798|
|801|Second-reach auxiliaries, verified winter-operator reuse|796|
|802|Regional/local change attribution on the fixed evaluation panels|795|

At03:21 CDT on2026-10-27,789 had completed all500 control epochs and exported
its cube at02:39.791,792 and794 had subsequently completed the residual fit
and both arms' summer/winter evaluation.790 was running at epoch37/500, with
the remaining comparison tasks queued behind it. Internal automatic validation
is deliberately disabled, so printed training-log validation NaNs are not
heldout performance results.796–801 had completed successfully. The first
residual worsens heldout summer endpoints and regional change alignment while
improving some winter-era scores; no primary winter-era point gain beats the
fixed-modern-feature placebo. Details and uncertainty are in the initial-results
report. A separately frozen partial comparison does not alter queued snapshots.

TACC job3535014 completed its separate stalled-fit gradient audit at17:07.
Job3535018 cleared the habitat-projection memory failure but was rejected by
the optimizer gate after4,584 iterations at18:14: recent median gradient norm
2.27e6 exceeded10,000 during the objective plateau. Its final saved-point
gradient norm is4,413; this is distinct from the recent-trajectory gate statistic.
It remains an invalid refinement, with outputs retained, and has no accepted
validation figures. Reports are copied under
`research/experiments/final_shared_round1_20261025/evening_20261026/`.

Pueue dependencies will execute fitting and scoring without
another manual dispatch; a failed prerequisite stops its dependent tasks.

The control and time candidate use the same permitted community observations,
fresh teachers and initial baseline architecture. Both use a fixed500-epoch
budget and checkpoints; neither selects an epoch on the hidden historical
outcomes. Internal automatic spatial splitting is disabled for these runs because
the external source masks implement the evaluation split. This is not training
on evaluation observations or a claim that500 epochs is optimal.

The time candidate modifies only the late environmental mixing block through a
rank-two modulation. Year enters explicitly in each forward/checkpoint call,
not through mutable global state. Its cubic temporal basis spans1902–2025,
with20-year knot spacing and a2006–2025 mean anchor. Initial amplitude and annual
curvature penalties are.01 and1000. These are declared pilot choices, not fitted
ecological calibration. No0.14 temporal/spatial coefficient ratio is used.

The residual uses two orthonormal latent directions and65 regional functions
(64 coverage-based Gaussian centers at300km scale plus a continental function),
with the same temporal basis. It fits the existing seasonal coordinate/kernel
and early-list objectives while holding the encoder fixed. It records the
initial weighted gradient contribution of early presence instead of assuming
the inherited.01 observation-loss weight supplies effective early supervision.
Its low-rank score process is shared across seasonal loadings; early summer
transfer remains an assumption tested by the summer-hidden experiment.

All source snapshots are separate from production outputs and contain SHA256
manifests. The time trainer/exporter changes are recorded as isolated patches;
existing production encoder source was not modified for these candidates.

## Evaluation and checks

Evaluation reuses the established ABBA splitting, log1p of mean raw counts,
spatial ridge calibration, centered cross-half covariance and circle bootstrap
helpers. It fits one spatial coefficient vector per species and uses it over
time. No independent temporal readout is introduced.

The first summer scorer uses a spatial epoch-mean readout for a consistent
epoch-mean count target. Its absolute numbers are not the old annual-pooled
readout's scores. It reports correlation, truth-on-prediction slope, gain over
no change, mean change and independent historical/modern endpoint risks. Paired
candidate comparisons require exactly matching cells, truth arrays and species.

Winter evaluation fits its count readout and seasonal offsets on modern training
circles, then scores the excluded circles with the existing conditional list
operator. It reports the fixed-modern-feature placebo, temporal gain and the
same-circle/low-counter sensitivities. It does not treat missing species as
verified ecological absences or winter lists as summer truth.

Thirteen focused tests pass. They include poisoning evaluation records without
changing training targets/benchmarks; relocation exclusion; nested-null equality
to the existing encoder; planted rank-two recovery; a complete synthetic residual
fit including early-winter gradients; effort-matched independent evaluation;
paired endpoint reporting with historical-only circles; and exact regional/local
signal decomposition under spatial block resampling. The isolated time
trainer separately completed two full CPU epochs with checkpointed backward.
These are implementation checks, not ecological null/recovery certification.

The supplementary regional report separates means within the existing6x6 grid
blocks from within-block departures. It uses the same evaluation cells with at
least two cells per block and the same paired block bootstrap. Regional means
are repeated at each cell, so signal and gain decompose exactly on that subset;
these are surveyed-cell weights, not geographic-area weights. Singleton blocks
are explicitly counted and excluded from this diagnostic. It does not fit a
regional model or use a regional score to choose hyperparameters. Local departures
can include observation effects, and the result is not a spatial-decay estimate.

Still required: actual pilot outcomes, correction-null/injected-signal checks
on the ecological observation design, second-seed/second-reach comparisons,
winter-availability ablations, and direction-dependent historical metric
calibration. No candidate is ready for promotion or early-summer truth claims.

## Locations

Runtime/data/source snapshots and logs:
`C:/Dev/houfin-range-model/.research_runtime/community_z_repair_20261026/`.

The automatic paired report will be written to
`.research_runtime/community_z_repair_20261026/evaluation/backcast40/PAIRED_COMPARISON.md`
and its JSON. Individual stage logs remain beside their artifacts if a job fails.
The supplementary report will be written beside it as
`REGIONAL_CHANGE.md` and `REGIONAL_CHANGE.json`.

Implementation: research/exp/z_repair_{data,teacher,aux,models,control,time_run,
residual_fit,evaluate,winter_evaluate,compare,regional}.py; wrappers in research/bin/.
