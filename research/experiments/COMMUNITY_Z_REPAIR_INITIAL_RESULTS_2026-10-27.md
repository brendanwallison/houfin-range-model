# Community Z repair: first masked pilot results

Checked at03:21 CDT on2026-10-27. The covariate-only control finished500 epochs
and exported its cube at02:39. The rank-two regional residual fit and both arms'
summer/winter evaluations completed. The separate rank-two time-conditioned
encoder is running, last checked at epoch37/500. It has no heldout results yet.

The initial regional residual is not a successful general Z repair. It improves
some winter scores but worsens summer backcasts and their historical endpoints.
No candidate is promoted, and these are development results from previously
used ecological records under new exclusion masks, not fresh confirmation.

## Heldout summer: direction and endpoint failure

The hidden target is1966–1985, with no summer training observations before1996.
Known-place modern training and evaluation observations are disjoint. These
are nonsealed community/development species, not wholly unseen species.
The primary panel has exactly two survey years in each independent half of
each epoch; four-year halves are a prespecified sensitivity on a smaller panel.
Each arm fits one modern spatial readout and reuses its coefficients over time.

| Primary panel | Cells | Control change correlation | Residual change correlation | Historical MSE, control | Historical MSE, residual |
|---|---:|---:|---:|---:|---:|
|Known places|1,262|0.2233|0.2127|0.11027|0.12790|
|Withheld places|284|0.1862|0.1739|0.16073|0.18408|

Paired spatial-block bootstrap change-correlation differences are−0.0106
[−0.0161,−0.0056] at known places and−0.0122 [−0.0250,−0.0016] at withheld places.
Historical MSE increases by0.0176 [0.0148,0.0207] and0.0233 [0.0149,0.0313],
respectively. Modern endpoint changes are much smaller. Four-year halves show
the same historical-error deterioration; their withheld-place correlation
difference interval includes zero.

Both arms have negative change-MSE gain over predicting no change on all four
panels. On the primary known-place panel, gain is−0.0050 for the control and
−0.0259 for the residual. Truth-on-prediction centered slopes are0.352 and0.306.
Thus nonzero directional correlation does not establish useful change size.
These heldout diagnostic slopes are not fitted or applied as a calibration.

## Regional versus local change

The supplementary diagnostic uses the existing6x6 grid blocks, excludes singleton
blocks and weights surveyed cells. On the1,207 replicated-block known-place
cells, the control's regional correlation is0.3181 [0.2932,0.3408], versus0.0970
[0.0784,0.1160] for within-region departures. The residual scores0.2952 and0.0961.
At withheld places, the corresponding control correlations are0.2771 and0.0738;
the residual scores0.2515 and0.0723.

The cross-half centered observed-change signal decomposes approximately into
13.13 regional and12.02 local units on the primary known-place subset. Neither
component is negligible. This supports distinguishing regional structure from
local departures; it does not show the regional component is nearly perfectly
predictable. This pilot loses regional alignment. The block scale is a fixed
diagnostic convention, not an inferred ecological correlation length; local
departures can include observation effects. No causal or spatial-decay claim
follows, and the result does not validate pre1966 summer reconstruction.

## Winter: improvements over control, but not a demonstrated useful backcast

Winter evaluates conditional recorded composition at excluded circles. The
modern-feature placebo holds each cell's features fixed at its2006–2024 mean.
Positive temporal gain favors the backcast over that placebo. The following
are the primary circle-bootstrap results, not latent-abundance accuracy.

| Era | Lists / circles | Control temporal gain | Residual temporal gain | Paired gain difference,95% CI |
|---|---:|---:|---:|---|
|1902–1919|117 /50|−0.0398|−0.0380|+0.0018 [−0.0109,+0.0147]|
|1920–1939|195 /60|−0.1050|−0.0594|+0.0456 [+0.0243,+0.0707]|
|1940–1952|209 /51|−0.1345|−0.0968|+0.0377 [+0.0189,+0.0590]|
|1953–1965|383 /68|−0.0179|−0.0434|−0.0254 [−0.0409,−0.0089]|
|1966–1985|1,258 /110|−0.1847|−0.0586|+0.1261 [+0.0917,+0.1556]|

All primary era point estimates of the residual's temporal gain remain negative.
In1902–1919 and1920–1939, its interval includes zero; this is not evidence that
the backcast beats the fixed-modern placebo. Same-circle and low-counter
sensitivities are retained in the paired JSON. Early same-circle support is
small, so pooled winter improvements cannot certify historical ecological truth.

## Interpretation and remaining work

This particular low-rank, smooth residual does not provide the desired seasonal
transfer. Winter gains alone would give an incomplete assessment; the independent
summer targets reveal historical deterioration. The outcome does not rule out
other community-level residuals or the time-conditioned network.

The residual shares a space-time score process across learned seasonal loadings.
It has no early-summer supervision in this emulation. Whether the summer harm
comes from that coupling, teacher geometry, prior strength or observation-model
bias remains unresolved. Its initial weighted early-presence score gradient
norm was0.00078, versus0.01935 for the winter kernel; this quantifies one initial
minibatch's imbalance, not the whole optimization trajectory or a calibrated
replacement weight. The inherited0.01 loss weight is not treated as established.

The time-conditioned encoder continues on the local GPU. Second-reach and
winter-availability teacher/auxiliary sources are ready, but their candidate
fits and scores have not run. Ecological-design null/injection checks and a
second seed remain necessary before selection. No House Finch results select
these candidates, and the0.14 temporal/spatial ratio is not used.

## Artifacts

The ignored runtime directory is
`.research_runtime/community_z_repair_20261026/evaluation/backcast40/`.
`PARTIAL_RESIDUAL_COMPARISON.{md,json}` contains paired endpoint and winter-era
effects; `PARTIAL_RESIDUAL_REGIONAL.{md,json}` contains the decomposition.
Each arm's raw panel predictions and summaries remain under `control/` and
`residual_r2/`. A separate frozen `partial_residual_code/` snapshot generated
these reports without changing queued fitting/evaluation sources. The eventual
three-arm comparison remains queued behind the time-conditioned evaluation.

## Time-conditioned DESK result (790, time_r2_s1; added 2026-10-27 evening)

The rank-two time modulation (cubic time spline, 20-yr knots, gating two hidden directions; no position input; no
pre-1996 summer rows) finished 500 epochs locally. Learned gates reach ~0.11 (time_weights max 0.112), so the module
trained; it did not change change prediction.

Paired vs control (PAIRED_COMPARISON.md): change correlation k2 known -0.003 [-0.005, -0.001], k2 withheld +0.000
[-0.007, +0.008]; endpoint MSEs unchanged.

Regional / local, residency tiers (assim/backcast40/ASSIM_TIME.md; withheld places, k2, all / regional / local):

| arm | sedentary common | sedentary | all species |
|---|---|---|---|
| control | .313 / .443 / .105 | .276 / .403 / .092 | .186 / .277 / .074 |
| time_r2 | .311 / .440 / .107 | .273 / .398 / .092 | .186 / .275 / .075 |

Known places: sedentary common regional 0.451 vs 0.458; all species 0.314 vs 0.318.

Reading (likely): with no summer observation before 1996 nothing identifies what a time term should do in 1966-1985,
and the winter objectives alone did not move the summer change. Consistent with the in-era interpolation result
(ASSIM_FIELD_2026-10-27.md): observing the era is what moves regional change, not adding a time input. Null; not a
candidate.

## 808 change100 vs 809 continue100 (2026-10-27 ~22:55)

Ran on TACC (task 14; 100 epochs, 8.3 s/epoch, VRAM 24.3 GB). The step-1 gradient calibration was computed beforehand in
two .backward() passes (research/exp/z_change_calib.py: g0 0.040132, g1 0.000143376, weight 279.907). The run loaded that
file (identical bytes on TACC) and skipped calibration. The local 808 had failed at calibration (OOM), so local
812/814 were re-added as 863/864.

Withheld-place species change (assim/backcast40/ASSIM_CHANGE100.json, paired vs continue100, k2):
| stratum | regional [95% CI] | all | local |
|---|---|---|---|
| sedentary common (21) | +0.019 [-0.009, +0.051] | +0.011 [-0.005, +0.032] | -0.004 [-0.028, +0.015] |
| sedentary (108) | +0.017 [-0.007, +0.045] | +0.010 [-0.004, +0.028] | -0.004 [-0.024, +0.014] |
| all species (369) | +0.023 [+0.001, +0.043] | +0.015 [+0.002, +0.028] | +0.005 [-0.007, +0.016] |
Levels: all-species regional 0.277 -> 0.300; sedentary-common regional 0.442 -> 0.461. Training places: all-species
regional +0.029 [+0.020, +0.038], local +0.016 [+0.009, +0.024]. continue100 equals control_s1 (all differences
<= 0.003), so extra training or the optimizer restart contribute nothing.
Winter (paired gain differences): 1920-1939 +0.030 [+0.019, +0.040]; the other eras' CIs span 0.

Observation audit (observation_comparison/MATCHED_CHANGE_LOSS_COMPARISON.md; cell-level community change, pooled):
k2 withheld change corr +0.024 [+0.013, +0.036], historical-endpoint MSE -0.008 [-0.011, -0.006], modern MSE -0.001.
k4 withheld +0.023 [+0.010, +0.034].

Reading: the loss works in the intended direction, but the gain is small. It is about 1/6 of desk_interp's +0.145 from
pre-1996 summer rows (ASSIM_FIELD_2026-10-27.md) and about 1/5 of the field's further gain over desk_interp. Likely
reason: its targets are 2,303 1996+ contrasts from 1,245 training cells, much less change information than the 31,693
pre-1996 rows. The sedentary-common CI spans zero.
