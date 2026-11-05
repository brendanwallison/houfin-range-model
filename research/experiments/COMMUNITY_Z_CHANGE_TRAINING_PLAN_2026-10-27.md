# Next intervention: learn community change, preserve the spatial anchor

The objective is a material improvement on the current heldout change estimand,
with0.5 a target to test, not a promised result. The current regional correlation
is0.318 and pooled correlation0.223; report both rather than substituting one
for the other. The observed-teacher oracle near0.76 establishes representable
signal, not its predictability from available historical inputs.

## Evidence that determines the intervention

- The matched teacher diagnostic localizes a major shortfall to learned temporal
  reconstruction: DESK summer64 versus observed teacher change is0.310 at known
  and0.237 at withheld places. Actual observed-teacher coordinates passed through
  a modern spatial species readout recover change at0.760/0.758.
- E043 already oversampled same-cell cross-year Ruzicka pairs; it did not provide
  a material general improvement and incurred a spatial cost. Do not repeat it
  as the proposed solution.
- E073 PartD proposed a separate within-cell vector-deviation loss, but the
  audited neural training stage was not run. Its implemented community covariate
  delta and later corrections provide a cheap diagnostic/control to reuse.
- E086r/r-b tested species-wise CBC conditioning and did not establish a remedy.
  The proposed seasonal transfer must be a shared community map, not a repeated
  species-specific correction.
- The new joint regional residual improved some winter periods while degrading
  summer, especially the historical endpoint. Extra capacity and a shared score
  process alone did not establish the correct seasonal transfer.

## Execution order

1. **Cheap matched change-learning control on CPU.** Reuse E073's covariate-history
   extraction and training-only standardization/PCA. Fit community-coordinate
   changes on repeated-cell epoch contrasts, not species-specific downstream
   temporal readouts. Fit only permitted training observations, with nested
   spatial and temporal blocks for hyperparameters. Compare covariate changes
   alone with available community winter information. Report whether the new
   loss actually improves heldout teacher-change recovery before investing in
   another full neural run. This is a current-data reproduction/control of E073,
   not a new claim that a covariate delta has never been tried.

2. **A change-supervised DESK pilot.** Start from the completed mask-valid control
   in a separate source/output snapshot. Retain absolute modern coordinate/kernel
   losses and add an explicit vector loss:

       Lchange = || [mean Z(B) - mean Z(A)]
                     - [observed teacher(B) - observed teacher(A)] ||².

   Use nonoverlapping multiyear epochs and adequate repeated observations,
   independent-half reliability diagnostics, and training-only uncertainty
   weights. No cosine-only target: direction and size both matter. This gives
   temporal differences an explicit objective and budget rather than relying
   on absolute-coordinate MSE dominated by spatial offsets or scalar similarity
   pairs. Report its gradient contribution; choose its strength inside training
   blocks rather than supplying another fixed seasonal-amplitude ratio.
   The existing same-cell scalar kernel sampler is not this intervention.

3. **Constrain the seasonal bridge with observations.** Learn winter-to-summer
   change transfer jointly across community coordinates from modern overlapping
   training records. Use a regularized low-rank map, with rank/strength supported
   by inner heldout epochs and regions. Transfer only supported directions;
   unsupported summer corrections shrink toward zero. Evaluate a coarse regional
   community residual with this vector objective/bridge, rather than assuming
   a common winter/summer temporal correction is sufficient. First publish the
   support remaining after pairing and blocking; do not assume overlap identifies
   a rich map. Early winter information remains subject to its observation
   operator and quality checks. There is no invented early-summer label.

   The result is one community-trained frozen representation. House Finch
   receives that Z with shared coefficients and does not fit the bridge,
   temporal correction, or seasonal weights.

4. **Matched acceptance test.** Keep current outer masks and independent modern
   evaluation records. Require improved historical endpoints and change-MSE,
   alongside correlation, preserved modern spatial accuracy, and regional/local
   reporting. Fit and test target-era-winter availability and sparse coverage
   variants. Use training-block validation to choose strength; the already opened
   outer panels are exploratory. A second reach and seed follow any promising
   result before adoption. Null and injected-change checks must verify that
   survey damage or a modern-end improvement cannot masquerade as historical
   reconstruction.

## Resources and boundaries

The ongoing time-conditioned pilot continues on the local GPU. It supplies a
capacity comparison, but its outcome cannot alone distinguish missing covariate
information from an inadequate change-learning objective. New preparation and
the small community change/transfer fits use CPU. The next GPU arm is the
isolated change-supervised pilot after its inputs, masking and gradient checks
pass; no additional rank/architecture sweep is justified by these results.

This plan does not make0.5 inevitable. The observed oracle is given the bird
responses; our historical model is not. A successful change-learning fit in
modern blocked data followed by failure in historical reconstruction would
point toward historical support/observation/transfer limitations, rather than
justify endlessly enlarging the encoder. Failure even in supported modern
blocks would direct attention to inputs/objective/estimation first.

Status update: the CPU community-change/seasonal-transfer pilot has completed;
see `COMMUNITY_Z_CHANGE_CPU_INITIAL_2026-10-27.md`. It improves change magnitude
calibration without a clear directional gain. The observation-EMA audit is also
complete and must govern future observation comparisons. A competing learned
mean-map/deviation neural prototype is implemented/tested, documented in
`MEAN_AND_DEVIATION_DESK_2026-10-27.md`. Full trainer/exporter integration and
the change-supervised neural fits remain outstanding; none is claimed running.
The original time-conditioned GPU pilot continues in its frozen snapshot.

Dispatch update2026-10-27: the explicit vector-loss trainer has now passed a
complete CPU training smoke and is queued as808, alongside a matched
continuation control809. Both await the active GPU task; neither is claimed
running. Observation-aligned summer/winter scoring follows automatically. See
`COMMUNITY_Z_NEURAL_CHANGE_DISPATCH_2026-10-27.md`. The mean/deviation architecture's
dedicated full trainer/exporter integration remains outstanding.

References: E073.md PartD and later correction sections; E043.md final results;
E086.md E086r/r-b/s/u/v; COMMUNITY_Z_BOTTLENECK_2026-10-27.md;
COMMUNITY_Z_REPAIR_INITIAL_RESULTS_2026-10-27.md.
