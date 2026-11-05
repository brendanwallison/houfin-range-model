# Active plan: repair community Z and test controlled space-time DESK

This restores the core research question omitted from the synthesis. The work is
not just seasonal reweighting or another House Finch readout. All ecological
expressiveness described here is fitted using community observations; the
resulting Z is frozen before House Finch inference. The latter uses shared
spatial/temporal coefficients. No focal observations select a Z transform,
seasonal mixture, time modulation, residual, or prior strength.

Execution update on2026-10-26: the masks, training-only sources, first fresh
teachers and winter auxiliaries are complete. The regional residual and
time-conditioned network are implemented and tested; the matched control,
time-conditioned pilot, residual fitting and subsequent scoring are queued with
dependencies. Historical direction-dependent metric calibration remains
unimplemented. See [the execution record](COMMUNITY_Z_REPAIR_EXECUTION_2026-10-26.md)
for actual completed/running/queued stages and limitations.

## 1. The scientific comparison

Can community observations identify slowly changing regional structure that the
covariates miss, and can that improve a frozen representation at locations and
periods whose evaluation observations were withheld?

Three distinct mechanisms must be separated:

1. Reweight information already present in Z: a common PSD metric correction.
2. Add community-informed variation absent from the covariate map: a low-rank
   regional space-time residual.
3. Change how environmental variables are interpreted over time: low-rank
   time modulation inside DESK.

The first cannot manufacture missing information. It is a cheap control, not a
prerequisite that must succeed before trying the other two. The residual is the
first explicit-space-time candidate; the time-conditioned network is the second.
An unrestricted latitude/longitude/year concatenation is not an initial arm.

## 2. Validation before choosing architecture strength

Use the existing ABBA/count-quality logic, conditional CBC list scorer, spatial
and circle blocking, and endpoint controls. Do not replace them with raw counts
as noiseless labels or ordinary random-row validation.

### A. Summer-hidden, winter-assisted historical emulation

Use two modern-period backcast tasks with substantial breeding evaluation data:
target1966–1985 with breeding training1996–2025, and target1986–2005 with breeding
training2006–2025. Remove all earlier breeding supervision, not just the scored
target interval. Winter observations remain available in the target period,
using fixed spatial/effort thinning to emulate sparse historical coverage.
The tasks share some later observations and are not independent replications.
Publish exact support/masks before looking at candidate outcomes; inadequate
coverage is reported rather than silently altering a scored comparison.

Crucially, refit the covariate-only control and candidate encoders under these
masks. A current final encoder trained on the hidden summer years is not an
honest initializer for this experiment. Apply exclusion to every source of
supervision: direct counts, species heads, ESK/coordinate targets and landmark
fitting, trend reconstructions, pair pools, completeness estimation and any
cached teacher targets. Environmental covariates and calendar/position inputs
remain available because those are predictors, not hidden observations.

Separate two evaluation tasks: known-place historical reconstruction, and
withheld-region reconstruction. The first must reserve independent modern
evaluation observations from encoder training and readout fitting as well as
withholding historical summer labels. Where ABBA estimates need two evaluation
halves, keep training observations separate from both evaluation halves. This
prevents a better modern fit from masquerading as a better historical backcast.
Keep the existing endpoint-swap/memorizer diagnostics, with their coordinate-
alignment caveats, rather than merely scoring a net change correlation.

Vary winter information availability with one prespecified sparse setting and
a fuller-data sensitivity. Include a no-target-era-winter ablation: a claimed
winter-assisted improvement should respond to whether those observations exist.

### B. Actual historical winter reconstruction

Score1902–1939 winter community composition and subsequent eras using withheld
circles/regions and withheld observation blocks within eras. Community winter
records at other training locations in the same historical era may be used;
that is reconstruction with sparse historical information, not blind temporal
extrapolation. Also test removal of target-era winter data to distinguish those
tasks. Keep relocated circles out of both train and test under different labels.

Reuse the established effort-aware observation operators, recorded-list
conditioning, same-circle/low-counter sensitivities and circle uncertainty.
Mask derived completeness and winter targets using training records only.
Training/calibration uses development species; reserved species remain unopened.
Do not describe development species as absent from all representation training
if an encoder teacher used them.

Early summer has no direct validation. Improvement in winter is not sufficient
evidence of correct early summer; task A tests this bridge where summer truth
exists. Actual early-summer predictions remain model-dependent, with uncertainty
and seasonal-transfer assumptions reported explicitly.

### C. Matched comparators and failure controls

Score covariate-only DESK, fixed-modern/persistence controls and the repaired
regional space-time comparators on the same records and information sets. Reuse
E033/E039/E040 implementations and their corrections. If a new community model
cannot outperform those comparators, report that; do not compare unrelated
historical headline scores.

Reuse historical-data damage/observation checks from E070/E086v-b. Add correction-
null and injected-correction recovery checks on the actual observation design:
covariate-only truth should not create systematic space-time departures; a
known small regional departure should be recovered with appropriate shrinkage.
These checks do not establish that all coherent real-data trends are ecological.

## 3. Candidate M: metric correction of existing features

Use Z_M = Z0 M^(1/2), with M positive semidefinite and shared across all space
and time. Fit using community targets, not House Finch. Start with identity and
the already tested seasonal weights as controls; add a small, identity-shrunk
direction-dependent family. Rotation alone changes no isotropic kernel and is
not a repair.

The missing extension is joint seasonal/historical calibration. Include winter
development observations and earlier blocked contrasts with their appropriate
observation operators, rather than using a modern breeding Gram proxy alone.
Protect modern spatial fidelity while measuring historical endpoint, change
direction, change size and mean-change capture separately. Report the tradeoff
instead of hiding it behind a single objective score.

Any candidate trace scaling is an explicit comparison convention, not an
unannounced input normalization or an empirically established temporal budget.
No .14 ratio is supplied. This arm can correct geometry only within the existing
span; failure does not rule out residual learning or a time-conditioned encoder.

## 4. Candidate R: community-learned regional space-time departure

First hold a mask-valid covariate encoder fixed and fit

    Z_R(s,t) = f_theta(c(s,t)) + U a(s,t).

U has only2 or4 latent directions in the initial family. The scores a(s,t) use
coarse spatial basis functions and a small smooth temporal basis over the actual
1902–2025 domain. Zero residual amplitude reproduces the covariate model. Center
the departure relative to a modern reference window available in that task's
training data; retain the modern spatial anchor and assess it independently.

Fix basis normalization and loading scale conventions explicitly so U and a
cannot trade an arbitrary multiplicative scale. Begin with a frozen backbone
to isolate the residual's contribution. Joint encoder fine-tuning is a later
ablation if the residual earns a reproducible improvement.

Rank, temporal curvature and regional spatial scale are explicit prior choices,
not knowledge extracted from the single .56 nearby statistic. Start with a
small family of regional spatial scales and multidecadal temporal bases; choose
strength on blocked development validation. Include zero correction and
stronger-shrinkage controls. No independent trajectory for each cell, species
or survey. No unconstrained extrapolated polynomial before the first observations.
Weakly observed periods must remain uncertain rather than acquire an arbitrary
deterministic trend.

This is new relative to the audited downstream correction experiments because
community observations estimate the correction and it is frozen into Z before
the focal model. It is not new merely because it is written as prediction plus
residual. Its viability is the masked reconstruction test, not the formula.

## 5. Candidate T: low-rank time conditioning inside DESK

Modulate a small part of the environmental response rather than giving the
network unrestricted coordinates. For an environmental hidden code h(c):

    h_t(c) = h(c) + U diag(g(t)) V^T h(c).

Begin at rank2 or4 in a late encoder block; g(t) is a small smooth temporal
basis with shrinkage toward zero. Zero modulation is exactly the covariate-only
control. The same environmental covariates can then have somewhat different
community meanings in different periods. Time enters this constrained channel,
not all branches and the decoder indiscriminately.

Train with the same community observations, masks and seasonal heads as R.
Keep the teacher/coordinate contract and modern spatial constraints visible.
Do not force winter and breeding responses to be identical: estimate their
coupling only where overlap provides information. Historical winter can constrain
the winter component without pretending it directly observes early summer.

A coarse regional modulation g(s,t), or a combination of R and T, is a subsequent
ablation if the initial comparison identifies a specific need. Do not launch an
unrestricted coordinate network and a large grid of combinations immediately.

## 6. What prior work changes about this plan

| Prior evidence | Consequence |
|---|---|
|E073 anchored covariate delta; no reliable early-CBC transfer | Reuse its training-only support/novelty logic; do not call a covariate-delta readout new. R explicitly admits non-covariate information; T changes the response with calendar time. |
|E086e change basis and unsuccessful readout-prior follow-up | Do not assume leading modern change directions span early change, or that rotating/reordering Z is an untried cure. Preserve full baseline support. |
|E033/E039/E040 position-time and GP comparators | Use them as information-matched baselines; community sharing must earn its benefit. |
|E055g/h modern-end memorization | Reserve independent modern observations and score endpoints. Known-place change correlation alone is inadequate. |
|E086u target-era winter ablation | Test explicitly whether historical winter information supplies a gain. |
|E070/E086v-b historical quality and observation-operator failures | Correct the observation route before treating a smooth fitted trend as ecological. |
|E066/E079/E082 winter-summer temporal disagreement | Winter success cannot certify breeding backcasts; task A is essential. |
|Age-model E057/E067 drift/correction tracks | These were focal/downstream changes, not frozen community-Z repairs; some real fits were invalid. Their pre1966 time hold does not answer this plan's early-winter-supported reconstruction question. |
|New multi-span coherence analysis | Supports testing a slow regional component; does not determine its amplitude, spatial length scale or recoverability from sparse data. |

## 7. Selection, resources and immediate deliverables

Do not gate the community program on House Finch refinement. Use local CPU for
record masks, winter targets, uncertainty and fixed-encoder residual fitting;
use the local GPU queue for the mask-valid encoder control and low-rank
modulation pilots. Use checkpointed two-hour TACC dev tasks for the same frozen
experiments when available. Respect the existing GPU queue rather than running
two memory-heavy processes concurrently on the local card.

First deliverables, in order:

1. Fixed observation/mask manifests and an audit that all derived supervision
   honors them, including a report of support lost to blocking and thinning.
2. Baseline reproduction on those exact masks, plus historical winter target/
   observation-operator checks. Reuse archived baselines where truly compatible;
   do not label contaminated current encoders mask-valid.
3. Frozen-backbone R pilot and metric control on community development data.
4. T pilot and a matched covariate-only encoder, followed by a second seed for
   promising arms and the second backcast reach before selecting a model.
5. A comparison reporting historical endpoint accuracy, regional/local change,
   direction, magnitude, mean trend, modern spatial fidelity, seasonal transfer,
   and null/recovery behavior. Hyperparameters are selected inside development
   folds, not from opened historical final scores or House Finch maps.

Only then freeze the selected representation/transform and build provenance-
checked local and path-integrated cubes for a shared-readout population test.
No automatic promotion follows a small proxy gain. If candidate differences
are not distinguishable at historical support, retain the simpler model and
report what remains unidentified.

Existing code to reuse: research/exp/anchored_delta.py,
research/exp/e086e_change_basis.py, research/exp/fair_rivals.py,
research/exp/final_winter_score.py, research/exp/e069_presence_score.py,
research/exp/cbc_prepare.py, research/exp/cbc_completeness.py,
src/community_encoder/train_DESK/validation_core.py,
src/community_encoder/train_DESK/model_arch.py and desk_training.py.

Prior-work audit: COMMUNITY_Z_EXPERIMENT_MAP_2026-10-26.md and
DESK_EXPLICIT_SPACETIME_AUDIT_2026-10-26.md. The execution record distinguishes
implemented/queued pilots from stages still proposed; queued DESK jobs are not
reported as already running.
