# Overnight priority: established-state injection with CBC supervision

November 1, 2026, 10:55 p.m. America/Chicago. Authorized by the user: prioritize later initialization and better CBC supervision; native departure is secondary. Frozen run specifications: [CBC_ESTABLISHED_RUNS_20261101.json](CBC_ESTABLISHED_RUNS_20261101.json). Live execution: [CBC_ESTABLISHED_STATUS_20261101.md](CBC_ESTABLISHED_STATUS_20261101.md).

## Actual queue and resources

Twenty `step38_cbc1951_*` jobs are published to the existing shared TACC queue, priorities 0615–0634, ahead of the pending HMC screen. Eighteen include CBC and two are matched no-CBC/free-abundance controls. All inherit the no-native-departure established-state control, not the native-departure branch. Both existing A100 allocations consume the shared queue; no additional allocation was requested. At 10:51 p.m. the allocations had about 47 minutes and 31 hours remaining. They were finishing previously started no-NestWatch comparisons, so queued CBC jobs were not yet running.

The batch is ordered to deliver the central comparisons first, followed by matched sensitivities and reserves. Existing comparable fits took roughly an hour each; the new likelihood adds work, so this is a backlog sufficient for overnight use, not a promise that every fit will finish. Each attempt is capped at three hours. The existing float32 MAP and float64 refinement both checkpoint/resume; the executor allows re-entry with identical frozen source after allocation loss. Previously queued work is preserved.

| Order | Comparison | Seeds |
|---|---|---|
| 1–4 | CBC + free established abundance; CBC + fixed middle abundance | 41, 42 |
| 5–6 | Matched no-CBC + free abundance | 41, 42 |
| 7–8 | Fixed low/high abundance profiles with CBC | 41 |
| 9–11 | 600-km winter observation scenario; broader circle heterogeneity; uncertain NestWatch annual conversion | 41 |
| 12 | Additional cold free-abundance replicate | 43 |
| 13–20 | Matched seed-42 sensitivities, 600-km fixed-middle controls and additional middle-abundance replicate | 41–43 |

## Demographic treatment

The initialization is one established release-cell population in 1951. There is no 1940 release pulse in this branch and no ten-year Allee exemption. The old CBC founder-backcast pin is not used. Fixed profiles are 280/560/1,120 conditional bird-equivalents, not observed censuses. Free-abundance fits estimate `log_inv_total` with a broad reference distribution centered at the middle scenario (log SD 2); its prior is not independent historical evidence. We will examine profile likelihood contributions and seed sensitivity rather than treating its fitted value as an identified founder census.

Yearling breeding, shared habitat, absolute Allee and the existing movement process remain. The inherited equal adult/young initialization and release-cell spatial support remain assumptions. Two reproductive sensitivities allow lognormal annual NestWatch conversion (log SD .3) rather than fixing all regional conversions at 2.5.

## CBC likelihood and observation mapping

The input uses real count-day House Finch counts and surveyed nondetections from the original CBC circle-year export. It is not a pseudo-observation or founder backcast. There are 98,207 eligible rows at 3,773 circles after date/count/effort/grid/coordinate checks. Missing measured party-hours are excluded in this first branch, never assigned zero. Relocated circles with >.1-degree coordinate changes are excluded. The source export is retained unchanged.

December of year y and January of y+1 observe the postbreeding state for summer y. No CBC record before the 1951 initialized state is given a likelihood. A 7.5-mile-radius circle is mapped by 64 deterministic equal-area disk quadrature points to the 54-km cells; coastal non-land fractions are retained rather than silently renormalized. A circle effect represents persistent local sampling heterogeneity: a CBC circle is not a census of its whole model cell.

The count likelihood is NB2 with a source-specific scale, circle effects, a monotone Box–Cox party-hour response, feeder effort and feeder-missingness terms. Circle-effect prior SD is fixed at 1 in the primary branch and 2 in a sensitivity, avoiding a freely optimized joint-MAP variance scale. There are no unconstrained CBC year effects that could simply replace the demographic trajectory.

Winter density uses an observation-only conservative mixture of local state and southwest redistribution. The migrant fraction and young availability are estimated observation parameters. The 300/600-km displacement scenarios, three southwest bearings, latitude/longitude gate and nearest-land assignment are explicitly provisional sensitivity choices; literature supports partial southwest migration, not these exact numerical kernels. This operator never changes the population trajectory, permanent dispersal or next year's breeding state. A static kernel cannot reconstruct the documented historical evolution of migration distance. A source fit exploiting extreme availability/fraction values is a diagnostic to report, not evidence of a validated annual-cycle model.

## Supervision and validation

Training contains 74,397 CBC rows, including 6,932 observations in 1951–1965; 2,183 of those are east of 80W. Crucially, **127 historical training observations have circle footprints overlapping the release cell**. Those are the actual later observations intended to supervise accumulated growth and establishment, with effort and sampling heterogeneity represented.

Spatial holdouts use deterministic 216-km blocks, with the geographically defined founder block retained for supervision; 16,722 rows remain spatially withheld. The 2021–2025 temporal test has 7,088 additional rows. The existing BBS comparison is auxiliary-aided prediction, not a fully independent cross-source spatial-transfer experiment, because BBS has not been withheld in these same CBC blocks.

The first generic split happened to withhold the founder block. Preflight identified this before any fit started. All twenty unstarted `step37` entries were moved to `queue/held/cbc_founder_split_v1`; their source and inputs remain preserved. The corrected `step38` batch has separate immutable inputs and source manifests. It supersedes that staging rather than silently modifying it.

Three focused tests pass: Box–Cox limits/derivatives, seasonal readout versus explicit matrices with mass conservation/gradient checks, and NB2 agreement with an independent reference plus held-out-count isolation. The real-data observation-only calibration at the frozen later-state population converged by the optimizer's relative objective criterion after 1,175 iterations (max absolute gradient .093; all finite). Its first 400-iteration attempt was finite but unfinished and is retained separately. Predictive adequacy remains imperfect: conditional temporal mean prediction is about 284 versus 123 observed, and fitted young availability is .096. Thus numerical convergence does not establish a validated winter mapping. Joint fits test whether changing demography improves this correspondence; if the source model remains miscalibrated or achieves fit through implausible availability, it must be revised before biological adoption. These queued joint fits are experiments, not an accepted CBC model. Final held-out source scoring integrates unobserved circle effects using quadrature; the preliminary fixed-population summary used point effects and is not a full marginal spatial prediction.

The full frozen joint-model integration check also passes on real inputs: both the objective and checked gradients are finite, and switching the CBC training factor on/off changes gradients for adult survival and `log_inv_total`. This verifies that the observation block actually supervises establishment and demography; it is not merely calibrating an unrelated source index. Receipt: `.research_runtime/cbc_joint_inputs_v2/FULL_MODEL_SMOKE.json`. The first harness attempt needed its saved NumPy constants converted to JAX arrays for differentiation; the model code was unchanged.

Every completed fit runs the full pathology readout, invasion timing, contribution/BIDE/isolation and source-specific CBC scoring. Persistent watchers collect outputs and queue matched BBS scoring. Existing movement followups and common cross-model diagnostics apply as the new budgets arrive. Review East, Great Plains, West consistently; do not select a configuration from one favorable demographic scalar or source/sink threshold.

## Literature basis and project-specific assumptions

- Link, Sauer & Niven (2006), [hierarchical CBC analysis](https://doi.org/10.1093/condor/108.1.13): effort response and hierarchical site effects. The project uses the published effort-response family with a shared-population offset, not a claim to reproduce their complete model.
- Meehan, Michel & Rue (2019), [spatial CBC models](https://doi.org/10.1002/ecs2.2707): source effort and spatial heterogeneity matter; CBC counts are relative-abundance observations.
- Able & Belthoff (1998), [eastern House Finch partial migration](https://pmc.ncbi.nlm.nih.gov/articles/PMC1689500/): southwest breeding-to-winter movement, partial migration and changing migration distance. This motivates explicit seasonal sensitivity; it does not validate our particular observation kernel.

The experiment asks whether later-state initialization plus historical observation supervision reduces pathologies and stabilizes the inferred growth/movement processes. It does not assume that CBC alone identifies initial abundance, nor that winter counts directly measure breeding density.
