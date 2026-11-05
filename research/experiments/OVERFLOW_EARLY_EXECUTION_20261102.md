# Movement, capacity floor and historical CBC: execution, November 2

**20:10 CST queue correction:** no K-floor fit had started. The four standalone floor fits were promoted to the next four pending positions (priorities0611/0612/0613), ahead of the remaining step41 departure and step42 curvature fits. The active fit was not interrupted. Dispatch metadata and watcher specifications were updated; frozen scientific source/configurations were unchanged. Two floor-plus-strict-overflow fits remain behind these standalone tests. Receipt: remote `monitor/KFLOOR_PRIORITY_20261102.json`.

**Correction following historical-data review:** the four unstarted early-CBC fits are now held in `queue/held_early_design`; the other twelve new fits remain queued. The missing-effort and broad-envelope assumptions do not adequately implement the intended early absence/lag supervision. See [the coverage audit and correction](EARLY_CBC_DESIGN_CORRECTION_20261102.md). The original specifications below are retained as execution history, not a current endorsement of the held CBC design.

Sixteen immutable fits are published on TACC. These are experiments, not completed results or endorsed configurations. Specifications: [all runs](OVERFLOW_EARLY_RUNS_20261102.json); actual queue/GPU observations: [live status](OVERFLOW_EARLY_LIVE_20261102.md).

## Implemented and queued

| Track | Matched changes | Fits |
|---|---|---:|
| Strict smooth overflow | Existing Veit control, proportional shared-age overflow, adult-priority overflow; seeds 41/42 | 6 |
| Historical CBC detection | Surveyed detections/nondetections only, then plus buffered exterior range constraints; seeds 41/42 | 4 |
| Small capacity floor | Existing departure with additive floors 1.25 and 4 times the fixed absolute Allee density scale; seeds 41/42 | 4 |
| Combined structural constraint | Adult-priority overflow plus the 4-times floor; seeds 41/42 | 2 |

All inherit the 1951 established-state/no-native-departure configuration, yearling breeding, shared habitat field, absolute aggregated Allee (`aggregation_k=3`) and NestWatch. Main movement/floor comparisons fix the introduced-state multiplier at 1, the inherited nominal 102.8 bird-equivalent convention. This is a conditional reference, not an observed census. Historical CBC fits let that multiplier vary. Within-cell support and the 50/50 initial age split remain inherited assumptions. Modern all-years CBC counts are disabled throughout this batch.

These are cold MAP fits followed by float64 refinement. New departure rules add no learned scalar. Width is fixed at 5% of K, in the same pre-departure total-population currency as the inherited model. Adult priority means adults fill the retention budget first, followed by young. Existing convolution kernels still include some within-cell retention: entering the dispersal pool is not identical to successful off-cell export. Compare actual successful movement, not just departure fractions.

## Capacity floor: intended restriction

The point is to learn inhospitable habitat through demographic performance while restricting tiny carrying capacities as an artificial absence switch under absolute Allee. The floor is not a requirement that every cell support a viable population. Genuine demographic sinks remain allowed.

Implemented as `K_final = K_disease_adjusted + K_floor`, a positive differentiable link. The inherited Allee scale remains fixed at approximately 0.05921 route-count units (0.002961 internal units at population scale 20). Floors are approximately 0.07401 and 0.23684 route-count units, or 0.003701 and 0.011842 internal units. This parameter is an Allee density scale, not an exactly measured extinction boundary or exact half-mating density under the aggregated form.

Applying the additive floor after disease prevents disease from lowering final K beneath it. The unfloored baseline K and inherited native-seeding formula are unchanged, so the floor itself does not add native founders. Fitted baseline K can still vary between experiments, as it did previously. The restriction affects N/K-dependent crowding and movement in marginal habitat; those are intended, scientifically consequential effects to measure. It is not assumed harmless by definition.

## Historical CBC supervision

There are 14,740 eligible circle-years from the raw 1902–1966 winter records, aligned to summer 1902–1965: 10,511 training, 2,461 spatial holdout and 1,768 nonspatial temporal holdout rows. Training includes 5,377 observations in 1951–1965, with 103 overlapping the release-cell footprint. Counts enter only as detection/nondetection, never count magnitudes. Unknown focal counts and invalid dates are excluded. Missing effort uses a reference exposure and an estimated missingness effect, not zero effort.

The binary likelihood is Bernoulli with complementary-log-log encounter link, `p = 1-exp(-encounter_rate)`. Encounter rate reads the simulated population through the existing positive seasonal observation operator, with source calibration, circle heterogeneity and Box-Cox effort response. This is a conventional encounter model; it does not claim that a single visit independently identifies occupancy and detection. Fixed seasonal settings (300 km SW scenario, migrant fraction .35, young availability 1) avoid adding those uncertain scalars to this first historical test. They are explicit sensitivity assumptions, not identified movement estimates.

The introduced state begins in 1951. Therefore eastern/central 1940–1950 CBC rows are excluded rather than scored against an introduced population the model has not represented. The native western series remains available. Pre-1940 eastern detections/nondetections can supervise the absence of the introduced range.

The inherited input already contains 68,932 pre-1940 hull/buffer pseudozeros and **no** post-1950 pseudozeros. Those are preserved. New envelope constraints cover 1951–1962 only, using separate western and eastern positive-observation hulls and the inherited conservative 700 km buffer. They exclude surveyed-circle support in each five-year bin and use only training detections to define geometry. Each bin's later training detections define a retrospective conservative extent for that bin, not an annually observed range boundary. No continental hull connects the two ranges. New exterior rows have an explicit fixed weight .1 and are assessed as an additional assumption via the surveyed-only arm. The 1963–1965 temporal holdout receives neither CBC fitting factors nor new envelope factors.

The buffer follows the project's existing conservative bound approach. The 330 km juvenile kernel parameter is a characteristic distance of a long-tailed distribution, not a literal finite maximum. No new dispersal-distance estimate or hard cutoff is inferred here.

## Validation and automatic analysis

Six numerical tests passed: allocation bounds/conservation, adult priority, empty-cell/capacity-boundary derivatives, probability/pool agreement, and the binary likelihood against Poisson encounter probabilities. Four full-data checks cover shared overflow, adult priority, the larger floor and CBC plus envelope. All have finite objective/gradient values for survival, reproduction, capacity and introduced abundance. Diagnostic trajectory replay errors are zero; maximum BIDE residual is below 1.4e-15. This caught and corrected an attribution mismatch in the numerical K safeguard before queue publication.

Runge/BIDE attribution freezes both adult and young departure probabilities at the fitted ambient pools before following origin contributions. It does not substitute a nonlinear movement Jacobian for contributions. Automatic followups collect full pathology readouts, matched BBS rescoring, contribution/budgets, isolation, permanent movement distances and finite-propagule establishment checks. Historical CBC scoring reports detection calibration, log score and Brier score on both holdouts, integrating unseen circle effects by quadrature. These are conditional MAP checks, not posterior identification.

Cross-model process comparisons include the new matched movement, floor and historical-CBC contrasts. They retain the same metrics/support across East, Great Plains and West, with per-cell changes and reversals alongside regional means. Vital-rate plausibility alone will not select a winner. Evaluate actual-state replacement, Runge contribution, immigration dependence, finite-propagule outcomes, low-K/Allee suppression, Plains source connectivity, successful permanent dispersal distances, invasion timing, and BBS/NestWatch correspondence together.

## Compute and persistence

Existing A100 small job **3544518** is retained; its current fit is not interrupted. The already-authorized HMC timing screen is promoted next on that longer allocation. Development job **3546708** requests one three-GPU A100 development node for two hours; three separate GPU lanes claim distinct new fits atomically. It was scheduler-pending on submission, with a provisional 10:10 a.m. Central start estimate. That estimate can change.

Ten unstarted, redundant all-years CBC sensitivities are preserved in `queue/held_new_priority`, giving these branches priority. A persistent resource watcher records all allocated GPUs, recovers interrupted claims and may submit up to three development waves total while at least three new fits remain pending. Longer HMC is excluded from development workers. Partial MAP/refinement checkpoints are retained if an allocation ends. Existing and new collection/audit watchers are active; model source and accepted checkpoints from older runs are unchanged.

## Methodological basis

The density-dependent departure baseline is tied to [Veit and Lewis (1996)](https://doi.org/10.1086/285924); excess-only allocation and adult priority are explicit structural alternatives, not asserted House Finch measurements. [Link, Sauer and Niven (2006)](https://doi.org/10.1093/condor/108.1.13) supports modeling CBC effort rather than treating raw counts as comparable abundance. [MacKenzie et al. (2002)](https://www.usgs.gov/publications/estimating-site-occupancy-rates-when-detection-probabilities-are-less-one) establishes why nondetection cannot simply be equated with absence. Hull exterior pseudozeros and a K floor remain openly stated project assumptions, tested through matched sensitivity arms rather than presented as independently observed biological facts.
