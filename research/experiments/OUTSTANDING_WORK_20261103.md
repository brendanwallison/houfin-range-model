# Outstanding work and execution register — November 3

This register distinguishes validated queue entries, unfinished analysis, and conditional research. Lambda inflation means excessive change in growth through time. High growth levels are a separate concern. No single diagnostic selects a model.

## Published primary tests

The 28 field/abundance fits in FLEXIBILITY_EXECUTION_20261103.md are validated and published. Nine test small field departures, eighteen compare tiny independent versus regional abundance innovations, and one adds a third common-control seed. All comparisons use seeds 41/42/43 and retain the same fixed 1951 initialization, original departure and larger K floor. Publication receipts are MINOR_FLEXIBILITY_PUBLICATION_20261103.json. A queue entry is not a running or completed fit.

## Remaining matched transfers: registered before staging

Twelve additional fits are now validated and published:

1. Six small transport tests: redistribute only 5% of already-arrived birds within the 3x3 land neighborhood. Compare uniform land weights against capacity weights, seeds 41/42/43. Hold expected arrivals, stage survival, boundary treatment, original departure, fixed 1951 initialization and larger floor constant. Residents are untouched. The existing 25% capacity-placement fits are a larger-budget comparator, not the matched control. Uniform weights must enter redistribution, return-to-origin attribution and distance accounting consistently.
2. Six historical-supervision/floor transfers: add the larger floor separately to the original pseudozero model, pre-1940 range replacement, and the replacement plus dated 1951–1965 invasion constraints, seeds 41/42. All six retain their parents' **free** established-state abundance. Reuse the six corresponding unfloored parents. Comparing these directly against the fixed-abundance floor arm would confound supervision with initialization. No CBC count likelihood or new winter movement process enters.

The floor is additive after disease in final K and does not alter native initialization K. It blocks extremely small capacity as an explanatory route; it does not force positive demographic growth or establishment everywhere. Both genuinely poor habitat and residual deterministic establishment problems must still be measured.

The staging implementation is ../exp/stage_outstanding_transfer.py. All five scientific configurations passed full-data finite objective/gradient and trajectory accounting. Both placement forms also passed independent dense transport, mass-conservation, origin-offset and composite-diagonal checks. All twelve seed sources/configurations matched the tested groups before publication. The transfer batch is lower priority than the 28 primary fits and above remaining secondary native-departure fits.

## Existing unfinished execution

Finish float64 refinement and follow-ups for the four reproductive-variation and four 25% capacity-placement fits, each with/without floor and two seeds. Local float32 checkpoint handoff does not complete these fits. Retain the four native-departure sensitivities as secondary; do not promote them into the preferred model. Two strict adult-priority/floor fits remain legacy stress tests, rather than a proposed biological solution.

Previously held early-effort CBC designs and ten redundant all-years CBC variants remain held. They are not missing execution to restart indiscriminately. Failed queue status is reconciled against refinement and diagnostic receipts: a post-fit diagnostic failure differs from failed optimization convergence.

## Analysis required across all completed families

Use one comparative scorecard across original and altered crowding, departure, floors, reproductive variation, placement, historical supervision, and the new flexibility tests. Include unsuccessful and nonconverged attempts with their status; do not silently treat them as accepted comparisons.

- Historical growth change: fixed cells and weights, early and modern levels, each adjacent era, and post-1966 change; exhaustive East/Great Plains/West within 25–49N plus explicitly labeled legacy masks.
- Components versus emergent processes: vital rates, effective mover/stayer survival, realized reproduction, local replacement, origin-attributed breeding contribution, export, immigrant dependence and regional persistence; continuous differences and cellwise reversals across model classes.
- Geographic artifacts: tiny K, K/Allee failure, finite-propagule responses, genuinely poor habitat, Plains source connectivity with both edge/diagonal adjacency, and permanent movement distances/tails.
- Timing and feedback: fitted and frozen-environment invasion fronts, arrival thresholds, establishment at observed locations, density-feedback oscillations, and exodus/regathering.
- Process-correction reliance: biological and full fitted growth/contribution, explicit correction budget, cumulative effects, maps, and zero-innovation trajectories/fronts. Small annual SD alone is insufficient evidence of a small effect.
- Evidence and inference: matched real-record blocked prediction with paired uncertainty; NestWatch annual conversion; founder abundance/age/support assumptions; seed/precision robustness; source/input provenance and HMC cost/geometry.

These checks are part of evaluation, not a license to discard a useful improvement because some parameter remains non-identifiable. Stability of higher-level growth and movement must itself be measured.

## Conditional scientific work, not queue entries

1. Combine complementary components only after their matched single-component results are available. Pre-register the actual combination and its compensating failure predictions; do not launch an unrestricted Cartesian product.
2. Extend independent noise to earlier years only if the modern-window experiment is useful and its innovation reliance is acceptable. The first batch deliberately cannot directly correct 1902–1965 states.
3. If private K helps, test small reproductive-private or survival-versus-reproduction grouped fields. These are not already covered by fixed all-field rho .99.
4. Probe reduced historical-growth-change targets at approximately 25% and 50% below matched controls, reoptimizing compensators from multiple starts. Separate data cost, prior and profile penalty. This is a diagnostic constraint, not a claim that historical habitat must be constant.
5. Compare independent versus clustered arrivals at equal mean movement in forward capability tests before proposing a fitted stochastic establishment model. Specify model-unit to bird-count conversion, group/age/sex assumptions and how to avoid double-counting the existing clumped mating correction. This is not implemented as an HMC likelihood.
6. Use a Gaussian redistribution kernel as the first diffusion contrast, matching displacement variance and boundary treatment. Do not replace the full inference stack with an unsupported PDE branch. Nonlocal tails and stage/survival timing remain explicit differences.

## Observation and HMC branches still unfinished

NestWatch annual conversion remains uncertain; reported nesting attempts do not identify attempts per female per year. Revisit the uncertain conversion jointly with useful floor/supervision combinations rather than assuming the existing conversion is an annual census.

The BBL differentiable recovery pilot is not yet coupled to the spatial simulator. Complete stage/region/year exposure, permanent-versus-seasonal movement and reporting consistency checks before a joint fit. Existing eBird/CBC calibration pilots do not yet establish a calibrated shared population readout; seasonal availability and blocked transfer need validation. Historical range supervision is already a useful separate branch and does not wait on a full seasonal process model.

One older diagonal NUTS timing pilot is complete; three old sampler comparisons remain pending, with four older geometry results available. These do not establish HMC feasibility of any new candidate. Use gradient/HVP cost and geometry first on shortlisted current fits, then budget a resumable sampler pilot using measured cost. The ~237,490 iid latent innovations make automatic full NUTS on every noise case an inappropriate first screen. Evaluate mixing for regional growth, contribution and movement as well as component parameters.

## Verified publication

All **40 new fits** are published: 28 primary flexibility tests and 12 matched transfers. The direct shared-queue check found 39 pending and the third-seed shared control running locally. See OUTSTANDING_PUBLICATION_20261103.json and the auto-updated OUTSTANDING_WORK_STATUS_20261103.md. Validation is not fit convergence or biological acceptance.

The local worker reloaded at a completed-MAP boundary, preserving the prior placement fit and its verified handoff. Pending TACC workers use the updated shared registry. Dispatch samples distinct seed-41 scientific hypotheses before their seed-42/43 repeats, while preserving batch priorities. At the last scheduler check no TACC allocation was running; four remained pending. The development estimate had moved to 10:30 CST; other requests had no estimate. No new allocation was requested for this work.

## Execution safeguards

Existing TACC requests consume the atomic shared queue; local 3090 performs float32 MAP only and hands verified checkpoints back for TACC float64 refinement. Never interrupt an active fit to reload workers. Monitor allocation state, lane claims, manifests, refinement receipts, rescoring and missing diagnostics separately. No additional allocation is implied by this register.
Later local update: the new shared-control seed-43 attempt terminated with signal 11 around MAP step 97, without a reported numerical diagnosis. Its logs were retained and its original task returned to the TACC pending queue; it is excluded from immediate local retries. The worker then claimed the independent cell-year sigma .01 seed-41 fit. This is an execution failure, not evidence against the scientific configuration. The live status table supersedes the earlier running-control snapshot.

Legacy reconciliation: six old CBC queue failures have accepted refinements and recovered budgets; their failure flags are not six failed fits. The 600-km all-years sensitivity remains genuinely nonconverged and excluded from accepted comparisons. The older later-state no-NestWatch/native fit already has an accepted refinement but lacks final diagnostics; CPU recovery task 1235 was scheduled to reuse that checkpoint, without repeating optimization. Follow-up scheduling now includes missing supported older initialization families, while avoiding already completed movement audits.

Native-crash investigation update: a second consecutive local MAP signal-11 failure was discovered. Further local claims are now held while the active fit finishes. The known firmware gap, native-library evidence, bounded repeatability checks, tested failure circuit and unresolved diagnosis are documented in [LOCAL_NATIVE_CRASH_INVESTIGATION_20261103.md](LOCAL_NATIVE_CRASH_INVESTIGATION_20261103.md). The local lane is not certified healthy; TACC queue entries remain available.
