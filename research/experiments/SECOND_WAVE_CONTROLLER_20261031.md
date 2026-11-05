# Overnight experiment controller — expanded October 31, 2026

Objective: find parsimonious House Finch configurations that reduce joint biological pathologies while retaining useful prediction. This is a registered experiment dispatcher, not automatic model selection. Expansion follows the user's instruction to provide productive work through 10 a.m.

## Matched experiment queue

All arms retain yearling breeding, the shared demographic habitat field, the observation model, real NestWatch and the CBC founder anchor unless the named sensitivity changes its value. Relative and absolute Allee laws have separate fixed parents (`step33_nestcbc_s41` and `step33_nestcbc_abs_s41`). Every new arm is a cold fit from the same frozen source.

| Group | New fits | Question |
|---|---:|---|
| Absolute-Allee independent seed 42 | 1 | Does the initial absolute-law result reproduce? |
| Founder low/high and nest attempts low/high, under each Allee law | 8 | Does auxiliary-data uncertainty move the distortion into survival, growth, capacity or establishment? |
| Distance hazard, each Allee law, seeds 41/42 | 4 | Does CBC plus nesting information change the benefit or inferred magnitude of travel mortality? |
| Segmented compounded path survival, each law, seeds 41/42 | 4 | Does accounting for heterogeneous journeys reduce front/habitat compensation? |
| Native departure under absolute Allee, seeds 41/42 | 2 | Does reduced departure in the native range help under the preferred mate-finding law? Relative-law counterparts are already first-wave fits. |
| Distance hazard or compounded survival crossed with native departure, each law, seeds 41/42 | 8 | Does apparent travel mortality disappear when staying versus leaving is represented, or do the terms merely compensate? Compare both single-term controls before considering the extra scalar justified. |
| DESK v2, each Allee law, seeds 41/42 | 4 | Are apparent demographic/movement improvements dependent on the chosen habitat basis? |
| DESK v2 crossed with distance hazard or native departure, each law, seeds 41/42 | 8 | Do movement conclusions transfer to the alternative habitat basis, or depend on habitat compensation? |
| Conditional founder × nest-attempt corners | up to 4 | Do individually influential auxiliary assumptions interact to recreate pathologies? |
| Conditional reference seeds 43/44 | up to 2 | Is the fixed reference materially basin-dependent? |

There are **39 nonconditional configurations and up to six conditional follow-ups**. Eight compounding configurations are currently held by the failed feature-equivalence audit, leaving **31 input-ready planned fits plus six conditional follow-ups**, each gated on an accepted parent and complete biological/front readouts. The existing independent absolute repeat counts toward these totals. Nonconditional means scientifically warranted in advance; it does not bypass input readiness, source validation, time or spending limits.

Founder log pins are -1.79 / +1.23; fixed nest attempts are 2.0 / 3.125 versus 2.5 centrally. Movement settings reproduce the existing source-supported experiments: distance hazard HalfNormal(.05) with existing band means [80,286,867] km; four-segment endpoint-reference compounding with scale .25 and 100 km length; native-departure Beta(1.5,1.5). No new free spatial field is introduced. The distance-only model remains an approximation to travel mortality.

Conditional corners require all four relative-law main sensitivities to have accepted complete readouts and at least one change in adult-survival median >.05, realized fecundity at nests >20%, population share with resident Sj/Sa >.95 >.10, or median fitted-front timing >5 years. Additional seeds require identical-configuration initial seeds to differ in adult survival >.05, fecundity >20%, or front timing >5 years. Missing results cannot trigger a positive claim.

## Input preparation and contracts

Native-habitat segment features are prepared on the local GPU in float32 with the existing frozen builder, using at most .55 allocator fraction. The existing builder checks median reproduction error for every year. An additional equivalence gate is required because median error alone can hide large localized discrepancies. Transfer verifies SHA256, exact input metadata identity, full dimensions and file size. Existing differing files are never overwritten. Compounding fits remain gated until complete segment files and reproduction metadata are present. Float64 fitting/refinement stays on TACC.

Before any v2 fit is staged, compare the complete input metadata recursively: observations, route/observer identifiers, years, grid/land ordering, initial conditions, kernels and other dynamics must match exactly. Only feature provenance/build identifiers may differ, and kernel/rank/centering contracts must agree. Preserve metadata hashes in each run's INPUT_CONTRACT.json. An invalid arm is reported as blocked without blocking other eligible work.

## Resource limits and reporting

The authorized budget is **36 additional TACC GPU-hours**, replacing the earlier assistant-selected 12-hour bound. Reserve two hours for every queued/running attempt and replace that with measured elapsed time at completion, including failures. Publish additional eligible arms as reservations are released; the full candidate list is not a promise to exceed the budget. The local feature-preparation task is separate from this TACC fit budget.

Each attempt has a two-hour timeout, frozen source/config manifest, exclusive ownership and durable receipt. Do not automatically retry or relax acceptance gates. Use existing allocations only. All current worker submissions explicitly use the same GRID54 queue; separate atomic tasks let any free worker claim a different fit. Two nodes are active and the third is pending at registration. Its scheduler start estimate is not a guarantee of overnight availability.

Stop staging at **2026-11-01 10:00 CST** (`2026-11-01T10:00:00-06:00`), or when monitor/STOP_SECOND_WAVE exists. In-flight work is preserved. Record staging, GPU-hour reservations, missing inputs, blocked arms and budget deferrals in the persistent ledger. Every accepted completed fit enters the existing paired held-out scorer and whole-pathology/front reporting. Predictive gain alone does not select a model; inspect the complete checklist and biological parameter compensation. The bounded queue extends productive work without guaranteeing hardware utilization if fits fail or allocations do not start.

### Segment-equivalence hold and replacement comparisons

The native float32 segment build passed the historical median-only check (max normalized median error 1.09e-8), but sampled relative L2 errors were about 2% and maximum absolute error reached .548. A one-year float64 probe still had max absolute error .237 and normalized p999 error .0208. This is an input-equivalence failure, not evidence about biological compounding. All eight compounding fits are held before any attempt; the controller requires `overnight/native_segments_equivalence.json` with `passed: true` before publishing them. Existing fits are unaffected. Newly created segment files are retained as unvalidated artifacts.

Eight paired v2 × movement tests (distance hazard or native departure, both Allee laws, two seeds) replace the held compounding work in the input-ready supply. Total registry: 45 arms; 31 input-ready nonconditional comparisons, six conditional follow-ups, eight compounding arms held for a later input correction. Spending remains capped at 36 additional TACC GPU-hours. No automatic scientific adoption or attempted-input repair is implied.
