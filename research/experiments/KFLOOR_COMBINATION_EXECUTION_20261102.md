# Floored combinations: implemented and queued November 2

> **Objective correction, November 3:** Lambda inflation denotes excessive historical change, not a high modern growth level. Reducing the reproductive ceiling is a separate mechanistic hypothesis and does not demonstrate a reduction of temporal inflation. Primary temporal readout: modern-minus-early potential lambda on the same cells with equal weights, reported for East, Great Plains and West. Report absolute levels separately and retain the full pathology assessment. This correction changes interpretation; it does not alter frozen fits or their source/configuration.


The user authorized the combinations in KFLOOR_COMBINATION_RETEST_PLAN_20261102.md. All18 new fits are published and verified pending on TACC. Specifications: [KFLOOR_COMBINATION_RUNS_20261102.json](KFLOOR_COMBINATION_RUNS_20261102.json). Seeds41 and42 throughout. Every new run is a cold MAP followed by float64 refinement, with immutable code and source manifests.

## Design

| Family | New fits | Matched comparison |
|---|---:|---|
| Floored crowding/departure factorial | 6 | Power1/rational-adult20, power2/Veit and power2/rational-adult20, each with the4x floor. Existing step39 floored Veit/power1 supplies the other two controls. Compare with the corresponding unfloored step41/42 runs too. |
| Modest reproductive variation | 4 | Fixed sigma.1 log-fecundity variation, eight regional basis functions,1966â€“2020, zero in forecasts, with constant and linear temporal components removed; floor0 versus4x. |
| Limited local placement of arrivals | 4 |25% of arrivals allocated over their cell and immediate neighboring cell centers according to capacity, with maximum added search distance about76km; floor0 versus4x. Residents stay in place. |
| Native retention, secondary | 4 | Existing single native-departure scalar on the same fixed1951 state; floor0 versus4x. Placed behind the primary combinations. |

The common floor is4 times the fixed absolute mate-finding density scale:0.0118421053 internal units,0.2368421 route-count units under population scalar20. This choice fixes one common restriction for the combination test; it is not selected from favorable fitted maps. The separate1.25x and4x standalone runs remain the floor-strength comparison. No guarantee of viability is imposed on truly poor demographic habitat.

All inherit yearling breeding, shared habitat, fixed weak absolute aggregated mate limitation, NestWatch with the same conversion assumptions, the fixed nominal1951 introduced state, no release-cell exemption and original pseudozeros. CBC counts/range likelihood is not newly added in these comparisons. The K floor is added after the existing field projection, and the native seed remains based on unfloored baseline capacity. No currently running fit was edited.

## Why these particular ports

The modest-variation branch uses the existing fecundity-target mechanism rather than a post-reproduction abundance multiplier. Births, reproductive observations and contribution accounting therefore include its effect explicitly. It is a current-base test of the earlier useful variation mechanism, not an exact replay of E070's abundance-noise version. Its scale is fixed, not optimized; it adds no learned amplitude and cannot carry a constant/linear temporal trend through its basis coefficients. It can still influence population histories substantially, so a fixed-parameter replay zeroing the variation is included in the follow-up. Potential-growth and realized reproductive diagnostics include the fitted variation; stripping it from a diagnostic is not counted as biological improvement.

Local placement is a deliberately bounded alternative to perfect-information settlement. A fraction.75 remains at the original arrival destination; fraction.25 is distributed among local land cells using positive capacity weights normalized per source. It does not change juvenile stayer survival or apply Sj again to arriving young. Journey survival is paid before the short local placement; additional search mortality is assumed negligible. Adults receive local annual survival at the final destination, as in the existing simulator. This is a stated assumption and a limited variant of the older settlement idea, not the exact old3x3 block algorithm at a different grid resolution.

The local operator excludes non-land intermediate cells. Otherwise it could incorrectly relocate arrivals that the original land-conditioned transport would discard. Independent finite-grid checks caught that distinction in origin-retention accounting before publication. The operator's columns conserve incoming land mass and remain nonnegative. Mean successful movement distances and tails are calculated for the combined final displacement, rather than labeling first-leg distance as endpoint distance.

## Validation and diagnostics

All nine scientific configuration types (the three factorial additions plus three mechanisms each with/without the floor) passed full-data float64 likelihood and gradient evaluations. The reproductive effect was nonzero in validation and its directional derivative was checked; native departure was nonzero and its derivative was included. Maximum birth/death/immigration/emigration residual over these checks was1.34e-15 (below1.5e-15); adult/young replay errors were at most3.34e-16. These are numerical checks, not evidence of biological improvement.

The independent local-placement matrix comparison checked conservation (maximum error2.22e-16), the exact diagonal of long movement plus local placement (3.47e-18), and full endpoint-distance/tail moments (1.11e-16 /2.78e-17). Checks are implemented in `verify_local_settlement.py`. The updated contribution audit is frozen inside every new run and used by its runner, leaving the shared remote audit for existing runs unchanged.

Local and remote validation receipts are under `.research_runtime/floor_combination_smoke/VALIDATED_LOCAL.json` and `VALIDATED_TACC.json`, with remote `monitor/floor_combination_VALIDATED.json`. Critical source hashes, scientific population configurations and cold-start seed flags were checked for all18 runs before publication; source manifests are rechecked when a fit starts.

Persistent collectors recognize step43, download each frozen source, rescore accepted fits and schedule movement/establishment follow-ups. The common-cell audit includes floor-on/off contrasts within each mechanism, mechanism contrasts at the same floor and both floored crowding/departure interactions. Oscillation flags use the existing80-year local-propagule trajectories. All East/Great Plains/West comparisons retain the same definitions and supports within the audit. The complete pathology checklist remains required, including growth inflation, habitat gating, effective survival, successful exchange/immigration dependence, K/Allee artifacts, bridges, establishment, prediction, seed stability and HMC geometry.

Implementation: `stage_floor_combinations.py`, `local_arrival_settlement.py`, `smoke_floor_combinations.py`, extended `contribution_budget.py` and `run_movement_followup.py`.

## Scheduler receipt

Verified at approximately20:44 CST. Single-A100 job3544518 remained active, with about9h19m of allocation left. Three-GPU requests3547292 (four hours regular) and3548096 (two hours development) remained pending. An additional bounded four-hour/three-GPU regular request,3548331, was submitted for the expanded authorized backlog and was pending on resources. [Dispatch receipt](KFLOOR_COMBINATION_DISPATCH_20261102.json).

All18 new fits were pending at verification; none is represented as started or completed. Existing standalone K-floor fits keep the first pending positions. New factorial fits follow the current departure/curvature batch; reproductive variation and local placement follow, with native retention secondary. All worker lanes claim from the same queue atomically, preventing duplicate fits across allocations. Queue delay remains uncertain, so complete overnight execution is not guaranteed.
