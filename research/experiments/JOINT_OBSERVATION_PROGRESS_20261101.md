# Joint-observation implementation — first results

2026-11-01T11:30:12.012945-06:00

This is a data/observation-model pilot, not a completed joint population refit. Existing population fits and TACC queue are unchanged. Methods and amendments are registered in JOINT_OBSERVATION_PILOT_20261101.md.

| Source | Preparation | Calibration attempt |
|---|---|---|
| CBC | complete | finished; review required |
| eBird Canada | complete | finished; review required |
| eBird US | complete | finished; review required |

CBC focal export: 106,894 circle-years at 4,256 circles, 1940–2025. Raw date/year conflicts: 3; excluded from primary calibration. Missing historical effort is retained as missing.

Standard NB2 pilots use survey effort, geography/time and an optional circle panel. Counts remain winter observations. These are source-calibration baselines, not the final shared-population likelihood.

| Model | Converged | Temporal-test n | Temporal log score | Mean observed / predicted |
|---|---|---:|---:|---|
| linear_effort | True | 9640 | -4.9976 | 116.78 / 133.86 |
| curved_effort | True | 9640 | -4.9969 | 116.78 / 137.61 |
| circle_panel | True | 367 | -5.6736 | 177.44 / 130.86 |

The global polynomial baselines overpredict average counts, and the circle panel underpredicts the temporal-test mean. These simplified geographic/trend baselines are not adequate production calibrations. The circle-panel prediction is conditional on previously observed circles, selected using survey availability and geography, and is not a spatial-transfer test. The next likelihood comparison must use the ecological/population readout and retain blocked evaluation.

NestWatch audit reconciles 1,089 training attempts (1,065 exposure rows; 528 successful-count rows). 70.1% date from 2000 onward.
There are 318 location-years with multiple reported attempts, but no individual-female or complete-season denominator in the inspected schema. Annual attempts and breeding participation are not identified by those reporting counts.

BBL inputs retained: 296,896 adult bands and 782 dead recoveries. A differentiable port of the existing coarse regional recovery likelihood now exists and matches the independent implementation/finite-difference gradients in tests. It retains never-recovered birds, reporting eras and partial exposure. It is not yet connected to the spatial population simulator; the full movement/survival adapter and approximation checks remain required.

eBird CA: effort-aware hurdle calibration results

| Season | n after thinning | Encounter converged | Positive-count converged | Temporal Brier | Temporal observed / predicted detection |
|---|---:|---|---|---:|---|
| breeding | 45641 | True | True | 0.0404 | 0.0454 / 0.0323 |
| winter | 13567 | True | True | 0.0723 | 0.0857 / 0.0607 |

Convergence is not acceptance. The Canadian baselines underpredict temporal-test encounter rates. A low Brier score for a rarely detected species does not by itself establish calibration; inspect observed versus predicted rates and compare the shared population/ecological readout before integration.

eBird US: effort-aware hurdle calibration results

| Season | n after thinning | Encounter converged | Positive-count converged | Temporal Brier | Temporal observed / predicted detection |
|---|---:|---|---|---:|---|
| breeding | 200000 | True | True | 0.1481 | 0.2104 / 0.1862 |
| winter | 97226 | True | True | 0.1481 | 0.2093 / 0.1939 |

Convergence is not acceptance. The Canadian baselines underpredict temporal-test encounter rates. A low Brier score for a rarely detected species does not by itself establish calibration; inspect observed versus predicted rates and compare the shared population/ecological readout before integration.

The likelihood tests cover checklist denominator selection, group deduplication, X detections, winter-year alignment, zero-truncated NB2 conditioning, and the differentiable recovery kernel. A successful source fit is not a biological identification gate.

Artifacts: `/mnt/e/Datasets/houfin/joint_observations_20261101`. Preparation and calibration jobs run persistently in the observation-io, observation-cpu and observation-watch pueue groups. Full-country scans are required before zero-filled focal products become ready.

Next: completed eBird exports automatically trigger the registered effort-aware encounter/positive-count calibration, using the pinned 216-km spatial holdout and 2021–25 temporal holdout. Then evaluate the shared annual population readout and modern seasonal mapping. No founder prior or demographic parameter has been changed by this pilot.
