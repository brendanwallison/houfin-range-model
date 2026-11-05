# Fixed crowding-curvature test: execution specification

> **Objective correction, November 3:** Lambda inflation denotes excessive historical change, not a high modern growth level. Reducing the reproductive ceiling is a separate mechanistic hypothesis and does not demonstrate a reduction of temporal inflation. Primary temporal readout: modern-minus-early potential lambda on the same cells with equal weights, reported for East, Great Plains and West. Report absolute levels separately and retain the full pathology assessment. This correction changes interpretation; it does not alter frozen fits or their source/configuration.


Authorized November 2 after LAMBDA_INFLATION_SYNTHESIS_20261102.md. Primary question: can reproduction at intermediate density be fitted with less inflated maximum growth by changing the density response, while maintaining credible movement and sourceâ€“sink processes?

## Matched design

Two seeds, 41 and 42, in each cell of a two-by-two comparison:

| Departure | Crowding power 1 control | New crowding power 2 |
|---|---|---|
| Gradual Veit, shared by age | step39_veit_fixed_s41/s42 | step42_power2_veit_s41/s42 |
| Rational gradual departure, adult multiplier .2 | step41_rational_adult20_s41/s42 | step42_power2_rational_adult20_s41/s42 |

Four new cold MAP + float64 refinement fits; four controls are reused. Source, configuration and checkpoint provenance remain separate per run. The power-2 Veit arm does not depend on the new departure law succeeding.

All inherit the fixed nominal 1951 established state (multiplier 1; roughly 102.8 bird-equivalents under the inherited conversion), equal adult/young initialization, no release-cell Allee exemption, yearling breeding, one shared habitat field, weak fixed absolute clumped mate limitation, NestWatch and its inherited conversion assumptions, original historical pseudozeros, no CBC likelihood, no capacity floor and no native-departure scalar. These are matched scientific assumptions, not newly validated demographic facts. Historical-CBC transfer and K-floor interactions remain subsequent conditional tests.

## Change

The existing standard reproduction becomes:

```
F_effective = Fmax / (1 + c * (N_reproduction/K)^power)
c = max(Fmax*Sj/(1-Sa+1e-6) - 1, 0)
```

The fixed power is 2 in the new fits and 1 in the inherited controls. Actual births still apply mate limitation, the actual surviving breeding pool and the existing disease/movement accounting. Cells unable to replace themselves retain c=0; the experiment does not force viability. K retains its original meaning, including the limitation that mate finding and disease can move the realized equilibrium below nominal K. There is no new fitted scalar or spatial field.

Implementation is isolated to frozen run source: `density_regulation.py` supplies a backend-preserving density argument to both `age_forward.reproduction_age_structured` and `src/vis/age_model_math.allee_viability`. The at-K equilibrium calculation is unchanged because both powers equal 1 there. Potential lambda remains Sa+Sj*Fmax. Contribution, regional-isolation, finite-propagule and invasion replays invoke each run's own forward primitive. Unsupported subcell/occupancy/settlement and alternate reproduction branches are excluded by staging assertions.

## Validation required before publication

- Exact identity of the power-one crowding argument; original full-data objective/gradient parity on the newer movement lineage.
- Independent fecundity values and finite-difference derivatives through survival, fecundity, capacity and density.
- Two-age equilibrium and local feedback eigenvalues on the illustrative case.
- Viability peak search compared directly with the fitted forward reproductive function, including an intrinsically nonviable case.
- Full-data float64 likelihood/gradients and frozen movement/contribution mass balance for both movement lineages.
- Hash agreement between validated source/configuration and all four staged runs; manifests checked again at publication and fit start.

Tests do not establish biological superiority or global stability. Stronger overcompensation above K is a specific risk. Local finite-propagule follow-ups retain 80-year trajectories; tail variability and fitted biological curves must be inspected before selecting a candidate. Numerical acceptance does not pass that biological check.

## Evaluation

Compare both movement-matched power contrasts and the movement contrast under power 2. The cross-model audit uses identical cells, years, regional definitions and metrics for East, Great Plains and West. Retain the complete pathology checklist: potential growth and upper tails, historical growth shifts, realized contribution/replacement, effective survival, successful export and immigration dependence, K/Allee artifacts and bridges, establishment and invasion timing, reproductive observations, prediction, seed sensitivity, parsimony and HMC geometry. A lower Fmax with compensating survival is not sufficient. A productive Plains bridge is not excused by a favorable regional mean.

Queue priority 0614 places the new tests after the existing departure comparison and before the older floor sensitivities. Shared queue claims are atomic across the small A100 worker and all three development/regular GPU lanes. Persistent status, source collection, predictive rescoring, movement follow-ups and direct cross-model contrasts are extended to step42.

Implementation: `research/exp/density_regulation.py`, `stage_crowding_curvature.py`, `verify_crowding_curvature.py`. Publication and live scheduler receipts are appended after validation.

## Published and verified

At 20:01 CST November 2, all four step42 scripts were verified in TACC's pending queue. Run specifications: [CROWDING_CURVATURE_RUNS_20261102.json](CROWDING_CURVATURE_RUNS_20261102.json). Local validation receipt: `.research_runtime/crowding_smoke/VALIDATED_TACC.json`; the corresponding remote receipt and manifests are under the grid54 monitor/run directories.

Both exact frozen movement lineages passed independent reproductive/viability checks and full-data float64 gradients. Maximum birth/death/immigration/emigration balance errors were 1.33e-15 for Veit and 1.44e-15 for rational age-specific departure; both adult and juvenile replay discrepancies were exactly zero. The power-one rational control reproduced the previous full-data objective and its five checked gradients exactly. The shared-power helper preserved the original power-one argument exactly. The illustrative full two-age density-feedback eigenvalues were inside the unit circle; this is not a stability certificate for all fitted cells.

Persistent watchers restarted as local tasks 1152/1153. New step42 runs are eligible for all development/regular GPU lanes and the existing single-A100 worker. Fixed-cell process audits now include direct curvature contrasts under both departure rules and the departure contrast under power2. A separate post-fit score inspects the final 20 years of the existing80-year local propagule trajectories for persistent variation; it retains the trajectory outputs and treats a large range as a diagnostic flag, not automatic proof of a stable cycle.

Scheduler check: small job3544518 running step41_rational_adult20_s41 since19:55:38, with10h02m left; three-GPU jobs3547292 (four hours regular) and3548096 (two hours development) remained pending on priority. No step42 fit had started at this receipt. Existing completed/running fits were not edited.
