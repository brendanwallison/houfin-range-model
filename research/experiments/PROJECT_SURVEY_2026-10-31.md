# Project survey and dispersal review, 2026-10-31

Read-only survey of the Windows checkout, the isolated WSL ladder checkout, existing experiment records, and the active TACC queue. Live checks were approximately 16:39–16:47 CDT. No fits, jobs, configurations, or existing records were changed. This is an assessment, not an adoption decision or a completed research-protocol verification gate.

## Project map and current source of truth

The project connects two inference problems. ESK/DESK learns ecological similarity from bird communities and historical environmental covariates; the population model uses the resulting feature covariance to regularize spatial and temporal demographic fields. The population simulation combines age structure, dispersal, density dependence, mate finding, introductions, disease, and a BBS count observation model. Optimization currently supplies MAP points, including float64 L-BFGS refinement; these are not posterior uncertainty estimates.

The open October 26 STATUS.md is historical. Follow these instead:

- `GRID54_LADDER_2026-10-29.md`: intervention sequence, current preregistrations, and compensator constraints.
- `DISPERSAL_SYNTHESIS_2026-10-31.md`: compact dispersal results, with some live results still uncollected.
- `FIELD54_FIELD_2026-10-30.md` and the companion covariate/DESK records: current habitat construction.
- `FECUNDITY_CONSTRAINT_SCOPING_2026-10-31.md` and `SURVIVAL_BBL_SCOPING_2026-10-31.md`: new demographic observations.

The Windows branch is `research/desk-temporal`, with substantial existing uncommitted research work. The live population code is a separate WSL checkout at `/home/breallis/houfin/work/houfin/research/GRID54_LADDER_20261029/code`, HEAD `d4435fb` at inspection. Individual fits use frozen snapshots. Reading only Windows `src/model` would miss current mechanisms and the nest likelihood.

## Habitat has improved, with a specific interpretation limit

The native 54 km pipeline now includes seasonal community supervision and a separately fitted regional space-time field over a frozen DESK prediction. In the withheld-place twin, regional change correlation for sedentary common species is .475 for raw DESK, .566 for DESK EMA, and .651 for the field using EMA DESK as input. The comparable averaged-27 km field is .623. The primary stratum has only 21 species; use the recorded paired intervals, not these point estimates alone.

Production exports the **raw field output**. EMA on its DESK input and EMA on its final output are different interventions; the latter harmed transfer in the 27 km comparison. The coastal climate correction and independent retraining also move downstream scores by roughly .01, exceeding the quoted MAP-seed floor. Mechanism comparisons therefore need identical input provenance.

Better prediction of shared community change does not establish its cause. Habitat, observation changes, and community dynamics can all contribute. Conversely, shrinking all historical change would force real habitat change into demographic parameters. Preserve the existing observer-continuity checks, placebos, and independent species evaluation.

## What dispersal currently establishes

The recorded results support a practical identification problem under the current sampling and parameterization, rather than proof that the biological processes are absent.

| Intervention | Recorded outcome | Interpretation |
|---|---|---|
| Distance mortality J1 | Planted .03 recovered near .002 | Fecundity/founders can absorb the cost |
| Water mortality J2 | Planted 1 recovered near .14 | Weakly distinguished from destination/coastal effects |
| Conspecific attraction | Planted .3 recovered near .008 | A null real estimate is not evidence against attraction |
| Learned band weights | Mean movement around 625 km; 60% beyond 483 km | Slight score gain is purchased with biologically suspect redistribution |
| MDD 110/200 km | Higher fecundity and founders; poorer paired held-out scores | Counts alone prefer a tail that conflicts with season-filtered banding evidence |
| Native departure | Planted .6 recovered .779, just beyond the specified upper gate | Distinctive western signature is detected, but recovery is biased; provisional |

The frozen-habitat release front remains about 30 years early in the recorded dispersal comparisons. A displacement warp diagnoses the timing defect but is not a mechanistic solution. A static-habitat synthetic outside truth did not automatically induce lambda inflation, so the equilibrium confound alone does not explain the real-data result.

Float64 optimization exposes these tradeoffs more fully; its success criterion establishes numerical acceptability, not biological plausibility. Compare prediction, biological plausibility, and parameter recovery separately. Raw lambda changes also cannot be compared across delayed and yearling breeding life cycles without accounting for their different local matrices; use suitable-land fractions and front timing alongside the correct eigenvalue.

## Live status, newer than the synthesis

The existing WSL SSH master was active. TACC workers 3541710 and 3542329 were running; 3544518 was pending for priority. Worker presence is not proof that every member fit is making progress.

- Directional-choice bundles 0375 (guards) and 0380 (real data) are done. Real, null, recovery, and water cross-leak runs have refined diagnostics and `valid: true`. Their mechanism parameter recovery still needs collection before any identification verdict.
- Real compounding bundle 0390 is done, `valid: true`. Its diagnostic land-mean early-to-modern lambda change is **+.744856**, close to the recorded step-3 +.743. This alone does not resolve its effect on front timing or held-out prediction.
- Compounding guard bundle 0385 remains in the running queue. The null has accepted diagnostics; the recovery suite is incomplete.
- CBC founder bundle 0397 is running. Four arms bracket the founder scale on yearling and delayed-breeding bases.
- NestWatch bundles 0400, 0401, 0402, and 0405 are pending. They cover null/recovery, loose or wrong brood assumptions, reporting biases, and real-data fits. There is no completed nest-based identification result yet.
- Local step28c, the slow static-truth refit with density-dependent concentration disabled, finished successfully according to pueue and has refined diagnostic artifacts. Its scientific readout remains to be collected.

## Audit the observation-to-process mapping before extending it

Three issues in the BBL scoping design need resolution. These are mathematical/code-review concerns, not measured failures of an implemented BBL block.

1. **Avoid counting juvenile survival twice.** The live forward code explicitly applies `S_j` to stayers and adds surviving arrivals directly (`age_forward.py:1024–1027`). Q already supplies survival for departing juveniles. The proposed first interval `Sj^f × Q-bar` therefore cannot be imported literally as annual survival. Construct the marked-bird transition from the same stayer/mover branches, departure probability, origin edge correction, and destination Q used by the simulator. Partial-year exposure then needs an explicit timing/hazard assumption. Verify that the resulting transition reproduces the simulator's juvenile survival exactly at full exposure.
2. **Recovery frequency is not conditional reporting probability.** The reported .26% is dead recoveries divided by bandings over observed follow-up. In `P(recovered in k) = previous survival × (1-S_k) × r_k`, r is conditional on death. Centering r directly on .3% requires derivation using follow-up, cohort survival, exclusions, and truncation. Also marginalize unknown movement locations; substituting survival at a bird's expected position generally does not equal averaging its survival over positions.
3. **Movement and encounter opportunity remain confounded.** The proposed natal location likelihood identifies `K(origin,destination) × encounter opportunity(destination)`. Adult displacements also contain an unknown movement distribution, so adult records alone do not guarantee identification of reporting bias. Use sensitivity envelopes and synthetic joint recovery of both components. Include the stayer atom, spatial rounding, age/season selection, and multiple elapsed years; a later breeding encounter is not always one application of the natal kernel. If the same encounter contributes to survival and location blocks, use a joint event/location likelihood or its exact conditional factorization.

For NestWatch, the live block is tied to realized births per breeder, which is preferable to equating observed nests with Fmax. Its central 2.5 attempts per female and 32-day cycle are structural assumptions. The pre-registered loose/wrong-brood and failure-reporting arms are essential: extra nest records cannot identify annual fecundity if attempts, breeding participation, and sex ratio remain freely interchangeable. The yearling base already predicts roughly the observed per-attempt output; expect constraints on pattern and compensator trades, not necessarily a large mean shift.

The CBC founder band is useful but is a derived anchor, not a direct release census. Its conversion uses detection, sampled area, release-cell land fraction, and assumed early growth. Retain the low/high brackets. Do not later count those same CBC records as independent validation of the anchored model.

## Additional data and the next discriminating work

Prioritize observations that measure different parts of the life cycle over another broad abundance dataset.

1. **Finish the queued NestWatch/CBC guards, then add BBL survival with the mapping above corrected.** The existing local reconnaissance reports about one million HOFI bandings, 2,608 dead encounters, and sparse routine recapture coverage. The public [USGS release](https://www.usgs.gov/data/north-american-bird-banding-program-dataset-1960-2025-retrieved-2025-07-11) independently confirms the dataset and access. Begin with broad adult survival and reporting strata; the crude .53 is a diagnostic, not a ready-made precise prior. The decisive test remains planted J1 recovery with nests, founder constraints, and survival information together, allowing the habitat coefficients to refit.
2. **Use existing early CBC geography as an arrival-time observation, beyond a founder prior.** The records already identify an initial local lag and subsequent neighboring-cell detections. Model repeated surveyed nondetections and detections with uncertain detection and winter-to-breeding correspondence. This targets the poorly observed 1940–66 spread directly. Do not treat first detection as exact colonization or reuse the founder-anchor evidence independently.
3. **Scope FeederWatch as an external winter/disease check.** [Public checklist and site-description downloads](https://feederwatch.org/explore/raw-dataset-requests/) include observation-effort fields and repeated sites. The [House Finch Disease Survey history](https://feederwatch.org/house-finch-eye-disease/history-research/) confirms disease monitoring continued within FeederWatch after the standalone survey ended in 2008. First test winter redistribution and disease timing; an annual breeding-state model cannot directly explain winter feeder counts. Disease-field access/coverage needs separate verification. Count data require checklist zero-filling, effort modeling, and feeder-site selection treatment.
4. **Requestable station capture histories are a later option.** MAPS or original bander histories could distinguish recapture from survival better than the public sparse live-encounter stream. Access, HOFI sample sizes, and spatial coverage remain unverified here; no requests were sent.

Before committing more GPU time, collect the already completed directional and compounding parameters, guard outcomes, fronts, and paired scores. Then report the compensator changes in the same table. For each additional data block, test the planted mechanism both with that block alone and cumulatively with the others. Include deliberately wrong reporting, brood, and founder assumptions: apparent recovery under perfectly specified synthetic observations is insufficient.

Finally, profile the journey parameter while reoptimizing the compensators, and examine the weak parameter combinations on manageable subsets. Posterior sampling can quantify remaining uncertainty, but cannot manufacture information missing from the likelihood. Until recovery succeeds, interpret the dispersal mechanisms as defensible structural alternatives rather than estimated biological causes.
