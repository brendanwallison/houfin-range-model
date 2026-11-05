# Parsimonious configuration search — overnight amendment

Registered 2026-10-31 after the real-data NestWatch fits and before these new combination fits. This supersedes the earlier overnight plan's scope of collection/objective audits only. Goal: fewer joint pathologies at reasonable predictive cost, not identification of a distance-hazard parameter as an end in itself.

## Evidence selecting these experiments

- Real NestWatch with and without an added distance hazard gives very similar held-out prediction; the extra fitted hazard is ~.001. Use **nest-only** as the simpler starting point, preserving the real reproductive observations.
- Fecundity and survival move together; a score alone cannot tell whether the overall demography improved. The nest-only fit's reference adult-survival logit falls to about -.77 while its juvenile ratio offset rises above 3. Read actual spatial/population-weighted rates before interpreting those intercepts.
- CBC founder anchors retain yearling-model prediction across a wide bracket, but demography changes. Combining that independent anchor with nest observations removes a compensating free introduction parameter at the central pin; it is a sensitivity/configuration choice, not a claim that the release size is known exactly.
- Native departure (E077 / ladder step25) is a one-parameter biological intervention that previously helped Plains capacity collapse and has a western spatial signature. Test it on the current native habitat/yearling/nest base rather than extrapolating from its older base.
- E078 showed that apparently promising mechanisms can undo one another's benefits. Hence matched nested combinations and seeds, with the entire pathology scorecard, rather than stacking every plausible mechanism.
- Absolute density is the standing mate-finding interpretation (C1). Relative Allee remains a diagnostic baseline. Revisit the original absolute law on the improved demographic-data base, while explicitly monitoring the old low-K/Allee-dead failure.

## Candidate matrix (five new fits maximum)

All use the frozen **step31_nestonly source**, native v1 54 km habitat, yearling breeding, shared habitat field, unchanged observer model, real NestWatch, and no additional journey-distance hazard. No free journey-survival intercept/slope, band reweighting, warp, process-noise field or new habitat trend is introduced.

| Run | Change from nest-only | Seed | Main question |
|---|---|---:|---|
| step33_nestcbc_s41 | CBC central founder pin | 41 | Does combining reproductive and introduction information reduce demographic compensation? |
| step33_nestcbc_s42 | same | 42 | Is the combined configuration stable to initialization? |
| step33_nestcbc_natdep_s41 | CBC pin + existing native departure | 41 | Does one spatially specific departure parameter improve K/sinks without worsening demography? |
| step33_nestcbc_natdep_s42 | same | 42 | Is that tradeoff reproducible? |
| step33_nestcbc_abs_s41 | CBC pin + original 54 km absolute Allee law | 41 | Can the preferred mate-finding structure now avoid low-K and Allee-dead behavior? |

The CBC pin is log_inv_total=-.283 (~77.5 birds under the existing conversion). Relative cases retain their original pinned n50. The absolute case copies the original step0 54 km Allee block and pinned n50_raw=-2.7969046060186082 (~.0592 BBS route-count units), explicitly restoring its units, rather than reinterpreting a relative-to-K threshold as an absolute one. It is a sensitivity to the documented absolute configuration; a bad result does not reject every absolute Allee model.

These are complete configurations to compare. They are not automatically promoted by a score threshold or synthetic hazard-recovery gate. Realism-based changes may be useful with neutral predictive scores.

## Readouts and decision branches

First generate the same float64 biological scorecard for the existing yearling base, nest-only, nest+distance cost, free-attempts nest fit and CBC-pinned yearling base, then for every candidate. Report:

- actual Sa, stayer Sj, Sj/Sa saturation, Fmax and lambda0 by region/era, area- and population-weighted;
- realized fecundity and fledglings-per-attempt at the actual nest locations;
- early-East suitability and historical change, western sink/subsidy pattern, low-K fields, Allee-dead extent and existing barrier/front diagnostics;
- founder numbers and early release history where available;
- paired held-out scores, uncertainty, numerical validity, field complexity and initialization stability.

Conditional reading: if the extra native-departure parameter buys no material biological improvement, favor the simpler CBC+nest configuration. If absolute Allee revives low-K/threshold behavior, report that failure explicitly and retain its biology as an unresolved structural issue. If nest information moves implausibility into survival, advance BBL survival integration as the next data task rather than adding a compensating biological field. Do not automatically tighten survival priors around the coarse BBL pilot.

No configuration is selected solely for reducing lambda change, eliminating every sink, improving raw likelihood or recovering one synthetic parameter. The report must identify improvements, regressions and unresolved concerns separately. A provisional favorable configuration still requires the appropriate maps and supporting sensitivity checks.

## Execution and bounds

Frozen snapshots and new overlays only; no live model source edits or production changes. Existing A100 workers perform float64 refinement and readouts. The local 3090 may precompute the two seed-41 float32 MAP stages, with fingerprint/hash-verified transfer and exclusive queue reservation. Seed-42 and absolute-law fits run on TACC. Failed local handoff returns the original bundle to TACC. Existing synthetic objective audits finish first/in parallel on available workers.

Five new fits, no automatic additional model family or allocation requests. The staged TACC runners have a hard bound of two hours per configuration; failures/timeouts remain visible and are not automatically retried with weaker gates. This is a bounded overnight configuration comparison; novelty beyond these branches awaits a reasoning session.

## Synthetic-test contract check (registered before evaluating the means)

The objective audit found very large generating-versus-fitted differences, especially on the planted-hazard data. Source inspection shows that synthetic counts are substituted **before** occupied-habitat centering is computed. Original generating parameters were centered using original positive observations; the refit may use a different center. This is a candidate test artifact, not yet an explanation.

Quantify both centers on CPU, then compare saved generating expected counts (`mu` in the synthetic NPZ) with predictions under the recorded truth using (a) the refit's center, (b) the original center. Do not refit or edit old checkpoints. Mean matching tolerance: rtol 1e-5, atol 1e-6; report continuous errors as well. Two fixed-point GPU checks, at most 20 minutes each, on an existing worker. If only the original center reproduces the saved means, qualify previous recovery interpretations and plan a corrected synthetic test rather than interpreting the discrepancy as biological non-identification. This does not change the real-data candidate fits.
