# Dispersal collection, 2026-10-31 late afternoon

Authorized follow-up to PROJECT_SURVEY_2026-10-31.md. Sources are the frozen TACC run checkpoints, accepted refine reports, and diagnostic JSONs, collected in `DISPERSAL_COLLECTION_20261031.json`. NumPy host reconstruction reads checkpoint arrays without starting JAX on the login node. No existing fit, task, or configuration was changed. The following applies the existing gates; it is not a new adoption decision or a full independent skeptic gate.

## Directional choice: real-data interpretation blocked by guards

| Arm | Planted steering kappa | Fitted kappa | Pre-registered gate | Result |
|---|---:|---:|---|---|
| Null | 0 | .213704 | <= .06 | FAIL |
| Recovery | .5 | .737244 | [.25,.75], plus compensator checks | Inside parameter range only |
| Water cross-leak | 0; water hazard 1 | .140910 | <= .06 | FAIL |

All three refinements have `valid: true`. Thus their failure is an identification/specificity result at the accepted optimization points, not a rejected numerical fit. Recovery alone is insufficient: the mechanism also estimates steering in its absence and responds to another mechanism's signal. The pre-registration says real kappa is read only if all four guards pass. It does not pass that rule, so its estimate must not be presented as biological evidence for steering or promoted by its score.

For the recovery versus null fits, alpha_f is 9.014 vs 9.087, alpha_a .237 vs .311, alpha_j 2.218 vs 1.904, and log founders -1.677 vs -1.936. These are raw fitted-site values, not automatically physical rates; their transforms and habitat contributions matter.

## Compounding: partial recovery, full verdict pending

Null fitted compounding lambda=.006523 (gate <=.05: passes). Recovery of planted .25 is .140130 (parameter gate [.125,.375]: inside, near the lower edge). Both have accepted refinements. Recovery versus null: alpha_f 8.748 vs 9.396, alpha_a .262 vs .313, alpha_j 2.492 vs 2.036, log founders -1.738 vs -2.062. The compensators still move materially, especially the juvenile-survival offset.

The real fit has lambda=.004611 and diagnostic land-mean lambda increase +.744856. Its numerical validity is established, but its biological interpretation awaits the fecundity cross-leak, compensator gate, and current-model comparison. Those reports were absent at collection. A near-zero real fit cannot yet establish that compounding is absent or unnecessary. Realism-based structural adoption is a separate decision already contemplated by the project.

## Pending

CBC founder bundle 0397 is running. NestWatch 0400/0401/0402/0405 remain queued. These are already scheduled; this follow-up did not submit duplicates. The independent BBL adult pilot is recorded in `BBL_ADULT_PILOT_2026-10-31.md` and is not a production constraint yet.

## Update checked 17:55 CDT

New snapshot: `DISPERSAL_UPDATE_20261031_1755.json`. Three founder arms now have accepted refinements: central CBC pin on the yearling base, central pin on the delayed-breeding native base, and the low-founder yearling bracket. The high-founder bracket is still in MAP (383/400 at the check). NestWatch's four bundles remain pending. Compounding's fecundity cross-leak is entering refinement; the current-model comparison has not produced a report.

The central founder pin changes yearling demography more than the pre-registration predicted: alpha_f 7.602 -> 6.754 (about -11% on this raw site, approximately fecundity at these positive values), alpha_a -.405 -> -.662, alpha_j 1.053 -> 3.185. Its diagnostic early-to-modern lambda increase drops 1.345 -> 1.177, while suitable-land fraction changes only slightly (.101 -> .443 unanchored; .103 -> .446 anchored). The low-founder bracket gives a similar 1.152 increase and alpha_f 6.757. Thus nearly unchanged suitable area conceals substantial demographic compensation. Whether these runs represent a different optimum or a stable founder effect still needs the high bracket, held-out rescore, and matched-seed checks.

On the delayed-breeding native base, lambda change moves only .853 -> .863, not the large increase anticipated in the prediction. This comparison uses the native baseline from the same collection; the older +.743 refers to the averaged-Z reference and is not the matched comparator. No founder setting is adopted from these point diagnostics alone.
