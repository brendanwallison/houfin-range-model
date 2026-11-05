# Executable overnight plan, 2026-10-31

Continue the current demographic-data/dispersal work. The seasonal branch remains a reference plan.

**Amended scope:** [PARSIMONIOUS_CANDIDATES_20261031.md](PARSIMONIOUS_CANDIDATES_20261031.md) now registers five bounded configuration fits and whole-pathology readouts, selected from the prior experiments. This replaces the initial collection-and-audit-only scope. The objective is a parsimonious configuration with fewer joint biological distortions; parameter recovery is supporting evidence, not the sole target.

**Archive/reanalysis extension:** [RETROSPECTIVE_AND_PRECISION_PLAN_20261031.md](RETROSPECTIVE_AND_PRECISION_PLAN_20261031.md) adds CPU review of the retained fit archive, eight bounded biological readouts at old fitted points, and four completion-triggered float32/float64 stress tests. These add no fitted models or new parameters. Cached TACC and local comparison tables are already produced; synthetic-observation runs are kept separate from the real-data retrospective.

## Already running and next automatic actions

1. Existing TACC jobs finish the real-data NestWatch fits (pinned nest attempts + distance cost; uncertain nest attempts + distance cost; nest observations without the added distance cost).
2. Accepted refined checkpoints and complete diagnostics trigger local CPU held-out scoring against the same yearling/native baseline. All three arms are now registered in the dispatcher. Preserve numerical validity, identical held-out design, and prediction parity checks. A failed fit produces a recorded failure, not a relaxed gate or automatic promotion.
3. Every polling cycle updates `OVERNIGHT_RESULTS_20261031.md` and machine-readable local-monitor artifacts, showing real-fit distance cost, fecundity intercept, juvenile-survival offset, habitat summaries and paired prediction scores. Retain the positive simulated-data improvement alongside its limitations; no binary gate will erase that result.

## Fixed-point synthetic objective audit — registered before running

Question: is the generating truth a better solution than the fitted synthetic point under the identical objective, or does the fitted point improve the objective while moving away from the generating demographics?

Use the **existing frozen code and overlays**, synthetic observations, and nuisance-truth JSON for `step31_nestnull` and `step31_nestrec`. Recover the generating population parameters from the recorded step29_yb_native refined checkpoint, override the recorded hazard, and add the exact recorded nest nuisance parameters. No randomly invented missing latent sites. Missing sites abort the audit.

Evaluate, without optimization:

- the constrained log joint and its per-site contributions at generating and fitted points;
- the same transformed potential used by refinement at both points;
- agreement of the fitted-point potential with the saved refinement report.

The null hazard is exactly zero in the constrained calculation. Its unconstrained-potential calculation requires an explicitly reported interior approximation h=1e-8. These two objective conventions must not be conflated. A finite sample and the priors need not put the optimum at the generating parameters.

Interpretation rules:

- If the generating point has a substantially lower **matched refinement potential**, inspect optimization/initialization before claiming a data-identification problem.
- If the fitted point improves that potential while trading demographic parameters, this documents the tradeoff; it does not prove that every mechanism is globally unidentifiable.
- Separate BBS, nest and prior contributions. An objective improvement dominated by priors is different from a likelihood preference.
- If fitted-potential agreement fails, investigate the objective/serialization mapping before interpretation. Preserve all outputs.

Execution: a single serial TACC queue task on an existing A100 worker, two arm audits, **30-minute hard limit per arm**, no new fits or Slurm allocations. Float64 stays on TACC. Record timeout/failure receipts in the overnight directory. This is a bounded diagnostic (~1 GPU-hour worst case), not an unattended search.

## What tomorrow's report should answer

- What did actual nest observations change in reproduction, survival, distance cost and inferred habitat history?
- Do those changes retain predictive compatibility with held-out bird counts?
- How much depends on treating attempts per female per season as known?
- Does the synthetic discrepancy look like an optimizer issue, posterior/prior tradeoff, or unresolved information?

These are registered executable tasks, extended by the five configuration comparisons above. Scientific adoption and experiments outside those registered branches still require a subsequent reasoning session. Seasonal models and production promotions are outside this overnight plan.

## Second-wave execution update — October 31, 22:27 CDT

The bounded controller is running persistently on TACC, with the local scorer registered for all 11 possible derivatives. See [the executable decision tree and limits](SECOND_WAVE_CONTROLLER_20261031.md). It currently waits for accepted first-wave fits with complete pathology/front readouts. Both local float32 candidate MAP stages finished successfully; TACC retains float64 refinement. Two A100 allocations are running and a third remains pending.

The second wave fixes the CBC+NestWatch configuration as its matched sensitivity reference. It tests founder and nest-attempt assumptions, independently repeats absolute Allee, and adds interactions or basin checks only when registered thresholds warrant them. It is capped at 12 additional GPU-hours and stops publishing at 10:00 Chicago time. This extends useful work beyond the first wave; it does not guarantee continuously occupied hardware or automate scientific adoption. Eleven watcher/controller checks passed before enablement.

## Expanded overnight queue — supersedes the initial 12 GPU-hour/11-arm bound

At the user's request, the controller budget is now **36 additional TACC GPU-hours** with the existing **10 a.m. CST staging cutoff**. The registry contains **31 nonconditional fits and six conditional follow-ups**. See [the current registered comparisons](SECOND_WAVE_CONTROLLER_20261031.md) for exact settings and gates.

The expansion adds distance-hazard and segmented-compounding experiments under both Allee laws, matched native-departure interactions, absolute-law founder/attempt sensitivities, and paired DESK-v2 fits. Exact v1/v2 survey/grid/dynamics identity passed preflight on TACC. Native segment features are being prepared locally in float32 and require verified transfer before the compounding arms become eligible. All float64 fitting remains on TACC. New fits publish separately to the common worker queue as input and budget gates allow; the entire registry need not run if the budget or deadline intervenes. Thirteen controller/watcher checks passed before enabling the expanded controller.

The segment-equivalence audit subsequently found localized errors hidden by the historical median-only check; all eight compounding arms are held before any fit attempt. Eight paired v2 × movement comparisons replace that work. Current supply is 31 input-ready planned fits plus six conditional follow-ups; the separate eight held arms bring the registry to 45. See the current controller registration for the audit and exact gates. This supersedes the earlier 37-arm registry count.
