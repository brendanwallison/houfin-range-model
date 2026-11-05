# Final downstream numerical audit — 2026-10-25

Direct audits reuse the existing unconstrained age-model potential, observations, priors and data. No heldout score is optimized. They evaluate retained failed/accepted points without modifying them.

## Findings so far

| Point | Recomputed potential | Raw gradient norm | Coordinate roundtrip max error |
|---|---:|---:|---:|
| Control64, finite restart |141371.47642474|59406|initial audit used checkpoint parameters|
| Full v2, finite restart |141651.64023293|21318|2.8e-16|
| Final v1, accepted restart |136319.31532677|65379|1.3e-15|

The latter two audits use the exact saved optimizer vector and reproduce the saved potential. Thus reconstructing unconstrained coordinates from saved constrained parameters does not explain the large gradient norms.

For control64, the unit-gradient directional derivative is59406 by autodiff and59516 at central-difference step1e-7 (~.18% discrepancy); the largest manifold coordinate agrees to~.025%. Larger steps cross very high curvature and give dramatically different slopes. Dominant gradient blocks are adult-survival intercept alpha_a (~53314 norm), manifold_factor (~20564), juvenile-survival intercept alpha_j (~9871), and manifold_idio (~8109).

Full-v2 and final-v1 remain much more step-sensitive through1e-7. Their unit-gradient central differences at that step are~717209 and~2891802 versus autodiff21318 and65379. Do not interpret those as proven bad gradients or proven smooth curvature: smaller steps1e-8/1e-9/1e-10 are scheduled to distinguish the cases. Repeated potential evaluations and per-coordinate checks are retained in the JSON artifacts.

## Isolated conditioning pilot

The control64 exact-state audit additionally estimates diagonal curvature with16 fixed-seed Rademacher Hessian-vector probes (seed20261025). A scalar per site is the mean estimated diagonal curvature. Coordinate scales are1/sqrt(clip(abs(block_mean),1,1e12)). This is an optimizer preconditioner, not a density/prior change.

Source snapshot `code_precondition1` adds optional coordinate scaling to L-BFGS. Optimizer steps use u=(x-origin)/scale, while the potential remains the original potential at x. Returned/saved vectors remain x; raw gradient norms are recovered as grad_u/scale for every original convergence/non-finite gate. No Jacobian constant is added to the reported potential. The default unscaled code path is preserved.

The isolated snapshot's manifest SHA256 is `acf160701a61da88e92b0c4f26e1150f5dd7be72dcd0d0912766ce9e7dee698d`; all entries verified on TACC. Relative to the original handoff, its density-related change is the earlier host-side float32 cast that eliminates a temporary float64 GPU allocation. The additional refiner change concerns optimizer coordinates only. Original snapshots and failed outputs remain retained.

The CPU validation solves a quadratic with1e10 condition ratio and verifies the original-coordinate optimum, checkpoint coordinates and raw-gradient gate. A transformed gradient smaller than the stopping tolerance cannot cause false convergence when its raw gradient is large.

Task9970 dispatches only after the curvature audit9968 succeeds and atomically publishes finite positive scales. It starts at the retained control64 finite point, uses a separate blockscale1 output tag and has a2000-iteration pilot budget. All original tolerances remain unchanged. An iteration cap or failed guard remains invalid; this pilot is not an automatic model promotion.

Fine-step audits9971/9972 cover full-v2 and final-v1. Existing-allocation expiration recovery includes these tasks and the pilot; no new Slurm allocation is submitted.

Sources/results: `/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025/{diagnostics,gradient_audit,code_precondition1}`. Existing observation-aware predictive/ecological comparisons are recorded separately in FINAL_DOWNSTREAM_INTERIM_2026-10-25.md.

Update at15:08 CDT: the exact control-state audit reproduces U and all three repeated evaluations exactly; its constrained-coordinate roundtrip error is2.8e-16. The largest estimated block curvature is alpha_a≈6.4e11, producing scale≈1.25e-6. Signed stochastic block estimates are only preconditioner heuristics, not a certification that the Hessian is positive/negative definite.

The pilot reached2000iterations with U141371.4764→140951.4624 (~420nats improvement), versus only~.005nats from the earlier unscaled fresh-history restart. It remains invalid/not_converged at the pilot cap, with raw gradient norm~43600. Source-checkpoint continuation to8000iterations is queued as9975 after the fine-step audits. Dispatch requires an iteration-cap stop and >1nat progress; stalled/non-finite stops are not retried. Original tolerances remain unchanged. Initial logs stay in blockscale1/refine.log; continuation logs use refine_continue8000.log. The CPU validation also verifies continuation from a scaled optimizer checkpoint.

Fine-step audits completed: at1e-9, the full-v2 unit-gradient central difference agrees within~.75%, and final-v1 within~.037%. At1e-8 discrepancies are~25%/4.8%; at1e-10 they are~3.5%/.16%, where subtraction/roundoff can matter. These pointwise tests support severe conditioning rather than an obvious autodiff error; they are not a proof for all gradient components. Continuation9975 has passed2350iterations with U≈140924.71 (about447nats below the original stalled point), still without convergence.

The actual submitted worker script for surviving allocation3530555 was inspected through `scontrol write batch_script`. It already owns orphan recovery. The external watch is now read-only with respect to queue files, avoiding a second coordinator racing the native worker; no earlier custom queue move occurred before this change. Native recovery resumes checkpointed work when the surviving worker reaches its next task boundary.

Validation: the patched refiner passes all5 existing refiner convergence/exact-resume/finite-point tests (20 unrelated tests deselected), run in the local complete age-source environment. The TACC login-node CPU attempt aborted while creating the JAX CPU backend, before test collection; it is not counted as a test result. The targeted local rerun passed. The conditioned run reached4320logged iterations and U≈140618 before the second short allocation expired; its checkpoint remains queued for native resumption.
