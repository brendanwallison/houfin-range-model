# GPU follow-up priorities — 2026-10-25

At 12:23–12:33 CDT the three existing TACC workers remain busy with control64, final-v1 and full final-v2 float64 refinements. The final-v2 32+32 fit is pending. E081a's isolated numerical restart completed with valid=true. No new Slurm allocation is requested by this follow-up.

## Priority 1: matched completeness-only retraining

Two seeds (1 and2), each for final v1 and final v2. Preserve the original 500-epoch budget, selection rule, bases, covariates, optimizer, loss weights and seed. Change only the CBC completeness benchmark to training species and refit the associated observation operators. Reuse the existing ESK bases. Preserve the original Ruzicka supervision; preparation must pass an array-level equality/tolerance check against it before dispatch.

The original final-run spatial masks were fetched from TACC. The local generic `desk` directory held placeholder masks; an initial preparation attempt was stopped before training and rerun with the actual final masks. The corrected Ruzicka calibration reproduces a≈0.0167114, b≈0.8383575, weight≈2.81142. Both local/TACC basis projection hashes agree: breeding `412dee91653fafd82f2f718d9597da7313b2cdb3099fb353c9f386b1ba4b822a`; winter `7e5eeb4ebe6dc70a340d15f5e52bc05684e2fcac47812b7246631a9c016c2884`.

Local run root: `/home/breallis/houfin/work/houfin/research/FINAL_CLEAN_20261025`.
TACC run root: `/work/07980/bwa386/ls6/houfin/research/final_clean_20261025`.
Frozen encoder code manifest SHA256: `e3f0c3eaa3f51268105a15880bc1a8e76567b6094f39ae84dc8bc028e336c4e0`.
Preparation and dispatch scripts are preserved separately beside the frozen code. `aux_manifest.json` records input hashes when preparation passes.

- Local pueue754: corrected operator preparation on CPU.
- Local pueue755: dispatch after preparation and source verification; does not submit allocations.
- Local pueue757: v1 seed1 on RTX3090, after sparse repeats. If the unchanged recipe exceeds VRAM, preserve its log and dispatch task9960 to an existing TACC worker; do not shorten the time window or change the training recipe.
- Existing queue2 tasks9961/9962/9963: v1 seed2, v2 seed1, v2 seed2, queued after the new inputs pass validation. Pending House Finch fit9953 retains priority.

## Priority 2: sparse-data replication while inputs prepare

Local pueue756 runs thinning seeds3 and4 for v1(64), v2(32+32), and v2(64+64), at the original 3% row-thinning rate and 1966–1986 to2005–2025 comparison. Same noise-corrected truth, fixed rivals, calibration, paired block bootstrap and solver settings. Outputs are separate under the earlier run's `sparse_followup/`, not replacements for the original26 scores. This addresses the inconsistent within-cell gain in the first two thinning seeds. It is a sensitivity analysis on already opened holdouts.

## Evaluation scheduled after training

Local pueue758 gathers original and clean mean features using the same seeds1/2. It repeats the primary summer comparison for v1(64), v2(32+32), v2(64+64), both at full calibration data and 3% thinning seeds1/2, then the existing eight-era winter evaluation. It reuses the original scoring logic, taxonomy, completeness/observation handling, location exclusion and modern-placebo comparisons. Original and clean feature adapters/results are isolated under `evaluation/{original2,clean2}`. This avoids comparing a new two-seed ensemble against an original four-seed ensemble.

These are reused evaluation observations; fixing the training leakage does not turn them into a new untouched confirmation set. No encoder or downstream setting is automatically promoted.

## Deferred until these comparisons resolve

The all-species eBird auxiliary product and smooth regional / human-associated trend corrections remain unstarted. They require additional preparation and independent model changes. Filling GPUs with those before finishing the matched leakage correction would make attribution harder. Further clean seeds3/4 can follow if the first matched repeats support the recipe and time permits.

The current House Finch refinements still require assessment of gradient behavior, diagnostics and ecological fit, beyond their valid flag. Control64 has large gradients while its objective decreases; it is not yet a converged comparator.

Validation at12:35 CDT: all four resolved clean configs match their original seed's resolved recipe exactly after undoing only the declared auxiliary/output path replacements and ignoring comments. Epoch budget500, spatial_tiles1 and seed are unchanged. Fourteen focused export and observation-evaluation tests pass. Local sparse repeats were verified actively executing at98% GPU utilization (~12GB VRAM), while the clean winter operator prepared on CPU.

Additional identity checks: local/TACC covariate schema hashes, state files for1902/1966/2025, BBS target arrays, and both ESK projection matrices match. Winter coordinates reconstruct exactly. Ruzicka calibration has ~1.7e-7 CPU/GPU/linear-algebra rounding in its intercept; the finalizer validates the reconstruction and copies the exact original Ruzicka artifact and winter coordinate targets rather than training against rounded recomputations. It also checks unchanged list locations, years, presence matrices and species columns, finite operators/completeness, and writes the auxiliary/mask hashes. Local pueue759 performs this finalization after the already-running operator builder; source and original strict-tolerance logs remain retained. The local GPU model can still differ numerically from an original TACC run because the hardware differs; it uses the same recipe and seed, not a claim of bitwise-identical optimization.

Update at12:43–12:49 CDT: finalization759 and dispatch755 completed successfully. The first builder's overly strict floating-point reconstruction check stopped as recorded; finalization validated the reconstruction and preserved the exact original pair/coordinate artifacts. Clean v1 seed2 (9961, allocation3530554) and clean v2 seed1 (9962, allocation3530553) have reached training epochs3/2. Clean v2 seed2 remains pending. The long allocation3530555 is running the32+32 House Finch fit.

Newly detected downstream failures: control64 and final-v1 stopped at the unchanged large-gradient stall guard; full128 finished MAP but OOMed during float64 diagnostic input transfer, before refinement. Isolated source `code_memoryfix1` casts Z on host before GPU transfer. The dedicated recovery wrapper permits this documented density-preserving source-hash difference, retaining all model/optimizer guards and using finite retained points. Recovery jobs9957–9959 are queued ahead of remaining clean jobs; details are in FINAL_AGE_HANDOFF_2026-10-25.md. Initial source-drift rejections9954–9956 and their logs are retained. Numerical failures are not counted as converged results.

Read-only age status collection now runs as pueue760. Pueue761 checks our checkpointed follow-up tasks once a minute; only after an owning Slurm allocation is absent for120seconds does it preserve the old ownership record and requeue that task to surviving existing workers. It never retries an optimizer failure, interrupts an active allocation, or submits a new Slurm job. This protects useful partial training if a short allocation expires before500epochs.

## Update at15:08 CDT

Clean v1 seed2 and v2 seed1 completed500epochs and cube export. V1 seed1 failed locally during backward with a CUDA driver unknown error, rather than an explicit OOM; task9960 now trains the unchanged recipe on TACC. V2 seed2 remains in training as9963. Latest verified progress: v1 seed1 epoch267, v2 seed2 epoch312. Existing-allocation expiration recovery now runs as pueue765 and includes the new fits/audits; previous watch761 was replaced so it does not stop when only proxy evaluation finishes.

All six sparse follow-up scores completed; see FINAL_SPARSE_REPLICATION_2026-10-25.md. All nine original two-seed summer comparisons are now complete, with winter completion still being checked. Local controller763 moves the clean-model dependency wait off the GPU queue if needed. During that wait it runs a fixed exploratory10% calibration comparison (seeds1/2, three representations), then queues clean-only evaluation when the four clean cubes exist. This intermediate fraction probes the data-size dependence already seen at3% versus full calibration; it is not a new heldout confirmation.

The stalled age fits are now being investigated through direct density/gradient audits. One tested, density-preserving optimizer-coordinate pilot lowered control64's potential by~420nats over2000iterations and reached its cap without convergence. Its guarded checkpoint continuation is queued. Details: FINAL_NUMERICAL_AUDIT_2026-10-25.md; current heldout/ecological diagnostics: FINAL_DOWNSTREAM_INTERIM_2026-10-25.md.

To use surviving GPU time after clean training, pueue764 waits for the two clean v2 cubes, then dispatches9973 to build their full128 mean-feature age inputs; successful ingestion dispatches9974 for cold MAP→float64 L-BFGS→diagnostics. This is a **diagnostic clean two-seed candidate**, not a controlled clean-effect comparison to the earlier four-seed age fits. The separate proxy evaluation does compare matched old/clean two-seed ensembles. New source `code_clean2` retains the original optimizer and the host-cast memory fix; it does not enable the conditioning pilot. Manifest SHA256 `5b96c92002e19bdbfda0163f83037cee4bcd99cf0f719653cf3aaf319d12ade6`, all entries verified. Model configs and outputs are isolated under fd_clean_v2_m2 names. No production encoder is promoted.

Source-checkpoint continuation9975 is conditional on a capped, improving conditioning run; it does not retry a failed convergence guard. Current age/audit status monitor is pueue762. New per-step JSON audit artifacts are also retained in the workspace under final_age_handoff_20261025/gradient_audit/.

## Update at15:33 CDT

The original two-seed summer comparison completed all9scores. Its winter phase initially rejected overlapping circle identities because the copied analysis snapshot predated the already-tested relocated-circle exclusion. The active analysis copy now uses the current scorer (SHA256 `cdacde323763d4e7b53ea27b5d7971310b0be0617c8a49f5bd3dabf8f616ad13`). The summer scorer hash exactly matches the current corrected scorer, so its completed results did not need rerunning. Winter-only task767 completed successfully at15:18: five readouts,44529calibration lists and6243test lists. Original task758's failure and the pre-fix winter source are retained.

Calibration-size exploration768 is running on the local GPU while clean models finish; clean-only scoring is now dependency-dispatched by769. The superseded wait-controller763 was stopped. Collector770 will write FINAL_CLEAN_RESULTS.json/.md in the workspace when matched old/clean evaluations complete. Its strict support checks and paired bootstrap summaries passed an identity comparison: comparing the same model to itself returns exactly zero differences and intervals for every supported winter subset and all summer regimes. The summary records source/feature-manifest hashes and refuses mismatched supports.

Allocation3530553 has expired, interrupting clean v2 seed2 after logged epoch399. Its readable checkpoint retains epoch390 and best_epoch124; the last9epochs will be replayed on recovery. The actual surviving batch script was inspected and confirms native orphan recovery. The external watch now **observes only** and leaves all queue movement to that native mechanism; this avoids two coordinators racing. Existing snapshots/outputs remain unchanged. The observer continues until both proxy evaluation and downstream follow-ups are terminal, rather than stopping after proxy evaluation alone.

The conditioning pilot's cap continuation9975 is active beyond2350iterations with further improvement, still subject to the original raw-gradient guards. The two remaining short/long allocations will determine how much finishes today; unfinished checkpointed stages can resume on surviving workers. No new allocation has been submitted.

## Latest verified state after both short allocations ended

Three clean encoders and their cubes are complete. V1 seed1 selected epoch142 and finished its export. The surviving allocation3530555 natively reclaimed v2 seed2, restored its checkpoint and advanced to epoch404, beyond its earlier interrupted log. The last short allocation ended with the conditioned control at4320logged iterations (about753nats below its original stalled potential); continuation9975 is pending on the surviving worker. No threshold has been relaxed and no converged control is claimed.

The remaining work is chained: finish clean v2 seed2; build/run its diagnostic mean2 House Finch case; resume conditioned control; evaluate matched clean/original ensembles on the local GPU. The local10% calibration exploration runs while those inputs prepare. Collector771 replaces770 and uses a frozen comparison source tree (`code_compare`, manifest SHA256 `aefa98f0b0bc67bf7beb859dcc87ad0a8734653c783effe8a63853cd2796353a`). Its identity test passed in that frozen environment. Final reports retain both helper and feature-manifest hashes, software versions and strict pairing checks; source checksums are verified before aggregation.

## Verified update at16:42 CDT

All four clean encoders completed500epochs and cube export. The matched old/clean two-seed evaluation is active locally as pueue773. All three full-calibration summer cases are complete; sparse and winter scoring remain pending, so these are interim results on reused heldout observations. Clean minus original differences (paired95% bootstrap intervals):

| Representation | Within | Pooled |
|---|---:|---:|
| v1 summer64 | -0.00128 [-0.00620,0.00283] | -0.00281 [-0.00802,0.00257] |
| v2 summer32+winter32 | -0.00410 [-0.00708,-0.00114] | +0.00323 [-0.00028,0.00628] |
| v2 summer64+winter64 | +0.00084 [-0.00245,0.00402] | +0.00039 [-0.00398,0.00432] |

This main summer comparison does not show a large full-width v2 sensitivity to completeness membership. It does not establish winter robustness, independence of the already-reused holdout, or generalization across encoder seeds. Collector771 will replace partial inspection with the full paired report after all regimes finish.

The10% calibration exploration completed as768, with results in FINAL_CALIBRATION_CURVE.md. Ensemble-size sensitivity is in FINAL_ENSEMBLE_SENSITIVITY_2026-10-25.md: early winter can shift materially between the fixed two- and four-seed ensembles, reinforcing the need for matched ensemble comparisons.

TACC input build9973 completed successfully. Clean mean2 full128 House Finch MAP/refinement9974 is running on allocation3530555, with about two hours remaining at16:41. Conditioned control continuation9975 remains pending. Both short allocations expired normally; the native worker recovered interrupted clean training, which finished. No new Slurm allocation was submitted.

Local fallback preparation: pinned jax-cuda12-plugin/pjrt0.8.1 installed only under FINAL_LOCAL_JAX_20261025/runtime, leaving the base environment unchanged. JAX0.8.1 detected the RTX3090 and passed a tiny dot-product check with preallocation disabled. Task774 stages and hashes immutable control64 input files. This is preparation only: no local House Finch continuation has been launched, and VRAM fit, checkpoint compatibility and objective agreement must be verified first.

## Update at17:34 CDT

Clean evaluation773 and paired collector771 completed successfully at17:27. FINAL_CLEAN_RESULTS.md/.json now contain all9summer cases and8winter eras. Full-width v2 summer differences clean-minus-original remain near zero across full and sparse regimes. Its winter temporal-gain difference is -0.021[-0.037,-0.006] in1902�1919, -0.009[-0.020,+0.001] in1920�1939, and +0.008[+0.005,+0.010] in1966�1985. This is mixed sensitivity, not a universal benefit from cleaning; same-circle and low-counter subsets remain in JSON. No fresh confirmation set was used.

Local audit775 completed successfully. Exact retained control potential agrees with TACC to displayed precision (absolute difference0); relative gradient-norm difference is2.0e-12. Control inputs774 passed all checksums. Local continuation776 atomically claimed pending remote9975 into queue2/local_handoff before copying its immutable checkpoint; it cannot be claimed simultaneously by the TACC worker. The copied checkpoint files also passed SHA256 verification. The same source, versions, raw-coordinate optimizer state, scale specification and convergence guards are retained; only the backend and memory fraction change. Local refinement is active; no convergence is claimed. TACC clean mean2 MAP completed400steps with final loss144619.875 and proceeds to float64 refinement. Existing allocation3530555 has about69minutes left at17:32.
