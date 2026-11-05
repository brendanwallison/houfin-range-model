# Final DESK House Finch handoff — 2026-10-25

The final-candidate downstream runs are now dispatched to the three existing small-queue workers. No new Slurm allocation was submitted. This is a diagnostic comparison; the current encoders still have the CBC completeness contamination documented in FINAL_HELDOUT_INTERPRETATION.md.

## Representation and matched comparisons

The handoff uses arithmetic mean **features** from the four coordinate-anchored seeds, matching the artifacts used in the completed heldout evaluations. This is distinct from the older E081 proposal to average kernels and project to rank64. Explicit shared-coordinate mode requires matching hashed basis and coordinate targets, and retains the seed prediction-disagreement diagnostics.

| Arm | Input | Readout width |
|---|---|---|
| fd_control64 | Earlier CBC1902 DESK | 64 |
| fd_v1_r64 | Final v1 mean features | 64 summer |
| fd_v2_r128 | Final v2 mean features | 64 summer + 64 winter |
| fd_v2_r64 | Same full v2 source, explicit per-block prefixes | 32 summer + 32 winter |

All four retain the existing 80h correlated spatial/temporal readout, population biology, dispersal, observation model and optimization objective. They use a cold quick90 MAP followed by float64 L-BFGS with a 16000-iteration ceiling and the existing convergence guards. Equal isotropic feature-coefficient variances induce the sum of seasonal kernels. No new seasonal amplitude parameter is fitted. The earlier E081g comparator inherited rank32; fd_control64 supplies the width-matched baseline.

## Frozen source and validation

Base commit: `d9bd47ac8730b295416f79b75da58e6d1f89c897`.

Frozen patched source: `/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025/code`.

Manifest SHA256: `f23a2cb5fbe8b95c8a3cf34634f526443d505b63335a704e39385eac3db07bbe`. All manifest entries passed checksum verification on TACC. The patch bundle in `research/age_handoff` documents the reproducible overlay; the other agent's checkout and production configs were not edited.

Patches add seasonal contract ingestion/validation and explicit per-block feature selection. They refuse a global width cut that would discard winter, and preserve source/selected block provenance. The downstream wrapper requires a valid refinement before producing final diagnostics; capped optimization can no longer silently become a successful fit.

Eight focused export/seasonal-selection tests and 18 existing kernel/split/tied-readout tests passed on the frozen source. One unchanged numerical-tolerance diagnostic is excluded: its landmark error is approximately1.23e-5 against a1e-5 threshold. The rebuilt mean cubes are checked against the heldout-evaluation feature cache at 5969 cells and five years; v2 passed with maximum absolute difference 0.0.

## Dispatch and recovery

Existing queue: `/work/07980/bwa386/ls6/houfin/research/age_model/queue2`.

- 9942: E081a recovery, using its retained last finite point and fresh optimizer history, with original model and source. Original failed result preserved; output tag `finite_restart1`. At 11:05 CDT it had advanced beyond 2770 finite iterations, U≈136683 versus the original retained U≈144095.
- 9943/9944: v1/v2 ensemble build, path integration and model-input ingest. Both running at 11:05 CDT.
- 9950: fd_control64 fit, pending.
- 9951: fd_v1_r64, automatically dispatched only after v1 build succeeds.
- 9952/9953: fd_v2_r128 and fd_v2_r64, automatically dispatched only after v2 build succeeds.

Fit outputs: `/work/07980/bwa386/ls6/houfin/research/age_model/FINAL_DESK_MAP/<arm>`; each retains resolved config, base commit, code manifest, MAP/refinement logs, diagnostics and western-sink audit. Build/fit completion markers live beside the frozen code.

A bounded read-only WSL collector (pueue750) records queue state, final refinement reports and recent fit logs every minute in `research/experiments/final_age_handoff_20261025/status.json`. `MONITOR_TERMINAL.json` records terminal task states or a failed build. The collector does not dispatch additional allocations or alter model results.

The earlier capped refinements are now optimizer-converged: 81c at iteration8568 (U≈139197.22), 81e at iteration12783 (U≈138430.70). Both completed their follow-up diagnostics. These validity flags certify the configured optimizer gates, not ecological adequacy.

The remaining scientific decision is comparative fit quality and behavior after these new runs finish. No final encoder is promoted by this dispatch.

Update at 12:23–12:35 CDT: both input builds completed. Control64, final-v1 and full final-v2 completed cold MAP and entered float64 refinement; the32+32 arm remains pending. E081a's isolated restart completed at4056 iterations with valid=true and follow-up diagnostics. Its initial objective matched the retained finite point exactly (144094.7171325632). Additional GPU work and the clean-completeness sensitivity are tracked in [FINAL_CLEAN_FOLLOWUP_2026-10-25.md](FINAL_CLEAN_FOLLOWUP_2026-10-25.md).

Correction from the live logs at12:42 CDT: full128 finished MAP but failed during float64 MAP-diagnostic data loading, before refinement. Control64 and final-v1 reached stalled optimizer states, with recent median gradients≈16100/13200 above the unchanged10000 plateau guard; neither is a valid refined comparison. The32+32 case is now running.

Recovery source `code_memoryfix1` preserves the original snapshot and changes one expression in data_loading: cast large Z arrays to float32 on host before transferring them, rather than briefly allocating float64 on GPU and then casting. The old path tried to allocate24.42GiB as a temporary array. This changes transfer residency/peak memory, not the inputs, shapes, final dtypes or density. Manifest SHA256 `e82d13b21a73bcf03060d47641b31c922d1fa19d92b628c48ba232770abae6a0`; all entries verified. Numerical recovery uses the same finite points and fresh optimizer history, with the original convergence guards retained. Full128 starts refinement from its completed MAP. Diagnostics follow refinement so visualization cannot prevent fitting.

Tasks9954–9956 initially encountered the expected source-hash drift guard before optimization. Tasks9957–9959 explicitly permit the recorded density-preserving host-cast change; this override is scoped to the recovery wrapper. No likelihood/prior or optimization threshold is changed. Recovery outputs use `finite_restart1` and the old failures are retained. Local monitor760 includes these jobs. Until these recoveries pass, the64/128 comparisons remain incomplete.
