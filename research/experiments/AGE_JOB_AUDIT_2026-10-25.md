# House Finch downstream job audit — 2026-10-25

Finding: the expected overnight jobs did run, but there were two workflow gaps: incomplete refinements were left in the done queue, and no build/fit tasks were installed for the final v1/v2 ensembles.

## What ran

The source deployed on TACC is research/age-model commit d9bd47a (2026-10-24 21:04 CDT). E081 consumes the earlier CBC-supervised `cube_desk_cbc1902` and matched `cube_desk_ctl1902`, both at 64 dimensions. Their input builds completed October 24 at 21:50 and 22:07 CDT (Slurm 3532629/3532630).

| Arm | DESK | Refinement result before this audit |
|---|---|---|
| 81a | cbc1902, free temporal readout | Failed: non-finite objective/gradient at iteration 169; last finite point retained; valid=false |
| 81b | cbc1902, tied temporal readout | Optimizer-reported converged, iteration 5461 |
| 81c | cbc1902, static readout | Hit iteration cap 8000; not_converged; valid=false |
| 81d | ctl1902, free temporal readout | Optimizer-reported converged, iteration 6972 |
| 81e | ctl1902, tied temporal readout | Hit iteration cap 8000; not_converged; valid=false |
| 81f | ctl1902, static readout | Optimizer-reported converged, iteration 6282 |
| 81g | cbc1902, correlated temporal readout | Optimizer-reported converged, iteration 3088 |
| 81h | ctl1902, correlated temporal readout | Optimizer-reported converged, iteration 6814 |

All eight have MAP-stage diagnostics. Seven proceeded through refinement-stage diagnostics, but that does not make all seven valid: 81c and 81e explicitly record valid=false. The wrapper treats a max-iteration stop as exit success and has no continuation scheduler; the worker consequently moved them to done. Non-finite/stalled refinements raise an error, so 81a correctly entered failed.

The 81a failure was numerical, not OOM or a Slurm timeout: peak device memory was about 12.5 GiB of 37.5 GiB available. Its retained last-finite gradient norm was ~3132. Do not compare its incomplete refined objective with converged arms as if both were optima.

E082 robustness fits 82a/82b/82c also completed MAP/refinement/diagnostics. Their optimizer reports label them converged by objective-plateau criteria, not gradient <1. In particular, final gradient norms can remain large (82b ~16960, 82c ~14542); the configured plateau gate uses the recent median gradient, not the final value. This audit does not certify scientific adequacy from a valid flag alone.

## Why workers were idle this morning

The last age-model tasks finished at approximately 06:33 (worker 3530553), 06:11 (3530554), and 07:28 (3530555) CDT. No further age tasks were pending, including in the retired queue. The workers stayed allocated and polled empty queues. DESK evaluation tasks subsequently used the existing allocations from 08:12 through 09:06.

## Missing final-candidate handoff

Only `numpyro_input_cbc1902_r64` and `numpyro_input_ctl1902_r64` exist for the new-DESK age-model comparison. I found no input builds, MAP/refinement queue entries, or watchers for `desk_final_s1..s4` or `desk_final_v2_s1..s4` in the deployed downstream code/queues.

The E081 note says the final ensemble will be built when full-window models land, but its stated ensemble is a mean *kernel* (stack seed features / sqrt(n), then top-64 eigenfeatures). The DESK candidate plan supplies an arithmetic mean *feature* cube, with v2 retaining explicit summer/winter halves. These are different kernels and the handoff needs to resolve that difference. The current downstream source in this checkout also requires explicit seasonal-block support for the v2 contract. This is missing scheduling/ingestion work, not evidence that final-ensemble MAP jobs crashed.

## Recovery performed

Resumed the already authorized 81c and 81e refinements on the existing workers, with original code snapshots and model settings and only the iteration ceiling raised from 8000 to 16000. No new cold MAP runs or Slurm allocations were submitted. The original refiner explicitly supports this continuation and excludes max_iter from its resume fingerprint.

Before dispatch, original optimizer states, refined outputs, reports and logs were backed up under:
`/work/07980/bwa386/ls6/houfin/research/age_model/continuation_audit_20261025/{81c,81e}_before_16000/`.

Tasks: queue2 `9940_continue_81c_16000.sh` and `9941_continue_81e_16000.sh`. Both were claimed. Logs: E081/{81c,81e}/refine_continue_16000.log. New diagnostics go to separate viz_ref_continue16000 directories. The continuation wrappers explicitly require valid=true before running diagnostics, so another iteration-cap stop cannot silently be reported as a successful refinement.

81a's terminal non-finite state and original failure are preserved for diagnosis. No automatic identical rerun was installed for it. No final-ensemble downstream experiment was invented or launched during this audit.

Verified at 09:24 CDT: both continuations passed their original-source checks and advanced beyond iteration 8,150, with finite objectives still decreasing. Task 9940 (81c) is running inside allocation 3530555; task 9941 (81e) inside allocation 3530554. The remaining allocation is available. Both pending queues were empty before these continuation tasks were installed.

Subsequent authorized recovery: 81c converged at iteration8568 and 81e at12783; both completed follow-up diagnostics. A separate 81a numerical restart now uses its retained finite point and fresh optimizer history, preserving its failed output. Final v1/v2 handoff builds and dependent matched MAP/refinement tasks have also been dispatched. See [FINAL_AGE_HANDOFF_2026-10-25.md](FINAL_AGE_HANDOFF_2026-10-25.md) for the source manifest, model arms and live status location.
