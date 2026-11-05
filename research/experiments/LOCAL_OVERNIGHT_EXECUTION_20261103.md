# Unsupervised local GPU and TACC handoff, November 3

The RTX 3090 lane is running under persistent pueue supervision, independently of the conversation. It takes float32 cold-MAP stages from the existing validated step41/42/43 experiment queue; TACC retains float64 refinement, diagnostics and biological replay. This is operational infrastructure for the already registered comparisons. It does not claim that the newly proposed private-field or cell-year-noise experiments have been implemented.

## Verified live execution

- Local worker task **1166**, label `range-autonomy/local3090-overnight`, started at 01:11 CST.
- Its first reservation is `step41_rational_shared_s41`: the departure-curve-only control, with the registered seed 41 and unchanged frozen scientific configuration.
- CUDA preflight passed. The fit advanced beyond checkpoint 100/400, with finite recorded losses and approximately three seconds per optimizer step after compilation. About 18.8 GB of total GPU memory was allocated, including pre-existing usage; allocation is not peak scientific-array memory or sustained GPU utilization.
- The actual checkpoint-100 library versions, model source identities and input identities were compared against TACC: **no mismatches**. This is a provenance check of an in-progress checkpoint, not acceptance of an incomplete fit.
- Full checkpoint handoff will occur only after all 400 steps and finite-parameter checks. No completed real-job handoff is claimed at this snapshot.
- TACC development allocation **3548096** started by 01:16 CST on c301-003. Its GPU lanes 0/1/2 independently claimed `step41_rational_shared_s42`, `step41_veit_adult20_s41` and `step41_veit_adult20_s42`. The existing A100-small allocation **3544518** also remains active. Thus five GPU lanes are now productive: one local and four on TACC. Two regular allocations remain pending.

## Scheduling and automatic actions

The local lane has a **10:00 a.m. CST November 3 deadline**, with a start buffer to allow finishing and transfer. Serial morning-continuation task **1169** loads the finalized 40-minute start buffer after the original process exits, without interrupting the active fit. It shares the same GPU lock, state, registration exclusions and deadline. There is never a second concurrent local fit from these tasks.

1. Choose an eligible pending step41/42/43 job whose frozen manifest matches its validation receipt and which has no existing TACC MAP attempt or completed MAP marker.
2. Atomically move its original task script from `pending` to `local_running`; this arbitrates with all TACC GPU workers. Persist ownership token, registered seed and heartbeat.
3. Mirror and verify the frozen source. Run only float32 quick90 MAP on the local GPU, with capped CPU threads, a bounded GPU memory fraction and a timeout. A compatible local checkpoint supports restart; an unrelated existing TACC partial attempt is not adopted.
4. Upload into an isolated directory. Verify the full frozen manifest; checkpoint/parameter checksums and agreement; all 400 finite losses; finite AutoDelta parameters; seed receipt; configuration; optimizer schedule; precision/profile; input residency; library versions; model/input identities; and NestWatch hash. There is no fingerprint-drift exception.
5. Atomically place the accepted MAP, write MAP_DONE, then return the **same immutable task script** to the shared pending queue. Its existing guard skips MAP and performs diagnostics and float64 refinement on TACC. Existing collectors then rescore and audit the accepted fit.

No new Slurm allocation was requested by this setup, and no frozen model or overlay was edited. The validated backlog is sufficient for repeated local work; queued fits are not assumed to finish overnight merely because they exist.

## Failure handling and persistence

- One shared local GPU flock and a pueue group with parallelism one prevent cooperative local overlap. The fit subprocess inherits the lock so a supervisor crash cannot immediately launch another GPU fit beside an orphan.
- The supervisor restarts a failed worker after a bounded delay. Status, ownership token, completed/failed exclusions and current task survive restarts.
- Lost handoff responses are reconciled against durable remote receipts; an already accepted checkpoint is not recomputed or overwritten.
- Local failure preserves source, logs and checkpoints and returns the task for TACC to run. The local worker excludes that failed job from immediate re-claim, rather than retrying the same failure indefinitely.
- The local fit stops after prolonged loss of ownership heartbeats. Resource watcher **1167** recovers local reservations abandoned for more than 20 minutes, so a dead workstation cannot indefinitely withhold a task from TACC.
- Temporary Windows system-sleep inhibition is active until 10 a.m. It does not keep the display on or alter persistent power-plan settings; the helper releases its request on exit.
- These processes survive the assistant turn ending. They are not a claim that an unattended reasoning agent will invent and implement new scientific models while the conversation is idle.

## Collection and interpretation

Collector **1154** recognizes `local_running` reservations and continues downloading accepted refinements and scheduling existing follow-ups. Archive watcher **1168** refreshes the full pathology audit, source-sink comparisons, connectivity, movement/establishment reports and the temporal-compensation audit. That last audit now discovers completed registered departure, curvature and floor-combination variants automatically, while keeping missing budgets explicit.

Temporal lambda inflation remains modern-minus-early growth on fixed cells and equal weights, separately from modern growth level. Regional maps, stage survival, correction dependence, immigration/export, establishment, invasion timing, observation fit, seed sensitivity and available HMC geometry remain part of interpretation. Starting a fit, completing float32 MAP, or passing refinement convergence does not establish biological improvement.

## Evidence and controls

**21 operational tests passed**, covering the new dispatcher and existing local acceptance/monitor utilities. The new checks include concurrent claims, source mutation, wrong ownership, abandoned leases, incomplete MAP rejection, successful fixture handoff/reconciliation and library mismatch rejection. These test queue/provenance behavior, not ecological truth.

Implementation: [local_overnight_worker.py](../exp/local_overnight_worker.py), [local_map_dispatch.py](../exp/local_map_dispatch.py), [supervisor](../bin/local_overnight_supervisor.sh), [tests](../../tests/test_local_map_dispatch.py).

Local live state: `.research_runtime/local_overnight/status.json`; events: `events.jsonl`; verified real-checkpoint receipt: `LIVE_FINGERPRINT_VERIFIED.json`; Windows power-request state: `windows_awake.json`. Per-fit log: `/mnt/e/Datasets/houfin/grid54/runs/<step>/out/local3090_map.log`. Remote reservation/return receipts live in `queue/local_running` and the individual run directory.

Creating `.research_runtime/local_overnight/STOP` requests a local stop and releases the temporary sleep request. It does not cancel independent TACC allocations. The deadline also bounds the worker without user intervention.
