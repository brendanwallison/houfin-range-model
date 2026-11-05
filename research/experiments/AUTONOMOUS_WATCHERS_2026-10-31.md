# Persistent experiment monitoring and follow-ups

Enabled 2026-10-31 around 18:00 CDT, following explicit authorization to work autonomously and use the local GPU appropriately. This workflow continues independently of the chat process while WSL/TACC remain running. It does not survive a machine reboot automatically.

## Running services

- **TACC monitor:** `/work/07980/bwa386/ls6/houfin/research/grid54/monitor/range_watch.py`, detached on the login node. Polls every 60 seconds, queries Slurm and only allocated worker nodes, records GPU memory/utilization/processes, active bundles, progress logs, accepted/rejected refine reports, scalar parameters and diagnostic scores. Discovers allocations dynamically rather than hard-coding node IDs. At setup, workers 3541710/v330-003 and 3542329/v330-019 were running, with 3544518 pending.
- **Local mirror:** `.research_runtime/range_monitor/`. Polls remote snapshots, local GPU and pueue. Stale remote heartbeats (>180 seconds) are explicit health events. Both monitor processes have singleton locks, atomic JSON snapshots, heartbeat files, and append-only event/error logs. Run-log inactivity over 30 minutes is flagged for inspection, not automatically called optimizer failure.
- **Follow-up dispatcher:** pueue task **959**, group `range-watch`, label `range-autonomy/followups`. Watches accepted, fully diagnosed founder fits and queues their pre-registered held-out comparisons, serially in the separate `range-cpu` group. Durable labels prevent duplicate enqueueing after dispatcher restart. Downloaded checkpoints, remote completion markers and diagnostic parity are checked before a result is recorded as complete.

Primary live files are `latest.json`, `events.jsonl`, `action_required.json`, `heartbeat`, `followup_events.jsonl` and `followup_errors.jsonl` inside the local monitor directory. Equivalent remote files remain available even when the local SSH mirror disconnects. SSH probes cannot consume the parent launch script's standard input.

These services collect results and execute registered follow-ups; they do not autonomously interpret novel biology, relax convergence gates, kill apparent stalls, or submit extra Slurm allocations. Existing Slurm workers continue claiming and recovering their own queued bundles.

## Local GPU / TACC split

Pueue task **958**, `range-autonomy/local-float32-map`, reserves the already-registered two-arm NestWatch bias bundle `0402_bundle_nest_g3.sh`. Reservation is an atomic move from `queue/pending/` to `queue/local_running/`, so an A100 worker cannot start the same bundle concurrently.

Local RTX 3090 work is **float32 cold MAP only**, sequentially for `step31_nestsub` and `step31_nestdrop`, with a 55% XLA memory fraction (~13.2 GiB allocator limit). Source snapshots, seed 41, data and optimization schedule are unchanged. The first arm was observed actively advancing beyond step 100/400 on the GPU. No local float64 refinement is requested.

After each local MAP completes, the script uploads an isolated checkpoint directory. `accept_local_map.py` requires format v2, step/loss-history consistency, finite losses, the original fingerprint, matching config, library versions, source and model-input identities, and explicit synthetic/nest/manifest SHA256 hashes. It refuses to replace an existing TACC MAP. Only after validation does it publish the directory and the MAP_DONE marker. No drift exception is used.

After both transfers, the original bundle returns to the TACC pending queue for **float64 diagnostics and L-BFGS refinement**. On ordinary local failure/termination, the exit trap also returns the bundle so TACC can finish it; partial local work is retained. If connectivity prevents release, the reservation remains visible in `local_running` and the task log reports the failed release. A forced machine loss requires inspection of that reservation; the watcher does not guess and launch a duplicate.

## Verified automatic work

Founder comparisons for central yearling pin, central delayed-breeding pin and low-founder yearling bracket were collected and scored automatically. Final status-verified tasks **963–965** all succeeded, report accepted refinement, and pass prediction parity against the remote diagnostics.

Paired held-out real-BBS spatial mean log-score differences (90% block-bootstrap intervals):

| Arm versus matched base | Difference | Interval |
|---|---:|---|
| CBC central pin, yearling | +.0041 | [-.0072, +.0147] |
| CBC central pin, delayed/native | -.0132 | [-.0236, -.0039] |
| CBC low bracket, yearling | +.0038 | [-.0073, +.0144] |

These are the existing scorer's comparisons, not new adoption decisions. The central yearling pin is within the pre-registered .005 point-score floor; the delayed/native pin has a measurable cost. The high-founder comparison will enqueue when its accepted diagnostics finish. Remaining bracket sensitivity and identification checks still matter.

## Operations

Inspect local tasks with `pueue status --json`, filtering labels beginning `range-autonomy/`. Monitoring has its own process/group and does not occupy the local GPU queue. CPU rescoring uses two numerical threads and cores 8–11, leaving the local MAP's cores 0–7 available.

To stop monitoring, create `STOP` in the relevant monitor state directory. To stop only the follow-up dispatcher, create local `STOP_FOLLOWUPS` or stop task 959. To stop local GPU offload, terminate task 958 normally and verify that the bundle returned to TACC. Stopping a monitor does not cancel fits. Preserve state/event files for provenance.

Sources: `research/exp/range_watch.py`, `research/exp/range_followups.py`, `research/exp/accept_local_map.py`, and `research/bin/range_local_map.sh`. Ten targeted monitor/transfer tests pass; three real follow-up executions also passed status/parity checks.
