# Codex TACC control disabled — November 6, 2026

At the user's explicit request, Claude now has TACC scheduling control.
Do not restart Codex TACC submission or execution controllers without a new
explicit user request to resume them.

- Codex's resource/submission and later-state controller tasks were stopped.
- Persistent local stop markers also disable its HMC and audit watchers.
- The sole live Codex-owned Slurm request, pending dev allocation 3558841,
  was cancelled. No Codex-owned allocation remained running.
- All ten Claude-labelled running/pending jobs were preserved.
- Shared Slurm scripts, scientific queues, checkpoints, logs and the SSH
  connection were preserved for Claude's use. Some Claude jobs reuse the same
  Slurm command path, so command-path matching alone must not be used to cancel
  jobs or disable shared dispatchers.
- Local scientific compute remains disabled for Codex as previously requested.

Receipt: `CODEX_TACC_CONTROL_DISABLED_20261106.json` and
`.research_runtime/CODEX_TACC_DISABLED_LOCAL.json`.
