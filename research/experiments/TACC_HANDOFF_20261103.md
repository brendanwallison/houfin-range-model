# Codex execution handoff — November 3, 2026

The user assigned local runs to Claude. Codex must use TACC and must not restart local GPU/scientific workers without a later user instruction.

Codex local resources were released and checked: no outstanding remote local-worker lease, local GPU flock available, no active Codex GPU worker, expired Windows sleep helper absent. Local worker STOP is persistent. CODEX_LOCAL_COMPUTE_DISABLED prevents the status collector from launching local scientific analysis. Codex's full archive and HMC analysis watchers were stopped; lightweight TACC allocation/status collection remains active. Claude's processes and reservations were not modified.

At 15:50 CST, TACC development allocation 3548968 was running on c301-002, three A100 GPUs, with a two-hour allocation beginning 15:49:43. Three workers claimed step44_shared_s43, step44_kprivate10_s41 and step45_cell_iid_s010_s41. At 15:52 their Python processes were loading inputs and initializing diagnostics/MAP; no completed fit is claimed. The private-K task reuses its verified local float32 checkpoint. Regular jobs 3547292/3548331 and small 3548970 remain pending.

Further post-fit scientific calculations must run on TACC. The collector continues storing results but deliberately skips its former local analysis dispatch while the Codex marker exists. Frozen scientific sources and queue ownership remain authoritative.