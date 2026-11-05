# Local native-crash investigation — November 3, 2026

The earlier response stopped after reading the exit/log and returning the task to TACC. It did not diagnose the failure. That was incomplete. Investigation was triggered when the following local experiment also crashed.

## What failed

- `step44_shared_s43`: signal 11 during MAP around iteration 97, after finite losses. The kernel records a fault in `xla_cuda_plugin.so` for its Python process.
- `step45_cell_iid_s010_s41`: signal 11 around iteration 13, also after finite losses. The kernel records an invalid instruction-address access in its Python process; it does not resolve that fault to a named library.
- In the same interval, a separate CPU analysis process faulted in SciPy's `_ni_label` extension. An earlier compiler process also faulted while using the pinned CUDA 12.8 `ptxas`.

These are native process crashes, not optimizer convergence decisions. A population-model input or compiled computation can still trigger a native software bug; the logs do not prove a hardware cause.

## Checks performed

The loaded JAX, jaxlib, CUDA plugin and PJRT distributions are all 0.8.1. The worker already disables XLA command buffers and selects CUDA 12.8 ptxas. Those protections were already present; they are not new fixes for this incident. The GPU driver reports 621.01.

All five checked native binaries match the installed distribution RECORD hashes: SciPy `_ni_label`, jaxlib `_jax` and `libjax_common`, and both CUDA plugin components. This checks on-disk installation integrity, not correctness of the library implementation or RAM.

At inspection the GPU was 47 C with about 19.1/24.6 GB allocated, host memory had roughly 165 GiB free, and no OOM kill appeared in the inspected kernel log. No WHEA event was returned by the preceding-day Windows query. Neither absence of WHEA nor available memory proves hardware stability.

Two independent 90-second CPU-only checks, pinned to CPUs 12 and 18, performed 3,033 and 3,046 repetitions. They verified deterministic connected-component labels, an integer identity and memory hashes. Both passed. This did not reproduce the CPU failure and is not a full CPU/RAM stability test.

## Concrete firmware finding

The host is an Intel i9-13900K on an ASUS ROG MAXIMUS Z690 EXTREME, BIOS 3802 dated August 2024. Windows reports active microcode revision **0x129**. ASUS identifies 3802 as its 0x129 release and documents subsequent 0x12B and 0x12F updates on the [exact board's support page](https://www.asus.com/us/supportonly/rog%20maximus%20z690%20extreme/helpdesk_bios/).

[Intel's current guidance](https://www.intel.com/content/www/us/en/support/articles/000102331/processors.html) recommends Intel Default Settings and a BIOS containing **0x12F or later** for this processor family's Vmin-shift instability. The host is below that recommendation. This is a plausible contributor worth correcting; it does **not** establish that this CPU is damaged or that these particular crashes were caused by Vmin instability. RAM/WSL/native-library issues remain alternatives. No firmware, voltage or global power setting was changed.

## Changes made now

Further local claims were paused on the remote dispatcher after the second signal-11 failure was noticed. The currently progressing private-K fit was allowed to continue; it is not killed to reload supervision. Both failed tasks remain pending for TACC, with their scientific sources unchanged.

The worker now enables Python fault-handler output and captures native failure evidence. Two consecutive native crashes open a persistent local hold rather than silently moving through the scientific queue. Two new focused tests and nine existing dispatcher tests passed. Current hold state is `.research_runtime/local_overnight/NATIVE_CRASH_HOLD.json`; the remote PAUSE_LOCAL marker independently prevents further claims by the old in-memory worker. Boundary coordinator pueue task 1242 reloads the corrected worker only once the current fit finishes and transfers its checkpoint.

The host/runtime issue is **not resolved**. The local lane must not be reported healthy merely because another case runs. A controlled, separate repeat of a failed case with fault traces remains necessary after host/runtime remediation, followed by independent replay and the normal TACC validation. The TACC queue is unaffected by the local hold.

## Evidence

The adjacent `local_native_crash_20261103` directory contains the kernel log, host/firmware record, binary checksum checks, two CPU repeatability results, failed-log tails and a timestamped local-state snapshot. Full fit logs remain in their original run directories. BIOS review is a maintenance requirement involving a reboot and must be coordinated separately from the running scientific work.