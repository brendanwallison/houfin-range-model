# Local takeover execution — November 5, 2026

User direction: execute the takeover autonomously; prioritize structural changes.
Historical-rate priors are diagnostic comparators. Externally grounded sharp
disease timing may be fixed or strongly constrained; lambda change remains learned.

## Initial authoritative state

At 10:30 UTC (04:30 CST), R104 PID 343703 was alive, consuming approximately two
CPU cores with affinity 24–31. Its CPU-only environment explains the CUDA warning;
the warning is not evidence that the fit died. Its trace was empty because output
was buffered while the first optimization work continued. Do not restart it from
file age alone. The 3090 had no research process; about 165 GiB RAM was available.
WSL root had 14 GiB free, E: 3.5 TiB and C: 611 GiB. New artifacts go on mounted
drives, not the WSL root. Existing TACC watchers remain untouched.

The audit preserved the current tracked diff and indexed 390 sandbox summaries
under `.research_runtime/takeover/initial-normalized/`. Of 126 frozen Python source
files, 8 match bytes, 83 differ only in newlines, 26 differ in content and 9 are
missing from this checkout. Windows Git uses core.autocrlf=true; the first WSL Git
inventory falsely reported newline churn. A second inventory explicitly matches
that setting. Both manifests are retained; use the normalized one for source work.

## Scope and first checks

Added a read-only source/AST/artifact inventory and live host inventory. The audit
tests distinguish content changes, newline-only changes, missing files, stale
overlay comments and incomplete summary JSON. Narrow ignore rules exclude local
worktrees, disposable hidden root probes and downloaded R libraries; scientific R
scripts and experiment specifications remain visible.

Baseline replay is an audit, not a scientific intervention: use the frozen step43
source and original overlay, verify objective about 135590.8 and regional real-BBS
NLL about 110044.7, repeat a finite-state canary, and record loaded module hashes.
GPU reductions may not be bit-exact: an exact-hash failure must trigger numeric
state comparison before concluding a scientific or hardware problem.

Scientific launch gate: complete the execution map and historical intervention
crosswalk, reproduce anchors, and reconcile the source baseline. No structural
fit is claimed started by this initial record.
