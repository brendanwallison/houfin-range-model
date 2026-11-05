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

The first GPU replay completed the numerical work but failed JSON serialization:
the legacy harness overwrites its hash variable `h` with a held-out-row mask,
returning an array as `final_state_sha256`. The takeover replay now computes its
own hashes from actual density arrays, plus a numeric canary comparison. This is
a diagnostic bug, not evidence that the fit or GPU failed. The active R104 harness
is not edited while that process runs.

## Collected evidence

The frozen GPU replay passed at objective 135590.79802835 and BBS NLL
110044.700261, with a finite, byte-identical within-launch canary. The reconstructed
worktree replay gives the same objective and held-out scores to floating-point
precision. Full trace accounting closes to 3e-11. A GPU data-likelihood gradient
audit covers all 29 sites / 19,275 scalar entries: all finite, with exactly zero
data gradients only for the two inactive logistic departure parameters.

Reciprocal factor/amplitude scaling leaves both count and nest likelihoods
unchanged to about 4e-10 while changing the prior penalty. Source-only coordinate
cleanup cannot be claimed as new biological identification.

R104 completed at about 04:48 CST. The collector verified PID disappearance and
read its saved summary, checkpoint and final trace. Objective 135881.253412,
occupied-East potential lambda .54455 (1966) -> 1.34933 (2010), modern disease toll
.2590. The relative-change convergence flag accompanies a final gradient norm of
384; treat stationarity as unproven. It remains a direct-change-prior diagnostic.

Recovered model tests initially yielded 85 passes, one skip and three missing
ingestion-dependency failures. Those dependencies were recovered from the same
frozen source; the affected observation/disease rerun yielded 62 passes and one
skip. Test capture uses RAM-backed temporary files because drvfs's unlinked-file
handling broke pytest capture; caches and persistent outputs remain off WSL root.

Input audit hashes actual WSL-resolved feature files (137 MB local Z, 1.645 GB
path Z) and verifies 124 years, 4,318 land cells, 64 features and 12 kernels. The
used occupied center exactly matches all 33,519 positive cell-years, not only the
25,600 training-positive cell-years. Training-only studies must account for this
support difference explicitly; centering can alter prior anchoring even where
intercepts can preserve the likelihood family.

Publication: foundation commit f726f6a is local. Automatic approval review rejected
its push to the existing GitHub remote because exact destination authorization
was not considered established. An asynchronous destination-approval request is
pending; local work continues and no alternate publication route is attempted.
