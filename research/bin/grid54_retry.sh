#!/bin/bash
# Run grid54_fit.sh, retrying up to 3 times after a crash (segfault in ptxas / libcuda / XLA, seen on the local WSL
# box 2026-10-30 in several unrelated binaries). grid54_fit.sh skips stages whose *_DONE marker exists, so a retry
# resumes at the failed stage. Non-crash failures (Python errors, invalid refine) are not retried.
# Usage: grid54_retry.sh RUN_DIR STEP OVERLAY_NAME
R="$1"; shift
for i in 1 2 3 4; do
  bash "$R/grid54_fit.sh" "$@"; rc=$?
  [ $rc -eq 0 ] && exit 0
  crash=0
  [ $rc -ge 128 ] && crash=1
  grep -l -E 'ptxas exited with non-zero error code 139|Segmentation fault|SIGSEGV' "$R"/out/*.log >/dev/null 2>&1 && crash=1
  [ $crash -eq 1 ] && [ $i -lt 4 ] || exit $rc
  echo "[grid54_retry] attempt $i crashed (rc $rc) -- retrying $(date -Is)" | tee -a "$R/out/retries.log"
  sleep 30
done
