#!/bin/bash
# Independent of the conversation; restart a crashed lane until the deadline.
set -uo pipefail
REPO=/mnt/c/Dev/houfin-range-model
PY=/home/breallis/houfin/venv/bin/python
STATE="$REPO/.research_runtime/local_overnight"
mkdir -p "$STATE"
DEADLINE=$(TZ=America/Chicago date -d '2026-11-03 10:00:00' +%s)
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
while [ "$(date +%s)" -lt "$((DEADLINE-2400))" ] && [ ! -f "$STATE/STOP" ]; do
 "$PY" "$REPO/research/exp/local_overnight_worker.py" --deadline 2026-11-03T10:00:00
 RC=$?
 if [ "$RC" -eq 0 ]; then exit 0; fi
 printf 'Local lane restart after exit %s at %s\n' "$RC" "$(date -Is)"
 sleep 45
done
