#!/usr/bin/env bash
# Block until ANY pueue task that is running or queued right now finishes (or the queue is empty), then print what
# finished. Run it in the background at the end of every turn that leaves work in flight, so the session is woken at
# the first completion instead of sitting idle (2026-10-10: eight GPU runs finished unnoticed for ~7 hours).
#   bash research/bin/rx sh 'bash /mnt/c/Dev/houfin-range-model/research/bin/wait_any.sh [poll_seconds]'
#   WAIT_LABEL_RE='^E059/' ... only watch tasks whose label matches (2026-10-13: two sessions share the queue, and each
#   was woken by the other's jobs). Stashed tasks count as pending, not finished (a stash used to read as a finish).
PUEUE="$HOME/.local/bin/pueue"
POLL="${1:-60}"
RE="${WAIT_LABEL_RE:-.}"
snapshot() {
  "$PUEUE" status --json 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
for tid, t in d["tasks"].items():
    st = t["status"]
    s = list(st.keys())[0] if isinstance(st, dict) else st
    res = st[s].get("result", "") if isinstance(st, dict) and isinstance(st[s], dict) else ""
    print(tid, s, json.dumps(res), t["label"], sep="\t")
'
}
live() { awk -F'\t' -v re="$RE" '($2=="Running" || $2=="Queued" || $2=="Stashed") && $4 ~ re {print $1}'; }
pending=$(snapshot | live)
if [ -z "$pending" ]; then
  echo "queue empty: nothing running, queued or stashed (label filter: $RE)"
  exit 0
fi
echo "watching $(echo "$pending" | wc -l) running/queued/stashed tasks matching '$RE' (and any added later)"
while true; do
  sleep "$POLL"
  now=$(snapshot)
  # tasks enqueued after the waiter started are watched too (2026-10-10: a waiter armed before three new jobs were
  # queued would have slept through all of them)
  pending=$(printf '%s\n%s\n' "$pending" "$(echo "$now" | live)" | grep -v '^$' | sort -un)
  done_ids=""
  for id in $pending; do
    line=$(echo "$now" | awk -F'\t' -v id="$id" '$1==id')
    state=$(echo "$line" | cut -f2)
    if [ "$state" != "Running" ] && [ "$state" != "Queued" ] && [ "$state" != "Stashed" ]; then
      done_ids="$done_ids $id"
      echo "FINISHED: ${line:-$id (removed from the queue)}"
    fi
  done
  if [ -n "$done_ids" ]; then
    left=$(echo "$now" | live | wc -l)
    echo "still running, queued or stashed (matching '$RE'): $left"
    exit 0
  fi
done
