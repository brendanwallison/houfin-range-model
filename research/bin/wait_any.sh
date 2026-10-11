#!/usr/bin/env bash
# Block until ANY pueue task that is running or queued right now finishes (or the queue is empty), then print what
# finished. Run it in the background at the end of every turn that leaves work in flight, so the session is woken at
# the first completion instead of sitting idle (2026-10-10: eight GPU runs finished unnoticed for ~7 hours).
#   bash research/bin/rx sh 'bash /mnt/c/Dev/houfin-range-model/research/bin/wait_any.sh [poll_seconds]'
PUEUE="$HOME/.local/bin/pueue"
POLL="${1:-60}"
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
pending=$(snapshot | awk -F'\t' '$2=="Running" || $2=="Queued" {print $1}')
if [ -z "$pending" ]; then
  echo "queue empty: nothing running or queued"
  exit 0
fi
echo "watching $(echo "$pending" | wc -l) running/queued tasks"
while true; do
  sleep "$POLL"
  now=$(snapshot)
  done_ids=""
  for id in $pending; do
    line=$(echo "$now" | awk -F'\t' -v id="$id" '$1==id')
    state=$(echo "$line" | cut -f2)
    if [ "$state" != "Running" ] && [ "$state" != "Queued" ]; then
      done_ids="$done_ids $id"
      echo "FINISHED: $line"
    fi
  done
  if [ -n "$done_ids" ]; then
    left=$(echo "$now" | awk -F'\t' '$2=="Running" || $2=="Queued"' | wc -l)
    echo "still running or queued: $left"
    exit 0
  fi
done
