#!/bin/bash
set -euo pipefail
LOCAL=/home/breallis/houfin/work/houfin/research/FINAL_CLEAN_20261025
REMOTE=/work/07980/bwa386/ls6/houfin/research/final_clean_20261025
SSH=(ssh -S /home/breallis/.ssh/cm-tacc -o BatchMode=yes -o ConnectTimeout=15 bwa386@ls6.tacc.utexas.edu)
for tick in $(seq 1 120); do
 [ -f "$LOCAL/PREPARED" ] && break
 sleep 30
done
test -f "$LOCAL/PREPARED"
"${SSH[@]}" "mkdir -p $REMOTE/code"
rsync -a --exclude __pycache__ --exclude .pytest_cache -e 'ssh -S /home/breallis/.ssh/cm-tacc' "$LOCAL/code/" "bwa386@ls6.tacc.utexas.edu:$REMOTE/code/"
for d in cbc_aux_final_clean cbc_aux_final_v2_clean; do
 rsync -a -e 'ssh -S /home/breallis/.ssh/cm-tacc' \
  "/home/breallis/houfin/work/houfin/processed/encoder/$d/" \
  "bwa386@ls6.tacc.utexas.edu:/work/07980/bwa386/ls6/houfin/processed/encoder/$d/"
done
scp -q -o ControlPath=/home/breallis/.ssh/cm-tacc "$LOCAL/PREPARED" "$LOCAL/aux_manifest.json" "bwa386@ls6.tacc.utexas.edu:$REMOTE/"
"${SSH[@]}" "set -e; cd $REMOTE/code; sha256sum -c SOURCE_MANIFEST.sha256 > ../source_verification.log; python3 research/exp/final_clean_queue.py"
date -Is > "$LOCAL/DISPATCHED"
