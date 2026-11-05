#!/bin/bash
set -uo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_CLEAN_20261025
SOURCE=/mnt/c/Dev/houfin-range-model/research/exp/final_gpu_requeue.py
for tick in $(seq 1 600); do
 ssh -S /home/breallis/.ssh/cm-tacc -o BatchMode=yes -o ConnectTimeout=10 \
  bwa386@ls6.tacc.utexas.edu python3 - < "$SOURCE" || true
 if [ -f "$ROOT/EVALUATION_DONE" ] && ssh -S /home/breallis/.ssh/cm-tacc -o BatchMode=yes -o ConnectTimeout=10 \
  bwa386@ls6.tacc.utexas.edu 'test -f /work/07980/bwa386/ls6/houfin/research/final_clean_20261025/QUEUE_WATCH_DONE'; then exit 0; fi
 sleep 60
done
exit 2
