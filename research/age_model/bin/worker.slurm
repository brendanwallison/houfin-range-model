#!/bin/bash
#----------------------------------------------------------------------
# Self-terminating experiment worker. Runs task files from $QUEUE_DIR/pending/*.sh in name order
# (each moved to running/, then done/ or failed/, with its log), and EXITS after IDLE_MIN minutes
# with nothing pending -- so queued work starts the moment the node does and no SU is spent idle.
# Tasks are plain bash, executed from the research worktree with env.sh's environment.
#
#   sbatch -A DEB23008 -p gpu-a100-dev -t 02:00:00 research/age_model/bin/worker.slurm
#
# Several workers (any partitions) may share one QUEUE_DIR: tasks are claimed atomically.
#----------------------------------------------------------------------
#SBATCH -J age_worker
#SBATCH -o age_worker.o%j
#SBATCH -e age_worker.e%j
#SBATCH -N 1
#SBATCH -n 1
set -uo pipefail
WT="${RESEARCH_WORKTREE:-$WORK/houfin/age-research}"
QUEUE_DIR="${QUEUE_DIR:-$WORK/houfin/research/age_model/queue}"
IDLE_MIN="${IDLE_MIN:-15}"
source "$WT/scripts/tacc/env.sh"          # sets HOUFIN_* and cd's to the MAIN repo ...
cd "$WT"                                   # ... so come back to the research worktree
module load python/3.12.11 2>/dev/null || true
export PYTHONPATH="$WT:$WT/src:${PYTHONPATH:-}"
mkdir -p "$QUEUE_DIR"/{pending,running,done,failed,logs}
echo "[worker] $(hostname) job $SLURM_JOB_ID start $(date); worktree $WT @ $(git rev-parse --short HEAD)"
idle_since=$(date +%s)
while :; do
    task=$(ls "$QUEUE_DIR"/pending/*.sh 2>/dev/null | sort | head -1)
    if [ -z "$task" ]; then
        if [ $(( $(date +%s) - idle_since )) -ge $(( IDLE_MIN * 60 )) ]; then
            echo "[worker] idle ${IDLE_MIN} min -- exiting $(date)"; break
        fi
        sleep 20; continue
    fi
    # Claim by rename: atomic on one filesystem, so when several workers share a queue only
    # one wins a task; the losers' mv fails and they look again.
    name=$(basename "$task" .sh); claimed="$QUEUE_DIR/running/$name.sh"
    mv "$task" "$claimed" 2>/dev/null || continue
    echo "[worker] START $name $(date)"; t0=$(date +%s)
    if bash "$claimed" > "$QUEUE_DIR/logs/$name.log" 2>&1; then
        mv "$claimed" "$QUEUE_DIR/done/"; st=done
    else
        mv "$claimed" "$QUEUE_DIR/failed/"; st=FAILED
    fi
    echo "[worker] $st $name ($(( ($(date +%s) - t0) / 60 )) min)"
    idle_since=$(date +%s)
done
