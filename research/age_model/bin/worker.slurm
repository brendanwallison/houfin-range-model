#!/bin/bash
#----------------------------------------------------------------------
# Self-terminating experiment worker. Runs task files from $QUEUE_DIR/pending/*.sh in name order
# (each moved to running/, then done/ or failed/, with its log), and EXITS after IDLE_MIN minutes
# with nothing pending -- so queued work starts the moment the node does and no SU is spent idle.
# Tasks are plain bash, executed from the research worktree with env.sh's environment.
#
#   sbatch -A DEB23008 -p gpu-a100-small -t 24:00:00 research/age_model/bin/worker.slurm
#   (only gpu-a100-dev is capped at 2 h; small/normal take long limits -- the worker exits when
#   idle, so a long limit costs nothing and stops tasks being cut off mid-refine)
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
QUEUE_DIR="${QUEUE_DIR:-$WORK/houfin/research/age_model/queue2}"   # queue2 is the live queue; queue/ is retired
IDLE_MIN="${IDLE_MIN:-15}"
source "$WT/scripts/tacc/env.sh"          # sets HOUFIN_* and cd's to the MAIN repo ...
cd "$WT"                                   # ... so come back to the research worktree
module load python/3.12.11 2>/dev/null || true
export PYTHONPATH="$WT:$WT/src:${PYTHONPATH:-}"
mkdir -p "$QUEUE_DIR"/{pending,running,done,failed,logs}
echo "[worker] $(hostname) job $SLURM_JOB_ID start $(date); worktree $WT @ $(git rev-parse --short HEAD)"
idle_since=$(date +%s)
# A task whose worker hit its time limit is left in running/. Each claim records its owning job in
# running/<name>.owner; any worker returns tasks whose owner is no longer running to pending/.
recover_orphans() {
    for o in "$QUEUE_DIR"/running/*.owner; do
        [ -e "$o" ] || continue
        owner=$(cat "$o"); n=$(basename "$o" .owner)
        if ! squeue -h -j "$owner" -t R,CG 2>/dev/null | grep -q .; then
            mv "$QUEUE_DIR/running/$n.sh" "$QUEUE_DIR/pending/" 2>/dev/null && \
                echo "[worker] requeued orphan $n (owner job $owner gone) $(date)"
            rm -f "$o"
        fi
    done
}
while :; do
    recover_orphans
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
    echo "$SLURM_JOB_ID" > "$QUEUE_DIR/running/$name.owner"
    echo "[worker] START $name $(date)"; t0=$(date +%s)
    if bash "$claimed" > "$QUEUE_DIR/logs/$name.log" 2>&1; then
        mv "$claimed" "$QUEUE_DIR/done/"; st=done
    else
        mv "$claimed" "$QUEUE_DIR/failed/"; st=FAILED
    fi
    rm -f "$QUEUE_DIR/running/$name.owner"
    echo "[worker] $st $name ($(( ($(date +%s) - t0) / 60 )) min)"
    idle_since=$(date +%s)
done
