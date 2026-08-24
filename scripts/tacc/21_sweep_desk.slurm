#!/bin/bash
#----------------------------------------------------------------------
# One packed batch of DESK sweep runs: SWEEP_TASKS_PER_NODE independent trainings per node,
# one per GPU. Driven by SWEEP_JOBLIST, a TSV of  run_id \t overlay_path \t output_dir  written
# by submit_sweep.sh. SUBMIT VIA THAT WRAPPER, not bare sbatch -- this account has multiple
# projects, the `#SBATCH -A` line is deliberately disabled, and SLURM rejects a job with no
# allocation. The wrapper injects -A, sizes -N/--ntasks-per-node, and handles resume.
#
# DESK training is single-device (plain .to(device); no DataParallel), so an unpacked job idles
# two of a gpu-a100 node's three A100s while billing the whole node -- charging is per
# NODE-hour. Packing three tasks per node is 3x cheaper per GPU-hour and 3x more parallel.
#
# Each task must get its own GPU, its own output directory, and its own overlay. GPU binding is
# CUDA_VISIBLE_DEVICES=$SLURM_LOCALID: local task 0/1/2 -> GPU 0/1/2. Output isolation is the
# overlay's paths.desk_output_dir, NOT an environment root -- env.sh exports
# HOUFIN_DATA/HOUFIN_PROCESSED UNCONDITIONALLY, so a value set in the submitting shell is
# silently discarded by every job, and three tasks would write to one directory while looking
# like they had worked.
#
# NOTE on gpu0/gpu1/gpu2 topology: gpu0 is on socket 0 and gpu1/gpu2 on socket 1. srun's
# default binding may put two tasks on one socket's memory; this matters for host-side staging
# (DESK stages its whole year window through host memory before the device copy) but not for
# correctness. If a host OOM appears, drop TASKS_PER_NODE to 2 rather than chasing affinity.
#----------------------------------------------------------------------
#SBATCH -J houfin_sweep
#SBATCH -o houfin_sweep.o%j
#SBATCH -e houfin_sweep.e%j
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=3
#SBATCH -t 04:00:00
#SBATCH --mail-type=all
##SBATCH -A REPLACE_WITH_PROJECT
set -uo pipefail

: "${SWEEP_JOBLIST:?SWEEP_JOBLIST must name the batch TSV (submit via submit_sweep.sh)}"
TPN="${SWEEP_TASKS_PER_NODE:-3}"
STAGES_RUN="${SWEEP_STAGES:-desk}"

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
module load python/3.12.11 2>/dev/null || true
[ -n "${TACC_ALLOCATION:-}" ] && echo "allocation: $TACC_ALLOCATION"
date
echo "joblist: $SWEEP_JOBLIST ($(wc -l < "$SWEEP_JOBLIST") runs), $TPN task(s)/node, stages='$STAGES_RUN'"

# How many GPUs this node actually has, and their memory. Printed once per job because the
# gpu-a100-small GPU count is NOT in the LS6 docs (the accounting page is an unexpanded
# template include), so the SU-per-GPU-hour comparison between queues rests on this measurement
# rather than on an inference from 1.5-of-3 SU, which does not divide into 3 GPUs.
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "=== GPUs visible to this job ==="
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
fi

# The per-task body. Written to a file and invoked by srun so each rank picks its own line of
# the joblist by SLURM_PROCID -- one producer of the assignment, so two ranks cannot claim the
# same run.
TASK="$SLURM_SUBMIT_DIR/.sweep_task_$SLURM_JOB_ID.sh"
cat > "$TASK" <<'INNER'
#!/bin/bash
set -uo pipefail
source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
# One GPU per task. SLURM_LOCALID is the rank's index WITHIN its node, which is exactly the
# GPU ordinal to bind; SLURM_PROCID is global across nodes and would index past the node's
# GPU count on node 2 onward.
export CUDA_VISIBLE_DEVICES="${SLURM_LOCALID}"
LINE="$(sed -n "$((SLURM_PROCID + 1))p" "$SWEEP_JOBLIST")"
if [ -z "$LINE" ]; then
    echo "[rank $SLURM_PROCID] no run assigned (joblist shorter than the task count) -- idle"
    exit 0
fi
RUN_ID="$(printf '%s' "$LINE" | cut -f1)"
OVERLAY="$(printf '%s' "$LINE" | cut -f2)"
OUT_DIR="$(printf '%s' "$LINE" | cut -f3)"
echo "[rank $SLURM_PROCID localid $SLURM_LOCALID gpu $CUDA_VISIBLE_DEVICES] $RUN_ID -> $OUT_DIR"

# Resume, checked again HERE and not only in the wrapper: a resubmission can race a job still
# finishing an earlier batch, and repeating a completed run would overwrite its checkpoint and
# its trajectory with a second, differently-seeded one.
if [ -f "$OUT_DIR/run_summary.json" ]; then
    echo "[$RUN_ID] already complete (run_summary.json present) -- skipping"
    exit 0
fi
mkdir -p "$OUT_DIR"
# Per-task scratch. PyTorch/CUDA and matplotlib both write into shared temp roots; three ranks
# sharing one is a documented source of intermittent cache corruption.
export TMPDIR="$OUT_DIR/tmp"; mkdir -p "$TMPDIR"
export MPLCONFIGDIR="$TMPDIR/mpl"
# W&B offline: compute nodes' outbound network is not documented and must not be assumed. Sync
# from a login node afterwards. `wandb agent` is deliberately unused -- it pulls configs from a
# server, which offline mode precludes, and Bayesian search works against a deterministic grid.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$OUT_DIR"

export ESK_DESK_CONFIG="$OVERLAY"
export STAGES="$SWEEP_STAGES"
LOG="$OUT_DIR/train.log"
set -o pipefail
bash "$SLURM_SUBMIT_DIR/scripts/tacc/pipeline.sh" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "[$RUN_ID] exit $rc"
# Record the failure where the manifest reader will find it. A run that dies leaves no
# run_summary.json, so resume retries it -- but without this there is nothing to distinguish
# "not yet run" from "ran and failed", and the plan calls for per-run exit status.
if [ "$rc" -ne 0 ]; then
    printf '{"run_id":"%s","exit_status":%d,"log":"%s"}\n' "$RUN_ID" "$rc" "$LOG" \
        > "$OUT_DIR/run_failed.json"
fi
exit "$rc"
INNER
chmod +x "$TASK"

export SWEEP_JOBLIST SWEEP_STAGES="$STAGES_RUN"
srun --ntasks-per-node="$TPN" "$TASK"
rc=$?
rm -f "$TASK"
echo "=== 21_sweep_desk batch complete (srun rc=$rc) ==="
date
# A nonzero srun status means at least one task failed; the others still finished and wrote
# their run_summary.json, so a resubmission retries only the gaps. Exit 0 so a single bad run
# does not mark the whole batch as lost in the scheduler's accounting.
exit 0
