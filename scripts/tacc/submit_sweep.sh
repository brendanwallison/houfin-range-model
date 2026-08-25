#!/bin/bash
# Submit a DESK hyperparameter-sweep stage: N runs packed 3-per-node onto gpu-a100.
#
#   DRY_RUN=1 bash scripts/tacc/submit_sweep.sh                  # default: plan, submit nothing
#   SWEEP_STAGE=1 DRY_RUN=0 bash scripts/tacc/submit_sweep.sh
#   SWEEP_STAGE=2 SWEEP_CONFIGS=base,sk0,hl4,w64 DRY_RUN=0 bash scripts/tacc/submit_sweep.sh
#   SWEEP_SEEDS=0,1,2 SWEEP_CONFIGS=sk0 DRY_RUN=0 bash scripts/tacc/submit_sweep.sh   # stage 3
#
# WHY PACKED. DESK training is single-device -- plain .to(device), no DataParallel -- so one
# job per node leaves two of a gpu-a100 node's three A100s idle while billing the whole node.
# Charging is per NODE-hour (3 SU/hr on gpu-a100, 1.5 on gpu-a100-small), so:
#   gpu-a100, 1 job/node   3.0 SU per GPU-hour, 12 concurrent GPUs (12 nodes/user)
#   gpu-a100, 3 jobs/node  1.0 SU per GPU-hour, 36 concurrent GPUs      <-- this
#   gpu-a100-small         1.5 / n SU per GPU-hour, n UNKNOWN -- MEASURE IT (see below)
# Packing is 3x cheaper per GPU-hour AND 3x more parallel than the same queue unpacked. The
# -small comparison cannot be computed: the LS6 accounting page is an unexpanded template
# include and the GPU count per virtual node appears nowhere in the docs. 1.5 of 3 SU implies
# half a node, which does not divide into 3 GPUs, so it cannot be inferred either. The first
# debug job prints `nvidia-smi --query-gpu=index,name,memory.total`; if -small turns out to
# give more than one GPU, recompute before committing the full grid.
#
# NOT PyLauncher: the TACC docs describe it as distributing serial commands pinned to specific
# CORES -- built for CPU high-throughput work, with no GPU affinity, so it would need the same
# CUDA_VISIBLE_DEVICES wrapper anyway. Its value is dynamic rebalancing across many short
# heterogeneous tasks; these are long homogeneous ones (the forwarded year window is
# 1940..label_year in EVERY cell, so a temporal holdout changes which years are supervised, not
# how many are forwarded -- every run costs the same). Reconsider only if measured durations
# turn out to vary by more than ~2x across the grid.
#
# RESUME. A run whose output dir already holds run_summary.json is skipped, so resubmitting
# fills gaps rather than repeating work. run_summary.json is written LAST by the desk stage,
# after the checkpoint and desk_meta.npz, so its presence means "this run finished" -- a
# desk_meta.npz alone can be left behind by a job killed between the two writes.
set -euo pipefail
source "$(dirname "$0")/env.sh"

SWEEP_NAME="${SWEEP_NAME:-desk_hp}"
SWEEP_ROOT="${SWEEP_ROOT:-$HOUFIN_PROCESSED/sweeps/$SWEEP_NAME}"
SWEEP_STAGE="${SWEEP_STAGE:-1}"
SWEEP_CONFIGS="${SWEEP_CONFIGS:-}"
SWEEP_SEEDS="${SWEEP_SEEDS:-}"
# Forwarded to the generator. Without this the wrapper's own regeneration silently DROPS a
# --stop-at applied by an earlier manual run of the generator, and every run goes full-length --
# the overlays on disk would say one thing and the submitted jobs another.
SWEEP_STOP_AT="${SWEEP_STOP_AT:-0}"
QUEUE="${QUEUE:-gpu-a100}"
TIME="${TIME:-04:00:00}"
TASKS_PER_NODE="${TASKS_PER_NODE:-3}"
# gpu-a100 allows 8 nodes/job and 12 nodes/user. Default to 4 so several stages can be in
# flight without one of them monopolising the per-user cap.
MAX_NODES="${MAX_NODES:-4}"
# Cap on how many batches this invocation submits. The staged progression the plan calls for
# needs exactly this: one batch on gpu-a100-dev (1 job/user, 2 h) to prove three packed
# processes do not collide, before the full grid goes to gpu-a100. Without it a 17-run stage
# computes 6 batches, exceeds the dev queue's 1-job cap, and is refused outright.
MAX_BATCHES="${MAX_BATCHES:-0}"
DRY_RUN="${DRY_RUN:-1}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
STAGES_ENV="${SWEEP_TRAIN_STAGES:-desk}"

A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"
submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }

cd "$HOUFIN_REPO"
PY="${HOUFIN_VENV}/bin/python"; [ -x "$PY" ] || PY="python"

echo "=== preflight ==="
# Only TRACKED changes under src/scripts/tests/config count: SLURM drops job logs and 30-second
# telemetry CSVs straight into this directory, so an untracked-inclusive check calls every
# post-run tree dirty. What matters is that a source edit mid-sweep makes runs incomparable --
# two configurations differing by code as well as by config is not a measurement of the config.
DIRTY="$(git status --porcelain --untracked-files=no -- src scripts tests config)"
if [ -n "$DIRTY" ] && [ "$ALLOW_DIRTY" != "1" ]; then
    echo "ERROR: tracked source/config changes are uncommitted:"; echo "$DIRTY"
    echo "A source edit mid-sweep makes runs incomparable. Commit or stash first,"
    echo "or set ALLOW_DIRTY=1 if the edits are genuinely inert."
    exit 1
fi
GIT_SHA="$(git rev-parse HEAD)"; echo "git sha: $GIT_SHA"

# The ESK basis. ONE basis serves every cell of the grid: it is fitted on the point set with no
# reference to the training mask or the holdout years, so no cell needs a refit. Checked here
# because a missing basis fails identically in all N runs, minutes in.
Z_DIR="$("$PY" -c "from src.config_utils import load_config; print(load_config()['desk']['z_dir'])")"
for f in Z.npy valid_mask.npy meta.json; do
    [ -f "$Z_DIR/$f" ] || { echo "ERROR: ESK basis incomplete: $Z_DIR/$f missing (run STAGES=spacetime-esk)"; exit 1; }
done
echo "esk basis: $Z_DIR"

echo "=== overlays ==="
GEN_ARGS=(--root "$SWEEP_ROOT" --stage "$SWEEP_STAGE")
[ -n "$SWEEP_CONFIGS" ] && GEN_ARGS+=(--configs "$SWEEP_CONFIGS")
[ -n "$SWEEP_SEEDS" ] && GEN_ARGS+=(--seeds "$SWEEP_SEEDS")
[ "$SWEEP_STOP_AT" -gt 0 ] && GEN_ARGS+=(--stop-at "$SWEEP_STOP_AT")
"$PY" scripts/sweep/generate_overlays.py "${GEN_ARGS[@]}"
MANIFEST="$SWEEP_ROOT/sweep_manifest.json"
[ -f "$MANIFEST" ] || { echo "ERROR: no manifest at $MANIFEST"; exit 1; }

# Any per-ema_tau yearly_states build the manifest asks for must EXIST before its runs start:
# ema_tau is applied when state_{year}.npz is written, not by DESK, so a missing states dir is
# not a slow path -- it is a FileNotFoundError per year, or worse, a silent fallback to a dir
# built at a different tau. state_schema.json now records ema_tau for exactly that reason.
echo "=== states dirs ==="
MISSING_STATES="$("$PY" - "$MANIFEST" <<'PYS'
import json, os, sys
m = json.load(open(sys.argv[1]))
need = sorted({r["requires_states_dir"] for r in m["runs"] if r["requires_states_dir"]})
for d in need:
    d = os.path.expandvars(d)
    if not os.path.isdir(os.path.join(d, "yearly_states")):
        print(d)
PYS
)"
if [ -n "$MISSING_STATES" ]; then
    echo "these yearly_states builds are required but absent:"
    echo "$MISSING_STATES" | sed 's/^/  /'
    echo "ema_tau is consumed at STATE-BUILD time (src/data/combine/streams.py applies it along"
    echo "the year axis as the arrays are written), not by DESK, so each tau variant is a"
    echo "different covariate dataset and needs its own build. Submit them with:"
    echo "    DRY_RUN=0 bash scripts/tacc/submit_tau_states.sh"
    echo "It reads them from this manifest, so no overlay path has to be retyped. Or drop the"
    echo "tau configurations from this stage: SWEEP_CONFIGS=<tags without tau0/tau1/tau4>."
    # A DRY RUN still prints the packing plan. Blocking here meant the one command whose whole
    # job is "show me what you would do" refused to answer until hours of unrelated
    # preprocessing had finished -- so the plan could not be reviewed before committing to it.
    # A real submission is still refused: those runs would fail per-year on a missing states
    # dir, or read a dir built at a different tau and measure the wrong thing entirely.
    if [ "$DRY_RUN" != "1" ]; then
        echo "REFUSING to submit. Build them first, or exclude them."
        exit 1
    fi
    echo "(dry run: continuing so the plan below can be reviewed -- these runs would be"
    echo " REFUSED on a real submission)"
else
    echo "all required states dirs present"
fi

# --- resume: drop runs that already finished -------------------------------------------------
# while-read, not `mapfile`: mapfile is a bash 4 builtin and this repo is
# developed on macOS, whose /bin/bash is 3.2 -- the script would pass `bash -n`
# and then die at this line on a dev machine while working on the cluster.
# submit_juv_mdd_sweep.sh already uses this form for the same reason.
PENDING=()
while IFS= read -r _line; do
    [ -n "$_line" ] && PENDING+=("$_line")
done < <("$PY" - "$MANIFEST" <<'PYP'
import json, os, sys
m = json.load(open(sys.argv[1]))
done = pend = 0
for r in m["runs"]:
    out = os.path.expandvars(r["desk_output_dir"])
    if os.path.exists(os.path.join(out, "run_summary.json")):
        done += 1
        continue
    pend += 1
    print(f'{r["run_id"]}\t{os.path.expandvars(r["overlay"])}\t{out}')
print(f"# {done} already complete, {pend} pending", file=sys.stderr)
PYP
)
N_PENDING="${#PENDING[@]}"
# Cross-check against the manifest, counted INDEPENDENTLY. A process substitution's exit status
# is invisible to the reading loop, so if the reader above dies the array is simply empty -- and
# an empty array is indistinguishable from "everything finished". That reads as a completed
# sweep and exits 0. Counting the completed runs separately and requiring the two halves to add
# up turns a crashed reader into an error instead of a false success.
N_TOTAL=$("$PY" -c "
import json; print(len(json.load(open('$MANIFEST'))['runs']))")
N_DONE=$("$PY" - "$MANIFEST" <<'PYD'
import json, os, sys
m = json.load(open(sys.argv[1]))
print(sum(os.path.exists(os.path.join(os.path.expandvars(r["desk_output_dir"]),
                                      "run_summary.json")) for r in m["runs"]))
PYD
)
if [ $((N_PENDING + N_DONE)) -ne "$N_TOTAL" ]; then
    echo "ERROR: manifest has $N_TOTAL runs but $N_PENDING pending + $N_DONE complete ="
    echo "       $((N_PENDING + N_DONE)). The resume reader did not enumerate every run --"
    echo "       treat this as a failure, not as a finished sweep."
    exit 1
fi
echo "=== $N_PENDING run(s) pending, $N_DONE complete, $N_TOTAL total ==="
[ "$N_PENDING" -gt 0 ] || { echo "nothing to do -- every run in the manifest is complete"; exit 0; }

# --- pack into node-sized chunks -------------------------------------------------------------
N_NODES=$(( (N_PENDING + TASKS_PER_NODE - 1) / TASKS_PER_NODE ))
[ "$N_NODES" -le "$MAX_NODES" ] || N_NODES="$MAX_NODES"
CHUNK=$(( N_NODES * TASKS_PER_NODE ))
echo "queue=$QUEUE time=$TIME nodes=$N_NODES tasks/node=$TASKS_PER_NODE -> $CHUNK per job"
# Computed on its own line with awk -v and SINGLE quotes. Inline as
# `$(awk "BEGIN{printf \"%.2f\", 3.0/$TPN}")` inside a double-quoted echo, the \" escapes are
# eaten by the outer quoting level and awk receives `BEGIN{printf %.2f, 3.0/3}` -- a syntax
# error, printed twice, with the cost figure blank. Not platform-specific: awk is awk.
SU_PER_GPU_HR=$(awk -v n="$TASKS_PER_NODE" 'BEGIN{printf "%.2f", 3.0/n}')
echo "packed cost: ${TASKS_PER_NODE} GPUs of 3 per node at 3 SU/node-hr = ${SU_PER_GPU_HR} SU per GPU-hour (unpacked would be 3.00)"

# Queue caps, from hpc/lonestar6.md. These behave DIFFERENTLY and the distinction matters:
# a nodes-per-user overrun just leaves jobs pending (QOSMaxNodePerUserLimit), but a
# jobs-per-user overrun is REJECTED at submit time -- so a large stage, or a resubmission while
# an earlier batch is still queued, silently loses its tail batches unless this is checked.
case "$QUEUE" in
    gpu-a100)       MAX_JOBS=8;  CAP_NODES=12 ;;
    gpu-a100-small) MAX_JOBS=12; CAP_NODES=3  ;;
    gpu-a100-dev)   MAX_JOBS=1;  CAP_NODES=2  ;;
    *)              MAX_JOBS=0;  CAP_NODES=0  ;;   # unknown queue: skip the check, don't guess
esac
N_BATCHES=$(( (N_PENDING + CHUNK - 1) / CHUNK ))
if [ "$MAX_BATCHES" -gt 0 ] && [ "$N_BATCHES" -gt "$MAX_BATCHES" ]; then
    echo "MAX_BATCHES=$MAX_BATCHES: submitting $MAX_BATCHES of $N_BATCHES batches"
    echo "  ($(( N_PENDING - MAX_BATCHES * CHUNK )) run(s) left for a later invocation --"
    echo "   resume picks them up, nothing is lost)"
    N_BATCHES="$MAX_BATCHES"
fi
if [ "$MAX_JOBS" -gt 0 ]; then
    # `set -euo pipefail` is active, so a non-zero squeue inside a command substitution ABORTS
    # the script -- and with the redirect swallowing its stderr, silently: the run stopped dead
    # after the cost line with no message and nothing submitted. squeue exits non-zero for
    # reasons that have nothing to do with this check (a slurmctld timeout, an unrecognised
    # partition), so its failure has to be absorbed and REPORTED rather than treated as zero.
    # Reporting matters: "no jobs queued" and "could not ask" pass the cap check identically,
    # and only one of them is safe.
    if EXISTING=$(squeue -u "$USER" -h -p "$QUEUE" 2>/dev/null); then
        EXISTING=$(printf '%s\n' "$EXISTING" | grep -c . || true)
    else
        EXISTING="unknown"
    fi
    echo "queue caps for $QUEUE: $MAX_JOBS jobs/user, $CAP_NODES nodes/user; $EXISTING already queued, $N_BATCHES to add"
    if [ "$EXISTING" = "unknown" ]; then
        echo "warning: squeue failed, so the jobs-per-user cap could NOT be checked. If this"
        echo "         submission is rejected, that is why -- wait for running jobs or raise"
        echo "         MAX_NODES to pack more runs into fewer jobs."
        EXISTING=0
    fi
    if [ $((EXISTING + N_BATCHES)) -gt "$MAX_JOBS" ]; then
        echo "ERROR: $N_BATCHES batches + $EXISTING already queued exceeds $QUEUE's"
        echo "       $MAX_JOBS-jobs-per-user cap. sbatch REJECTS the overflow rather than"
        echo "       queueing it, so the tail batches would be silently lost. Either wait for"
        echo "       running jobs, or raise MAX_NODES to pack more runs per job:"
        echo "         MAX_NODES=$(( (N_PENDING + TASKS_PER_NODE - 1) / TASKS_PER_NODE )) puts everything in one job."
        exit 1
    fi
    if [ $((N_BATCHES * N_NODES)) -gt "$CAP_NODES" ]; then
        echo "note: $N_BATCHES x $N_NODES = $((N_BATCHES * N_NODES)) nodes exceeds the"
        echo "      $CAP_NODES-nodes-per-user cap, so later batches will sit PENDING until"
        echo "      earlier ones finish. That is queueing, not rejection -- nothing is lost."
    fi
fi

JOBLIST_DIR="$SWEEP_ROOT/joblists"
mkdir -p "$JOBLIST_DIR"
i=0; batch=0
while [ "$i" -lt "$N_PENDING" ]; do
    batch=$((batch + 1))
    LIST="$JOBLIST_DIR/batch_${SWEEP_STAGE}_${batch}.tsv"
    : > "$LIST"
    n=0
    while [ "$i" -lt "$N_PENDING" ] && [ "$n" -lt "$CHUNK" ]; do
        printf '%s\n' "${PENDING[$i]}" >> "$LIST"
        i=$((i + 1)); n=$((n + 1))
    done
    nodes=$(( (n + TASKS_PER_NODE - 1) / TASKS_PER_NODE ))
    echo "  batch $batch: $n run(s) on $nodes node(s) -> $LIST"
    awk -F'\t' '{printf "      %s\n", $1}' "$LIST"
    if [ "$MAX_BATCHES" -gt 0 ] && [ "$batch" -gt "$MAX_BATCHES" ]; then
        echo "      (not submitted: MAX_BATCHES=$MAX_BATCHES)"
        break
    fi
    if [ "$DRY_RUN" = "1" ]; then continue; fi
    jid=$(SWEEP_JOBLIST="$LIST" SWEEP_TASKS_PER_NODE="$TASKS_PER_NODE" \
          SWEEP_STAGES="$STAGES_ENV" \
          submit $A -p "$QUEUE" -t "$TIME" -N "$nodes" \
                 --ntasks-per-node="$TASKS_PER_NODE" --export=ALL --parsable \
                 scripts/tacc/21_sweep_desk.slurm)
    [ -n "$jid" ] || { echo "ERROR: submit failed for batch $batch"; exit 1; }
    echo "      submitted: $jid  (log: houfin_sweep.o$jid)"
done
[ "$DRY_RUN" = "1" ] && echo "DRY RUN: nothing submitted. Re-run with DRY_RUN=0." || true
