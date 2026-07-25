#!/bin/bash
# Sensitivity sweep over juvenile mean dispersal distance.
#
# For each MDD point this submits a two-job chain -- path-features + ingest
# (25_model_prep) -> MAP fit (30_model_map) -- with every stage pointed at a
# per-point OVERLAY CONFIG. Points are independent chains, so the scheduler
# overlaps them. Diagnostics for ALL points then run in a SINGLE trailing job
# (32_sweep_viz), which also writes the cross-point summary.
#
#   DRY_RUN=1 bash scripts/tacc/submit_juv_mdd_sweep.sh    # default: write overlays, print sbatch lines
#   DRY_RUN=0 bash scripts/tacc/submit_juv_mdd_sweep.sh    # actually submit
#   MDD_POINTS="250 330" DRY_RUN=0 bash scripts/tacc/submit_juv_mdd_sweep.sh
#
# WHY OVERLAYS AND NOT HOUFIN_PROCESSED: env.sh exports HOUFIN_DATA/HOUFIN_PROCESSED
# UNCONDITIONALLY, so a value set in the submitting shell is silently discarded by
# every job. Isolation therefore has to live inside the config the job loads, via
# $AGE_MODEL_CONFIG (deep-merged over the committed config by src/config_utils.py).
# Do not try to isolate a sweep point with environment roots -- it will look like
# it worked and every point will write to the same directory.
#
# Each overlay must override FOUR path keys or points collide destructively:
#   path_features.output_dir  where Z_disp_{year}.npz is written
#   raw_z_dir                 where ingest READS them (must equal the above)
#   input_dir                 memmaps + metadata.pkl; the .dat files are uuid-named
#                             but metadata.pkl is NOT, so a shared input_dir means
#                             the last point silently owns the pointer
#   run_names.map             fit + viz output dir (both derive from this one key)
set -euo pipefail
source "$(dirname "$0")/env.sh"

MDD_POINTS="${MDD_POINTS:-200 250 300 330 350}"
SWEEP_NAME="${SWEEP_NAME:-juv_mdd}"
SWEEP_ROOT="${SWEEP_ROOT:-$HOUFIN_PROCESSED/sweeps/$SWEEP_NAME}"
PROFILE="${HOUFIN_MAP_PROFILE:-quick90}"
PRECISION="${HOUFIN_MODEL_PRECISION:-float32}"
# Everything stays on gpu-a100-small: every job is -N 1 -n 1 on ONE GPU, and a
# gpu-a100 node carries 3 A100s at 3 SU/hr, so putting a single-GPU job there pays
# double for a third of a node. small is 1.5 SU/hr and less contended.
# It caps submitted jobs per user at 12, which is why diagnostics for ALL points
# run in ONE job (32_sweep_viz.slurm) instead of one per point: 5 preps + 5 fits +
# 1 viz = 11 jobs for a 5-point sweep. Per-point viz would be 15 and overflow onto
# the main queue.
PREP_QUEUE="${PREP_QUEUE:-gpu-a100-small}"
PREP_TIME="${PREP_TIME:-02:00:00}"
MAP_QUEUE="${MAP_QUEUE:-gpu-a100-small}"
MAP_TIME="${MAP_TIME:-02:00:00}"
VIZ_QUEUE="${VIZ_QUEUE:-gpu-a100-small}"
VIZ_TIME="${VIZ_TIME:-02:00:00}"
MAP_RESUBMITS="${MAP_RESUBMITS:-0}"
MAX_CONCURRENT_POINTS="${MAX_CONCURRENT_POINTS:-3}"
DRY_RUN="${DRY_RUN:-1}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"

A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"
submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }

cd "$HOUFIN_REPO"
PY="${HOUFIN_VENV}/bin/python"
[ -x "$PY" ] || PY="python"

# ---------------------------------------------------------------- preflight
# All of this is seconds of CPU. Each sweep point is hours of GPU, so every
# failure worth catching should be caught here.
echo "=== preflight ==="
GIT_SHA="$(git rev-parse HEAD)"
# Only TRACKED changes under the code/config paths count. A plain
# `git status --porcelain` also lists untracked files, and SLURM drops job logs
# (houfin_*.o*) plus 30-second telemetry (gpu_*.csv) straight into this directory,
# which would make every post-run tree look dirty. What actually matters is that
# _run_fingerprint hashes source files, so editing one mid-sweep makes points
# incomparable and blocks chained resumes.
DIRTY="$(git status --porcelain --untracked-files=no -- src scripts tests config)"
if [ -n "$DIRTY" ] && [ "$ALLOW_DIRTY" != "1" ]; then
    echo "ERROR: tracked source/config changes are uncommitted:"
    echo "$DIRTY"
    echo "The MAP fingerprint hashes source files, so editing them mid-sweep makes"
    echo "points incomparable and blocks chained resumes. Commit/stash first, or set"
    echo "ALLOW_DIRTY=1 if you know the edits are inert."
    exit 1
fi
echo "git sha: $GIT_SHA"

CUBE_DIR="$("$PY" -c "from src.config_utils import load_age_model_config as c; print(c()['path_features']['input_dir'])")"
N_CUBE=$(ls "$CUBE_DIR"/Z_latent_*.npy 2>/dev/null | wc -l | tr -d ' ')
echo "encoder cube: $CUBE_DIR ($N_CUBE years)"
[ "$N_CUBE" -ge 1 ] || { echo "ERROR: no Z_latent_*.npy in $CUBE_DIR"; exit 1; }

# The committed defaults every overlay must move away from.
BASE_RAW_Z="$("$PY" -c "from src.config_utils import load_age_model_config as c; print(c()['raw_z_dir'])")"
BASE_INPUT="$("$PY" -c "from src.config_utils import load_age_model_config as c; print(c()['input_dir'])")"
echo "production dirs to protect: $BASE_RAW_Z | $BASE_INPUT"

N_POINTS=$(echo "$MDD_POINTS" | wc -w | tr -d ' ')
NEED_GB=$((N_POINTS * 14 + 20))
# -k is POSIX; -BG/--output are GNU-only and silently produce nothing on BSD/macOS.
AVAIL_GB=$(df -k "$HOUFIN_PROCESSED" 2>/dev/null | awk 'NR==2 {print int($4/1048576)}' || true)
if [ -n "${AVAIL_GB:-}" ]; then
    echo "disk: ${AVAIL_GB} GB available, ~${NEED_GB} GB needed for $N_POINTS points"
    [ "$AVAIL_GB" -ge "$NEED_GB" ] || { echo "ERROR: insufficient space"; exit 1; }
else
    echo "disk: could not determine free space; ~${NEED_GB} GB needed for $N_POINTS points"
fi

mkdir -p "$SWEEP_ROOT"
MANIFEST="$SWEEP_ROOT/sweep_manifest.json"

# ------------------------------------------------------- overlays + manifest
# Generated by python so the JSON is always valid and the resolved splits are
# recorded (and validated) by the same resolver the jobs will use.
echo "=== writing overlays ==="
"$PY" - "$SWEEP_ROOT" "$SWEEP_NAME" "$PROFILE" "$PRECISION" "$GIT_SHA" "$BASE_RAW_Z" "$BASE_INPUT" $MDD_POINTS <<'PYGEN'
import copy, datetime, json, os, sys

from src.config_utils import load_age_model_config
# src.model.dispersal_spec, NOT src.model.build_kernels: this runs on a LOGIN NODE,
# and build_kernels imports JAX, whose CPU-backend init aborts there
# (make_cpu_client). Same single resolver, no JAX.
from src.model.dispersal_spec import dispersal_spec

root, name, profile, precision, sha, base_raw_z, base_input = sys.argv[1:8]
points = [float(x) for x in sys.argv[8:]]
base = load_age_model_config()
cell_km = 27.0  # only used for the resolution note below
entries = []

for mdd in points:
    tag = f"mdd{int(round(mdd))}"
    # ${HOUFIN_PROCESSED} is left literal: config_utils expands it at load time,
    # inside the job, where env.sh has already fixed the root.
    pdir = f"${{HOUFIN_PROCESSED}}/sweeps/{name}/{tag}"
    overlay = {
        "_sweep": {"point": tag, "sweep": name, "juvenile_mdd_km": mdd,
                   "git_sha": sha, "profile": profile, "precision": precision,
                   "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
        "dispersal": {"juvenile_mdd_km": mdd, "juvenile_radial_splits_km": "derive"},
        "path_features": {"output_dir": f"{pdir}/latent_avian_paths"},
        "raw_z_dir": f"{pdir}/latent_avian_paths",
        "input_dir": f"{pdir}/numpyro_input",
        "path_diagnostics_dir": f"{pdir}/path_diagnostics",
        "run_names": {"map": f"sweep_{name}_{tag}_age_map_{{precision}}"},
    }
    # Resolve exactly as the jobs will, to (a) surface the splits now and (b) fail
    # here rather than two GPU-hours in.
    merged = copy.deepcopy(base)
    merged["dispersal"].update(overlay["dispersal"])
    spec = dispersal_spec(merged)
    splits = spec["juvenile_radial_splits_km"]

    real_dir = os.path.join(os.environ["HOUFIN_PROCESSED"], "sweeps", name, tag)
    for key, val in (("raw_z_dir", real_dir + "/latent_avian_paths"),
                     ("input_dir", real_dir + "/numpyro_input")):
        if val in (base_raw_z, base_input):
            raise SystemExit(f"ABORT: {tag} {key} resolves to a production dir ({val})")
    os.makedirs(real_dir, exist_ok=True)
    path = os.path.join(real_dir, "config.json")
    with open(path, "w") as fh:
        json.dump(overlay, fh, indent=2)

    run_dir = f"sweep_{name}_{tag}_age_map_{precision}"
    if profile != "standard":
        run_dir = f"{run_dir}_{profile}"
    flag = "  <-- inner cohort under-resolved" if 0 < splits[1] < 2 * cell_km else ""
    print(f"  {tag}: splits={[round(s,1) for s in splits[:-1]]}+inf  run={run_dir}{flag}")
    entries.append({"point": tag, "juvenile_mdd_km": mdd, "overlay": path,
                    "resolved_splits_km": splits, "run_dir": run_dir,
                    "path_features_dir": real_dir + "/latent_avian_paths",
                    "input_dir": real_dir + "/numpyro_input",
                    "git_sha": sha, "profile": profile, "precision": precision})

with open(os.path.join(root, "sweep_manifest.json"), "w") as fh:
    json.dump({"sweep": name, "git_sha": sha, "profile": profile,
               "precision": precision, "points": entries}, fh, indent=2)
print(f"  manifest -> {os.path.join(root, 'sweep_manifest.json')}")
PYGEN

# ------------------------------------------------------------------- submit
echo "=== submitting (DRY_RUN=$DRY_RUN) ==="
# Read tags/overlays back from the manifest rather than re-deriving them here: bash
# ${mdd%.*} truncates where python's int(round(mdd)) rounds, so a non-integer point
# (237.5) would silently submit against a nonexistent overlay path. One producer of
# the naming, one consumer.
POINT_ROWS=()
while IFS= read -r line; do
    [ -n "$line" ] && POINT_ROWS+=("$line")
done < <("$PY" -c "
import json
m=json.load(open('$MANIFEST'))
for p in m['points']:
    print(p['point'], p['overlay'], sep='\t')
")
[ "${#POINT_ROWS[@]}" -eq "$N_POINTS" ] || { echo "ERROR: manifest has ${#POINT_ROWS[@]} points, expected $N_POINTS"; exit 1; }

i=0
declare -a PREP_IDS=()
declare -a MAP_IDS=()
for row in "${POINT_ROWS[@]}"; do
    tag="${row%%$'\t'*}"
    overlay="${row#*$'\t'}"
    [ -f "$overlay" ] || { echo "ERROR: overlay missing for $tag ($overlay)"; exit 1; }

    # Throttle: point k waits on point (k - MAX_CONCURRENT_POINTS)'s prep. afterany
    # here because this is a queue-pressure gate, not a correctness dependency.
    gate=""
    if [ "$MAX_CONCURRENT_POINTS" -gt 0 ] && [ "$i" -ge "$MAX_CONCURRENT_POINTS" ]; then
        prev_idx=$((i - MAX_CONCURRENT_POINTS))
        [ -n "${PREP_IDS[$prev_idx]:-}" ] && gate="--dependency=afterany:${PREP_IDS[$prev_idx]}"
    fi

    # Per-job settings travel as ENV PREFIXES, not inside --export=. STAGES's value
    # contains a space ("path-features model-ingest") and --export takes a
    # comma-separated list, so embedding it there is fragile; --export=ALL already
    # propagates the submitting environment, which is how submit_model_prep.sh
    # passes STAGES too.
    prep_args=($A -p "$PREP_QUEUE" -t "$PREP_TIME" -J "${tag}_prep" $gate
               --export=ALL --parsable scripts/tacc/25_model_prep.slurm)
    # afterok, NOT afterany: a failed ingest must never launch a 90-minute fit on
    # stale or absent inputs.
    map_args=($A -p "$MAP_QUEUE" -t "$MAP_TIME" -J "${tag}_map"
              --export=ALL --parsable scripts/tacc/30_model_map.slurm)

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [$tag] AGE_MODEL_CONFIG=$overlay STAGES='path-features model-ingest' \\"
        echo "         sbatch ${prep_args[*]}"
        echo "  [$tag] HOUFIN_MAP_PROFILE=$PROFILE HOUFIN_MAP_FRESH=1 \\"
        echo "         sbatch --dependency=afterok:<prep> ${map_args[*]}"
        PREP_IDS+=("dry$i")
        MAP_IDS+=("dry_map$i")
    else
        prep=$(AGE_MODEL_CONFIG="$overlay" STAGES="path-features model-ingest" \
               submit "${prep_args[@]}")
        [ -n "$prep" ] || { echo "ERROR: prep submit failed for $tag"; exit 1; }
        map=$(AGE_MODEL_CONFIG="$overlay" HOUFIN_MAP_PROFILE="$PROFILE" HOUFIN_MAP_FRESH=1 \
              submit --dependency=afterok:"$prep" "${map_args[@]}")
        [ -n "$map" ] || { echo "ERROR: map submit failed for $tag"; exit 1; }
        last="$map"
        for _ in $(seq 1 "$MAP_RESUBMITS"); do
            # Within the fit chain, afterany + FRESH=0: a wall-clock kill is the
            # expected exit and the next window resumes the checkpoint.
            last=$(AGE_MODEL_CONFIG="$overlay" HOUFIN_MAP_PROFILE="$PROFILE" HOUFIN_MAP_FRESH=0 \
                   submit --dependency=afterany:"$last" "${map_args[@]}")
            [ -n "$last" ] || { echo "ERROR: chained map submit failed for $tag"; exit 1; }
        done
        echo "  [$tag] prep=$prep map=$map"
        PREP_IDS+=("$prep")
        MAP_IDS+=("$last")
    fi
    i=$((i + 1))
done

# ONE diagnostics job for the whole sweep (see 32_sweep_viz.slurm): it loops over
# every point's overlay, then writes the cross-point summary. afterok on ALL fits
# would skip diagnostics entirely if a single point failed, so gate with afterany
# and let the per-point loop and the summary mark stragglers excluded.
viz_dep="$(IFS=:; echo "afterany:${MAP_IDS[*]}")"
viz_args=($A -p "$VIZ_QUEUE" -t "$VIZ_TIME" -J "sweep_viz"
          --export=ALL --parsable scripts/tacc/32_sweep_viz.slurm)
if [ "$DRY_RUN" = "1" ]; then
    echo "  [all]    SWEEP_MANIFEST=$MANIFEST HOUFIN_MAP_PROFILE=$PROFILE \\"
    echo "           sbatch --dependency=$viz_dep ${viz_args[*]}"
else
    viz=$(SWEEP_MANIFEST="$MANIFEST" HOUFIN_MAP_PROFILE="$PROFILE" \
          submit --dependency="$viz_dep" "${viz_args[@]}")
    [ -n "$viz" ] || { echo "ERROR: sweep viz submit failed"; exit 1; }
    echo "  [all]    viz=$viz (all points + summary, after all fits)"
fi

echo
echo "manifest: $MANIFEST"
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN -- nothing submitted. Re-run with DRY_RUN=0 to launch."
else
    echo "watch:     squeue -u \$USER"
    echo "aggregate: python scripts/viz/juv_mdd_sweep_summary.py --manifest $MANIFEST"
fi
