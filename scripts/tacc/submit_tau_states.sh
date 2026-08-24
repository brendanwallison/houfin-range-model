#!/bin/bash
# Build the per-ema_tau yearly_states directories a DESK sweep needs -- one 04_states job per
# tau, read from the sweep manifest.
#
#   DRY_RUN=1 bash scripts/tacc/submit_tau_states.sh          # default: show the plan
#   DRY_RUN=0 bash scripts/tacc/submit_tau_states.sh
#
# WHY THIS EXISTS. ema_tau is consumed at STATE-BUILD time (src/data/combine/streams.py applies
# it along the year axis as the arrays are written), not by DESK, so a tau variant is not a
# config change to a training run -- it is a different covariate dataset. Each one needs its own
# yearly_states build into its own paths.hist_dir, and the sweep's preflight refuses to submit
# until they exist.
#
# Driven by the manifest rather than by three hand-typed overlay paths, for the same reason the
# overlays are generated: one producer of the naming. The manifest already records exactly which
# runs need which states dir (requires_states_dir), so a tau added to the grid is picked up here
# with no second edit -- and a mismatch between "the dir the run will read" and "the dir we
# built" is impossible rather than merely unlikely.
#
# EACH BUILD RE-READS THE SAME SOURCE RASTERS. The three taus differ only in one EMA
# coefficient applied along the year axis; the underlying climate/land-use/HYDE/BUI grids are
# identical. Building them in one pass would be cheaper but would mean teaching run_states to
# write N output dirs from one read, and a states build is well under an hour against the
# sweep's ~57 GPU-hours -- so this pays the I/O three times on purpose. Revisit only if the
# measured wall time makes it matter.
#
# STAGES=states ONLY. 04_states.slurm defaults to the full pre-encoder chain (bbs_trend,
# bbs_abund, ebird_trend, bbs_points, trend_reference), and every one of those products is
# tau-INDEPENDENT -- they are built from community data and never read a covariate state. Running
# them three more times would waste hours and, worse, rewrite the shared training point set
# while sweep runs are reading it.
set -euo pipefail
source "$(dirname "$0")/env.sh"

SWEEP_NAME="${SWEEP_NAME:-desk_hp}"
SWEEP_ROOT="${SWEEP_ROOT:-$HOUFIN_PROCESSED/sweeps/$SWEEP_NAME}"
QUEUE="${QUEUE:-normal}"
TIME="${TIME:-02:00:00}"
DRY_RUN="${DRY_RUN:-1}"
MANIFEST="$SWEEP_ROOT/sweep_manifest.json"

A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"
submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }

cd "$HOUFIN_REPO"
PY="${HOUFIN_VENV}/bin/python"; [ -x "$PY" ] || PY="python"
[ -f "$MANIFEST" ] || { echo "ERROR: no manifest at $MANIFEST -- run submit_sweep.sh (DRY_RUN=1) first"; exit 1; }

# The production states dir, which no tau build may target. A tau build writing there would
# overwrite the covariates every non-tau run in the grid -- and every existing checkpoint -- was
# normalized against, and at tau 2 the arrays would be byte-similar enough that nothing would
# look wrong.
PROD_STATES="$("$PY" -c "from src.config_utils import load_config; print(load_config()['paths']['hist_dir'])")"
echo "production states (protected): $PROD_STATES"

# One overlay per required states dir. A dir is claimed by several runs only if they share a
# tau, so taking the first run that names it is well defined; asserting that is cheaper than
# discovering later that two taus wrote to one directory.
mapfile -t ROWS < <("$PY" - "$MANIFEST" <<'PYS'
import collections, json, os, sys
m = json.load(open(sys.argv[1]))
by = collections.defaultdict(list)
for r in m["runs"]:
    if r["requires_states_dir"]:
        by[os.path.expandvars(r["requires_states_dir"])].append(r)
for d, runs in sorted(by.items()):
    taus = {r["config"] for r in runs}
    if len({t for t in taus}) != 1:
        raise SystemExit(f"ABORT: {d} is claimed by more than one configuration: {sorted(taus)}. "
                         f"Two taus writing to one states dir would leave the second silently "
                         f"overwriting the first.")
    print(f'{runs[0]["config"]}\t{os.path.expandvars(runs[0]["overlay"])}\t{d}')
PYS
)
[ "${#ROWS[@]}" -gt 0 ] || { echo "no per-tau states builds required by this manifest"; exit 0; }

# Disk. A states dir is 86 years of per-year npz over ~295 channels; three of them is not a
# rounding error on $WORK, and running out mid-build leaves a partial dir that the sweep's
# preflight would accept (it checks for the yearly_states directory, not its year count).
AVAIL_GB=$(df -k "$HOUFIN_PROCESSED" 2>/dev/null | awk 'NR==2 {print int($4/1048576)}' || true)
if [ -d "$PROD_STATES/yearly_states" ]; then
    ONE_GB=$(du -sk "$PROD_STATES/yearly_states" 2>/dev/null | awk '{print int($1/1048576)+1}')
    NEED_GB=$(( ONE_GB * ${#ROWS[@]} ))
    echo "disk: ${AVAIL_GB:-?} GB available; ~${NEED_GB} GB needed (${ONE_GB} GB x ${#ROWS[@]} builds)"
    if [ -n "${AVAIL_GB:-}" ] && [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
        echo "ERROR: insufficient space. Drop the tau configurations from the grid"
        echo "       (SWEEP_CONFIGS on submit_sweep.sh) or free space first."
        exit 1
    fi
fi

echo "=== ${#ROWS[@]} states build(s) (DRY_RUN=$DRY_RUN) ==="
for row in "${ROWS[@]}"; do
    tag="$(printf '%s' "$row" | cut -f1)"
    overlay="$(printf '%s' "$row" | cut -f2)"
    outdir="$(printf '%s' "$row" | cut -f3)"
    [ -f "$overlay" ] || { echo "ERROR: overlay missing for $tag ($overlay)"; exit 1; }

    # The overlay's hist_dir must be what the manifest says AND must not be production.
    # Resolved through the same loader the job will use, so this cannot disagree with the run.
    resolved="$(ESK_DESK_CONFIG="$overlay" "$PY" -c \
        "from src.config_utils import load_config; print(load_config()['paths']['hist_dir'])")"
    if [ "$resolved" != "$outdir" ]; then
        echo "ERROR: $tag overlay resolves hist_dir to '$resolved' but the manifest says '$outdir'"
        exit 1
    fi
    if [ "$resolved" = "$PROD_STATES" ]; then
        echo "ERROR: $tag would build into the PRODUCTION states dir ($PROD_STATES)"; exit 1
    fi
    tau="$(ESK_DESK_CONFIG="$overlay" "$PY" -c \
        "from src.config_utils import load_config
c=load_config(); t={s.get('ema_tau') for s in c['states']['streams'] if s.get('ema_tau') is not None}
print(sorted(t)[0] if len(t)==1 else 'MIXED')")"
    [ "$tau" != "MIXED" ] || { echo "ERROR: $tag has more than one ema_tau across its streams"; exit 1; }

    if [ -d "$outdir/yearly_states" ]; then
        echo "  $tag (tau=$tau): already built -> $outdir  [skipping]"
        continue
    fi
    echo "  $tag (tau=$tau) -> $outdir"
    [ "$DRY_RUN" = "1" ] && continue
    mkdir -p "$outdir"
    jid=$(ESK_DESK_CONFIG="$overlay" STAGES=states \
          submit $A -p "$QUEUE" -t "$TIME" --export=ALL --parsable scripts/tacc/04_states.slurm)
    [ -n "$jid" ] || { echo "ERROR: submit failed for $tag"; exit 1; }
    echo "      submitted: $jid  (log: houfin_states.o$jid)"
    echo "      verify:    grep -E 'build_states.*->|TOTAL' houfin_states.o$jid"
done
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN: nothing submitted. Re-run with DRY_RUN=0."
else
    echo "When these finish, re-run: DRY_RUN=1 bash scripts/tacc/submit_sweep.sh"
fi
