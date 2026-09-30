#!/bin/bash
# Submit a stage of the HMC trial (docs/TACC.md §3g). One GPU per job, as for MAP.
#
#   STAGE=probe                                  34_model_hmc_probe.slurm
#   STAGE=vi                                     35_model_vi_from_map.slurm
#   STAGE=hmc ARM=map METRIC=<diag|laplace_fixed|laplace_adapt> CHAINS=2
#   STAGE=hmc ARM=neutra CHAINS=2                36_model_hmc.slurm, array 0..CHAINS-1
#
# RESUBMITS=N chains N more jobs, each afterany the previous (a timeout included), that
# resume from checkpoints -- same pattern as submit_map.sh. AFTER=<jobid> makes the
# first job wait for another to finish OK (e.g. the probe, or VI before ARM=neutra).
# FRESH=1 archives existing output for the first job only.
#
# WHICH MAP: the one age_run_map would write under the current AGE_MODEL_CONFIG overlay
# + HOUFIN_MAP_PROFILE, so export the same pair the MAP was fitted with. Both pass through
# --export=ALL; outputs are keyed on that MAP run's name (e.g. ..._run_17_new_z_quick90).
#   export AGE_MODEL_CONFIG=config/overlays/map_new_z.json HOUFIN_MAP_PROFILE=quick90
#
#   STAGE=probe bash scripts/tacc/submit_hmc.sh
#   STAGE=hmc ARM=map METRIC=laplace_fixed CHAINS=2 RESUBMITS=2 AFTER=<probe jid> bash scripts/tacc/submit_hmc.sh
#   STAGE=vi RESUBMITS=1 bash scripts/tacc/submit_hmc.sh
#   STAGE=hmc ARM=neutra CHAINS=2 RESUBMITS=2 AFTER=<vi jid> bash scripts/tacc/submit_hmc.sh
#   # smoke: HOUFIN_HMC_WARMUP=10 HOUFIN_HMC_SAMPLES=10 HOUFIN_HMC_CHUNK=5 TIME=00:30:00 ...
set -euo pipefail
source "$(dirname "$0")/env.sh"

STAGE="${STAGE:?set STAGE=probe|vi|hmc}"
QUEUE="${QUEUE:-gpu-a100-small}"
TIME="${TIME:-02:00:00}"
RESUBMITS="${RESUBMITS:-0}"
CHAINS="${CHAINS:-2}"
AFTER="${AFTER:-}"
FRESH="${FRESH:-0}"
A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"

case "$STAGE" in
    probe) SCRIPT=scripts/tacc/34_model_hmc_probe.slurm; EXTRA=(); FRESH_VAR=HOUFIN_PROBE_FRESH ;;
    vi)    SCRIPT=scripts/tacc/35_model_vi_from_map.slurm; EXTRA=(); FRESH_VAR=HOUFIN_VI_FRESH ;;
    hmc)
        SCRIPT=scripts/tacc/36_model_hmc.slurm
        EXTRA=(--array="0-$((CHAINS - 1))")
        FRESH_VAR=HOUFIN_HMC_FRESH
        export HOUFIN_HMC_ARM="${ARM:-map}"
        [ -n "${METRIC:-}" ] && export HOUFIN_HMC_METRIC="$METRIC"
        ;;
    *) echo "STAGE must be probe, vi or hmc"; exit 2 ;;
esac

echo "MAP selection: AGE_MODEL_CONFIG=${AGE_MODEL_CONFIG:-<committed config>} HOUFIN_MAP_PROFILE=${HOUFIN_MAP_PROFILE:-standard} HOUFIN_MODEL_PRECISION=${HOUFIN_MODEL_PRECISION:-float32}"
[ -z "${AGE_MODEL_CONFIG:-}" ] && echo "  [warn] no overlay: this targets run_names.map of the COMMITTED config, not an A/B run"

submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+' | tail -1; }

DEP=()
[ -n "$AFTER" ] && DEP=(--dependency=afterok:"$AFTER")
jid=$(submit $A -p "$QUEUE" -t "$TIME" ${EXTRA[@]+"${EXTRA[@]}"} ${DEP[@]+"${DEP[@]}"} \
             --export=ALL,"$FRESH_VAR"="$FRESH" --parsable "$SCRIPT")
[ -n "$jid" ] || { echo "submit failed (no job id captured)"; exit 1; }
echo "submitted $STAGE ${HOUFIN_HMC_ARM:-} ${HOUFIN_HMC_METRIC:-} ($QUEUE, $TIME): $jid"

prev="$jid"
for _ in $(seq 1 "$RESUBMITS"); do
    # For an array, afterany waits for EVERY task; finished chains exit fast on resume.
    nxt=$(submit $A -p "$QUEUE" -t "$TIME" ${EXTRA[@]+"${EXTRA[@]}"} --export=ALL,"$FRESH_VAR"=0 --parsable \
                 --dependency=afterany:"$prev" "$SCRIPT")
    [ -n "$nxt" ] || { echo "chained submit failed"; exit 1; }
    echo "  chained resume job (afterany:$prev): $nxt"
    prev="$nxt"
done
echo "watch: squeue -u \$USER ; compare: python scripts/diagnostics/compare_hmc_trials.py"
