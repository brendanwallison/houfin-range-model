#!/bin/bash
#----------------------------------------------------------------------
# Shared stage library for the houfin pipeline. Two distinct kinds of stage:
#
#   PREPROCESSING (CPU, torch-free) -- one dev/normal-queue job (00_preprocess_all):
#     preprocess climate climate_grid states ebird_cache bbs amplitude
#     This is the default STAGES and produces every input the encoder needs.
#
#   ENCODER (needs torch; ESK/DESK are heavy -> GPU) -- separate job(s)
#   (20_encoder), submitted after preprocessing and typically ONE AT A TIME so
#   ESK and DESK can be sized/queued independently:
#     esk desk cube validate
#
# Select stages with STAGES (space-separated, in order). Runnable standalone on a
# login node for a quick partial stage. bbs/amplitude/validate are for
# bbs_mode=validate (default); for bbs_mode=off drop them (enrich not yet wired).
#     STAGES=esk  bash scripts/tacc/pipeline.sh          # just ESK
#     STAGES="climate_grid states ebird_cache" bash scripts/tacc/pipeline.sh
#----------------------------------------------------------------------
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO/scripts/tacc/env.sh"
cd "$REPO"

# Both roots on the path so data (src.*) and encoder (community_encoder.*) stages
# resolve; run_encoder.py also sets this, but export it for the -m data stages too.
export PYTHONPATH="$REPO:$REPO/src:${PYTHONPATH:-}"
export HOUFIN_PREPROCESS_WORKERS="${HOUFIN_PREPROCESS_WORKERS:-48}"
# climr is parallelized WITHIN one process via nthread (its DuckDB serializes across
# processes). HOUFIN_CLIMATE_WORKERS = R processes (default 1), HOUFIN_CLIMATE_THREADS
# = climr threads/process (default node cpus / workers). Pass through if the caller set them.
export HOUFIN_CLIMATE_WORKERS="${HOUFIN_CLIMATE_WORKERS:-}"
export HOUFIN_CLIMATE_THREADS="${HOUFIN_CLIMATE_THREADS:-}"

# Quiet REMORA's defaults that are irrelevant on a CPU node and flood the log:
# GPU/CUDA monitoring (no GPU here) and the Lustre/network collectors (this node
# type doesn't expose /proc/fs/lustre, so they spam "No such file"). Unknown vars
# are harmless (ignored). CPU + memory sampling stay on.
export REMORA_CUDA=0 REMORA_GPU=0 REMORA_LUSTRE=0 REMORA_NETWORK=0
MON="/usr/bin/time -v"; command -v remora >/dev/null 2>&1 && MON="remora"
DATA="$HOUFIN_DATA"
echo "resource monitor: $MON ; Rscript: ${HOUFIN_RSCRIPT:-Rscript}"

# Guaranteed memory visibility in the job log regardless of REMORA: print node-wide
# memory + load every MEM_HEARTBEAT_SEC (default 120s). REMORA still writes its full
# time series to remora_<jobid>/ and its Max-Memory summary at job end.
mem_heartbeat () {
    while true; do
        sleep "${MEM_HEARTBEAT_SEC:-120}"
        echo "[mem $(date +%H:%M:%S)] $(free -g 2>/dev/null | awk '/Mem:/{print "used "$3"/"$2"G"}')" \
             "load:$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)"
    done
}
mem_heartbeat & _HB_PID=$!
trap 'kill "$_HB_PID" 2>/dev/null || true' EXIT

run () {  # run <stage-label> <command...>
    local s="$1"; shift
    echo "======== [$s] $* ========"; date
    # Drop REMORA's Lustre/lnet probe spam (this node type lacks those /proc paths).
    # The inner `|| true` guards the grep-emptied edge; pipefail still surfaces a
    # real command failure (rightmost non-zero) to the outer `||`.
    $MON "$@" 2>&1 | { grep --line-buffered -vE 'proc/(fs/lustre|sys/lnet)' || true; } \
        || { echo "STAGE FAILED: $s"; exit 1; }
}

stage_preprocess () {
    run ref_grid   python -m src.data.preprocess.build_ref_grid
    run land_mask  python -m src.data.preprocess.land_mask
    run ebird      python scripts/project_ebird
    run luh3       python -m src.data.preprocess.luh3
    run hyde       python -m src.data.preprocess.hyde
    run soilgrids  python -m src.data.preprocess.soilgrids
    run elevation  python -m src.data.preprocess.elevation
    run bbs_finch  python scripts/ingest_bbs_data.py
}
stage_climate () {
    # Centroids default from climate.mode (subgrid -> subcell_centroids.csv, auto-built
    # from the DEM; elev_quantile -> cell_centroids.csv). No --centroids here.
    run climate python scripts/climate_climr.py --out "$DATA/climate" --rscript "$HOUFIN_RSCRIPT"
}
stage_climate_grid () { run climate_grid python -m src.data.preprocess.climate_grid; }
stage_states       () { run states       python -m src.data.combine.build_states \
                            ${HOUFIN_STATES_WORKERS:+--write-workers "$HOUFIN_STATES_WORKERS"}; }
stage_ebird_cache  () { run ebird_cache  python scripts/run_encoder.py ebird-cache; }
stage_bbs () {
    run bbs_crosswalk python -m src.data.identify.bbs_crosswalk \
        --bbs-species "$DATA/bbs_2026_release/SpeciesList.csv"
    run bbs_community python -m src.data.preprocess.bbs_community
}
stage_amplitude () { run amplitude python scripts/run_encoder.py amplitude; }
stage_esk       () { run esk       python scripts/run_encoder.py esk; }
stage_desk      () { run desk      python scripts/run_encoder.py desk; }
stage_cube      () { run cube      python scripts/run_encoder.py cube; }
stage_validate  () { run validate  python scripts/run_encoder.py validate; }
stage_viz       () { run viz       python scripts/viz/quicklook_grids.py --climate; }

# Default = the CPU preprocessing chain only. Encoder stages (esk/desk/cube/
# validate) are opt-in via STAGES from the GPU encoder job.
STAGES="${STAGES:-preprocess climate climate_grid states ebird_cache bbs amplitude}"
echo "STAGES: $STAGES"
for s in $STAGES; do
    if ! declare -F "stage_$s" >/dev/null; then echo "unknown stage: $s"; exit 2; fi
    "stage_$s"
done
echo "======== pipeline complete ($STAGES) ========"; date
