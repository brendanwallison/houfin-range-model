#!/bin/bash
#----------------------------------------------------------------------
# Shared stage library for the houfin pipeline. Two distinct kinds of stage:
#
#   PREPROCESSING (CPU, torch-free) -- one dev/normal-queue job (00_preprocess_all):
#     preprocess climate climate_grid states bbs_trend bbs_abund ebird_trend trend_points
#     This is the default STAGES and produces every input the encoder needs for the
#     default trend.anchor_mode=trends-abd path. The trend community must already be
#     selected on a LOGIN node (download_all.sh runs select_trend_community ->
#     community_trend.csv; it needs the eBird trends REST listing).
#     The WEEKLY eBird stages ('ebird' = project_ebird, 'ebird_cache') are OPT-IN: they are
#     only needed for the legacy trend.anchor_mode=weekly anchor and for bbs_mode=off/validate.
#     trends-abd reconstructs from the trends 'abd' raster (ebird_trend stage), so neither
#     the weekly download nor these two stages are required. Add "ebird ebird_cache" to
#     STAGES to run them.
#
#   ENCODER (needs torch; ESK/DESK are heavy -> GPU) -- separate job(s)
#   (20_encoder), submitted after preprocessing and typically ONE AT A TIME so
#   ESK and DESK can be sized/queued independently:
#     spacetime-esk desk cube validate encoder-viz
#
# Select stages with STAGES (space-separated, in order). Runnable standalone on a
# login node for a quick partial stage. bbs_trend/ebird_trend/trend_points are the
# bbs_mode=trend community path; for bbs_mode=off drop them.
#     STAGES=spacetime-esk bash scripts/tacc/pipeline.sh   # just the joint ESK
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
# Climate runs one R process PER GEOGRAPHIC TILE via a process pool (offline, reading
# the warm cache). HOUFIN_CLIMATE_WORKERS = concurrent tile processes (climatena.py
# default 4; 02_climate.slurm sets 16, cap ~96 to fill a node -- raise it, don't keep
# it at 1); HOUFIN_CLIMATE_THREADS = climr threads/process (default node cpus/workers).
export HOUFIN_CLIMATE_WORKERS="${HOUFIN_CLIMATE_WORKERS:-16}"   # 16 tiles x nthread 8 on a 128-core node
export HOUFIN_CLIMATE_THREADS="${HOUFIN_CLIMATE_THREADS:-}"

# REMORA collectors. Enable GPU/CUDA monitoring only when a GPU is actually present
# (the encoder job on gpu-a100-dev -- where we want util data); keep it off on the CPU
# preprocess nodes, where it's irrelevant and floods the log. Lustre/network stay off
# (this node type doesn't expose /proc/fs/lustre, so they spam "No such file").
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    export REMORA_CUDA=1 REMORA_GPU=1
else
    export REMORA_CUDA=0 REMORA_GPU=0
fi
export REMORA_LUSTRE=0 REMORA_NETWORK=0
MON="/usr/bin/time -v"; command -v remora >/dev/null 2>&1 && MON="remora"
DATA="$HOUFIN_DATA"
echo "resource monitor: $MON ; Rscript: ${HOUFIN_RSCRIPT:-Rscript}"

# Per-stage product path(s) for OPTIONAL validation (a case, not an assoc array, so
# it runs on any bash). Only stages with an entry are validated, and only when
# HOUFIN_VALIDATE is set (the split 01_preprocess/02_climate jobs set it; the
# 00/04/20 wrappers don't). This is the single source of truth for both the stage
# sequences AND their validation targets.
_vpath () {
    case "$1" in
        ref_grid)  echo "$DATA/ref_grid_27km.tif" ;;
        land_mask) echo "$DATA/land_mask" ;;
        ebird)     echo "$DATA/ebird_weekly_2023_grid" ;;
        luh3)      echo "$DATA/luh3_grid" ;;
        hyde)      echo "$DATA/hyde35_grid" ;;
        bui)       echo "$DATA/bui_grid" ;;
        bbs_points) echo "$HOUFIN_PROCESSED/encoder/bbs_points/X_points.npy" ;;
        soilgrids) echo "$DATA/soilgrids_grid" ;;
        elevation) echo "$DATA/elevation" ;;
        subcell)   echo "$DATA/elevation/subcell_centroids.csv" ;;
        bbs_finch) echo "$DATA/bbs_2026_release/bbs_data_for_python.npz" ;;
        climate)   echo "$DATA/climate" ;;
    esac
}

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
    # REMORA (the monitor) returns 0 regardless of what it wrapped, so its exit code
    # cannot be trusted to detect a stage failure -- a crashed stage would read as
    # success. Capture the REAL command exit status via an inner shell that writes $?
    # to a file, so a partial/crashed stage fails loudly no matter what MON reports.
    # The trailing grep only cleans REMORA's Lustre/lnet probe spam from the log.
    local _rc; _rc="$(mktemp)"
    $MON bash -c 'set -o pipefail; "$@"; echo $? > "$0"' "$_rc" "$@" \
        2>&1 | { grep --line-buffered -vE 'proc/(fs/lustre|sys/lnet)' || true; }
    local _code; _code="$(cat "$_rc" 2>/dev/null || echo 1)"; rm -f "$_rc"
    [ "$_code" = 0 ] || { echo "STAGE FAILED: $s (exit $_code)"; exit 1; }
    if [ -n "${HOUFIN_VALIDATE:-}" ]; then
        local vp; vp="$(_vpath "$s")"
        if [ -n "$vp" ]; then
            python scripts/validate_products.py --stage "$s" --paths $vp \
                || { echo "VALIDATION FAILED: $s"; exit 1; }
        fi
    fi
}

# Individual preprocess sub-stages, each independently selectable via STAGES (e.g.
# STAGES=ebird to re-project just eBird on a compute node). stage_preprocess chains them.
stage_ref_grid  () { run ref_grid   python -m src.data.preprocess.build_ref_grid; }
stage_land_mask () { run land_mask  python -m src.data.preprocess.land_mask; }
stage_ebird     () { run ebird      python scripts/project_ebird; }
stage_luh3      () { run luh3       python -m src.data.preprocess.luh3; }
stage_hyde      () { run hyde       python -m src.data.preprocess.hyde; }
# BUI needs land_mask's Natural Earth admin-0 polygon (read as a CONUS *inclusion*), so it
# runs after it in the bundle below. It is CPU/memory-bound, not I/O-bound: one ~174 MB
# float64 fine grid per snapshot, reprojected then quantiled per model cell.
stage_bui       () { run bui        python -m src.data.preprocess.bui; }
stage_soilgrids () { run soilgrids  python -m src.data.preprocess.soilgrids; }
stage_elevation () { run elevation  python -m src.data.preprocess.elevation; }
stage_subcell   () { run subcell    python -m src.data.preprocess.subcell_centroids; }
stage_bbs_finch () { run bbs_finch  python scripts/ingest_bbs_data.py; }
stage_preprocess () {
    # NOTE: 'ebird' (project_ebird, weekly status grids) is NOT in this bundle -- it is
    # opt-in (weekly anchor / off / validate only). Run it explicitly: STAGES="... ebird ...".
    stage_ref_grid; stage_land_mask; stage_luh3; stage_hyde; stage_bui
    stage_soilgrids; stage_elevation; stage_subcell; stage_bbs_finch
}
stage_climate () {
    # Centroids default from climate.mode (subgrid -> subcell_centroids.csv, auto-built
    # from the DEM; elev_quantile -> cell_centroids.csv). No --centroids here.
    run climate python scripts/climate_climr.py --out "$DATA/climate" --rscript "$HOUFIN_RSCRIPT"
}
stage_climate_grid () { run climate_grid python -m src.data.preprocess.climate_grid; }
stage_states () {
    # vm-small / shared nodes can trade wall time for lower transient I/O and
    # compression memory: HOUFIN_STATES_READ_WORKERS=4 HOUFIN_STATES_WORKERS=1.
    #
    # MEMORY, since monthly climate: the training bag is
    # samples_per_year x n_years x n_channels float32, accumulated as a list and
    # then np.concatenate'd (so peak is ~2x the final array). At ~250 channels the
    # 20k default is ~3 GB final / ~6 GB peak. Lower it via
    # HOUFIN_STATES_SAMPLES (6000 is ample: the bag only fits per-channel mu/sd
    # and the reconstruction term, not the spatial model).
    local args=()
    [ -n "${HOUFIN_STATES_READ_WORKERS:-}" ] && args+=(--read-workers "$HOUFIN_STATES_READ_WORKERS")
    [ -n "${HOUFIN_STATES_WORKERS:-}" ] && args+=(--write-workers "$HOUFIN_STATES_WORKERS")
    [ -n "${HOUFIN_STATES_SAMPLES:-}" ] && args+=(--samples-per-year "$HOUFIN_STATES_SAMPLES")
    run states python -m src.data.combine.build_states "${args[@]}"
}
stage_ebird_cache  () { run ebird_cache  python scripts/run_encoder.py ebird-cache; }
stage_bbs () {
    # bbs_crosswalk writes a standalone AOU<->eBird crosswalk CSV + match diagnostics;
    # bbs_community is self-sufficient (recomputes the crosswalk in-process) and writes
    # the gridded community_matrix the amplitude cube consumes. The crosswalk stage is
    # kept for its printed match report; its CSV is diagnostic (not read downstream).
    run bbs_crosswalk python -m src.data.identify.bbs_crosswalk \
        --bbs-species "$DATA/bbs_2026_release/SpeciesList.csv"
    run bbs_community python -m src.data.preprocess.bbs_community
}
# Trend-product community path (replaces the deprecated amplitude path). select_community
# needs the eBird trends REST listing (internet) so it runs on a LOGIN node (download_all.sh),
# NOT here; the align + point-build steps below are CPU preprocessing on a compute node.
stage_bbs_trend        () { run bbs_trend        python -m src.data.preprocess.bbs_trend; }
stage_bbs_abund        () { run bbs_abund        python -m src.data.preprocess.bbs_abund; }
stage_ebird_trend      () { run ebird_trend      python -m src.data.preprocess.ebird_trend; }
stage_select_community () { run select_community python -m src.data.identify.select_trend_community; }
stage_trend_points     () { run trend_points     python scripts/run_encoder.py trend-points; }
# The LIVE training target: raw route-level BBS community counts plus the eBird trend product
# inside its own window, put on one scale by an explicit calibration. Replaces trend_points as
# the thing the encoder learns from -- trend_points now only builds the sanity-check reference.
# CPU; reads the BBS release and the eBird trend grid.
stage_bbs_points       () { run bbs_points       python scripts/run_encoder.py bbs-points; }
# The reference for validate_reference: the same trend products WITHOUT our spatial blur, in a
# separate points_dir so it and the training target coexist. CPU.
stage_trend_reference  () { run trend_reference  python scripts/run_encoder.py trend-reference; }
stage_esk       () { run esk       python scripts/run_encoder.py esk; }
stage_spacetime_esk () { run spacetime_esk python scripts/run_encoder.py spacetime-esk; }
stage_desk      () { run desk      python scripts/run_encoder.py desk; }
# Targets-only diagnostic: the bars a DESK run has to clear. No GPU, no checkpoint.
stage_desk_baselines () { run desk_baselines python scripts/run_encoder.py desk-baselines; }
# Multi-epoch direction panel: model vs inverse-distance, per epoch pair, plus curvature.
# Needs the saved DESK checkpoint (encodes the year span), so run it on the GPU queue.
stage_validate_epochs () { run validate_epochs python scripts/run_encoder.py validate-epochs; }
stage_cube      () { run cube      python scripts/run_encoder.py cube; }
stage_validate  () { run validate  python scripts/run_encoder.py validate; }
# Route-level BBS validation. Separate from `validate` because it answers a different question:
# `validate` grades against the IDW-interpolated BBS trend surface, this grades against raw route
# counts at surveyed cell-years only, where reproducing an interpolator cannot win. GPU queue.
stage_bbs_route_validate () { run bbs_route_validate python scripts/run_encoder.py bbs-route-validate; }
# Five comparisons against the trend products as a REFERENCE rather than as truth: temporal
# direction and rank (which use the products' direction and ordering, not their magnitudes),
# per-species trend sign, and the full and spatial similarity structures. Reported separately
# because the reference is smoothed spatially and closed-form temporally, so a poor spatial
# score is partly the reference. GPU queue: one encode pass over the EMA span.
stage_validate_reference () { run validate_reference python scripts/run_encoder.py validate-reference; }
stage_encoder_viz () {
    # Always reconstruct the comparison arrays in a submitted diagnostics run.
    # Direct invocations may reuse the provenance-checked cache.
    run encoder_viz python scripts/viz/encoder_diagnostics.py --recompute-projection
}
stage_path_features () { run path_features python src/processing/generate_all_path_features.py; }
stage_model_ingest () { run model_ingest python scripts/ingest_model_data.py; }
stage_viz       () { run viz       python scripts/viz/quicklook_grids.py --climate; }

# NOTE on the default list: bbs_points (raw BBS + the eBird window) is the TRAINING TARGET and
# trend_reference is the sanity-check reference. `trend_points` -- the old trend-product target --
# is deliberately NOT in the default any more; run it explicitly only to reproduce the retired
# target for an A/B (and point target.points_dir at its output to train against it).
#
# Default = the CPU preprocessing chain only. Encoder stages (esk/desk/cube/
# validate) are opt-in via STAGES from the GPU encoder job.
STAGES="${STAGES:-preprocess climate climate_grid states bbs_trend bbs_abund ebird_trend bbs_points trend_reference}"
echo "STAGES: $STAGES"
for s in $STAGES; do
    fn="stage_${s//-/_}"      # accept either 'spacetime-esk' or 'spacetime_esk' (bash fn names can't have '-')
    if ! declare -F "$fn" >/dev/null; then echo "unknown stage: $s"; exit 2; fi
    "$fn"
done
echo "======== pipeline complete ($STAGES) ========"; date
