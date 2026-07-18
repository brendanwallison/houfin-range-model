#!/bin/bash
#----------------------------------------------------------------------
# Full off/validate pipeline as selectable stages, sized for the 2-hour dev queue.
# Sourced/run by 00_pipeline.slurm; also runnable standalone on a login node for
# partial runs. Select stages with STAGES (space-separated, in order); default is
# the whole chain. Skip already-done stages, e.g. re-run only the encoder:
#     STAGES="ebird_cache esk desk cube validate" bash scripts/tacc/pipeline.sh
#
# bbs_mode: the bbs/amplitude/validate stages are only meaningful for
# bbs_mode=validate (default). For bbs_mode=off, drop them:
#     STAGES="preprocess climate climate_grid states ebird_cache esk desk cube"
# (enrich adds run_spacetime_esk + a time-aware DESK — not yet wired.)
#----------------------------------------------------------------------
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO/scripts/tacc/env.sh"
cd "$REPO"

# Both roots on the path so data (src.*) and encoder (community_encoder.*) stages
# resolve; run_encoder.py also sets this, but export it for the -m data stages too.
export PYTHONPATH="$REPO:$REPO/src:${PYTHONPATH:-}"
export HOUFIN_PREPROCESS_WORKERS="${HOUFIN_PREPROCESS_WORKERS:-48}"
export HOUFIN_CLIMATE_WORKERS="${HOUFIN_CLIMATE_WORKERS:-32}"

MON="/usr/bin/time -v"; command -v remora >/dev/null 2>&1 && MON="remora"
DATA="$HOUFIN_DATA"
echo "resource monitor: $MON ; Rscript: ${HOUFIN_RSCRIPT:-Rscript}"

run () {  # run <stage-label> <command...>
    local s="$1"; shift
    echo "======== [$s] $* ========"; date
    $MON "$@" 2>&1 || { echo "STAGE FAILED: $s"; exit 1; }
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
    run climate python scripts/climate_climr.py \
        --centroids "$DATA/elevation/cell_centroids.csv" --out "$DATA/climate" \
        --rscript "$HOUFIN_RSCRIPT"
}
stage_climate_grid () { run climate_grid python -m src.data.preprocess.climate_grid; }
stage_states       () { run states       python -m src.data.combine.build_states; }
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

STAGES="${STAGES:-preprocess climate climate_grid states ebird_cache bbs amplitude esk desk cube validate}"
echo "STAGES: $STAGES"
for s in $STAGES; do
    if ! declare -F "stage_$s" >/dev/null; then echo "unknown stage: $s"; exit 2; fi
    "stage_$s"
done
echo "======== pipeline complete ($STAGES) ========"; date
