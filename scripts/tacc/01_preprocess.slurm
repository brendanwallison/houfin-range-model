#!/bin/bash
#----------------------------------------------------------------------
# 25 km preprocessing for the houfin pipeline (CPU-only; no internet needed).
# Reads raw data staged by download_all.sh under $HOUFIN_DATA; writes 25 km
# products under $HOUFIN_DATA and a per-stage JSON manifest under
# $HOUFIN_PROCESSED/validation. Submit AFTER downloads finish:
#     sbatch scripts/tacc/01_preprocess.slurm
# These stages are RAM-light (< ~1 GB each, validated locally); vm-small also
# works and is cheaper. LUH-3 is the long pole (many var-years).
#----------------------------------------------------------------------
#SBATCH -J houfin_prep
#SBATCH -o houfin_prep.o%j
#SBATCH -e houfin_prep.e%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH --mail-type=all
##SBATCH -A REPLACE_WITH_PROJECT     # uncomment + set, or rely on env.sh default
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
module load python/3.12.11 2>/dev/null || true
module load remora 2>/dev/null || true        # periodic CPU/mem/IO sampling per stage
[ -n "${TACC_ALLOCATION:-}" ] && echo "allocation: $TACC_ALLOCATION"
module list 2>&1 || true; date

# The reproject stages parallelize across the whole node (one multiprocessing
# program per node -- -N 1 -n 1 is correct; the program uses all cores). Cap RAM
# conservatively; raise toward $SLURM_CPUS_ON_NODE (128 on a normal node) once
# remora shows memory headroom.
export HOUFIN_PREPROCESS_WORKERS="${HOUFIN_PREPROCESS_WORKERS:-48}"

# Monitor: remora samples resource use over time (live, in remora_<jobid>/);
# fall back to /usr/bin/time -v (end-of-stage summary only) if remora is absent.
MON="/usr/bin/time -v"
command -v remora >/dev/null 2>&1 && MON="remora"
echo "resource monitor: $MON ; eBird workers: $HOUFIN_PREPROCESS_WORKERS"

run () {  # run() <stage-name> <paths-to-validate> -- <command...>
    local stage="$1"; local vpaths="$2"; shift 3
    echo "=== [$stage] $* ==="
    $MON "$@" 2>&1 || { echo "STAGE FAILED: $stage"; exit 1; }
    python scripts/validate_products.py --stage "$stage" --paths $vpaths
}

run ref_grid   "$HOUFIN_DATA/ref_grid_25km.tif"  -- python -m src.data.preprocess.build_ref_grid
run land_mask  "$HOUFIN_DATA/land_mask"          -- python -m src.data.preprocess.land_mask
run ebird      "$HOUFIN_DATA/ebird_weekly_2023_grid" -- python scripts/project_ebird
run luh3       "$HOUFIN_DATA/luh3_grid"          -- python -m src.data.preprocess.luh3
run hyde       "$HOUFIN_DATA/hyde35_grid"        -- python -m src.data.preprocess.hyde
run soilgrids  "$HOUFIN_DATA/soilgrids_grid"     -- python -m src.data.preprocess.soilgrids
run elevation  "$HOUFIN_DATA/elevation"          -- python -m src.data.preprocess.elevation
run bbs        "$HOUFIN_DATA/bbs_2026_release/bbs_data_for_python.npz" -- python scripts/ingest_bbs_data.py

echo "=== preprocessing complete. Manifests: $HOUFIN_PROCESSED/validation/ ==="
date
