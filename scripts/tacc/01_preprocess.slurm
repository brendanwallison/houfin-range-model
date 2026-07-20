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
module load remora 2>/dev/null || true
[ -n "${TACC_ALLOCATION:-}" ] && echo "allocation: $TACC_ALLOCATION"
date

# Thin wrapper over pipeline.sh (the single source of truth for stage sequences).
# Runs the `preprocess` stage group (ref_grid -> land_mask -> eBird -> LUH-3 -> HYDE
# -> SoilGrids -> elevation -> subcell -> BBS ingest) with per-stage product
# validation (HOUFIN_VALIDATE=1 -> manifests under $HOUFIN_PROCESSED/validation).
# The reproject stages use the whole node via HOUFIN_PREPROCESS_WORKERS.
export HOUFIN_PREPROCESS_WORKERS="${HOUFIN_PREPROCESS_WORKERS:-48}"
export HOUFIN_VALIDATE=1
export STAGES="${STAGES:-preprocess}"
bash "$SLURM_SUBMIT_DIR/scripts/tacc/pipeline.sh"
