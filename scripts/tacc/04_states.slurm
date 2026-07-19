#!/bin/bash
#----------------------------------------------------------------------
# CPU pre-encoder assembly (all torch-free): per-year N-stream covariate states,
# the reprojected eBird stack, the gridded BBS per-species community_matrix
# (crosswalk + community), and the amplitude cube (x = E * BBS-anomaly) -- every
# input the GPU encoder (20_encoder) consumes, so that job is purely GPU
# esk->desk->cube->validate. CPU-only but heavier than a login node should carry
# (build_states reads ~130 years of grids and compresses one npz per year in a fork
# pool), so it runs on a compute node. Depends on 01_preprocess + 02_climate outputs
# (climate_grid/, luh3_grid/, hyde35_grid/, soilgrids/, elevation/) and the raw BBS
# release. Submit via submit_states.sh (injects -A).
#----------------------------------------------------------------------
#SBATCH -J houfin_states
#SBATCH -o houfin_states.o%j
#SBATCH -e houfin_states.e%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH --mail-type=all
##SBATCH -A REPLACE_WITH_PROJECT
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
module load python/3.12.11 2>/dev/null || true
module load remora 2>/dev/null || true
[ -n "${TACC_ALLOCATION:-}" ] && echo "allocation: $TACC_ALLOCATION"

# Pin native thread pools to 1 for this CPU job. build_states parallelizes reads
# (thread pool) + writes (process pool) EXPLICITLY, and the reprojection is
# I/O-bound, so implicit GDAL/OpenBLAS/OMP threads add nothing but noise -- they
# were the source of the sys.excepthook flood at interpreter teardown. Pinning
# also stops each of the 32 read threads from spawning nested GDAL threads.
# Scoped to this job only (NOT the GPU encoder, which wants its BLAS threads).
export GDAL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# build_states compresses one npz per year in a fork process pool; use a chunk of
# the node's cores (compression is CPU-bound; the EMA/reads stay in the main proc).
# Overridable; 128-core node -> 16 parallel zlib writers is a good default.
export HOUFIN_STATES_WORKERS="${HOUFIN_STATES_WORKERS:-16}"
echo "states write workers: $HOUFIN_STATES_WORKERS"
date

# All CPU / torch-free pre-encoder assembly, in dependency order:
#   states       -> per-year N-stream covariate states + schema
#   ebird_cache  -> reprojected eBird stack E (needed by amplitude)
#   bbs          -> AOU<->eBird crosswalk + gridded per-species community_matrix
#   amplitude    -> x = E * BBS-anomaly point cube (needs ebird_cache + bbs)
# The GPU job (20_encoder) stays purely esk -> desk -> cube -> validate.
export STAGES="${STAGES:-states ebird_cache bbs amplitude}"
bash "$SLURM_SUBMIT_DIR/scripts/tacc/pipeline.sh"

echo "=== 04_states complete ($STAGES) -> states + eBird stack + community_matrix + amplitude cube ==="
date
