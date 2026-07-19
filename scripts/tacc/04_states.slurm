#!/bin/bash
#----------------------------------------------------------------------
# CPU pre-encoder assembly: build the per-year N-stream covariate states and the
# eBird stack cache -- the inputs the GPU encoder (20_encoder) consumes. Torch-free
# and CPU-only, but heavier than a login node should carry: build_states reads
# ~130 years of grids and compresses one npz per year IN PARALLEL (fork pool), so
# it runs on a compute node, not the login node. Depends on 01_preprocess +
# 02_climate outputs (climate_grid/, luh3_grid/, hyde35_grid/, soilgrids/,
# elevation/, ebird stack). Submit via submit_states.sh (injects -A).
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

# build_states compresses one npz per year in a fork process pool; use a chunk of
# the node's cores (compression is CPU-bound; the EMA/reads stay in the main proc).
# Overridable; 128-core node -> 16 parallel zlib writers is a good default.
export HOUFIN_STATES_WORKERS="${HOUFIN_STATES_WORKERS:-16}"
echo "states write workers: $HOUFIN_STATES_WORKERS"
date

export STAGES="${STAGES:-states ebird_cache}"
bash "$SLURM_SUBMIT_DIR/scripts/tacc/pipeline.sh"

echo "=== 04_states complete ($STAGES) -> yearly_states/ + state_schema.json + ebird stack ==="
date
