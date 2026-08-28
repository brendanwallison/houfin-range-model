#!/bin/bash
#----------------------------------------------------------------------
# Per-ESK-component predictability from the existing covariates
# (scripts/diagnostics/component_predictability.py). SUBMIT VIA
# scripts/tacc/submit_predictability.sh, not bare sbatch: this account has multiple projects,
# the `#SBATCH -A` line below is deliberately disabled, and SLURM rejects a job with no
# allocation.
#
# NOT A LOGIN-NODE JOB, even though it needs no GPU. The invocation line in the script's own
# docstring was copied from basis_domain_gap.py, which IS login-node safe -- its peak kernel
# block is ~0.04 GiB after the subsample-first fix. This one is a different shape of work:
#
#   * a dense multi-target ridge whose interaction rung has d = 42C + pairs (~6,200 columns at
#     C=100), so the Gram accumulation is an n*d^2 dgemm that saturates every core it is given
#     for minutes at a time -- exactly the sustained multi-core BLAS a shared login node's
#     process reaper exists to kill;
#   * ~1 GB resident for the feature matrix plus its standardized copy;
#   * the ESK projection of the sampled communities against ~16k landmarks over all species,
#     on CPU because this script deliberately takes no GPU.
#
# CPU-only, one node, so the development queue is the cheap right answer (same reasoning as
# 04_states). Unlike 04_states the native thread pools are NOT pinned to 1: that job is
# I/O-bound and parallelizes reads and writes itself, whereas every heavy step here is a
# single BLAS call that wants the whole node.
#
# Needs: the target point set, ${HOUFIN_PROCESSED}/encoder/states/yearly_states, and the saved
# ESK basis in desk.z_dir. Reuses the trained DESK run's holdout/buffer masks and mu/sd when
# desk_output_dir has them, which is what makes the R^2 comparable to DESK's own val numbers.
#----------------------------------------------------------------------
#SBATCH -J houfin_predict
#SBATCH -o houfin_predict.o%j
#SBATCH -e houfin_predict.e%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=all
##SBATCH -A REPLACE_WITH_PROJECT
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
module load python/3.12.11 2>/dev/null || true
[ -n "${TACC_ALLOCATION:-}" ] && echo "allocation: $TACC_ALLOCATION"
date

# Give the BLAS the node. Left unset, OpenBLAS/MKL pick their own default, which on a 128-core
# node has been either the whole machine (fine here, but only by luck) or a single core (which
# turns the interaction rung from seconds into an hour).
NTHREADS="${PREDICT_THREADS:-${SLURM_CPUS_ON_NODE:-32}}"
export OMP_NUM_THREADS="$NTHREADS" OPENBLAS_NUM_THREADS="$NTHREADS" \
       MKL_NUM_THREADS="$NTHREADS" VECLIB_MAXIMUM_THREADS="$NTHREADS" \
       NUMEXPR_NUM_THREADS="$NTHREADS"
echo "BLAS threads: $NTHREADS"

PREDICT_MAX_ROWS="${PREDICT_MAX_ROWS:-300000}"
PREDICT_PAIRS="${PREDICT_PAIRS:-2000}"
PREDICT_PCA_DIM="${PREDICT_PCA_DIM:-48}"
PREDICT_RFF_WIDTH="${PREDICT_RFF_WIDTH:-2048}"
PREDICT_OUT="${PREDICT_OUT:-$HOUFIN_PROCESSED/encoder/component_predictability.json}"
mkdir -p "$(dirname "$PREDICT_OUT")"

echo "max_rows=$PREDICT_MAX_ROWS pairs=$PREDICT_PAIRS pca_dim=$PREDICT_PCA_DIM"
echo "rff_width=$PREDICT_RFF_WIDTH out=$PREDICT_OUT"

python "$SLURM_SUBMIT_DIR/scripts/diagnostics/component_predictability.py" \
    --max-rows "$PREDICT_MAX_ROWS" \
    --pairs "$PREDICT_PAIRS" \
    --pca-dim "$PREDICT_PCA_DIM" \
    --rff-width "$PREDICT_RFF_WIDTH" \
    --out "$PREDICT_OUT"

echo "=== 23_predictability complete ==="
date
