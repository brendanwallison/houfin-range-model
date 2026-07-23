#!/bin/bash
#----------------------------------------------------------------------
# MAP fit of the age-structured model (NumPyro AutoDelta / SVI point estimate).
# GPU job. Reads the ingested inputs at age_model_config.input_dir
# (numpyro_input/: metadata.pkl + Z_gathered.dat + Z_disp_gathered.dat) and writes
# to results_dir/<run_names.map>/.
#
# CHECKPOINTED: age_run_map writes map_checkpoint.pkl every HOUFIN_MAP_CKPT_EVERY
# steps and resumes from it automatically. So on the 2 h dev queue, if the wall
# clock kills the job mid-fit, just resubmit -- it continues from the last
# checkpoint. Use submit_map.sh (which can auto-chain resubmissions).
#
# PREREQS (must exist first): the DESK cube -> path features (Z_disp) -> BBS npz ->
# scripts/ingest_model_data.py. This job does NOT build them.
#
#   bash scripts/tacc/submit_map.sh                       # gpu-a100-dev, 2 h
#   HOUFIN_MAP_STEPS=1800 RESUBMITS=3 bash scripts/tacc/submit_map.sh
#----------------------------------------------------------------------
#SBATCH -J houfin_map
#SBATCH -o houfin_map.o%j
#SBATCH -e houfin_map.e%j
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=all
##SBATCH -A REPLACE_WITH_PROJECT
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
module load python/3.12.11 2>/dev/null || true
[ -n "${TACC_ALLOCATION:-}" ] && echo "allocation: $TACC_ALLOCATION"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
# JAX (NumPyro) job -- confirm a GPU build is importable before burning queue time.
python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())" || \
    { echo "jax not importable — install a GPU jax build in the venv first"; exit 1; }
date

# Checkpoint cadence + total steps are read from the env by age_run_map (defaults
# 900 steps / ckpt every 100). Leave a safety margin under the 2 h wall so the final
# checkpoint always flushes.
export HOUFIN_MAP_STEPS="${HOUFIN_MAP_STEPS:-900}"
export HOUFIN_MAP_CKPT_EVERY="${HOUFIN_MAP_CKPT_EVERY:-100}"
python -m src.model.age_run_map

echo "=== 30_model_map complete (steps=$HOUFIN_MAP_STEPS) ==="
date
