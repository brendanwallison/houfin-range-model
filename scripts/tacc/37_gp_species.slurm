#!/bin/bash
#----------------------------------------------------------------------
# GP species validation (src/community_encoder/train_DESK/validate_gp_species.py): DESK as a GP
# kernel, graded on per-species abundance of out-of-community BBS species in held-out blocks.
# SUBMIT VIA scripts/tacc/submit_gp_species.sh, not bare sbatch (the -A line is disabled).
#
# A compute-node job for the same reason as 22_rescore: the output EMA is a causal scan, so the
# encode forwards every year from ema_warmup_start, ~86 whole-grid passes. The desk GP is exact
# low-rank algebra and takes minutes. The spacetime and covariate baselines each fit a shared
# shape (an n_fit x n_fit eigendecomposition per iteration) and then one eigendecomposition per
# held-out block, and the thinning arm repeats both per fraction -- that is most of the wall time.
#----------------------------------------------------------------------
#SBATCH -J houfin_gpsp
#SBATCH -o houfin_gpsp.o%j
#SBATCH -e houfin_gpsp.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --mail-type=all
##SBATCH -A REPLACE_WITH_PROJECT
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
module load python/3.12.11 2>/dev/null || true
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
cd "$SLURM_SUBMIT_DIR"
date
: "${ESK_DESK_CONFIG:?ESK_DESK_CONFIG must name the overlay (submit via submit_gp_species.sh)}"
echo "overlay: $ESK_DESK_CONFIG  n_boot=${GPSP_N_BOOT:-1000}"
python -m src.community_encoder.train_DESK.validate_gp_species --n-boot "${GPSP_N_BOOT:-1000}" \
    ${GPSP_ARGS:-}
# The regressions and thinning curves need only the saved outputs plus AVONET; a failure here
# leaves the validation itself intact and can be rerun anywhere with the gp_species directory.
GP_DIR="$(python -c "from src.config_utils import load_config; import os; print(os.path.join(load_config()['paths']['desk_output_dir'], 'gp_species'))")"
python -m src.community_encoder.train_DESK.gp_species_analysis "$GP_DIR" || \
    echo "WARNING: gp_species_analysis failed; rerun: python -m src.community_encoder.train_DESK.gp_species_analysis $GP_DIR"
echo "=== 37_gp_species complete ==="
date
