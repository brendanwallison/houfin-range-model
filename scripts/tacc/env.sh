#!/bin/bash
# Shared environment for the houfin-range-model data-processing pipeline on TACC
# Lonestar6. Source this from the login-node download script and from every SLURM
# job:  source "$(dirname "$0")/env.sh"
#
# EDIT the three marked values for your account, then leave the rest.
# ---------------------------------------------------------------------------

# --- EDIT ME -------------------------------------------------------------
export TACC_ALLOCATION="REPLACE_WITH_PROJECT"        # sbatch -A ...
export HOUFIN_REPO="$WORK/houfin/houfin-range-model"  # where you `git clone`d
export HOUFIN_VENV="$WORK/houfin/venv"                # uv-created venv
# -------------------------------------------------------------------------

# Portable dataset roots consumed by config/*.json (src/config_utils.py expands
# ${HOUFIN_DATA}/${HOUFIN_PROCESSED}). Raw + processed 25 km products go on
# $SCRATCH (fast, large, PURGED after 10 days); manifests + persistent outputs on
# $WORK. Promote confirmed processed products to $WORK before the purge (see
# docs/TACC.md).
export HOUFIN_DATA="$SCRATCH/houfin/data"
export HOUFIN_PROCESSED="$WORK/houfin/processed"

# eBird API key (or put it in config/secrets.json as {"ebird_key": "..."}).
# export EBIRD_KEY="..."

# climr writes its reference cache here; keep it on a shared FS so batch nodes
# (which have no internet) can read the cache warmed on the login node.
export HOUFIN_CLIMR_CACHE="$WORK/houfin/climr_cache"

mkdir -p "$HOUFIN_DATA" "$HOUFIN_PROCESSED" "$HOUFIN_CLIMR_CACHE"

# R interpreter for the climr climate step. TACC's Rstats/4.0.3 is too old for
# climr's CRAN dep tree (dplyr/tidyr/ggplot2/... need R >= 4.1) and climr is
# GitHub-only with native deps (terra/sf/RPostgres), so setup builds a modern
# userspace R at $WORK/houfin/renv via micromamba + conda-forge (LS6 has no
# conda/mamba module; see docs/TACC.md). Auto-detect that env if present; otherwise
# fall back to PATH Rscript. Override by exporting HOUFIN_RSCRIPT before sourcing.
if [ -z "${HOUFIN_RSCRIPT:-}" ]; then
    if [ -x "$WORK/houfin/renv/bin/Rscript" ]; then
        export HOUFIN_RSCRIPT="$WORK/houfin/renv/bin/Rscript"
    else
        export HOUFIN_RSCRIPT="Rscript"
    fi
fi

# Activate the Python environment (uv-managed venv on $WORK).
if [ -f "$HOUFIN_VENV/bin/activate" ]; then
    source "$HOUFIN_VENV/bin/activate"
fi
cd "$HOUFIN_REPO"
