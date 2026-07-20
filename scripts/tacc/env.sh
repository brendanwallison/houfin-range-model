#!/bin/bash
# Shared environment for the houfin-range-model data-processing pipeline on TACC
# Lonestar6. Source this from the login-node download script and from every SLURM
# job:  source "$(dirname "$0")/env.sh"
#
# EDIT the three marked values for your account, then leave the rest.
# ---------------------------------------------------------------------------

# --- EDIT ME -------------------------------------------------------------
export TACC_ALLOCATION="DEB23008"                    # sbatch -A ...
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

# climr writes its reference cache here; keep it on a shared, PERSISTENT FS ($WORK,
# not the 10-day-purged $SCRATCH) so batch nodes (no internet) read the cache warmed
# on the login node, and so it survives the purge -- warm ONCE and every later run is
# warm (this is what makes the one-shot 00_preprocess_all achievable on re-runs).
# climr resolves its cache via tools::R_user_dir("climr","cache"), which reads
# R_USER_CACHE_DIR; export it or climr silently falls back to $HOME/.cache (small
# quota, wrong FS). It appends /R/climr under this root.
export HOUFIN_CLIMR_CACHE="$WORK/houfin/climr_cache"
export R_USER_CACHE_DIR="$HOUFIN_CLIMR_CACHE"

mkdir -p "$HOUFIN_DATA" "$HOUFIN_PROCESSED" "$HOUFIN_CLIMR_CACHE"

# R interpreter for the climr climate step. TACC's Rstats/4.0.3 is too old for
# climr's CRAN dep tree (dplyr/tidyr/ggplot2/... need R >= 4.1) and climr is
# GitHub-only with native deps (terra/sf/RPostgres), so setup builds a modern
# userspace R at $WORK/houfin/renv via micromamba + conda-forge (LS6 has no
# conda/mamba module; see docs/TACC.md). Auto-detect that env if present; otherwise
# fall back to PATH Rscript. Re-detect on every source UNLESS HOUFIN_RSCRIPT already
# points at an executable: this honors a deliberate override (a full path) while
# self-correcting the case where an early source — during one-time setup, before the
# renv is built — pins the "Rscript" fallback that a bare -z guard would then keep.
HOUFIN_RENV="$WORK/houfin/renv"
if [ ! -x "${HOUFIN_RSCRIPT:-}" ]; then
    if [ -x "$HOUFIN_RENV/bin/Rscript" ]; then
        export HOUFIN_RSCRIPT="$HOUFIN_RENV/bin/Rscript"
    else
        export HOUFIN_RSCRIPT="Rscript"
    fi
fi

# The renv Rscript is invoked directly (not via `micromamba activate`), so the
# PROJ/GDAL data dirs that activation would set are missing -> terra/sf can't find
# proj.db ("proj_create: ... problem with the PROJ installation") and reprojections
# misbehave. Point them at the env's data dirs so the climate step gets a working PROJ.
if [ -d "$HOUFIN_RENV/share/proj" ]; then
    export PROJ_DATA="$HOUFIN_RENV/share/proj"
    export GDAL_DATA="$HOUFIN_RENV/share/gdal"
fi

# The venv's interpreter links against the python/3.12.11 module's libpython, so
# load the module before activating (harmless if already loaded, e.g. in SLURM
# jobs) -- otherwise `python` in an interactive shell fails with
# "libpython3.12.so.1.0: cannot open shared object file".
module load python/3.12.11 2>/dev/null || true

# Activate the Python environment (uv-managed venv on $WORK).
if [ -f "$HOUFIN_VENV/bin/activate" ]; then
    source "$HOUFIN_VENV/bin/activate"
fi
cd "$HOUFIN_REPO"
