#!/bin/bash
# Shared environment for the houfin-range-model data-processing pipeline on TACC
# Lonestar6. Source this from the login-node download script and from every SLURM
# job:  source "$(dirname "$0")/env.sh"
#
# EDIT the three marked values for your account, then leave the rest.
# ---------------------------------------------------------------------------

# Throttle uv/Cargo's rayon thread pool + concurrency. The LS6 login node's per-user
# process/memory cap makes uv's defaults trip "failed to initialize global rayon pool
# ... Resource temporarily unavailable" during installs. Harmless elsewhere.
# Reduce CUDA fragmentation OOMs on the DESK grid trainer (many whole-grid forwards per
# step); lets PyTorch grow segments instead of failing on a fragmented pool.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Flush python stdout live so per-epoch training lines appear in the log immediately
# (block-buffered stdout to a file otherwise hides progress for minutes -- looks stalled).
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# HDF5 file locking segfaults/hangs when the file lives on Lustre ($SCRATCH/$WORK), which
# is where every .nc (HYDE, LUH-3) sits. Disabling it is the standard TACC guard and lets
# xarray/netCDF4 open these files safely (a locking crash is a native segfault Python can't
# catch). Harmless off-Lustre.
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-2}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-4}"
# uv's cache defaults to $HOME/.cache (10 GB quota -> "Disk quota exceeded" on a
# multi-GB CUDA torch install) and, being on a different FS than the $WORK venv,
# forces slow full-copy installs. Put it on $WORK: big quota + same FS as the venv
# (hardlinks work).
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/houfin/uv_cache}"

# --- EDIT ME -------------------------------------------------------------
export TACC_ALLOCATION="DEB23008"                    # sbatch -A ...
export HOUFIN_REPO="$WORK/houfin/houfin-range-model"  # where you `git clone`d
export HOUFIN_VENV="$WORK/houfin/venv"                # uv-created venv
# -------------------------------------------------------------------------

# Portable dataset roots consumed by config/*.json (src/config_utils.py expands
# ${HOUFIN_DATA}/${HOUFIN_PROCESSED}). Raw + processed 27 km products go on
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

mkdir -p "$HOUFIN_DATA" "$HOUFIN_PROCESSED" "$HOUFIN_CLIMR_CACHE" "$UV_CACHE_DIR"

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
