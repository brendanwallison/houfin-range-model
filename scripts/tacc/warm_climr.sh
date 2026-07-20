#!/bin/bash
# Warm the climr reference cache for the WHOLE study region, on a LOGIN NODE.
# ---------------------------------------------------------------------------
# This is the one online step the offline climate stage depends on: compute nodes
# have no internet, so downscale(db_option="local") can only read an already-warmed
# cache. climate_climr.py --warm-cache downloads + caches the refmap and cru.gpcc obs
# surfaces, one R process PER geographic tile (process isolation; avoids the login
# node's ulimit -u), caching the refmap by a verbatim file copy + header-only band
# rename (survives the login cgroup), recording the requested bbox so the offline
# read hits at every tile. It is resumable -- re-running only fetches missing tiles.
#
# ORDER: run AFTER downloads (download_all.sh) AND after preprocessing has built the
# grid + DEM products, because the warm needs the sub-cell centroids, which are built
# from ref_grid + land_mask + DEM (auto-built here if absent but those must exist).
# Then the offline climate stage (02_climate / 00_preprocess_all) can run on compute.
#
#   ssh ls6 ; module load python/3.12.11 ; source scripts/tacc/env.sh
#   bash scripts/tacc/warm_climr.sh
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/env.sh"
echo "using Rscript: ${HOUFIN_RSCRIPT:-Rscript}"
python scripts/climate_climr.py --warm-cache --rscript "${HOUFIN_RSCRIPT:-Rscript}"
echo "== climr cache warmed (study-region tiles). The offline climate stage can now run on compute. =="
