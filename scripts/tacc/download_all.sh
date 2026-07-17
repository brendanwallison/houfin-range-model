#!/bin/bash
# Data acquisition for the houfin pipeline. RUN ON A LOGIN NODE, NOT via sbatch:
# Lonestar6 compute nodes have no internet. Downloads land under $HOUFIN_DATA
# ($SCRATCH). Everything here is idempotent/resumable, so re-running only fetches
# what's missing.
#
#   ssh ls6 ; module load python/3.12.11 ; source scripts/tacc/env.sh
#   nohup bash scripts/tacc/download_all.sh > download.log 2>&1 &
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "== AVONET (traits+crosswalk+phylogeny) + eBird taxonomy + urban tolerance =="
python scripts/download_avonet.py
echo "== Build the ranked reference species list (feeds the eBird download) =="
python scripts/avonet_pipeline.py

echo "== eBird (needs EBIRD_KEY / secrets.json; reads the species list above) =="
python scripts/download_ebird.py --top-n 100      # top-100 ranked reference community

echo "== BBS (US/Canada + Mexico) =="
python scripts/download_bbs.py --dataset bbs --extract
python scripts/download_bbs.py --dataset bbs_mexico

echo "== LUH-3 (Zenodo, ~8 GB) =="
python scripts/download_zenodo.py --dataset luh3

echo "== HYDE 3.5 =="
python scripts/download_hyde.py

echo "== SoilGrids =="
python scripts/download_soilgrids.py

echo "== DEM (ETOPO 2022, ~0.5 GB) =="
python scripts/download_dem.py

echo "== Natural Earth 10 m land (coastline source for the land mask) =="
NE_DIR="$HOUFIN_DATA/land_source"
if [ ! -f "$NE_DIR/ne_10m_land.shp" ]; then
    mkdir -p "$NE_DIR"
    curl -fSL -o "$NE_DIR/ne_10m_land.zip" \
        "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
    ( cd "$NE_DIR" && unzip -o ne_10m_land.zip )
fi

echo "== Warm the climr reference cache (online; needed for the offline batch step) =="
# Uses $HOUFIN_RSCRIPT (set in env.sh: the micromamba conda-forge env if present,
# else PATH Rscript). A tiny climr run downloads + caches the reference surfaces. If
# climr isn't installed yet, see docs/TACC.md (install it into the userspace R first).
echo "using Rscript: $HOUFIN_RSCRIPT"
"$HOUFIN_RSCRIPT" -e 'if (requireNamespace("climr", quietly=TRUE)) { library(climr); climr::downscale(data.frame(id=1, lon=-98, lat=39, elev=300), obs_years=2020) ; cat("climr cache warmed\n") } else cat("climr not installed yet\n")' || true

echo "== downloads complete -> $HOUFIN_DATA =="
