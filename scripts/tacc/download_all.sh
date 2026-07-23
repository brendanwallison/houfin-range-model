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
echo "== Build the ranked reference species list (feeds community selection) =="
python scripts/avonet_pipeline.py

echo "== BBS (US/Canada + Mexico) + BBS trend maps =="
python scripts/download_bbs.py --dataset bbs --extract
python scripts/download_bbs.py --dataset bbs_mexico
python scripts/download_bbs.py --dataset bbs_trends --extract    # 516 tr{AOU}.tif (%/yr, 27 km ESRI:102003)
python scripts/download_bbs.py --dataset bbs_abundance --extract  # 520 ra{AOU}.tif (birds/route) -- method-B deep scale

echo "== Select the trend community: top-N HF-similar present in BOTH trend products =="
# Needs the ranked list + BBS SpeciesList & trend rasters (above) + the eBird trends REST
# listing (EBIRD_KEY). Writes community_trend.csv, which drives the eBird downloads below.
python -m src.data.identify.select_trend_community

echo "== eBird (needs EBIRD_KEY / secrets.json; reads community_trend.csv) =="
COMMUNITY="$HOUFIN_DATA/avonet/community_trend.csv"
# Status & Trends 'trends' parquets (abd_ppy %/yr + abd midpoint reference) for the community.
# REQUIRED: the default trend anchor (trend.anchor_mode=trends-abd) reconstructs from `abd`
# forward-extrapolated to the reference year, so the parquets are the only eBird product needed.
python scripts/download_ebird.py --trends --species-list "$COMMUNITY" --workers 4
# Weekly status abundance: OPTIONAL, only for the LEGACY trend.anchor_mode=weekly anchor (and
# for bbs_mode=off/validate). trends-abd needs none, and --require-weekly's 52-week gate drops
# species that have a trend but no complete weekly stack. Uncomment to use the weekly anchor:
# python scripts/download_ebird.py --species-list "$COMMUNITY" --require-weekly --workers 4

echo "== LUH-3 (Zenodo, ~8 GB) =="
python scripts/download_zenodo.py --dataset luh3

echo "== HYDE 3.5 =="
python scripts/download_hyde.py

echo "== SoilGrids =="
python scripts/download_soilgrids.py

echo "== DEM (ETOPO 2022, ~0.5 GB) =="
python scripts/download_dem.py

echo "== Natural Earth 10 m land + polygonal lakes (terrestrial-mask sources) =="
NE_DIR="$HOUFIN_DATA/land_source"
if [ ! -f "$NE_DIR/ne_10m_land.shp" ]; then
    mkdir -p "$NE_DIR"
    curl -fSL -o "$NE_DIR/ne_10m_land.zip" \
        "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
    ( cd "$NE_DIR" && unzip -o ne_10m_land.zip )
fi
if [ ! -f "$NE_DIR/ne_10m_lakes.shp" ]; then
    mkdir -p "$NE_DIR"
    curl -fSL -o "$NE_DIR/ne_10m_lakes.zip" \
        "https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip"
    ( cd "$NE_DIR" && unzip -o ne_10m_lakes.zip )
fi

echo "== climr CONNECTIVITY CHECK (one point; NOT the study-region warm) =="
# This only verifies climr is installed and its server is reachable from the login
# node. It does NOT warm the study-region cache -- that is a separate step,
# scripts/tacc/warm_climr.sh, run AFTER preprocessing (it needs the sub-cell
# centroids). See docs/TACC.md. Uses $HOUFIN_RSCRIPT (set in env.sh).
echo "using Rscript: $HOUFIN_RSCRIPT"
"$HOUFIN_RSCRIPT" -e 'if (requireNamespace("climr", quietly=TRUE)) { library(climr); climr::downscale(data.frame(id=1, lon=-98, lat=39, elev=300), obs_years=2020) ; cat("climr reachable\n") } else cat("climr not installed yet\n")' || true

echo "== downloads complete -> $HOUFIN_DATA =="
echo "== NEXT: preprocess (submit_preprocess.sh) -> warm cache (warm_climr.sh, login) -> climate + assemble =="
