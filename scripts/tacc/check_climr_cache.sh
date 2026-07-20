#!/bin/bash
# Exit 0 if the climr reference cache is warm, else explain how to warm it and exit 1.
# Run on a login node (where the cache is visible and warmable). The climate-bearing
# submit wrappers call this BEFORE queuing, so a cold-cache offline climate job is
# refused up front instead of failing deep in a compute job.
#   bash scripts/tacc/check_climr_cache.sh        # standalone check
set -euo pipefail
source "$(dirname "$0")/env.sh"
META="${R_USER_CACHE_DIR:-$HOME/.cache}/R/climr/reference/refmap_climr/meta_data.csv"
if [ -s "$META" ]; then
    echo "climr cache warm: $META"
    exit 0
fi
cat >&2 <<EOF
ERROR: climr cache is COLD -- no reference map at
    $META
The offline climate stage cannot download it (compute nodes have no internet), so
this job would fail. Warm it ONCE on THIS login node -- after preprocessing has
built the sub-cell centroids (submit_preprocess.sh) -- then re-submit:
    bash scripts/tacc/warm_climr.sh
The cache is on persistent \$WORK, so this is a one-time step (survives the \$SCRATCH purge).
EOF
exit 1
