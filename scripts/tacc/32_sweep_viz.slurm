#!/bin/bash
# Post-MAP diagnostics for EVERY point of a parameter sweep, in ONE job, plus the
# cross-point summary. Runs map_diagnostics.py once per point (~3 minutes each,
# sequentially) rather than burning a whole job per point: five separate viz jobs
# would push a 5-point sweep to 15 jobs, over gpu-a100-small's 12-submitted cap,
# and force the overflow onto the 3-SU main queue.
#
# Driven by SWEEP_MANIFEST (written by submit_juv_mdd_sweep.sh); each point's
# AGE_MODEL_CONFIG overlay is applied for its own invocation.
#SBATCH -J houfin_sweepviz
#SBATCH -o houfin_sweepviz.o%j
#SBATCH -e houfin_sweepviz.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=all
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
source "$SLURM_SUBMIT_DIR/scripts/tacc/gpu_monitor.sh"
module load python/3.12.11 2>/dev/null || true
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
gpu_preflight "sweep post-MAP diagnostics"
start_gpu_monitor "$SLURM_SUBMIT_DIR/gpu_sweepviz_${SLURM_JOB_ID}.csv" 30
trap stop_gpu_monitor EXIT
date

: "${SWEEP_MANIFEST:?SWEEP_MANIFEST must point at a sweep_manifest.json}"
export HOUFIN_MAP_PROFILE="${HOUFIN_MAP_PROFILE:-standard}"

# Read (point, overlay) pairs from the manifest.
POINTS=$(python -c "
import json
for p in json.load(open('$SWEEP_MANIFEST'))['points']:
    print(p['point'], p['overlay'], sep='\t')
")

failed=0
while IFS=$'\t' read -r tag overlay; do
    [ -n "$tag" ] || continue
    echo "=== diagnostics: $tag ($overlay) ==="
    # One point's failure (e.g. its fit never finished) must not abandon the rest,
    # so don't let set -e kill the loop; record and continue. The summary script
    # marks incomplete points excluded anyway.
    if AGE_MODEL_CONFIG="$overlay" /usr/bin/time -v \
           python scripts/viz/map_diagnostics.py --profile "$HOUFIN_MAP_PROFILE"; then
        echo "--- $tag ok"
    else
        echo "--- $tag FAILED (continuing)"
        failed=$((failed + 1))
    fi
done <<< "$POINTS"

echo "=== cross-point summary ==="
python scripts/viz/juv_mdd_sweep_summary.py --manifest "$SWEEP_MANIFEST" || true

echo "=== 32_sweep_viz complete (profile=$HOUFIN_MAP_PROFILE, $failed point(s) failed) ==="
date
