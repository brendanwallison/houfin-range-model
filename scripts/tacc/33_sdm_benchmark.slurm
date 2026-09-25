#!/bin/bash
# Correlative-SDM baselines for the benchmark against the dynamic range model.
#
# INDEPENDENT OF THE MAP RUN. These fits read BBS route-years and the covariate
# grids only -- they never touch model_results. The dynamic model is needed just
# for the final designation overlay (scripts/run_sdm_benchmark.py designation),
# which is a separate, cheap, CPU step run after a MAP fit has its diagnostics.
#
# Runs here rather than locally because the covariate grids are built on TACC:
#   standard -> $HOUFIN_DATA/climate_grid_monthly, luh3_grid, hyde35_grid,
#               soilgrids_grid, elevation
#   full     -> $HOUFIN_PROCESSED/encoder/states/yearly_states/state_{year}.npz
#   latent   -> $HOUFIN_PROCESSED/latent_avian_paths/Z_latent_{year}.npy
# A laptop checkout carries only retired 25 km ESRI:102039 leftovers, which the
# covariate loader refuses by geometry rather than sampling silently.
#
# GPU queue for biolith: nmixture enumerates latent N over 0..max_abundance
# (~900+ for House Finch, whose western route counts reach 490), so the NUTS
# scan is much happier on an A100. The BRT half is CPU-only and cheap.
#
# Tiers via HOUFIN_SDM_TIERS, spatial contrast (Test B) via HOUFIN_SDM_SPATIAL=1.
#SBATCH -J houfin_sdm
#SBATCH -o houfin_sdm.o%j
#SBATCH -e houfin_sdm.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --mail-type=all
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
module load python/3.12.11 2>/dev/null || true
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
cd "$SLURM_SUBMIT_DIR"
date

TIERS="${HOUFIN_SDM_TIERS:-standard full latent}"
OUT="${HOUFIN_SDM_OUT:-$HOUFIN_PROCESSED/sdm_benchmark}"
CHAINS="${HOUFIN_SDM_CHAINS:-4}"
SAMPLES="${HOUFIN_SDM_SAMPLES:-1000}"
mkdir -p "$OUT"
echo "tiers=[$TIERS] out=$OUT chains=$CHAINS samples=$SAMPLES"

# Fail in seconds, not hours. preflight exits nonzero if BBS will not load or a
# covariate grid is absent / on the wrong lattice, and `set -e` aborts the job.
python scripts/run_sdm_benchmark.py preflight

# The premise check is model-free and costs seconds; run it first so the observed
# zone structure is recorded alongside whatever the baselines conclude.
/usr/bin/time -v python scripts/run_sdm_benchmark.py premise --out "$OUT"

for tier in $TIERS; do
    echo "=== BRT tier=$tier ==="
    /usr/bin/time -v python scripts/run_sdm_benchmark.py brt --tier "$tier" --out "$OUT"

    echo "=== biolith tier=$tier ==="
    /usr/bin/time -v python scripts/run_sdm_benchmark.py biolith --tier "$tier" \
        --out "$OUT" --chains "$CHAINS" --samples "$SAMPLES"

    # Test B: the same model with a spatial GP added. If the Great Plains
    # designation only matches the data once space is in the model, the pattern
    # is not environmentally attributable -- and that conclusion does not depend
    # on trusting the dynamic model at all.
    if [ "${HOUFIN_SDM_SPATIAL:-0}" = "1" ]; then
        echo "=== biolith tier=$tier +spatial (Test B) ==="
        /usr/bin/time -v python scripts/run_sdm_benchmark.py biolith --tier "$tier" \
            --spatial --out "$OUT/spatial" --chains "$CHAINS" --samples "$SAMPLES"
    fi
done

echo "=== 33_sdm_benchmark complete ==="
ls -la "$OUT"
date
