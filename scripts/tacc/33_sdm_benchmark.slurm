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

# (TIERS/OUT are set above.) Fail in seconds, not hours. preflight exits nonzero if BBS will not load or a
# covariate grid is absent / on the wrong lattice, and `set -e` aborts the job.
python scripts/run_sdm_benchmark.py preflight --tiers $TIERS

# The premise check is model-free and costs seconds; run it first so the observed
# zone structure is recorded alongside whatever the baselines conclude.
/usr/bin/time -v python scripts/run_sdm_benchmark.py premise --out "$OUT"

# Per-stage failures must not abandon the tiers that would have succeeded: a
# `set -e` abort on the first tier previously meant full and latent never ran at
# all. Record failures and carry on; the exit status at the end reflects them.
FAILED=""
run_stage () {                      # run_stage <label> <cmd...>
    local label="$1"; shift
    echo "=== $label ==="
    if /usr/bin/time -v "$@"; then
        return 0
    fi
    echo "!!! $label FAILED (continuing to remaining stages)"
    FAILED="$FAILED $label"
    return 0
}

for tier in $TIERS; do
    run_stage "brt/$tier" python scripts/run_sdm_benchmark.py brt \
        --tier "$tier" --out "$OUT"

    # The abundance fit runs by default, via the direct marginal sum (Royle
    # 2004) rather than biolith's enumerated Categorical -- linear in the
    # ceiling (~0.3 GiB) instead of quadratic (55 GiB), same model, verified to
    # agree to MCMC noise. HOUFIN_SDM_NMIXTURE=0 skips it.
    NMIX=""
    [ "${HOUFIN_SDM_NMIXTURE:-1}" = "0" ] && NMIX="--no-nmixture"
    run_stage "biolith/$tier" python scripts/run_sdm_benchmark.py biolith \
        --tier "$tier" --out "$OUT" --chains "$CHAINS" --samples "$SAMPLES" $NMIX

    # Test B: the same model with a spatial GP added. If the Great Plains
    # designation only matches the data once space is in the model, the pattern
    # is not environmentally attributable -- and that conclusion does not depend
    # on trusting the dynamic model at all.
    if [ "${HOUFIN_SDM_SPATIAL:-0}" = "1" ]; then
        run_stage "biolith/$tier+spatial" python scripts/run_sdm_benchmark.py \
            biolith --tier "$tier" --spatial --out "$OUT/spatial" \
            --chains "$CHAINS" --samples "$SAMPLES" $NMIX
    fi
done

ls -la "$OUT"
date
if [ -n "$FAILED" ]; then
    echo "=== 33_sdm_benchmark finished WITH FAILURES:$FAILED ==="
    exit 1
fi
echo "=== 33_sdm_benchmark complete ==="
