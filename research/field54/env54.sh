# FIELD54 Stage A environment: the 54 km data root and overlays (sourced by every job).
# Points HOUFIN_DATA / HOUFIN_PROCESSED at the 54 km roots too, so anything that still
# expands ${HOUFIN_DATA} lands in the 54 km root, never in the 27 km products.
source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
module load python/3.12.11 2>/dev/null || true
export CODE=/work/07980/bwa386/ls6/houfin/research/field54/code
export HOUFIN_DATA27=/scratch/07980/bwa386/houfin/data
export HOUFIN_DATA54=/scratch/07980/bwa386/houfin/data54
export HOUFIN_PROCESSED54=/work/07980/bwa386/ls6/houfin/processed/field54
export HOUFIN_DATA="$HOUFIN_DATA54"
export HOUFIN_PROCESSED="$HOUFIN_PROCESSED54"
export DATA_CONFIG="$CODE/config/overlays/field54_data.json"
export ESK_DESK_CONFIG="$CODE/config/overlays/field54_states.json"
export PYTHONPATH="$CODE:$CODE/src"
export PYTHONUNBUFFERED=1
cd "$CODE"
echo "[env54] host $(hostname) job ${SLURM_JOB_ID:-none} $(date)"
python -c "from src.config_utils import load_data_config as L; c=L(); assert c['grid']['target_res_m']==54000 and c['datasets_root']=='$HOUFIN_DATA54', c['grid']; print('[env54] data config ok', c['datasets_root'], c['grid']['ref_raster'])"
