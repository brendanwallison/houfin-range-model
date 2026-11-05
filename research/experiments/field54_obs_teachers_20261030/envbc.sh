# FIELD54 Stage B/C environment (sourced by bc_*.slurm). Two data environments:
#   env27: the 27 km data root/config (BBS release + 27 km mask: route cells as every 27 km build)
#   env54: the 54 km data config (54 km mask / ref raster for packaging and the ESK grid)
source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
module load python/3.12.11 2>/dev/null || true
export F=/work/07980/bwa386/ls6/houfin/research/field54
export CODE=$F/code_bc
export IN27=$F/inputs27
export OBS=/work/07980/bwa386/ls6/houfin/processed/field54/obs
export EVAL=$F/runs/field54_eval
export PROD=$F/runs/field54_prod
export PYTHONPATH="$CODE:$CODE/src" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=
cd "$CODE"
env27() { unset DATA_CONFIG ESK_DESK_CONFIG; export HOUFIN_DATA=/scratch/07980/bwa386/houfin/data HOUFIN_PROCESSED=/work/07980/bwa386/ls6/houfin/processed; }
env54() { unset ESK_DESK_CONFIG; export HOUFIN_DATA54=/scratch/07980/bwa386/houfin/data54 HOUFIN_PROCESSED54=/work/07980/bwa386/ls6/houfin/processed/field54 HOUFIN_DATA=/scratch/07980/bwa386/houfin/data54 HOUFIN_PROCESSED=/work/07980/bwa386/ls6/houfin/processed/field54 DATA_CONFIG=$CODE/config/overlays/field54_data.json; }
echo "[envbc] host $(hostname) job ${SLURM_JOB_ID:-none} $(date)"
