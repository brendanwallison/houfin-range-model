# FIELD54 Stage E common settings (sourced by e*.slurm after envbc.sh + env54)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-32} MKL_NUM_THREADS=${OMP_NUM_THREADS:-32}
EVALARGS="--root $EVAL --base $OBS/base_eval --similarity $IN27/species_similarity.csv --cbc $OBS/cbc_eval54.npz --task backcast40"
DCUBE=$EVAL/cubes/backcast40/desk_interp_s1
ECUBE=$EVAL/cubes/backcast40/desk_interp_s1_ema
need() { for f in "$@"; do test -e "$f" || { echo "[stageE] missing $f -- prerequisite not ready"; exit 3; }; done; }
