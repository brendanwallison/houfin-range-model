#!/bin/bash
#SBATCH -J f54_climv2
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54_climv2.o%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A DEB23008
# Closes the climate sub-point gap: climr for the 2,082 missing fine-land sub-points
# (new chunk dir), regroup over the union of both chunk dirs, climate_grid -> *_v2 dirs.
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code/research/field54/env54.sh
E=$HOUFIN_DATA54/climate_extra
date; /usr/bin/time -v python research/exp/field54_climr_extra.py --chunks $E/_chunks --workers 16 --threads 8
python - <<'PY'
import pandas as pd, glob, numpy as np
fs = sorted(glob.glob("/scratch/07980/bwa386/houfin/data54/climate_extra/_chunks/out_*/climate_points.csv"))
p = pd.concat([pd.read_csv(f) for f in fs])
v = p.drop(columns=[c for c in ("id", "PERIOD", "DATASET") if c in p.columns])
print(f"[extra points] files {len(fs)} rows {len(p)} ids {p.id.nunique()} periods {p.PERIOD.min()}..{p.PERIOD.max()} "
      f"datasets {p.DATASET.unique() if 'DATASET' in p else None}; rows with any NaN {int(v.isna().any(axis=1).sum())}; "
      f"ids all-NaN {int(v.isna().all(axis=1).groupby(p.id).all().sum())}", flush=True)
PY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date; /usr/bin/time -v python -m src.data.preprocess.climate_regroup \
    --chunks "$HOUFIN_DATA27/climate/_chunks" "$E/_chunks" --factor 2 --coarse-width 112 \
    --out "$HOUFIN_DATA54/climate_v2" --workers 8 --tmp "$HOUFIN_DATA54/_climate_v2_regroup_tmp"
export GDAL_NUM_THREADS=1
date; /usr/bin/time -v python -m src.data.preprocess.climate_grid \
    --climate-dir "$HOUFIN_DATA54/climate_v2" --out "$HOUFIN_DATA54/climate_grid_monthly_v2"
date; ls $HOUFIN_DATA54/climate_grid_monthly_v2 | wc -l
