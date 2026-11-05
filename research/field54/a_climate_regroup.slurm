#!/bin/bash
#SBATCH -J f54_clim
#SBATCH -o /work/07980/bwa386/ls6/houfin/research/field54/logs/f54_clim.o%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A DEB23008
set -euo pipefail
source /work/07980/bwa386/ls6/houfin/research/field54/code/research/field54/env54.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# 1. fidelity: recompute one more 27 km chunk's quantiles from its stored points
python - <<'PY'
import pandas as pd, numpy as np, os
from src.data.acquire.climatena import quantile_aggregate
ch = os.environ["HOUFIN_DATA27"] + "/climate/_chunks"
for i in (20,):
    pts = pd.read_csv(f"{ch}/out_{i:03d}/climate_points.csv")
    cen = pd.read_csv(f"{ch}/chunk_{i:03d}.csv", usecols=["id", "parent_id"])
    for lvl, df in quantile_aggregate(pts, cen).items():
        st = pd.read_csv(f"{ch}/out_{i:03d}/climate_{lvl}.csv")
        a = df.sort_values(["id", "PERIOD"]).reset_index(drop=True)
        b = st.sort_values(["id", "PERIOD"]).reset_index(drop=True)
        assert list(a.columns) == list(b.columns) and len(a) == len(b)
        d = np.abs(a.iloc[:, 2:].to_numpy(float) - b.iloc[:, 2:].to_numpy(float))
        print(f"[fidelity] chunk {i} {lvl}: rows {len(a)} max|diff| {np.nanmax(d):.3e} "
              f"nan-mismatch {(np.isnan(a.iloc[:,2:].to_numpy(float)) != np.isnan(b.iloc[:,2:].to_numpy(float))).sum()}", flush=True)
    print("DATASET values:", pts["DATASET"].unique()[:5] if "DATASET" in pts else None,
          "PERIOD", pts["PERIOD"].min(), pts["PERIOD"].max(), pts["PERIOD"].nunique())
PY
# 2. regroup to 54 km parents over the union of sub-points
mkdir -p "$HOUFIN_DATA54/climate"
/usr/bin/time -v python -m src.data.preprocess.climate_regroup \
    --chunks "$HOUFIN_DATA27/climate/_chunks" --factor 2 --coarse-width 112 \
    --out "$HOUFIN_DATA54/climate" --workers ${F54_CLIM_WORKERS:-8} \
    --tmp "$HOUFIN_DATA54/_climate_regroup_tmp"
ls -la "$HOUFIN_DATA54/climate"
