"""Plan L, C1 (phase 1): Bird Banding Lab adult dead-recovery m-array on the grid54 land cells.

Design: research/experiments/SURVIVAL_BBL_SCOPING_2026-10-31.md s.3.1 (Brownie / Seber). Phase 1 = adults only.
- Cohorts: banding events (event_type B) of normal wild birds (bird_status 3) aged AHY or older (age 1, 5, 6, 7, 8). East of
  100W only breeding-season bandings (Apr-Aug) are used to assign a breeding cell (eastern partial migration); West all
  months. Banding year runs Apr-Mar.
- Recoveries: the FIRST dead encounter (present_condition 03-05) of a band, excluding band-only finds (how_obtained 98) and
  banding mortality (51); encounter year Apr-Mar. As in the crude MLE, birds recovered in their banding year (k = 0) are
  dropped from the cohort (the likelihood conditions on surviving the first partial year).
- Output npz: per cohort (land cell, banding year) n_released and recoveries m[k], k = 1..K (years after banding), plus the
  cohort's region group (west < 100W, central 100-80W, east > 80W) for the reporting-rate nuisance.
Usage (inside sbx_env.sh): python bbl_prep.py --out FILE.npz
"""
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as crs_transform
from harness import Sandbox

CSV = '/mnt/e/Datasets/houfin/bbl/raw_2025/NABBP_2025_grp_37.csv'


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sb = Sandbox(os.environ['SBX_REFINED'])
    from harness import load_data_config
    dcfg = load_data_config()
    cols = ['band', 'event_type', 'event_month', 'event_year', 'lat_dd', 'lon_dd', 'bird_status', 'age_code',
            'how_obtained_code', 'present_condition']
    df = pd.read_csv(CSV, usecols=cols, dtype=str)
    for c in ('event_month', 'event_year'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ('lat_dd', 'lon_dd'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['event_month', 'event_year', 'lat_dd', 'lon_dd'])
    df['by'] = np.where(df.event_month >= 4, df.event_year, df.event_year - 1).astype(int)
    B = df[(df.event_type == 'B') & (df.bird_status == '3') & (df.age_code.isin(['1', '5', '6', '7', '8']))].copy()
    B = B[(B.lon_dd < -100) | B.event_month.between(4, 8)]
    E = df[(df.event_type == 'E') & (df.present_condition.isin(['03', '04', '05']))
           & (~df.how_obtained_code.isin(['98', '51']))].copy()
    E = E.sort_values('by').drop_duplicates('band', keep='first')[['band', 'by']].rename(columns={'by': 'ey'})
    B = B.drop_duplicates('band', keep='first').merge(E, on='band', how='left')
    B['k'] = B.ey - B.by
    B = B[~(B.k == 0)]                       # recovered in the banding year: dropped (conditioning on the first partial year)
    B = B[(B.k.isna()) | (B.k >= 1)]
    # cell assignment
    with rasterio.open(dcfg['grid']['ref_raster']) as src:
        xs, ys = crs_transform('EPSG:4326', src.crs, list(B.lon_dd), list(B.lat_dd))
        rr, cc = rasterio.transform.rowcol(src.transform, xs, ys)
    rr, cc = np.asarray(rr), np.asarray(cc)
    ok = (rr >= 0) & (cc >= 0) & (rr < sb.lookup.shape[0]) & (cc < sb.lookup.shape[1])
    land = np.full(len(B), -1); land[ok] = sb.lookup[rr[ok], cc[ok]]
    B = B.assign(cell=land); B = B[B.cell >= 0]
    y0, y1 = int(sb.years[0]), int(sb.years[-1])
    B = B[(B.by >= max(y0, 1960)) & (B.by < y1)]
    K = y1 - int(B.by.min())
    grp = np.where(B.lon_dd < -100, 0, np.where(B.lon_dd < -80, 1, 2))
    B = B.assign(grp=grp)
    keys, inv = np.unique(np.c_[B.cell.values, B.by.values], axis=0, return_inverse=True)
    n = np.bincount(inv, minlength=len(keys)).astype(float)
    m = np.zeros((len(keys), K))
    rec = B.k.notna().values
    np.add.at(m, (inv[rec], (B.k.values[rec] - 1).astype(int)), 1.)
    g = np.zeros(len(keys), int); g[inv] = B.grp.values
    np.savez_compressed(a.out, cell=keys[:, 0].astype(int), by=keys[:, 1].astype(int), n=n, m=m, grp=g, K=K,
                        year0=y0, year1=y1)
    print(dict(cohorts=len(keys), released=int(n.sum()), recoveries=int(m.sum()), K=K,
               by_group={int(k): (float(n[g == k].sum()), float(m[g == k].sum())) for k in range(3)}))


if __name__ == '__main__':
    main()
