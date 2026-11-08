"""Collector-filtered specimen effort (Erickson & Smith 2021 'occupancy-collection models': restrict target-group effort to
collectors likely to collect the focal species; whether a collector ever collected it is a key predictor).

E_cd' = E_cd x (share of the cell's Aves effort made by 'finch-aware' collectors), where finch-aware = recordedBy name with
>= 1 House Finch specimen 1800-1939 anywhere (records.csv). The effort file's per-cell collector facet ('rec') is not split
by decade, so the share is per cell (an approximation). Rows with no finch-aware effort and no finch specimen get weight 0.
Names are matched on (surname, first initial) after removing 'Collector(s):' and punctuation; multi-collector strings use
the first name.

Usage: specimen_collector_cache.py IN_CACHE.npz OUT_CACHE.npz
"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

G = Path('/mnt/e/Datasets/houfin/external/gbif_hofi_1800_1940')


def key(name):
    if not isinstance(name, str) or not name.strip():
        return None
    s = re.sub(r'(?i)collector\(s\):', '', name)
    s = re.split(r'\s*(?:&|;| and |\|)\s*', s.strip())[0]
    s = re.sub(r'[^A-Za-z, ]', ' ', s).strip()
    if not s:
        return None
    if ',' in s:
        sur, rest = s.split(',', 1); first = rest.strip()[:1]
    else:
        parts = s.split(); sur, first = parts[-1], (parts[0][:1] if len(parts) > 1 else '')
    sur = sur.strip().lower()
    return (sur, first.lower()) if sur else None


def main():
    src, dst = sys.argv[1], sys.argv[2]
    R = pd.read_csv(G / 'records.csv', usecols=['year', 'recordedBy'])
    finch = {k for k in (key(n) for n in R[R.year < 1940].recordedBy.dropna()) if k is not None}
    finch_sur = {k[0] for k in finch}
    eff = json.loads((G / 'effort_aves_by_year_1880_1940.json').read_text())
    share = {}
    for c, v in eff.items():
        rec = v.get('rec', {}) or {}
        tot = float(sum(rec.values()))
        if tot <= 0:
            share[int(c)] = np.nan; continue
        fa = 0.
        for n, cnt in rec.items():
            k = key(n)
            if k is not None and (k in finch or (k[1] == '' and k[0] in finch_sur)):
                fa += cnt
        share[int(c)] = fa / tot
    C = {k: np.asarray(v) for k, v in np.load(src).items()}
    sh = np.array([share.get(int(c), np.nan) for c in C['cell']])
    sh = np.where(np.isfinite(sh), sh, 1.)          # no collector facet: keep the unfiltered effort
    E2 = C['E'].astype(float) * sh
    w2 = C['w'].astype(float).copy(); w2[(E2 <= 0) & (C['y'] <= 0)] = 0.
    E2 = np.where(E2 > 0, E2, np.maximum(C['E'].astype(float) * .01, 1e-3))   # finch rows with zero filtered effort: tiny effort
    C['E'] = E2; C['w'] = w2; C['collector_share'] = sh
    np.savez_compressed(dst, **C)
    print(json.dumps(dict(finch_collector_keys=len(finch), cells_with_facet=int(np.isfinite([share.get(int(c), np.nan) for c in np.unique(C['cell'])]).sum()),
                          median_share=float(np.median(sh)), rows=int(sh.size), rows_dropped=int(((w2 == 0) & (np.asarray(np.load(src)['w']) > 0)).sum()))))


if __name__ == '__main__':
    main()
