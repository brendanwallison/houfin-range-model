"""Unit-free front readout: when an eastern cell is first colonised, how full were the already-colonised cells within
2 grid steps, relative to their OWN later settled density (median 2010-2025)? Low ratios mean the front is carried by
cells far below what they eventually hold. Colonised = density >= frac x the reference arm's modern occupied median
(as R6). Reads saved *_density.npz + cells.npz; no simulation.
Usage: python front_fullness.py DIR ARM [ARM ...] [--frac .01]
"""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dir', type=Path); p.add_argument('arms', nargs='+'); p.add_argument('--frac', type=float, default=.01)
    p.add_argument('--ref', default='fitted')
    a = p.parse_args()
    c = np.load(a.dir / 'cells.npz'); rows, cols, lon, ok = c['rows'], c['cols'], c['lon'], c['ok']
    ref = np.load(a.dir / f'{a.ref}_density.npz'); years = ref['years']
    base_med = float(np.nanmedian(ref['total'][years >= 2010][:, ok]))
    thr = a.frac * base_med
    east = lon >= -95.
    lookup = -np.ones((rows.max() + 3, cols.max() + 3), int); lookup[rows, cols] = np.arange(len(rows))
    nbr = []
    for r, cc in zip(rows, cols):
        ids = lookup[max(r - 2, 0):r + 3, max(cc - 2, 0):cc + 3].ravel()
        nbr.append(ids[ids >= 0])
    out = {}
    for arm in a.arms:
        tot = np.load(a.dir / f'{arm}_density.npz')['total'].astype(float)
        settled = np.median(tot[years >= 2010], 0)
        occ = tot >= thr
        first = np.where(occ.any(0), np.argmax(occ, 0), -1)
        ratios, eras = [], []
        for i in np.flatnonzero(east & (first > 0)):
            t = first[i]; nb = nbr[i]
            src = nb[(first[nb] >= 0) & (first[nb] < t) & (settled[nb] > thr)]
            if len(src):
                ratios.append(float(np.max(tot[t - 1, src] / settled[src]))); eras.append(int(years[t]))
        ratios, eras = np.array(ratios), np.array(eras)
        summ = {}
        for a0, b0 in [(1940, 1965), (1966, 1979), (1980, 1993), (1994, 2025)]:
            s = (eras >= a0) & (eras <= b0)
            if s.sum() >= 5:
                q = np.percentile(ratios[s], [10, 50, 90])
                summ[f'{a0}-{b0}'] = dict(n=int(s.sum()), p10=round(q[0], 4), median=round(q[1], 4), p90=round(q[2], 4))
        out[arm] = summ
        print(arm, json.dumps(summ))
    (a.dir / f'front_fullness_frac{a.frac}.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
