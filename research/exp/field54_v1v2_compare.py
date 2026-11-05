"""FIELD54: how much did the states_v2 coastal climate fix change a 54 km cube? (v1 vs v2 builds of the same arm)

Affected cells = 54 km land cells where any covariate state differs between states/ and states_v2/ (checked on a few
years). Reports, over land cells x 124 years x all dims: correlation, max |change|, the share of cells that moved, and
the split between the affected (coastal) cells and the rest (any change there comes through the DESK's spatial
convolution and the field's smoothing, not from the covariates).

    python research/exp/field54_v1v2_compare.py --v1 CUBE --v2 CUBE --states1 S1 --states2 S2 --out out.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

YEARS = range(1902, 2026)


def affected_cells(s1, s2, shape, years=(1902, 1950, 2000, 2025)):
    aff = np.zeros(shape, bool)
    for y in years:
        a, b = np.load(Path(s1)/'yearly_states'/f'state_{y}.npz'), np.load(Path(s2)/'yearly_states'/f'state_{y}.npz')
        for k in a.files:
            x, z = np.asarray(a[k]), np.asarray(b[k])
            if x.shape != z.shape or x.dtype.kind not in 'fiu':
                continue
            d = ~np.isclose(x, z, equal_nan=True, rtol=0, atol=0)
            if x.ndim >= 2 and x.shape[:2] == shape:
                aff |= d.reshape(shape + (-1,)).any(-1)
            elif x.ndim >= 2 and x.shape[-2:] == shape:
                aff |= d.reshape((-1,) + shape).any(0)
    return aff


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for k in ('v1', 'v2', 'states1', 'states2', 'out'):
        p.add_argument('--' + k, required=True)
    a = p.parse_args()
    z0 = np.load(Path(a.v1)/'Z_latent_2000.npy')
    land = np.isfinite(z0).all(-1)
    aff = affected_cells(a.states1, a.states2, land.shape) & land
    cmax = np.zeros(land.shape)
    sxy = sxx = syy = sx = sy = 0.
    n = 0
    for y in YEARS:
        x = np.load(Path(a.v1)/f'Z_latent_{y}.npy')[land].astype('float64')
        z = np.load(Path(a.v2)/f'Z_latent_{y}.npy')[land].astype('float64')
        cmax[land] = np.maximum(cmax[land], np.abs(z - x).max(-1))
        sx += x.sum(); sy += z.sum(); sxx += (x*x).sum(); syy += (z*z).sum(); sxy += (x*z).sum(); n += x.size
    cov = sxy/n - sx/n*sy/n
    corr = cov/np.sqrt((sxx/n - (sx/n)**2)*(syy/n - (sy/n)**2))
    rest = land & ~aff
    out = {'v1': a.v1, 'v2': a.v2, 'land_cells': int(land.sum()), 'covariate_affected_cells': int(aff.sum()),
           'corr_all_values': float(corr), 'max_abs_change': float(cmax[land].max()),
           'affected_cells': {'max_abs_change': float(cmax[aff].max()) if aff.any() else None,
                              'median_max_abs_change': float(np.median(cmax[aff])) if aff.any() else None},
           'other_cells': {'max_abs_change': float(cmax[rest].max()), 'median_max_abs_change': float(np.median(cmax[rest])),
                           'n_moved_gt_1e-3': int((cmax[rest] > 1e-3).sum()), 'n_moved_gt_1e-2': int((cmax[rest] > 1e-2).sum()),
                           'n_moved_gt_5e-2': int((cmax[rest] > 5e-2).sum())},
           'cells_moved_gt_1e-2': int((cmax[land] > 1e-2).sum()),
           'note': 'max |change| per cell over 124 years x all dims; teacher-space units (|z| ~ 0.8)'}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
