"""Transient handle on the disease toll (review 8, C5): does the post-arrival DECLINE discriminate along the ridge?

For each ridge arm (ridge_maps.py npz: route rows y / expected mu / year / cell), per region: log of (sum over routes of
counts in years onset+2 .. onset+4) / (same routes, onset-3 .. onset-1), onset = the documented arrival map's per-cell year
(data disease_onset). Same routes in both windows (route-level pairing via cell + both windows present). Observed vs
each arm's expected. If arms differ by > ~.2 in the expected log decline while the observed is fixed, the decline is an
independent handle on the toll (a likelihood / prior on it would locate the ridge position).
Usage (inside sbx_env.sh): python decline_readout.py --maps DIR --arms base,c0.35,c0.25,c0.15
"""
import argparse, json, os
from pathlib import Path
import numpy as np
from harness import Sandbox


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--maps', required=True); p.add_argument('--arms', required=True)
    a = p.parse_args()
    sb = Sandbox(os.environ['SBX_REFINED']); yrs = np.asarray(sb.years)
    onset_land = yrs[0] + np.round(np.asarray(sb.data['disease_onset'], float))   # documented arrival, timesteps -> years
    lut = {(r, c): i for i, (r, c) in enumerate(zip(sb.rows, sb.cols))}
    out = dict(onset_source='data disease_onset (documented arrival map, timesteps) + year0', by_arm={})
    for arm in a.arms.split(','):
        d = np.load(Path(a.maps) / f'{arm}.npz')
        cell = np.array([lut.get((r, c), -1) for r, c in zip(d['orow'], d['ocol'])])
        ok = cell >= 0
        on = np.where(ok, onset_land[np.maximum(cell, 0)], np.nan)
        k = d['oy'] - on
        pre, post = (k >= -3) & (k <= -1), (k >= 2) & (k <= 4)
        res = {}
        for reg in ('east', 'plains', 'west'):
            inreg = ok & np.asarray(sb.region[reg])[np.maximum(cell, 0)]
            cells_both = np.intersect1d(np.unique(cell[inreg & pre]), np.unique(cell[inreg & post]))
            s_pre = inreg & pre & np.isin(cell, cells_both); s_post = inreg & post & np.isin(cell, cells_both)
            n_pre, n_post = s_pre.sum(), s_post.sum()
            if n_pre < 20 or n_post < 20:
                continue
            # per-row means so unequal row counts in the two windows do not bias the ratio
            res[reg] = dict(cells=int(cells_both.size), rows_pre=int(n_pre), rows_post=int(n_post),
                            obs_log_decline=float(np.log(d['y'][s_post].mean() / d['y'][s_pre].mean())),
                            exp_log_decline=float(np.log(d['mu'][s_post].mean() / d['mu'][s_pre].mean())))
        out['by_arm'][arm] = res
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
