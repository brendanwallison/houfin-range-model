"""Ceiling diagnostic: how much target-era summer community change can observed winter change carry?

ORACLE: the winter -> summer coordinate map is cross-fitted over 6x6-block folds ON THE EVALUATION PANEL, i.e.
it uses target-era summer truth for fitting. It bounds what a correctly built winter route could add; it is
never a candidate. Winter input: the cached regional winter teacher field (change_cpu winter_fold-1: training
circles only, 16 neighbours, Gaussian 300 km, +/-2 years), averaged over each panel cell's own reserved survey
years. Summer truth: the observed fixed-teacher projection (teacher_bottleneck), cross-half covariance.
Predictors compared under the SAME cross-fitted ridge: field_d summer change alone (recalibrated), winter change
alone, both. Known and withheld places; all and regional (6x6-block means).

    python research/exp/z_winter_transfer.py --root R --arm field_d_s1
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.exp.z_repair_evaluate import matched_groups
from research.exp.z_repair_regional import partition_change
from research.exp.z_repair_teacher_bottleneck import score
from research.exp.z_repair_residual_fit import load_base


def crossfit(X, y, cells, lams=(1e-3, 1e-2, 1e-1, 1., 10.)):
    blocks = cells[:, 0]//6*100000 + cells[:, 1]//6
    ub, inv = np.unique(blocks, return_inverse=True)
    fold = np.random.default_rng(7).permutation(len(ub))[inv] % 5
    best = None
    for lam in lams:
        pred = np.zeros_like(y)
        for f in range(5):
            tr, te = fold != f, fold == f
            mx, my = X[tr].mean(0), y[tr].mean(0)
            A = (X[tr]-mx).T @ (X[tr]-mx)
            B = np.linalg.solve(A + lam*np.trace(A)/A.shape[0]*np.eye(A.shape[0]), (X[tr]-mx).T @ (y[tr]-my))
            pred[te] = (X[te]-mx) @ B + my
        err = ((pred - y)**2).mean()
        if best is None or err < best[0]:
            best = (err, lam, pred)
    return best[2], best[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm', default='field_d_s1')
    a = p.parse_args()
    root, task = a.root, 'backcast40'
    masks = dict(np.load(root/'data'/task/'summer_masks.npz'))
    keys = masks['keys']
    W = np.load(root/'change_cpu/winter_fold-1.npy', mmap_mode='r')
    obs = np.load(root/'teacher_bottleneck/observed_projection.npz')
    byrow = {int(r): i for i, r in enumerate(obs['source_rows'])}
    cells, spatial, panel = matched_groups(keys, masks, 2)
    groups = [g[e][h] for e, h in [(0, 0), (0, 1), (1, 0), (1, 1)] for g in panel]
    zobs = np.stack([obs['Z'][[byrow[int(r)] for r in g]].mean(0) for g in groups]).reshape(4, len(cells), 64)
    za, zb = zobs[2] - zobs[0], zobs[3] - zobs[1]
    rows = np.unique(np.concatenate(groups))
    model = load_base(root/'cubes'/task/a.arm, keys[rows])[:, :64].astype('float64')
    br = {int(r): i for i, r in enumerate(rows)}
    zm = np.stack([model[[br[int(r)] for r in g]].mean(0) for g in groups]).reshape(4, len(cells), 64)
    dm = .5*((zm[2] - zm[0]) + (zm[3] - zm[1]))
    wv = np.stack([np.asarray(W[np.asarray(g)], 'float64').mean(0) for g in groups]).reshape(4, len(cells), 64)
    dw = .5*((wv[2] - wv[0]) + (wv[3] - wv[1]))
    y = .5*(za + zb)
    result = {'scope': __doc__, 'arm': a.arm, 'panels': {}}
    for label, keep in [('known', ~spatial), ('spatial', spatial)]:
        c = cells[keep]
        entry = {}
        for name, X in [('raw_model', None), ('model_recal', dm[keep]), ('winter', dw[keep]),
                        ('model+winter', np.hstack([dm[keep], dw[keep]]))]:
            if X is None:
                pred, lam = dm[keep], None
            else:
                pred, lam = crossfit(X, y[keep], c)
            e = {'lambda': lam, 'all': score(c, pred, pred, za[keep], zb[keep], True)}
            rk, parts = partition_change(c, pred, za[keep], zb[keep])
            e['regional'] = score(c[rk], parts[0][0], parts[0][0], parts[1][0], parts[2][0], True)
            e['local'] = score(c[rk], parts[0][1], parts[0][1], parts[1][1], parts[2][1], True)
            entry[name] = e
            print(label, name, {k: (v['corr']['point'] if isinstance(v, dict) and 'corr' in v else v) for k, v in e.items()}, flush=True)
        result['panels'][label] = entry
    # NON-ORACLE: learn the winter -> summer change map on training-era contrasts at training cells only
    # (z_change_cpu training_pairs: 1996-2005 / 2006-15 / 2016-25 epochs, training rows), apply to the panel.
    tp = np.load(root/'change_cpu/training_pairs.npz')
    trows, ttarget, tkind, tcells = tp['rows'], tp['target'], tp['contrast'], tp['cells']
    tw = np.stack([np.asarray(W[r[4:]], 'float64').mean(0) - np.asarray(W[r[:4]], 'float64').mean(0) for r in trows])
    result['non_oracle'] = {}
    for tname, use in [('all_contrasts', np.ones(len(trows), bool)), ('span20_only', tkind == 2)]:
        X, Y, C = tw[use], ttarget[use].astype('float64'), tcells[use]
        _, lam = crossfit(X, Y, C)
        mx, my = X.mean(0), Y.mean(0)
        A = (X - mx).T @ (X - mx)
        B = np.linalg.solve(A + lam*np.trace(A)/A.shape[0]*np.eye(64), (X - mx).T @ (Y - my))
        result['non_oracle'][tname] = {'pairs': int(use.sum()), 'lambda': lam}
        for label, keep in [('known', ~spatial), ('spatial', spatial)]:
            c = cells[keep]
            for name, pred in [('winter_map', dw[keep] @ B),
                               ('model+winter_map', dm[keep] + dw[keep] @ B)]:
                e = {'all': score(c, pred, pred, za[keep], zb[keep], True)}
                rk, parts = partition_change(c, pred, za[keep], zb[keep])
                e['regional'] = score(c[rk], parts[0][0], parts[0][0], parts[1][0], parts[2][0], True)
                result['non_oracle'][tname][f'{label}_{name}'] = e
                print('NON-ORACLE', tname, label, name, e['all']['corr']['point'], e['regional']['corr']['point'], flush=True)
    out = root/'assim'/task/f'WINTER_TRANSFER_{a.arm}.json'
    out.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
