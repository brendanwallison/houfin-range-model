"""Score any backcast40 cube arm with the established z_repair estimands, stratified by residency.

Reuses z_repair_evaluate's panel construction (matched_groups), readout (one modern spatial ridge per species,
reused over time) and cross-half estimand (bootstrap_components); z_repair_regional's 6x6-block partition; the
teacher-bottleneck observed summer projection; and z_repair_winter_evaluate's held-out-circle composition score.
Every arm is paired against control_s1 on identical cells and bootstrap draws.

Strata (memory/CONSIDERATIONS G11): sedentary common (AVONET migration 1, training-row presence >= 0.08) first,
then sedentary, then all nonsealed targets. Targets are community + dev species; no sealed species or House Finch.

    python research/exp/z_assim_evaluate.py --root R --base B --similarity S --arms control_s1 assim_d_s1 ...

FIELD54 options (defaults reproduce the 27 km scores): --task; --block-cells (regional partition, block bootstrap
and ridge-CV fold blocks; default 6 = 162 km at 27 km, 3 keeps 162 km at 54 km; caches are keyed by it);
--no-teacher ARM... (skip the summer64 teacher score for arms whose dims are not in this package's teacher space,
e.g. a block-averaged 27 km cube).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.exp.z_repair_evaluate import matched_groups, bootstrap_components, weights_for_cells
from research.exp.z_repair_regional import partition_change
from research.exp.z_repair_compare import summary
from research.exp.z_repair_residual_fit import load_base
from research.exp.e069_presence_score import ridge_cv
from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed

TASK = 'backcast40'
BC = 6  # block size in cells; set from --block-cells


def stats(pred, ta, tb, cells, cols=None):
    if cols is not None:
        pred, ta, tb = pred[:, cols], ta[:, cols], tb[:, cols]
    W, _ = weights_for_cells(cells)
    s = bootstrap_components(pred, ta, tb, W)
    return s


def regional_parts(cells, pred, ta, tb):
    keep, parts = partition_change(cells, pred, ta, tb, block_cells=BC)
    return keep, parts


def species_scores(root, base, arm, strata, Y, keys, masks):
    z = load_base(root/'cubes'/TASK/arm, keys).astype('float64')
    training, fitcells = [], []
    for row, col in np.unique(keys[masks['train'], :2], axis=0):
        rows = np.flatnonzero(masks['train'] & (keys[:, 0] == row) & (keys[:, 1] == col))
        if len(rows) >= 4:
            training.append(rows); fitcells.append((row, col))
    fitcells = np.asarray(fitcells, int)
    targets = epoch_mean_observed(Y, training).astype('float64')
    F = np.stack([z[g].mean(0) for g in training])
    predict, lam = ridge_cv(F, targets, fitcells[:, 0]//BC*100000 + fitcells[:, 1]//BC,
                            [1e-4, 1e-3, 1e-2, 1e-1, 1., 10.], k=5, seed=69)
    out = {}
    for k in [2, 4]:
        cells, spatial, panel = matched_groups(keys, masks, k)
        ea, eb, ma, mb = ([g[e][h] for g in panel] for e, h in [(0, 0), (0, 1), (1, 0), (1, 1)])
        E1, E2 = epoch_mean_observed(Y, ea).astype(float), epoch_mean_observed(Y, eb).astype(float)
        M1, M2 = epoch_mean_observed(Y, ma).astype(float), epoch_mean_observed(Y, mb).astype(float)
        early = predict(np.stack([z[np.r_[x, y]].mean(0) for x, y in zip(ea, eb)]))
        modern = predict(np.stack([z[np.r_[x, y]].mean(0) for x, y in zip(ma, mb)]))
        ta, tb, D = M1 - E1, M2 - E2, modern - early
        hist = .5*((early - E1)**2 + (early - E2)**2)
        mod = .5*((modern - M1)**2 + (modern - M2)**2)
        for label, keep in [('known', ~spatial), ('spatial', spatial)]:
            for sname, cols in strata.items():
                key = f'k{k}_{label}_{sname}'
                c = cells[keep]
                out[key] = {'all': stats(D[keep], ta[keep], tb[keep], c, cols)}
                rk, parts = regional_parts(c, D[keep][:, cols], ta[keep][:, cols], tb[keep][:, cols])
                if rk.sum() >= 20:
                    Wr, _ = weights_for_cells(c[rk])
                    for comp, i in [('regional', 0), ('local', 1)]:
                        out[key][comp] = bootstrap_components(parts[0][i], parts[1][i], parts[2][i], Wr)
                W, _ = weights_for_cells(c)
                for nm, arr in (('historical_mse', hist), ('modern_mse', mod)):
                    h = arr[keep][:, cols].mean(1)
                    out[key][nm] = {'corr': (W @ h)/W.sum(1)}
    return out, lam


def teacher_scores(root, arm, keys, masks):
    """Summer64 coordinate change vs the observed fixed-teacher projection on reserved rows (k=2 panel)."""
    from research.exp.z_repair_teacher_bottleneck import score
    obs = np.load(root/'teacher_bottleneck/observed_projection.npz')
    byrow = {int(r): i for i, r in enumerate(obs['source_rows'])}
    cells, spatial, panel = matched_groups(keys, masks, 2)
    groups = [g[e][h] for e, h in [(0, 0), (0, 1), (1, 0), (1, 1)] for g in panel]
    zobs = np.stack([obs['Z'][[byrow[int(r)] for r in g]].mean(0) for g in groups]).reshape(4, len(cells), 64)
    rows = np.unique(np.concatenate(groups))
    model = load_base(root/'cubes'/TASK/arm, keys[rows])[:, :64].astype('float64')
    br = {int(r): i for i, r in enumerate(rows)}
    zmod = np.stack([model[[br[int(r)] for r in g]].mean(0) for g in groups]).reshape(4, len(cells), 64)
    za, zb = zobs[2] - zobs[0], zobs[3] - zobs[1]
    d = .5*((zmod[2] - zmod[0]) + (zmod[3] - zmod[1]))
    out = {}
    for label, keep in [('known', ~spatial), ('spatial', spatial)]:
        c = cells[keep]
        out[label] = {'all': score(c, d[keep], d[keep], za[keep], zb[keep], True)}
        rk, parts = partition_change(c, d[keep], za[keep], zb[keep], block_cells=BC)
        if rk.sum() >= 20:
            for comp, i in [('regional', 0), ('local', 1)]:
                out[label][comp] = score(c[rk], parts[0][i], parts[0][i], parts[1][i], parts[2][i], True)
    return out


def winter_scores(root, base, arm, cbc):
    """z_repair_winter_evaluate's held-out-circle conditional composition score, for any cube arm."""
    from research.exp.e069_presence_score import fit_delta, list_ll
    from research.exp.final_winter_score import ERAS, paired_circle_summary
    meta = json.loads((base/'meta.json').read_text())
    allowed = set(meta['community']) | set(meta['dev_species'])
    D = dict(np.load(cbc, allow_pickle=False))
    if not set(map(str, D['species'])) <= allowed:
        raise ValueError('Nondevelopment targets')
    wm = dict(np.load(root/'data'/TASK/'winter_full/record_masks.npz'))
    keys = np.column_stack([D['row'], D['col'], D['year_end']]).astype(int)
    assert np.array_equal(keys, wm['keys'])
    train = wm['train'] & (keys[:, 2] >= 2006) & np.isfinite(D['party_hours']) & (D['party_hours'] > 0)
    keep = (D['Y'][train] > 0).sum(0) >= 10
    Y = D['Y'][:, keep]; P = Y > 0
    train &= P.sum(1) >= 2
    test = wm['evaluation'] & (P.sum(1) >= 2)
    tr, te = np.flatnonzero(train), np.flatnonzero(test)
    cube = root/'cubes'/TASK/arm
    z = load_base(cube, keys[np.r_[tr, te]]).astype('float64')
    ztr, zte = z[:len(tr)], z[len(tr):]
    target = np.log1p(Y[tr]*((np.median(D['party_hours'][tr])/D['party_hours'][tr])[:, None]**.8))
    predict, _ = ridge_cv(ztr, target, keys[tr, 0]//BC*100000 + keys[tr, 1]//BC, [1e-4, 1e-3, 1e-2, 1e-1, 1., 10.])
    delta = fit_delta(predict(ztr), P[tr])
    cells, index = np.unique(keys[te, :2], axis=0, return_inverse=True)
    modern = np.zeros((len(cells), 128))
    for year in range(2006, 2025):
        modern += load_base(cube, np.column_stack([cells, np.full(len(cells), year)])).astype('float64')/19
    back = list_ll(predict(zte), delta, P[te])
    fixed = list_ll(predict(modern[index]), delta, P[te])
    circles, years = D['circle_idx'][te], keys[te, 2]
    out = {}
    for era, lo, hi in ERAS:
        m = (years >= lo) & (years <= hi)
        out[era] = paired_circle_summary(circles[m], {'backcast_minus_modern': (back - fixed)[m], 'backcast': back[m]})
    return out, {'circle': circles, 'year': years, 'gain': back - fixed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--similarity', type=Path, required=True)
    p.add_argument('--arms', nargs='+', required=True)
    p.add_argument('--out', default='ASSIM_SCORES')
    p.add_argument('--cbc', type=Path, default=None)
    p.add_argument('--score-only', action='store_true', help='compute and cache raw scores for the arms, no combine')
    p.add_argument('--task', default='backcast40')
    p.add_argument('--block-cells', type=int, default=6)
    p.add_argument('--no-teacher', nargs='*', default=[], help='arms scored without the summer64 teacher score')
    a = p.parse_args()
    global TASK, BC
    TASK, BC = a.task, a.block_cells
    import research.exp.z_repair_evaluate as zre
    zre.BLOCK_CELLS = BC
    root = a.root
    masks = dict(np.load(root/'data'/TASK/'summer_masks.npz'))
    keys = masks['keys']
    meta = json.loads((a.base/'meta.json').read_text())
    species = meta['community'] + meta['dev_species']
    if 'houfin' in species or set(species) & set(meta['sealed_species']):
        raise ValueError('Forbidden targets')
    assert np.array_equal(keys, np.load(a.base/'keys.npy'))
    Y = np.hstack([np.load(a.base/'X_comm.npy'), np.load(a.base/'X_dev.npy')]).astype('float64')
    sim = pd.read_csv(a.similarity)
    mig = dict(zip(sim.species_code, sim.migration))
    prev = (Y[masks['train']] > 0).mean(0)
    sed = np.array([mig.get(s) == 1.0 for s in species])
    strata = {'sed_common': np.flatnonzero(sed & (prev >= .08)), 'sedentary': np.flatnonzero(sed),
              'all': np.arange(len(species))}
    result = {'strata_sizes': {k: int(len(v)) for k, v in strata.items()}, 'task': TASK, 'block_cells': BC,
              'prevalence': 'presence fraction over permitted summer training rows',
              'arms': {}}
    raw = {}
    import pickle
    rawdir = root/'assim'/TASK/'raw'
    rawdir.mkdir(parents=True, exist_ok=True)
    for arm in a.arms:
        cache = rawdir/(f'{arm}.pkl' if BC == 6 else f'{arm}_b{BC}.pkl')
        if cache.exists():
            sp, te, lam, wpack = pickle.loads(cache.read_bytes())
        else:
            sp, lam = species_scores(root, a.base, arm, strata, Y, keys, masks)
            te = None if arm in a.no_teacher else teacher_scores(root, arm, keys, masks)
            wpack = winter_scores(root, a.base, arm, a.cbc) if a.cbc is not None else None
            cache.write_bytes(pickle.dumps((sp, te, lam, wpack)))
        if a.score_only:
            print('cached', arm, flush=True)
            continue
        raw[arm] = (sp, te)
        species_summary = {}
        for k, d in sp.items():
            species_summary[k] = {comp: (summary(v['corr']) if comp.endswith('_mse')
                                         else {m: summary(arr) for m, arr in v.items()})
                                  for comp, v in d.items()}
        result['arms'][arm] = {'ridge_lambda': lam, 'species': species_summary, 'teacher': te}
        if wpack is not None:
            ws, wraw = wpack
            result['arms'][arm]['winter'] = ws
            raw[arm] = raw[arm] + (wraw,)
        print('scored', arm, flush=True)
    if a.score_only:
        return
    # Paired differences vs the first arm (control) on identical draws.
    ref = a.arms[0]
    result['paired_vs_' + ref] = {}
    for arm in a.arms[1:]:
        dd = {}
        for k, d in raw[arm][0].items():
            dd[k] = {}
            for comp in ('all', 'regional', 'local'):
                if comp in d and comp in raw[ref][0][k]:
                    dd[k][comp] = {m: summary(d[comp][m] - raw[ref][0][k][comp][m])
                                   for m in ('corr', 'mse_gain_over_no_change')}
            for nm in ('historical_mse', 'modern_mse'):
                dd[k][nm] = summary(d[nm]['corr'] - raw[ref][0][k][nm]['corr'])
        if len(raw[arm]) > 2:
            from research.exp.final_winter_score import ERAS, paired_circle_summary
            wa, wr = raw[arm][2], raw[ref][2]
            assert np.array_equal(wa['circle'], wr['circle']) and np.array_equal(wa['year'], wr['year'])
            dd['winter_gain_difference'] = {}
            for era, lo, hi in ERAS:
                m = (wa['year'] >= lo) & (wa['year'] <= hi)
                dd['winter_gain_difference'][era] = paired_circle_summary(wa['circle'][m], {'diff': (wa['gain'] - wr['gain'])[m]})
        result['paired_vs_' + ref][arm] = dd
    out = root/'assim'/TASK/(a.out + '.json')
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    # Compact table: k2 known/spatial, sedentary common first.
    lines = ['| arm | panel | stratum | all corr | regional corr | local corr | slope | MSE gain vs no change | teacher all | teacher regional |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    f = lambda d: 'NA' if d is None or d['point'] is None else f"{d['point']:.3f}"
    for arm, r in result['arms'].items():
        for panel in ('known', 'spatial'):
            for s in ('sed_common', 'sedentary', 'all'):
                e = r['species'].get(f'k2_{panel}_{s}')
                if not e:
                    continue
                t = (r['teacher'] or {}).get(panel, {'all': {'corr': None}})
                lines.append(f"| {arm} | {panel} | {s} | {f(e['all']['corr'])} | {f(e.get('regional', {}).get('corr'))} | "
                             f"{f(e.get('local', {}).get('corr'))} | {f(e['all']['truth_on_prediction_slope'])} | "
                             f"{f(e['all']['mse_gain_over_no_change'])} | "
                             f"{f(t['all']['corr']) if s == 'all' else ''} | {f(t.get('regional', {}).get('corr')) if s == 'all' else ''} |")
    (root/'assim'/TASK/(a.out + '.md')).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
