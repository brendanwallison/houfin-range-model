"""Build record-level reconstruction masks and training-only community sources.

Never reads sealed matrices, House Finch data, or existing encoder outputs.
Derived ESK bases and observation heads must be fitted from these sources anew.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.community_encoder.train_DESK.validation_core import abba_halves
from src.community_encoder.train_DESK.bbs_community_points import write_points

TASKS = {'backcast40': {'target': [1966, 1985], 'summer_start': 1996},
         'backcast20': {'target': [1986, 2005], 'summer_start': 2006}}
# --production (FIELD54_NATIVE_PLAN 2026-10-29): nothing withheld -- every valid summer row from 1966 on trains, every
# valid winter list trains, sealed species allowed (House Finch still forbidden). Same package layout, so the
# teacher / auxiliary / DESK entry points read it unchanged.
PRODUCTION = {'target': None, 'summer_start': 1966, 'production': True}
WINTERS = {'full': (1., False), 'sparse10': (.1, False), 'no_target': (1., True)}


def production_summer_masks(keys, land):
    """Every valid (on-land) row trains; no evaluation roles, no panel."""
    keys = np.asarray(keys)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError('Expected one raw BBS row per cell-year')
    zero = np.zeros(land.shape, bool)
    valid, ho, bf = grid_flags(keys, zero, zero, land)
    result = {name: np.zeros(len(keys), bool) for name in ['modern_a', 'modern_b', 'historical_a', 'historical_b']}
    result.update(train=valid.copy(), spatial_holdout=ho, buffer=bf, valid=valid,
                  panel=np.zeros((0, 7), int))
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def stable_fraction(identifier, fraction, salt='ZREPAIR20261026'):
    return int.from_bytes(hashlib.sha256(f'{salt}:{identifier}'.encode()).digest()[:8], 'big') / 2**64 < fraction


def grid_flags(keys, holdout, buffer, land):
    r, c = np.asarray(keys[:, 0], int), np.asarray(keys[:, 1], int)
    valid = (r >= 0) & (c >= 0) & (r < land.shape[0]) & (c < land.shape[1])
    rr, cc = np.clip(r, 0, land.shape[0]-1), np.clip(c, 0, land.shape[1]-1)
    valid &= land[rr, cc]
    return valid, valid & holdout[rr, cc], valid & buffer[rr, cc]


def summer_masks(keys, holdout, buffer, land, task):
    keys = np.asarray(keys)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError('Expected one raw BBS row per cell-year')
    valid, ho, bf = grid_flags(keys, holdout, buffer, land)
    names = ['train', 'modern_a', 'modern_b', 'historical_a', 'historical_b']
    result = {name: np.zeros(len(keys), bool) for name in names}
    panel = []
    for (r, c), group in pd.DataFrame({'r': keys[:, 0], 'c': keys[:, 1],
                                      'idx': np.arange(len(keys))}).groupby(['r', 'c'], sort=True):
        rows = group.idx.to_numpy()
        if not valid[rows[0]] or bf[rows[0]]:
            continue
        modern = rows[keys[rows, 2] >= task['summer_start']]
        historical = rows[(keys[rows, 2] >= task['target'][0]) & (keys[rows, 2] <= task['target'][1])]
        spatial = bool(ho[rows[0]])
        if spatial:
            if len(modern) >= 4:
                ma, mb = abba_halves(modern, keys[modern, 2])
            else:
                ma = mb = np.array([], int)
        elif len(modern) >= 8:
            # Independent modern training data, followed by two evaluation halves.
            tr, evaluation = abba_halves(modern, keys[modern, 2])
            ma, mb = abba_halves(evaluation, keys[evaluation, 2])
            result['train'][tr] = True
        else:
            result['train'][modern] = True
            ma = mb = np.array([], int)
        result['modern_a'][ma], result['modern_b'][mb] = True, True
        if len(historical) >= 4:
            ea, eb = abba_halves(historical, keys[historical, 2])
            result['historical_a'][ea], result['historical_b'][eb] = True, True
            if len(ma) >= 2 and len(mb) >= 2:
                panel.append((int(r), int(c), int(spatial), len(ea), len(eb), len(ma), len(mb)))
    total = sum(result.values())
    assert np.max(total, initial=0) <= 1
    assert not np.any(result['train'] & (keys[:, 2] < task['summer_start']))
    assert not np.any(result['train'] & (ho | bf | ~valid))
    result['spatial_holdout'] = ho
    result['buffer'] = bf
    result['valid'] = valid
    result['panel'] = np.asarray(panel, int).reshape(-1, 7)
    return result


def winter_masks(keys, circle, holdout, buffer, land, target, fraction=1., exclude_target=False):
    valid, ho, bf = grid_flags(keys, holdout, buffer, land)
    valid &= (keys[:, 2] >= 1902) & (keys[:, 2] <= 2024)
    # A circle may relocate: never train it under a different cell label.
    test_circles = np.unique(circle[ho])
    buffer_circles = np.setdiff1d(np.unique(circle[bf]), test_circles)
    evaluation = valid & np.isin(circle, test_circles) & ~bf
    train = valid & ~np.isin(circle, np.r_[test_circles, buffer_circles]) & ~bf
    selected = np.array([stable_fraction(int(c), fraction) for c in circle])
    train &= selected
    if exclude_target:
        train &= ~((keys[:, 2] >= target[0]) & (keys[:, 2] <= target[1]))
    assert not set(circle[train]) & set(circle[evaluation])
    return {'train': train, 'evaluation': evaluation, 'valid': valid,
            'excluded_circle_buffer': np.isin(circle, buffer_circles)}


def fit_benchmarks(presence, circle, years, lat, lon, train, ncircles, bench_from=1950):
    """Existing >=50%, >=5-year benchmark rule, fitted only on allowed lists.

    Nearest-circle borrowing uses the existing <=1-degree convention. Benchmark
    identities never use evaluation detections. All species are nonsealed.
    """
    P = np.asarray(presence, bool)
    late = np.asarray(train, bool) & (years >= bench_from)
    bench = np.zeros((ncircles, P.shape[1]), bool)
    source = np.full(ncircles, -1, int)
    for c in np.unique(circle[late]):
        rows = late & (circle == c)
        if len(np.unique(years[rows])) >= 5:
            bench[c] = P[rows].mean(0) >= .5
            source[c] = c
    eligible = np.flatnonzero(source >= 0)
    if not len(eligible):
        raise ValueError('No eligible training-only completeness benchmarks')
    _, first = np.unique(circle, return_index=True)
    coords = np.full((ncircles, 2), np.nan)
    coords[circle[first]] = np.column_stack([lat[first], lon[first]])
    good = eligible[np.isfinite(coords[eligible]).all(1)]
    tree = cKDTree(coords[good])
    for c in np.flatnonzero((source < 0) & np.isfinite(coords).all(1)):
        distance, index = tree.query(coords[c])
        if distance <= 1.:
            donor = good[index]
            bench[c], source[c] = bench[donor], donor
    return bench, source


def winter_points(D, train):
    """Reuse final_cbc_aux's count standardization, fitted on training lists only."""
    ph = np.asarray(D['party_hours'], float)
    use = train & (D['year_end'] >= 1966) & np.isfinite(ph) & (ph > 0)
    if not use.any():
        raise ValueError('No count-quality winter training lists')
    reference = float(np.median(ph[use]))
    counts = np.asarray(D['Y'][use], float) * (reference/ph[use, None])**.8
    tab = pd.DataFrame(counts)
    tab['row'], tab['col'], tab['year'] = D['row'][use], D['col'][use], D['year_end'][use]
    g = tab.groupby(['row', 'col', 'year'], sort=True).mean()
    keys = np.asarray(g.index.tolist(), np.int32)
    return np.log1p(g.to_numpy()).astype('float32'), keys, reference, np.flatnonzero(use)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ['base', 'points', 'cbc', 'out']:
        ap.add_argument('--'+k, required=True, type=Path)
    ap.add_argument('--holdout', type=Path, default=None)
    ap.add_argument('--buffer', type=Path, default=None)
    ap.add_argument('--tasks', nargs='+', choices=sorted(TASKS), default=sorted(TASKS),
                    help='masked tasks to package (default: all, as before)')
    ap.add_argument('--winters', nargs='+', choices=sorted(WINTERS), default=['full', 'sparse10', 'no_target'],
                    help='winter variants to package (default: all, as before)')
    ap.add_argument('--production', action='store_true',
                    help='one task "production": nothing withheld, sealed species allowed, House Finch forbidden')
    args = ap.parse_args()
    if args.production:
        if args.holdout is not None or args.buffer is not None:
            raise ValueError('--production withholds nothing; do not pass --holdout/--buffer')
    elif args.holdout is None or args.buffer is None:
        raise ValueError('--holdout and --buffer are required for masked tasks')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'DONE.json').exists():
        raise RuntimeError('Refuse to overwrite an existing completed package')
    meta = json.loads((args.base/'meta.json').read_text())
    pm = json.loads((args.points/'points_meta.json').read_text())
    allowed = set(meta['community']) | set(meta['dev_species'])
    forbidden = set(meta['sealed_species']) | {'houfin'}
    if args.production:
        allowed, forbidden = allowed | set(meta['sealed_species']), {'houfin'}
    if allowed & forbidden or not set(pm['species']) <= allowed:
        raise ValueError('Training species are not exclusively nonsealed/nonfocal')
    if pm.get('target_source') != 'bbs_raw' or pm['bbs'].get('temporal_ema_tau') != 0:
        raise ValueError('Require unsmoothed raw BBS cell-year points')
    keys = np.load(args.points/'point_index.npy')
    assert np.array_equal(keys, np.load(args.base/'keys.npy'))
    X = np.load(args.points/'X_points.npy', mmap_mode='r')
    W = np.load(args.points/'point_weights.npy')
    from src.data.preprocess.bbs import load_grid_reference, MASK_PATH
    land, *_ = load_grid_reference(MASK_PATH)
    land = np.asarray(land, bool)
    if args.production:
        holdout = buffer = np.zeros(land.shape, bool)
    else:
        holdout, buffer = np.load(args.holdout), np.load(args.buffer)
    assert land.shape == holdout.shape == buffer.shape
    D = dict(np.load(args.cbc, allow_pickle=False))
    assert set(map(str, D['species'])) <= allowed and not set(map(str, D['species'])) & forbidden
    ckeys = np.column_stack([D['row'], D['col'], D['year_end']]).astype(int)
    provenance = {str(p): digest(p) for p in [Path(__file__), args.base/'meta.json',
                  args.points/'points_meta.json', args.points/'point_index.npy',
                  args.points/'X_points.npy', args.points/'point_weights.npy',
                  args.cbc] + ([] if args.production else [args.holdout, args.buffer])}
    tasks = {'production': PRODUCTION} if args.production else {k: TASKS[k] for k in args.tasks}
    winters = ['full'] if args.production else args.winters
    summary = {'source_sha256': provenance, 'tasks': {},
               'encoder_initialization': 'random; current final checkpoints prohibited for masked emulation',
               'status': 'Training-only sources prepared; new teachers/heads/encoders not yet fitted.',
               'modern_split': 'Nested existing ABBA: training half, then two disjoint evaluation halves.',
               'known_evaluation_holdouts_reused': True}
    if args.production:
        summary['production'] = PRODUCTION
        summary['status'] = 'Production sources: nothing withheld; sealed species included; House Finch excluded.'
    for name, task in tasks.items():
        out = args.out/name
        out.mkdir(exist_ok=True)
        b = production_summer_masks(keys, land) if args.production else summer_masks(keys, holdout, buffer, land, task)
        np.savez_compressed(out/'summer_masks.npz', keys=keys, **b)
        use = b['train']
        pmeta = {**pm, 'n_rows': int(use.sum()), 'n_recent': int((keys[use, 2] == pm['recent_year']).sum()),
                 'n_hist': int((keys[use, 2] != pm['recent_year']).sum()),
                 'z_repair_task': task, 'source_rows_sha256': hashlib.sha256(np.flatnonzero(use).tobytes()).hexdigest(),
                 'bbs': {**pm['bbs'], 'n_rows': int(use.sum()),
                         'year_range': [int(keys[use, 2].min()), int(keys[use, 2].max())]},
                 'rebuild_required': ['ESK landmarks', 'window augmentation', 'coordinate targets', 'DESK initialization']}
        write_points(out/'summer_train_points', X[use], keys[use], W[use], pmeta)
        np.save(out/'summer_train_source_rows.npy', np.flatnonzero(use))
        np.save(out/'holdout_cells.npy', holdout)
        np.save(out/'buffer_cells.npy', buffer)
        pd.DataFrame({'species_code': pm['species']}).to_csv(out/'summer_species.csv', index=False)
        info = {'specification': task, 'summer_training_rows': int(use.sum()),
                'summer_panel_known_cells': int((b['panel'][:, 2] == 0).sum()),
                'summer_panel_spatial_cells': int((b['panel'][:, 2] == 1).sum()),
                'summer_role_counts': {k: int(b[k].sum()) for k in ['train','modern_a','modern_b','historical_a','historical_b']},
                'winter': {}}
        for label in winters:
            fraction, no_target = WINTERS[label]
            wout = out/('winter_'+label)
            wout.mkdir(exist_ok=True)
            masks = winter_masks(ckeys, D['circle_idx'], holdout, buffer, land, task['target'], fraction, no_target)
            np.savez_compressed(wout/'record_masks.npz', keys=ckeys, circle=D['circle_idx'], **masks)
            train = masks['train']
            # These arrays contain training records only; head builders never receive eval lists.
            rows = np.flatnonzero(train)
            listdata = {k: v[train] for k, v in D.items() if k not in ['species','reference','circles','strata']
                        and isinstance(v, np.ndarray) and len(v) == len(train)}
            np.savez_compressed(wout/'train_lists.npz', **listdata, species=D['species'],
                                source_rows=rows)
            xx, kk, ref_effort, countrows = winter_points(D, train)
            write_points(wout/'count_points', xx, kk, np.ones(len(kk), 'float32'),
                         {'source': 'cbc_winter', 'target_source': 'cbc_raw_training_only',
                          'species': list(map(str, D['species'])), 'n_species': len(D['species']),
                          'n_weeks': 1, 'n_rows': len(kk), 'n_hist': len(kk), 'n_recent': 0,
                          'recent_year': 2024, 'ruzicka_log1p': True,
                          'effort_reference_training_median': ref_effort,
                          'effort': 'party-hours^0.8 to training median', 'z_repair_task': task})
            np.save(wout/'count_source_rows.npy', countrows)
            benchmark, donor = fit_benchmarks(D['Y'] > 0, D['circle_idx'], D['year_end'],
                                             D['lat'], D['lon'], train, len(D['circles']))
            # Sufficient statistics support leave-one-species-out completeness at fit/eval time.
            bn = benchmark[D['circle_idx']].sum(1)
            bh = ((D['Y'] > 0) & benchmark[D['circle_idx']]).sum(1)
            np.savez_compressed(wout/'completeness.npz', in_bench=benchmark, benchmark_source=donor,
                                bench_n=bn, bench_hit=bh, species=D['species'],
                                training_source_rows=rows)
            info['winter'][label] = {'training_lists': int(train.sum()),
                'evaluation_lists': int(masks['evaluation'].sum()),
                'training_circles': int(len(np.unique(D['circle_idx'][train]))),
                'count_cell_years': len(kk), 'effort_reference': ref_effort,
                'target_era_training_lists': None if task['target'] is None else
                    int((train & (ckeys[:,2] >= task['target'][0]) & (ckeys[:,2] <= task['target'][1])).sum()),
                'early1902_1939_training_lists': int((train & (ckeys[:,2] <= 1939)).sum()),
                'benchmark_fit': 'only training lists, years>=1950; >=5 years; prevalence>=0.5; nonsealed species only',
                'existing_auxiliary_files_reused': False}
        (out/'manifest.json').write_text(json.dumps(info, indent=2))
        summary['tasks'][name] = info
        print(name, json.dumps(info), flush=True)
    (args.out/'DONE.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
