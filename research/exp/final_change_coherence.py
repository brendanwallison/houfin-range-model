"""Extend E014's measured change covariance across matched temporal spans."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.exp.route_nugget import KEYS, ruzicka

BINS = [(0, 10), (10, 20), (20, 30), (30, 60), (60, 150), (150, 300), (0, 30)]


def delta_cov(a, b, i, j, batch=4096):
    """Symmetric cross-half inner product of RKHS changes; a,b: route,epoch,species."""
    out = np.empty(len(i))
    for start in range(0, len(i), batch):
        sl = slice(start, start + batch)
        ii, jj = i[sl], j[sl]
        def one(x, y):
            return (ruzicka(x[ii, 0], y[jj, 0]) - ruzicka(x[ii, 0], y[jj, 1])
                    - ruzicka(x[ii, 1], y[jj, 0]) + ruzicka(x[ii, 1], y[jj, 1]))
        out[sl] = (one(a, b) + one(b, a)) / 2
    return out


def make_epochs(cov, R, starts):
    epochs = {}
    # Four years, two per half, with identical mean year in the two halves.
    for start in sorted(set(starts)):
        tab = cov[cov.Year.between(start, start + 3)]
        d = {}
        for key, g in tab.groupby(KEYS):
            g = g.sort_values('Year')
            if len(g) != 4 or not np.array_equal(g.Year.to_numpy(), np.arange(start, start + 4)):
                continue
            idx = g.k.to_numpy()
            a, b = np.log1p(R[idx[[0, 3]]].mean(0)), np.log1p(R[idx[[1, 2]]].mean(0))
            obs = g.observer.to_numpy()
            stable_obs = int(obs[0]) if obs[0] >= 0 and np.all(obs == obs[0]) else -1
            d[tuple(map(int, key))] = (a, b, stable_obs)
        epochs[start] = d
    return epochs


def ci(x):
    x = np.asarray(x)
    return np.nanpercentile(x, [2.5, 97.5]).tolist() if np.isfinite(x).sum() > 10 else None


def measure(keys, lags, anchor, epochs, route_info, W, block_index):
    n = len(keys)
    if n < 2:
        return {'n_routes': n, 'spans': {}}, {}
    coords = np.array([route_info[k][:2] for k in keys]) / 1000
    blocks = np.array([block_index[route_info[k][2]] for k in keys])
    pairs = cKDTree(coords).query_pairs(300, output_type='ndarray')
    i, j = pairs.T if len(pairs) else (np.array([], int), np.array([], int))
    dist = np.linalg.norm(coords[i] - coords[j], axis=1)
    out, boots = {'n_routes': n, 'spans': {}}, {}
    for lag in lags:
        e, m = epochs[anchor - lag], epochs[anchor]
        a = np.array([[e[k][0], m[k][0]] for k in keys])
        b = np.array([[e[k][1], m[k][1]] for k in keys])
        v = delta_cov(a, b, np.arange(n), np.arange(n))
        c = delta_cov(a, b, i, j)
        vpair = (v[i] + v[j]) / 2
        row = {'early': [anchor - lag, anchor - lag + 3],
               'modern': [anchor, anchor + 3], 'same_route_energy': float(v.mean()), 'bins': {}}
        boots[lag] = {}
        for lo, hi in BINS:
            label = f'{lo}_{hi}'
            s = (dist >= lo) & (dist < hi)
            nn = int(s.sum())
            item = {'n_pairs': nn, 'interpret': nn >= 30}
            row['bins'][label] = item
            if nn == 0:
                continue
            # Aggregate by unordered endpoint block pair before multiplier draws.
            bi, bj = np.minimum(blocks[i[s]], blocks[j[s]]), np.maximum(blocks[i[s]], blocks[j[s]])
            edges, inv = np.unique(np.column_stack([bi, bj]), axis=0, return_inverse=True)
            nblocks = int(len(np.unique(np.r_[bi, bj])))
            supported = nn >= 30 and nblocks >= 10
            weights = W[:, edges[:, 0]] * np.where(edges[:, 0] == edges[:, 1], 1, W[:, edges[:, 1]])
            cnt = np.bincount(inv, minlength=len(edges))
            cc = np.bincount(inv, weights=c[s], minlength=len(edges))
            vv = np.bincount(inv, weights=vpair[s], minlength=len(edges))
            mass = weights @ cnt
            cb, vb = (weights @ cc) / mass, (weights @ vv) / mass
            ratio = np.divide(cb, vb, out=np.full_like(cb, np.nan), where=vb > 0)
            boots[lag][label] = ratio
            C, V = float(c[s].mean()), float(vpair[s].mean())
            item.update(C=C, V_pair=V, local_difference_energy=V-C,
                        interpret=supported,
                        ratio=C/V if V > 0 else None,
                        ratio_E014_denominator=C/float(v.mean()) if v.mean() > 0 else None,
                        C_ci=ci(cb) if supported else None,
                        V_pair_ci=ci(vb) if supported else None,
                        ratio_ci=ci(ratio) if supported else None,
                        n_endpoint_blocks=nblocks)
        out['spans'][str(lag)] = row
    if 5 in boots:
        out['versus_5_years'] = {}
        for lag in lags:
            if lag == 5:
                continue
            out['versus_5_years'][str(lag)] = {
                k: {'ratio_difference': out['spans'][str(lag)]['bins'][k]['ratio']
                    - out['spans']['5']['bins'][k]['ratio'],
                    'interpret': out['spans'][str(lag)]['bins'][k]['interpret']
                    and out['spans']['5']['bins'][k]['interpret'],
                    'ci': ci(boots[lag][k] - boots[5][k])
                    if out['spans'][str(lag)]['bins'][k]['interpret']
                    and out['spans']['5']['bins'][k]['interpret'] else None}
                for k in boots[lag] if k in boots[5]
                and out['spans'][str(lag)]['bins'][k]['ratio'] is not None
                and out['spans']['5']['bins'][k]['ratio'] is not None}
    return out, boots


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-boot', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    root, out = Path(a.cache), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((root / 'DONE.json').read_text())
    R = np.load(root / 'counts.npy', mmap_mode='r')
    cov = pd.read_parquet(root / 'coverage.parquet')
    anchors = {2016: [5, 10, 20, 30, 40, 50], 2006: [5, 10, 20, 30, 40]}
    epochs = make_epochs(cov, R, [s for anchor, lags in anchors.items() for s in [anchor, *[anchor-t for t in lags]]])
    info = cov.drop_duplicates(KEYS)
    route_info = {tuple(int(r[k]) for k in KEYS): (r['x'], r['y'], (int(r['row'])//6, int(r['col'])//6))
                  for _, r in info.iterrows()}
    allblocks = sorted(set(v[2] for v in route_info.values()))
    block_index = {b: i for i, b in enumerate(allblocks)}
    W = np.random.default_rng(a.seed).exponential(size=(a.n_boot, len(allblocks)))
    result = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cache_metadata': meta, 'n_boot': a.n_boot, 'seed': a.seed,
              'epoch_width': 4, 'years_per_half': 2, 'anchors': {}}
    for anchor, lags in anchors.items():
        panels = {f'available_{lag}': [lag] for lag in lags}
        panels.update(all_spans=lags, through_30=[t for t in lags if t <= 30])
        panels.update({f'paired_5_{lag}': [5, lag] for lag in lags if lag != 5})
        result['anchors'][str(anchor)] = {}
        for name, spans in panels.items():
            keys = set(epochs[anchor])
            for lag in spans:
                keys &= set(epochs[anchor - lag])
            for stable in (False, True):
                kk = sorted(k for k in keys if not stable or (
                    epochs[anchor][k][2] >= 0 and
                    all(epochs[anchor-t][k][2] == epochs[anchor][k][2] for t in spans)))
                label = name + ('_same_observer' if stable else '')
                res, _ = measure(kk, spans, anchor, epochs, route_info, W, block_index)
                res['route_keys_sha256'] = hashlib.sha256(json.dumps(kk).encode()).hexdigest()
                result['anchors'][str(anchor)][label] = res
                print(anchor, label, len(kk), 'routes', flush=True)
        (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('Completed:', out / 'result.json', flush=True)


if __name__ == '__main__':
    main()
