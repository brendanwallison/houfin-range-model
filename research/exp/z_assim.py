"""Observation-anchored community field: Z = M(s) + A.D(s,t) + R(s,t), fitted in teacher space.

M  fine-grained per-cell level offset on the control's modern mean map, smoothed over grid neighbours so
   unobserved cells interpolate (a Gaussian Markov random field penalty).
D  the masked control DESK's own covariate-driven change (control minus its 2006-2025 mean), passed
   through a learned 64x64 recalibration per season (ridge toward identity).
R  regional space-time field: shared latent scores a(s,t) = S(s) C T(t) (Gaussian RBF x cubic B-spline,
   P-spline penalty), with SEASON-SPECIFIC loadings, so winter observations reach summer coordinates only
   through couplings estimated where both seasons are observed.

Fits only permitted training sources from the z_repair package (summer 1996+, training winter circles),
never evaluation records, sealed species or House Finch. Exports cubes in the control's layout
(dims 0-63 summer teacher space, 64-127 winter teacher space) for the established scorers.

    python research/exp/z_assim.py inner --root R            # inner backcast: hyperparameter grid
    python research/exp/z_assim.py fit --root R --arm NAME [--winter full|sparse10|no_target] [--cfg JSON]

FIELD54 options (defaults reproduce the 27 km runs): --task (package task, default backcast40), --control-cube (the D
cube under cubes/<task>/, default control_s1), --production (task 'production': summer sources from 1966 allowed; the
>= 1996 guard stays on otherwise), --block-cells (inner-grid regional blocks, default 6 = 162 km at 27 km),
--inner-grid (JSON list of [arm, overrides] replacing the default inner grid). The RBF basis is in km (projected
cell-centre coordinates / 1000 from the data config's reference raster; centres by farthest-point selection over
observed cells; width scale_km), so it keeps its km scale on any grid. Export writes cube_meta.json: the D cube's
contract with the field provenance added (the field overwrites D's land cells in the same teacher spaces).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.exp.z_repair_models import CubicTimeBasis, GaussianRegionalBasis

TASK = 'backcast40'
YEARS = np.arange(1902, 2026)
REF = (2006, 2025)
ARMS = {
    # name: (use D, use R, R uses winter, temporal basis)
    'fine_d': (True, False, False, 'spline'),        # fine map + recalibrated DESK change, no R
    'trend_summer': (False, True, False, 'linear'),  # position x linear time from summer only: placebo analog
    'field_summer': (False, True, False, 'spline'),  # regional spline field from summer only
    'assim': (False, True, True, 'spline'),          # summer + winter field, no covariate change
    'assim_d': (True, True, True, 'spline'),         # full candidate
    'field_d': (True, True, False, 'spline'),        # added after the inner grid: fine map + DESK change + summer field
    'field_d_sep': (True, True, 'separate', 'spline'),  # added after first scores: + an independent winter field
}
DEFAULT = {'rank': 16, 'centers': 128, 'scale_km': 200., 'knot': 10, 'lam_lap': 1.0, 'lam_u': 1e-3,
           'lam_A': 1e-2, 'lam_C': 1e-2, 'lam_curv': 1.0, 'winter_weight': 1.0, 'steps': 2000, 'lr': 0.02}


def grid_xy(rows, cols):
    from src.config_utils import load_data_config
    import rasterio
    with rasterio.open(load_data_config()['grid']['ref_raster']) as ds:
        x, y = rasterio.transform.xy(ds.transform, rows, cols)
    return np.column_stack([x, y]) / 1000.


class Data:
    def __init__(self, root, winter='full', summer='train', control='control_s1', production=False):
        self.root = root
        self.control = control
        data, aux = root/'data'/TASK, root/'aux'/TASK/winter
        from src.data.preprocess.bbs import load_grid_reference, MASK_PATH
        land, *_ = load_grid_reference(MASK_PATH)
        self.land = np.asarray(land, bool)
        self.cells = np.argwhere(self.land)
        self.cid = -np.ones(self.land.shape, int)
        self.cid[self.cells[:, 0], self.cells[:, 1]] = np.arange(len(self.cells))
        self.xy = grid_xy(self.cells[:, 0], self.cells[:, 1])
        sk = np.load(data/'summer_train_points/point_index.npy').astype(int)
        if production:
            if TASK != 'production' or summer != 'train':
                raise ValueError('--production needs task production and summer train (all rows are training rows)')
        else:
            assert (sk[:, 2] >= 1996).all()
        self.s_keys, self.s_y = sk, np.load(root/'aux'/TASK/'full'/'summer_training_coords.npy').astype('float32')
        if summer == 'interp':
            # In-era interpolation test: add 1966-1995 summer rows at training places (z_assim_interp_prep.py).
            x = np.load(root/'aux'/TASK/'interp'/'summer_extra.npz')
            self.s_keys = np.vstack([self.s_keys, x['keys'].astype(int)])
            self.s_y = np.vstack([self.s_y, x['Z'].astype('float32')])
        w = np.load(aux/'winter_coords.npz')
        self.w_keys = np.column_stack([w['row'], w['col'], w['year']]).astype(int)
        self.w_y = w['Z'].astype('float32')
        for k in (self.s_keys, self.w_keys):
            if (self.cid[k[:, 0], k[:, 1]] < 0).any():
                raise ValueError('Observation outside land support')
        # Control cube on land cells: (cells, years, 128).
        cube = root/'cubes'/TASK/control
        self.C = np.empty((len(self.cells), len(YEARS), 128), 'float32')
        for j, y in enumerate(YEARS):
            g = np.load(cube/f'Z_latent_{y}.npy', mmap_mode='r')
            self.C[:, j] = g[self.cells[:, 0], self.cells[:, 1]]
        if not np.isfinite(self.C).all():
            raise ValueError('Non-finite control cube on land')
        if winter != 'full':
            # Each winter variant has its own teacher; the control's winter dims live in the FULL teacher's space.
            # The winter block is then offset + field in the variant's own space, with no control winter input.
            self.C[:, :, 64:] = 0.
        ref = (YEARS >= REF[0]) & (YEARS <= REF[1])
        self.Dbar = self.C[:, ref].mean(1)
        # Land-grid 4-neighbour pairs for the fine-map smoothness penalty.
        pairs = []
        for dr, dc in ((0, 1), (1, 0)):
            a = self.cells
            b = a + [dr, dc]
            ok = (b[:, 0] < self.land.shape[0]) & (b[:, 1] < self.land.shape[1])
            ok[ok] &= self.land[b[ok, 0], b[ok, 1]]
            pairs.append(np.column_stack([np.flatnonzero(ok), self.cid[b[ok, 0], b[ok, 1]]]))
        self.pairs = np.vstack(pairs)

    def rows(self, keys):
        return self.cid[keys[:, 0], keys[:, 1]], keys[:, 2] - YEARS[0]


class Field(torch.nn.Module):
    def __init__(self, d, cfg, arm):
        super().__init__()
        self.use_D, self.use_R, self.use_winter, tb = ARMS[arm]
        n = len(d.cells)
        self.u = torch.nn.Parameter(torch.zeros(n, 128))
        self.A = torch.nn.Parameter(torch.eye(64).repeat(2, 1, 1)) if self.use_D else None
        obs_xy = d.xy[np.unique(np.r_[d.cid[d.s_keys[:, 0], d.s_keys[:, 1]], d.cid[d.w_keys[:, 0], d.w_keys[:, 1]]])]
        self.S = torch.as_tensor(GaussianRegionalBasis(obs_xy, cfg['centers'], cfg['scale_km'])(d.xy), dtype=torch.float32)
        if tb == 'linear':
            T = ((YEARS - (REF[0] + REF[1]) / 2) / 50.)[:, None]
        else:
            T = CubicTimeBasis(reference=REF, knot_spacing=cfg['knot'])(YEARS)
        self.T = torch.as_tensor(T, dtype=torch.float32)
        r = cfg['rank']
        g = torch.Generator().manual_seed(20261027)
        # 'separate': each season has its own latent field (no shared scores, no winter -> summer coupling).
        self.separate = self.use_winter == 'separate'
        shape = (2, self.S.shape[1], self.T.shape[1], r) if self.separate else (self.S.shape[1], self.T.shape[1], r)
        self.Cr = torch.nn.Parameter(torch.zeros(shape)) if self.use_R else None
        self.L = torch.nn.Parameter(torch.randn(2, r, 64, generator=g) * .1) if self.use_R else None
        self.Dbar = torch.as_tensor(d.Dbar)
        self.Cc = d.C  # numpy, gathered per row
        self.cfg = cfg

    def latent(self, ci, ti, q=0):
        G = torch.einsum('np,pqr->nqr', self.S[ci], self.Cr[q] if self.separate else self.Cr)
        return torch.einsum('nqr,nq->nr', G, self.T[ti])

    def season(self, ci, ti, q, dD=None):
        sl = slice(64 * q, 64 * (q + 1))
        if q == 1 and not (self.use_D or self.use_winter):
            return self.Dbar[ci, sl] + dD  # summer-only arms keep the control's winter block unchanged
        z = self.Dbar[ci, sl] + self.u[ci, sl]
        if self.use_D:
            z = z + dD @ self.A[q].T
        if self.use_R and (q == 0 or self.use_winter):
            z = z + self.latent(ci, ti, q) @ self.L[q]
        return z

    def penalty(self, pairs):
        c = self.cfg
        p = c['lam_lap'] * (self.u[pairs[:, 0]] - self.u[pairs[:, 1]]).square().mean() + c['lam_u'] * self.u.square().mean()
        if self.use_D:
            p = p + c['lam_A'] * (self.A - torch.eye(64)).square().mean()
        if self.use_R:
            p = p + c['lam_C'] * self.Cr.square().mean()
            tdim = 2 if self.separate else 1
            if self.Cr.shape[tdim] >= 3:
                p = p + c['lam_curv'] * torch.diff(self.Cr, n=2, dim=tdim).square().mean()
            p = p + 1e-3 * (self.L.square().mean() - 1.).square()  # loading scale fixes the C/L factorization
        return p


def fit(d, cfg, arm, summer_mask=None, log=None):
    torch.manual_seed(20261027)
    m = Field(d, cfg, arm)
    ref = (YEARS >= REF[0]) & (YEARS <= REF[1])
    groups = []
    for q, (keys, y) in enumerate([(d.s_keys, d.s_y), (d.w_keys, d.w_y)]):
        if q == 1 and not (m.use_winter or m.use_D):
            continue  # winter rows touch only winter M/D; skipped where neither season-coupling nor D applies
        keep = np.ones(len(keys), bool) if (q == 1 or summer_mask is None) else summer_mask
        ci, ti = d.rows(keys[keep])
        dD = torch.as_tensor(d.C[ci, ti, 64*q:64*(q+1)] - d.Dbar[ci, 64*q:64*(q+1)])
        groups.append((q, torch.as_tensor(ci), torch.as_tensor(ti), dD, torch.as_tensor(y[keep])))
    pairs = torch.as_tensor(d.pairs)
    opt = torch.optim.Adam(m.parameters(), lr=cfg['lr'])
    for step in range(cfg['steps']):
        opt.zero_grad()
        loss = 0.
        parts = {}
        for q, ci, ti, dD, y in groups:
            l = (m.season(ci, ti, q, dD) - y).square().mean()
            parts[q] = float(l)
            loss = loss + (cfg['winter_weight'] if q == 1 else 1.) * l
        loss = loss + m.penalty(pairs)
        loss.backward()
        opt.step()
        if log and (step % 500 == 0 or step == cfg['steps'] - 1):
            print(f'{arm} step {step} loss {float(loss):.6f} parts {parts}', file=log, flush=True)
    return m


@torch.no_grad()
def predict_rows(m, d, keys, q):
    ci, ti = d.rows(keys)
    dD = torch.as_tensor(d.C[ci, ti, 64*q:64*(q+1)] - d.Dbar[ci, 64*q:64*(q+1)])
    return m.season(torch.as_tensor(ci), torch.as_tensor(ti), q, dD).numpy()


def centred_corr(p, t, blocks=None):
    if blocks is not None:
        ub, inv = np.unique(blocks, return_inverse=True)
        n = np.bincount(inv)
        P = np.zeros((len(ub), p.shape[1])); T = np.zeros_like(P)
        np.add.at(P, inv, p); np.add.at(T, inv, t)
        keep = n >= 2
        p, t = (P / n[:, None])[keep], (T / n[:, None])[keep]
    p, t = p - p.mean(0), t - t.mean(0)
    return float((p * t).sum() / np.sqrt((p * p).sum() * (t * t).sum()))


def inner(root, out, grid, part='', control='control_s1', block_cells=6):
    """Hide summer 1996-2005 at training cells; backcast it. Not noise-corrected; ranking only."""
    d = Data(root, control=control)
    k = d.s_keys
    held = (k[:, 2] >= 1996) & (k[:, 2] <= 2005)
    late = k[:, 2] >= 2016
    cells = []
    for rc in np.unique(k[held, :2], axis=0):
        h = held & (k[:, 0] == rc[0]) & (k[:, 1] == rc[1])
        l = late & (k[:, 0] == rc[0]) & (k[:, 1] == rc[1])
        if h.sum() >= 2 and l.sum() >= 2:
            cells.append((rc, np.flatnonzero(h), np.flatnonzero(l)))
    obs = np.stack([d.s_y[l].mean(0) - d.s_y[h].mean(0) for _, h, l in cells])
    blocks = np.array([rc[0] // block_cells * 100000 + rc[1] // block_cells for rc, _, _ in cells])
    results = []
    log = open(out/f'inner{part}.log', 'a')
    ctrl = np.stack([d.C[d.cid[rc[0], rc[1]], k[l, 2] - 1902, :64].mean(0) - d.C[d.cid[rc[0], rc[1]], k[h, 2] - 1902, :64].mean(0)
                     for rc, h, l in cells])
    results.append({'arm': 'control', 'cfg': None, 'corr': centred_corr(ctrl, obs), 'regional': centred_corr(ctrl, obs, blocks)})
    print(json.dumps(results[-1]), file=log, flush=True)
    for arm, over in grid:
        cfg = {**DEFAULT, **over}
        m = fit(d, cfg, arm, summer_mask=~held, log=log)
        pred = predict_rows(m, d, k, 0)
        p = np.stack([pred[l].mean(0) - pred[h].mean(0) for _, h, l in cells])
        r = {'arm': arm, 'cfg': over, 'corr': centred_corr(p, obs), 'regional': centred_corr(p, obs, blocks),
             'slope': float(((p - p.mean(0)) * (obs - obs.mean(0))).sum() / ((p - p.mean(0)) ** 2).sum())}
        results.append(r)
        print(json.dumps(r), file=log, flush=True)
    (out/f'INNER{part}.json').write_text(json.dumps({'cells': len(cells), 'results': results, 'control_cube': control,
        'block_cells': block_cells,
        'design': 'summer 1996-2005 hidden at training cells; change 1996-2005 -> 2016-2025; centred corr over cells x 64 summer '
                  'teacher dims; regional = 6x6 block means (>=2 cells); not noise-corrected; the 2016-25 end is in the fit '
                  'for every arm; ranking only'}, indent=2))


def cube_meta(root, d, arm, extra):
    """The D cube's contract (cube_meta.json) with the field provenance added; None if the D cube has none."""
    src = root/'cubes'/TASK/d.control/'cube_meta.json'
    if not src.exists():
        return None
    c = json.loads(src.read_text())
    c['source_cube_dir'] = str(root/'cubes'/TASK/d.control)
    c['contract_origin'] = {'copied_from': str(src), 'copied_from_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
                            'why': 'z_assim overwrites the land cells of the D cube with field predictions in the SAME '
                                   'teacher spaces (dims 0-63 summer teacher, 64-127 winter teacher)'}
    c['assim_arm'] = {'arm': arm, 'export': 'z_assim.export (research/exp/z_assim.py)', 'task': TASK, **extra,
                      'model': 'Z(s,t) = Dbar(s) + u(s) + A.d(s,t) + R(s,t) on the D cube'}
    return c


def export(root, d, m, arm, extra=None):
    out = root/'cubes'/TASK/(arm + '_s1')
    out.mkdir(parents=True, exist_ok=True)
    ci = np.arange(len(d.cells))
    for j, y in enumerate(YEARS):
        g = np.load(root/'cubes'/TASK/d.control/f'Z_latent_{y}.npy').copy()
        keys = np.column_stack([d.cells, np.full(len(ci), y)])
        z = np.hstack([predict_rows(m, d, keys, 0), predict_rows(m, d, keys, 1)])
        g[d.cells[:, 0], d.cells[:, 1]] = z
        np.save(out/f'Z_latent_{y}.npy', g.astype('float32'))
    c = cube_meta(root, d, arm, extra or {})
    if c is not None:
        from src.community_encoder.cube_contract import validate_cube_contract
        validate_cube_contract(c, 128)
        (out/'cube_meta.json').write_text(json.dumps(c, indent=2))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['inner', 'fit'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm', choices=list(ARMS))
    p.add_argument('--winter', default='full', choices=['full', 'sparse10', 'no_target'])
    p.add_argument('--cfg', default='{}')
    p.add_argument('--tag', default='')
    p.add_argument('--summer', default='train', choices=['train', 'interp'])
    p.add_argument('--part', default=None, help='inner grid shard i/n')
    p.add_argument('--task', default='backcast40', help='package task under data/, aux/, cubes/ (default backcast40)')
    p.add_argument('--control-cube', default='control_s1', help='the D cube under cubes/<task>/ (default control_s1)')
    p.add_argument('--production', action='store_true', help='task production: allow summer sources before 1996')
    p.add_argument('--block-cells', type=int, default=6, help='inner-grid regional block size in cells')
    p.add_argument('--inner-grid', default=None, help='JSON list of [arm, overrides] replacing the default inner grid')
    p.add_argument('--threads', type=int, default=12)
    a = p.parse_args()
    global TASK
    TASK = a.task
    torch.set_num_threads(a.threads)
    out = a.root/'assim'/TASK
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'inner':
        grid = [(arm, {}) for arm in ARMS] + [
            ('assim_d', {'lam_C': 1e-3}), ('assim_d', {'lam_C': 1e-1}), ('assim_d', {'rank': 8}),
            ('assim_d', {'scale_km': 300., 'centers': 64}), ('assim_d', {'winter_weight': 0.3}),
            ('assim_d', {'lam_lap': 0.1}), ('assim_d', {'lam_A': 1.0})]
        if a.inner_grid is not None:
            grid = [(arm, over) for arm, over in json.loads(a.inner_grid)]
        if a.part is not None:
            i, n = map(int, a.part.split('/'))
            grid = grid[i::n]
            torch.set_num_threads(max(1, 30 // n) if a.threads == 12 else a.threads)
        return inner(a.root, out, grid, '' if a.part is None else '_' + a.part.replace('/', 'of'), a.control_cube,
                     a.block_cells)
    name = a.arm + (('_' + a.winter) if a.winter != 'full' else '') + a.tag
    cfg = {**DEFAULT, **json.loads(a.cfg)}
    d = Data(a.root, a.winter, a.summer, a.control_cube, a.production)
    with open(out/f'{name}.log', 'w') as log:
        m = fit(d, cfg, a.arm, log=log)
    cube = export(a.root, d, m, name, {'winter': a.winter, 'summer': a.summer, 'cfg': cfg, 'control_cube': a.control_cube,
                                       'production': a.production})
    torch.save({'state': m.state_dict(), 'cfg': cfg, 'arm': a.arm, 'winter': a.winter, 'summer': a.summer}, out/f'{name}.pt')
    (cube/'ASSIM_DONE.json').write_text(json.dumps({'arm': a.arm, 'winter': a.winter, 'summer': a.summer, 'cfg': cfg,
        'task': TASK, 'control_cube': a.control_cube, 'production': a.production,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sources': 'summer training points (1996+), training-circle winter coordinates, masked control cube; no evaluation records'}, indent=2))
    print('exported', cube)


if __name__ == '__main__':
    main()
