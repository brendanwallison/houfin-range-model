"""FIELD54 Stage D: native 54 km DESK with the 27 km desk_interp recipe (z_coord_run arm 'interp').

    prepare : copy the FROZEN 27 km coord_code (the exact code desk_interp_s1 trained with on TACC), write the 54 km
              config derived from its resolved z_coord_interp.json (paths -> the 54 km build; cell-counted settings
              rescaled; nothing else), record every difference with its reason, re-freeze the source manifest.
    smoke   : a few CPU epochs of the real 54 km config (separate output dirs) + cube export + EMA cube; writes
              COORD_TRAINING_SMOKE_DONE (the marker z_coord_run's run stage requires) only if all of it passes.

    python research/exp/field54_desk.py prepare --root R --task backcast40|production --source-root SRC \
        --data-config field54_data.json --data54 D54 --processed54 P54
    python R/coord_code/research/exp/field54_desk.py smoke --root R --task T --epochs 2
Training itself is the frozen entry point: z_coord_run.py run --root R --arm interp (GPU; resumable every 10 epochs).
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

SRC_DATA = '/scratch/07980/bwa386/houfin/data'
SRC_STATES = '/work/07980/bwa386/ls6/houfin/processed/encoder/states'
# Only settings counted in grid cells are rescaled (FIELD54 plan Stage D, amended 2026-10-30). Values are 27 km -> 54 km.
RESCALE = {('desk', 'spatial_conv', 'kernel'): (5, 3, 'conv kernel in cells: 135 km -> 162 km (odd kernel; nearest)'),
           ('augment', 'tile_cells'): (6, 3, 'covariate-masking tile in cells: 162 km kept'),
           ('trend', 'smooth_sigma_cells'): (1.0, 0.5, 'smoothing sigma in cells halved (27 km km-extent kept); not '
                                                       'read by DESK training or the cube (trend_community only)'),
           ('trend_reference', 'smooth_sigma_cells'): (0.0, 0.0, 'halved; 0 stays 0')}
# Cell-counted keys deliberately NOT rescaled (inert in this recipe), with the reason recorded in the diff report.
INERT_CELL_KEYS = {('desk', 'trend', 'block_cells'): 'validation blocks; inert: desk.trend.holdout_frac = 0 (no internal '
                                                     'validation split), so no block is drawn',
                   ('bbs', 'smooth_sigma_s'): 'old BBS smoother; not read by DESK training or the cube',
                   ('latent_cube', 'radius_km'): 'cube gap-fill radius is set in KM (100) and converted with the data '
                                                 'config resolution: round(100/27) = 4 px at 27 km, round(100/54) = 2 px '
                                                 'at 54 km -- already km-scaled, so no change'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def get(d, keys):
    for k in keys:
        d = d[k]
    return d


def put(d, keys, v):
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = v


def leaves(o, p=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, p + (k,))
    elif isinstance(o, list) and any(isinstance(v, (dict, list)) for v in o):
        for i, v in enumerate(o):
            yield from leaves(v, p + (i,))
    else:
        yield p, o


def build_config(src, root, task, data54, processed54, src_root, states='states', climate_grid='climate_grid_monthly'):
    """The 54 km config from the 27 km resolved one, plus a reason for every changed leaf."""
    cfg = copy.deepcopy(src)
    root = str(root)
    points = (f'{root}/data/production/summer_train_points' if task == 'production'
              else f'{root}/data/{task}_interp/summer_train_points')
    teach, aux = f'{root}/teachers/{task}', f'{root}/aux/{task}/full'
    reasons = {}

    def setp(keys, value, why):
        put(cfg, keys, value)
        reasons[keys] = why

    setp(('paths', 'data_dir'), data54, '54 km data root (Stage A)')
    setp(('paths', 'hist_dir'), f'{processed54}/{states}', f'54 km covariate states (Stage A build_states: {states})')
    setp(('paths', 'desk_output_dir'), f'{root}/models/{task}/interp_s1', '54 km build output')
    for i, s in enumerate(cfg['states']['streams']):
        if s.get('grid_dir', '').startswith(SRC_DATA + '/'):
            sub = s['grid_dir'][len(SRC_DATA):]
            if sub == '/climate_grid_monthly':
                sub = '/' + climate_grid
            setp(('states', 'streams', i, 'grid_dir'), data54 + sub,
                 '54 km covariate grids (= config/overlays/field54_states.json' +
                 ('' if states == 'states' else f'; {states}: field54_states_v2.json') + ')')
    setp(('target', 'points_dir'), points,
         '54 km summer points: twin = training rows + pre-1996 training-place rows (interp, as 27 km); '
         'production = every row' if task == 'production' else
         '54 km twin training rows + pre-1996 training-place rows (identical construction to 27 km interp)')
    setp(('trend', 'community_trend_list'), f'{root}/data/{task}/summer_species.csv', '54 km package species list')
    setp(('desk', 'z_dir'), f'{teach}/summer/spacetime', '54 km summer teacher (Stage C)')
    setp(('desk', 'aux_ruzicka', 'path'), f'{aux}/cbc_ruzicka.npz', '54 km winter auxiliaries (Stage C)')
    setp(('desk', 'aux_presence', 'path'), f'{aux}/cbc_presence_early_w.npz', '54 km winter auxiliaries (Stage C)')
    setp(('desk', 'aux_metric_heads', 0, 'path'), f'{aux}/cbc_ruzicka.npz', '54 km winter auxiliaries (Stage C)')
    setp(('desk', 'aux_metric_heads', 0, 'coord_target', 'basis_dir'), f'{teach}/winter_full/spacetime',
         '54 km winter teacher (Stage C)')
    setp(('desk', 'aux_metric_heads', 0, 'coord_target', 'path'), f'{aux}/winter_coords.npz',
         '54 km winter auxiliaries (Stage C)')
    lc = 'latent_cube'
    setp((lc, 'data_dir'), data54, '54 km data root')
    setp((lc, 'hist_dir'), f'{processed54}/{states}/yearly_states', f'54 km covariate states ({states})')
    setp((lc, 'z_dir'), f'{teach}/summer/spacetime', '54 km summer teacher')
    setp((lc, 'z_ref_path'), f'{teach}/summer/spacetime/Z.npy', '54 km summer teacher')
    setp((lc, 'mask_ref_path'), f'{teach}/summer/spacetime/valid_mask.npy', '54 km summer teacher')
    setp((lc, 'model_path'), f'{root}/models/{task}/interp_s1/env_model_semisup.pth', '54 km build output')
    setp((lc, 'water_mask_path'), f'{data54}/land_mask/ocean_mask_54km.tif', '54 km land mask (Stage A)')
    setp((lc, 'output_dir'), f'{root}/cubes/{task}/desk_interp_s1', '54 km build output')
    for keys, (old, new, why) in RESCALE.items():
        if get(cfg, keys) != old:
            raise ValueError(f'Unexpected 27 km value at {keys}: {get(cfg, keys)} (expected {old})')
        setp(keys, new, why)
    zr = copy.deepcopy(cfg['z_repair'])
    zr.update(data_package=f'{root}/data/DONE.json', task=task,
              arm='field54_desk_interp' if task != 'production' else 'field54_production_desk',
              field54={'plan': 'FIELD54_NATIVE_PLAN_2026-10-29.md Stage D', 'source_config_root': str(src_root),
                       'grid': '54 km native'})
    if task == 'production':
        zr['validation_policy'] = ('Production: nothing withheld; fixed 500 epochs; no epoch selection; not scored.')
    setp(('z_repair',), zr, 'provenance block: 54 km package, task, arm label')
    # Every leaf that still names the 27 km z_repair root would read a 27 km artifact: refuse.
    for p, v in leaves(cfg):
        if isinstance(v, str) and str(src_root) in v and p[:2] != ('z_repair', 'field54'):
            raise ValueError(f'27 km z_repair path left at {p}: {v}')
    return cfg, reasons


def diff_report(src, cfg, reasons):
    a, b = dict(leaves(src)), dict(leaves(cfg))
    rows = []
    for k in sorted(set(a) | set(b), key=str):
        if a.get(k, '<absent>') != b.get(k, '<absent>'):
            why = next((r for p, r in reasons.items() if k[:len(p)] == p), None)
            if why is None:
                raise ValueError(f'Unexplained config difference at {k}')
            rows.append({'key': '.'.join(map(str, k)), 'from_27km': a.get(k, '<absent>'),
                         'to_54km': b.get(k, '<absent>'), 'reason': why})
    unchanged_27km_paths = [{'key': '.'.join(map(str, k)), 'value': v} for k, v in b.items()
                            if isinstance(v, str) and (v.startswith(SRC_DATA) or '/processed/encoder' in v)]
    inert = [{'key': '.'.join(k), 'value': get(cfg, k), 'reason': r} for k, r in INERT_CELL_KEYS.items()]
    return {'changed': rows, 'cell_counted_unchanged': inert,
            'unchanged_27km_paths_not_read_by_desk_or_cube': unchanged_27km_paths}


def prepare(a):
    src_code = a.source_root/'coord_code'
    man = json.loads((src_code/'FROZEN_SOURCE.json').read_text())
    bad = [n for n, h in man.items() if sha(src_code/n) != h]
    if bad:
        raise ValueError(f'27 km frozen source changed: {bad[:5]}')
    code = a.root/'coord_code'
    if code.exists():
        raise RuntimeError('Refuse to overwrite ' + str(code))
    shutil.copytree(src_code, code, ignore=shutil.ignore_patterns('__pycache__'))
    here = Path(__file__).resolve().parent
    for name in ['field54_desk.py', 'z_ema_cube.py']:
        shutil.copyfile(here/name, code/'research/exp'/name)
    shutil.copyfile(a.data_config, code/'config/field54_data.json')
    cdir = code/'config'
    for name in ['z_coord_interp.json', 'z_coord_interp_coord.json', 'z_repair_control.json']:
        if (cdir/name).exists():
            (cdir/name).rename(cdir/name.replace('.json', '_27km_source.json'))
    src = json.loads((cdir/'z_coord_interp_27km_source.json').read_text())
    cfg, reasons = build_config(src, a.root.resolve(), a.task, a.data54, a.processed54, a.source_root,
                                a.states, a.climate_grid)
    (cdir/'z_coord_interp.json').write_text(json.dumps(cfg, indent=2))
    report = diff_report(src, cfg, reasons)
    report.update(source_config=str(a.source_root/'coord_code/config/z_coord_interp.json'),
                  source_config_sha256=man.get('config/z_coord_interp.json'),
                  source_code_identical_to_27km_frozen=True,
                  added_files=['research/exp/field54_desk.py', 'research/exp/z_ema_cube.py', 'config/field54_data.json'],
                  renamed_27km_configs=['config/z_coord_interp_27km_source.json',
                                        'config/z_coord_interp_coord_27km_source.json',
                                        'config/z_repair_control_27km_source.json'])
    (a.root/'FIELD54_CONFIG_DIFF.json').write_text(json.dumps(report, indent=1, default=str))
    newman = {str(q.relative_to(code)): sha(q) for q in sorted(code.rglob('*'))
              if q.is_file() and q.name != 'FROZEN_SOURCE.json' and '__pycache__' not in q.parts}
    (code/'FROZEN_SOURCE.json').write_text(json.dumps(newman, indent=2))
    changed_code = sorted(n for n in newman if n in man and newman[n] != man[n] and not n.startswith('config/'))
    assert changed_code == [], changed_code     # only configs renamed/added; no 27 km source file edited
    print(json.dumps({'changed': len(report['changed'])}), flush=True)
    for r in report['changed']:
        print(f"  {r['key']}: {r['from_27km']} -> {r['to_54km']}  [{r['reason']}]", flush=True)


def smoke(a):
    code = Path(__file__).resolve().parents[2]
    if code != (a.root/'coord_code').resolve():
        raise ValueError('Run the frozen copy')
    import numpy as np
    from src.config_utils import load_config, load_data_config
    dc = load_data_config()
    assert dc['grid']['target_res_m'] == 54000, dc['grid']
    cfg = load_config(code/'config/z_coord_interp.json')
    sm = a.root/'smoke'
    if sm.exists():
        shutil.rmtree(sm)
    out = sm/'model'
    cfg['paths']['desk_output_dir'] = str(out)
    cfg['latent_cube']['model_path'] = str(out/'env_model_semisup.pth')
    cfg['latent_cube']['output_dir'] = str(sm/'cubes'/a.task/'desk_interp_smoke')
    cfg['desk'].update(epochs=a.epochs, stop_at_epoch=a.epochs, checkpoint_every=1, snapshot_every=0)
    from src.community_encoder.train_DESK.desk_training import run_desk_experiment
    from src.community_encoder.build_final_z_cube import build_spacetime_cube
    t0 = time.time()
    run_desk_experiment(cfg)
    t1 = time.time()
    build_spacetime_cube(cfg)
    t2 = time.time()
    rs = json.loads((out/'run_summary.json').read_text())
    traj = [json.loads(l) for l in (out/'train_trajectory.jsonl').read_text().splitlines() if l.strip()]
    assert len(traj) == a.epochs and all(np.isfinite(t['loss_total']) for t in traj), traj
    cube = Path(cfg['latent_cube']['output_dir'])
    z = np.load(cube/'Z_latent_2025.npy')
    import rasterio
    with rasterio.open(dc['grid']['ref_raster']) as r:
        assert z.shape[:2] == (r.height, r.width) == (67, 112), z.shape
    land = np.isfinite(z).all(-1)
    # EMA cube through the same tool the full run uses (models/<task>/<model> layout inside the smoke dir)
    (sm/'models'/a.task).mkdir(parents=True, exist_ok=True)
    (sm/'models'/a.task/'model').symlink_to(out)
    import subprocess
    subprocess.run([sys.executable, str(code/'research/exp/z_ema_cube.py'), '--root', str(sm), '--task', a.task,
                    '--model', 'model', '--cube', 'desk_interp_smoke'], check=True)
    ze = np.load(sm/'cubes'/a.task/'desk_interp_smoke_ema'/'Z_latent_2025.npy')
    assert ze.shape == z.shape and np.isfinite(ze[land]).all()
    rep = {'epochs': a.epochs, 'train_s': round(t1 - t0, 1), 'cube_s': round(t2 - t1, 1),
           'losses': [{k: t[k] for k in ('epoch', 'loss_total', 'loss_metric', 'loss_recon', 'loss_presence',
                                          'loss_ruzicka', 'loss_mhead_winter_half') if k in t} for t in traj],
           'n_train_cells': rs['n_train_cells'], 'n_train_cell_years': rs['n_train_cell_years'],
           'n_params': rs['n_params'], 'spatial_kernel': rs['spatial_kernel'], 'ema_half_life': rs['ema_half_life'],
           'cube_shape': list(z.shape), 'cube_land_cells_finite': int(land.sum()),
           'checkpoint_written': (out/'resume_checkpoint.pt').exists() or (out/'resume_checkpoint.pt.completed').exists(),
           'data_config_res_m': dc['grid']['target_res_m']}
    (a.root/'FIELD54_SMOKE.json').write_text(json.dumps(rep, indent=1))
    (a.root/'COORD_TRAINING_SMOKE_DONE').write_text(
        f'FIELD54 54 km smoke: {a.epochs} CPU epochs of the real config + cube + EMA cube passed (FIELD54_SMOKE.json).\n')
    print(json.dumps(rep, indent=1), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('stage', choices=['prepare', 'smoke'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--task', choices=['backcast40', 'production'], required=True)
    p.add_argument('--source-root', type=Path, default=Path('/work/07980/bwa386/ls6/houfin/research/z_repair_20261026'))
    p.add_argument('--data-config', type=Path)
    p.add_argument('--data54', default='/scratch/07980/bwa386/houfin/data54')
    p.add_argument('--processed54', default='/work/07980/bwa386/ls6/houfin/processed/field54')
    p.add_argument('--states', default='states', help='states dir under processed54 (v2 rebuild: states_v2)')
    p.add_argument('--climate-grid', default='climate_grid_monthly',
                   help='climate grid dir under data54 recorded in states.streams (v2: climate_grid_monthly_v2)')
    p.add_argument('--epochs', type=int, default=2)
    a = p.parse_args()
    if a.stage == 'prepare':
        prepare(a)
    else:
        smoke(a)


if __name__ == '__main__':
    main()
