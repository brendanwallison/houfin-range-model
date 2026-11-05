"""Comparable-conditions DESKs: the in-era interpolation recipe (desk_interp: masked control + 1966-1995 summer rows
at training places) with the z_assim field's own regional basis inside DESK (z_field_models.py).

    fieldout        A: field term added to summer Z (after decoder), trained jointly
    fieldin         B: the same basis as a coordinate input to the pointwise hidden layer
    fieldout_change C: A + the change-vector loss, contrasts extended to 1966-2025 decades
    rff_fast        D: desk_interp_coord (256 RFF) with the coordinate map in the fast group

New modules train in their own AdamW group (lr FAST_LR, no decay): DESK takes ONE full-batch step per epoch (500
updates at peak lr 1.5e-3), against the field's 2000 Adam steps at 0.02. Field penalties are scaled by the
stabilizing weight so their ratio to the summer-teacher fit matches z_assim's.

    prepare : snapshot control_code -> field_code, patch trainer/cube, write basis and change targets, configs
    smoke   : two CPU epochs per kind through the patched trainer
    run     : train + export one arm's cube (GPU)

Only WITHHELD-place scores are valid: the added pre-1996 rows include known-place historical evaluation rows.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys

ARMS = ('fieldout', 'fieldin', 'fieldout_change', 'rff_fast')
FAST_LR = 0.02
FIELD = {'rank': 16, 'centers': 128, 'scale_km': 200., 'knot': 10, 'lam_C': 1e-2, 'lam_curv': 1.0, 'seed': 20261027}
EPOCHS_EXT = [(1966, 1975), (1976, 1985), (1986, 1995), (1996, 2005), (2006, 2015), (2016, 2025)]
CONTRASTS_EXT = [(i, i + 1) for i in range(5)] + [(i, i + 2) for i in range(4)]
CHANGE_WEIGHT_SOURCE = 'models/backcast40/change100/change_loss_calibration.json'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Source context changed: ' + old[:100])
    return text.replace(old, new)


def patch_trainer(text):
    from research.exp.z_coord_run import patch_trainer as coord_patch
    text = coord_patch(text)
    text = replace_once(text, '''        model = CoordDESK(stream_dims, latent_dim, spatial_kernel, dropout,
                          hidden_width, mlp_expansion, private_dim=private_dim,
                          coord_spec=coord_conditioning, grid_xy=_gxy).to(device)''',
                        '''        from research.exp.z_field_models import make_model
        model = make_model(coord_conditioning, _gxy, stream_dims, latent_dim, spatial_kernel, dropout,
                           hidden_width, mlp_expansion, private_dim=private_dim).to(device)''')
    old = ('        model = MultiStreamAutoencoder(stream_dims, latent_dim, spatial_kernel, dropout,\n'
           '                                       hidden_width, mlp_expansion, private_dim=private_dim).to(device)')
    text = replace_once(text, old, old + '''
    _change_loss = None
    if coord_conditioning and coord_conditioning.get("change_supervision"):
        from research.exp.z_change_neural import ChangeVectorLoss
        _change_loss = ChangeVectorLoss(coord_conditioning["change_supervision"], window_years, device)
        if _change_loss.balance is None:
            raise ValueError("Change-loss weight must be fixed in advance")''')
    text = replace_once(text, '''            if not p.requires_grad:
                continue
            if p.ndim <= 1''', '''            if not p.requires_grad:
                continue
            from research.exp.z_field_models import is_fast as _is_fast
            if _is_fast(name):
                continue
            if p.ndim <= 1''')
    old = ('    opt = torch.optim.AdamW(_param_groups([model, ema] + aux_heads + ([pres_mod] if pres_mod is not None else [])\n'
           '                                          + mh_mods, weight_decay), lr=lr)')
    text = replace_once(text, old, old + '''
    from research.exp.z_field_models import is_fast as _is_fast
    _fast = [p for n_, p in model.named_parameters() if p.requires_grad and _is_fast(n_)]
    if _fast:
        opt.add_param_group({"params": _fast, "weight_decay": 0.0, "lr": float(coord_conditioning["fast_lr"])})
        print(f"[field] fast group: {sum(p.numel() for p in _fast):,} parameters at lr {coord_conditioning['fast_lr']}", flush=True)''')
    text = replace_once(text, '            # A non-finite loss here would silently poison every weight via the optimizer step, and',
                        '''            if hasattr(model, "field_penalty"):
                _fp = weights["stabilizing"] * model.field_penalty()
                loss = loss + _fp
                if ep % 10 == 1:
                    print(f"[field] epoch {ep}: weighted penalty {float(_fp.detach()):.6g}", flush=True)
            if _change_loss is not None:
                _wc = _change_loss.weighted(z_ema, None, None)
                loss = loss + _wc
                if ep % 10 == 1:
                    print(f"[change] epoch {ep}: weighted vector loss {float(_wc.detach()):.6g}", flush=True)

            # A non-finite loss here would silently poison every weight via the optimizer step, and''')
    return text


def patch_cube(text):
    from research.exp.z_coord_run import patch_cube as coord_patch
    text = coord_patch(text)
    return replace_once(text, '''        from research.exp.z_coord_models import CoordDESK, grid_xy_km
        _model_class = CoordDESK
        _coord_args = {"coord_spec": _coord_spec, "grid_xy": grid_xy_km((H, W))}''',
                        '''        from research.exp.z_coord_models import grid_xy_km
        from research.exp.z_field_models import make_model
        _gxy = grid_xy_km((H, W))
        _model_class = lambda *a_, **k_: make_model(_coord_spec, _gxy, *a_, **k_)''')


def write_basis(root, out):
    """Centres from exactly z_assim.Data(root, 'full', 'interp')'s observation cells (field_d_interp)."""
    import numpy as np
    from research.exp.z_assim import grid_xy
    from research.exp.z_repair_models import CubicTimeBasis, GaussianRegionalBasis
    from src.data.preprocess.bbs import load_grid_reference, MASK_PATH
    land = np.asarray(load_grid_reference(MASK_PATH)[0], bool)
    s = np.load(root/'data/backcast40/summer_train_points/point_index.npy').astype(int)
    x = np.load(root/'aux/backcast40/interp/summer_extra.npz')
    w = np.load(root/'aux/backcast40/full/winter_coords.npz')
    keys = np.vstack([s, x['keys'].astype(int), np.column_stack([w['row'], w['col'], w['year']]).astype(int)])
    if not land[keys[:, 0], keys[:, 1]].all():
        raise ValueError('Observation outside land support')
    cells = np.unique(keys[:, :2], axis=0)
    basis = GaussianRegionalBasis(grid_xy(cells[:, 0], cells[:, 1]), FIELD['centers'], FIELD['scale_km'])
    T = CubicTimeBasis(reference=(2006, 2025), knot_spacing=FIELD['knot'])(np.arange(1902, 2026))
    np.savez(out, centers=basis.centers, scale_km=basis.scale_km, T=T.astype('float32'), year0=1902)
    return out


def write_change(root, out):
    """change100's construction (z_change_neural.prepare) over desk_interp's rows and six decades."""
    import numpy as np
    from research.exp import z_change_cpu as zc
    zc.EPOCHS, zc.CONTRASTS = EPOCHS_EXT, CONTRASTS_EXT
    m = dict(np.load(root/'data/backcast40/summer_masks.npz'))
    keys = m['keys']
    use = np.zeros(len(keys), bool)
    use[np.load(root/'data/backcast40_interp/summer_train_source_rows.npy')] = True
    teacher = np.full((len(keys), 64), np.nan, 'float32')
    teacher[np.load(root/'data/backcast40/summer_train_source_rows.npy')] = np.load(root/'aux/backcast40/full/summer_training_coords.npy')
    x = np.load(root/'aux/backcast40/interp/summer_extra.npz')
    teacher[x['source_rows']] = x['Z']
    assert np.isfinite(teacher[use]).all() and not np.isfinite(teacher[~use]).any()
    cells, kind, pairs = zc.build_pairs(keys, use)
    pidx = np.array([np.r_[*a, *b] for a, b in pairs])
    assert use[pidx].all()
    target = zc.delta_rows(teacher, pairs)
    unique, count = np.unique(cells, axis=0, return_counts=True)
    cm = {tuple(c): n for c, n in zip(unique, count)}
    np.savez_compressed(out, year=keys[pidx, 2], row=cells[:, 0], col=cells[:, 1], target=target,
                        weight=np.array([1 / cm[tuple(c)] for c in cells], 'float32'), source_rows=pidx, contrast=kind)
    return {'contrasts': len(pairs), 'cells': len(unique), 'by_contrast': np.bincount(kind, minlength=len(CONTRASTS_EXT)).tolist(),
            'earliest_year': int(keys[pidx, 2].min())}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'smoke', 'run'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS)
    a = p.parse_args()
    root = a.root.resolve()
    code = root/'field_code'
    if a.stage == 'prepare':
        sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[2]/'src')]
        if code.exists():
            raise RuntimeError('Refuse to overwrite a source snapshot')
        inputs = root/'field_inputs'
        inputs.mkdir(exist_ok=True)
        basis = write_basis(root, inputs/'field_basis.npz')
        change_path = inputs/'summer_change_ext.npz'
        change_info = write_change(root, change_path)
        src_cal = json.loads((root/CHANGE_WEIGHT_SOURCE).read_text())
        shutil.copytree(root/'control_code', code, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ['z_coord_run.py', 'z_coord_models.py', 'z_field_models.py', 'z_field_run.py', 'z_change_neural.py']:
            shutil.copyfile(Path(__file__).with_name(name), code/'research/exp'/name)
        patches = {}
        for name, fn in [('src/community_encoder/train_DESK/desk_training.py', patch_trainer),
                         ('src/community_encoder/build_final_z_cube.py', patch_cube)]:
            path = code/name
            before = path.read_text()
            after = fn(before)
            compile(after, str(path), 'exec')
            path.write_text(after)
            patches[name] = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=name, tofile=name))
        from research.exp.z_coord_run import COORD
        field = {**FIELD, 'basis_path': str(basis), 'basis_sha256': sha(basis), 'fast_lr': FAST_LR}
        base = json.loads((code/'config/z_repair_control.json').read_text())
        for arm in ARMS:
            cfg = json.loads(json.dumps(base))
            cfg['target']['points_dir'] = str(root/'data/backcast40_interp/summer_train_points')
            out = root/f'models/backcast40/{arm}_s1'
            spec = {'rff': {**COORD, 'kind': 'rff', 'fast_lr': FAST_LR},
                    'fieldin': {**field, 'kind': 'basis_input'}}.get(arm.replace('rff_fast', 'rff'), {**field, 'kind': 'field_output'})
            if arm == 'fieldout_change':
                spec['change_supervision'] = {'path': str(change_path), 'sha256': sha(change_path),
                                              'calibration_path': str(out/'change_loss_calibration.json')}
                out.mkdir(parents=True, exist_ok=True)
                (out/'change_loss_calibration.json').write_text(json.dumps({
                    'weight': src_cal['weight'], 'target_sha256': sha(change_path),
                    'status': 'Fixed in advance: the change100 (808) initial-gradient weight, reused; not recalibrated for '
                              'this target or a from-scratch start.', 'source': CHANGE_WEIGHT_SOURCE}, indent=2))
            cfg['desk']['coord_conditioning'] = spec
            cfg['paths']['desk_output_dir'] = str(out)
            cfg['latent_cube']['model_path'] = str(out/'env_model_semisup.pth')
            cfg['latent_cube']['output_dir'] = str(root/f'cubes/backcast40/desk_{arm}_s1')
            cfg['z_repair']['arm'] = 'desk_' + arm
            cfg['z_repair']['validation_policy'] += ' Pre-1996 training-place summer rows added: score withheld places only.'
            (code/f'config/z_field_{arm}.json').write_text(json.dumps(cfg, indent=2))
        (code/'ISOLATED_PATCHES.json').write_text(json.dumps(patches, indent=2))
        (inputs/'PREPARED.json').write_text(json.dumps({'basis': str(basis), 'change': change_info,
                                                        'change_weight': src_cal['weight']}, indent=2))
        manifest = {str(q.relative_to(code)): sha(q) for q in sorted(code.rglob('*')) if q.is_file() and q.name != 'FROZEN_SOURCE.json'}
        (code/'FROZEN_SOURCE.json').write_text(json.dumps(manifest, indent=2))
        print(code, json.dumps(change_info))
        return
    if Path(__file__).resolve().parents[2] != code:
        raise ValueError('Run the frozen entry point')
    sys.path[:0] = [str(code), str(code/'src')]
    if a.stage == 'smoke':
        import tempfile
        import numpy as np
        import torch
        from src.community_encoder.train_DESK.desk_training import train_model_ema
        from research.exp.z_coord_run import COORD
        from research.exp.z_repair_models import CubicTimeBasis
        rng = np.random.default_rng(4)
        years = [2006, 2007]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bp = tmp/'basis.npz'
            np.savez(bp, centers=np.array([[0., 0.], [54., 54.]]), scale_km=50.,
                     T=CubicTimeBasis(reference=(2006, 2025), knot_spacing=10)(np.arange(1902, 2026)).astype('float32'), year0=1902)
            cp = tmp/'change.npz'
            np.savez(cp, year=np.array([[2006, 2006, 2007, 2007]]), row=np.array([0]), col=np.array([0]),
                     target=np.ones((1, 4), 'float32') * .1, weight=np.ones(1, 'float32'))
            (tmp/'cal.json').write_text(json.dumps({'weight': 279.907, 'target_sha256': sha(cp)}))
            field = {**FIELD, 'rank': 2, 'basis_path': str(bp), 'basis_sha256': sha(bp), 'fast_lr': FAST_LR, 'smoke_grid': True}
            specs = {'field_output': {**field, 'kind': 'field_output'}, 'basis_input': {**field, 'kind': 'basis_input'},
                     'change': {**field, 'kind': 'field_output',
                                'change_supervision': {'path': str(cp), 'sha256': sha(cp), 'calibration_path': str(tmp/'cal.json')}},
                     'rff': {**COORD, 'kind': 'rff', 'fast_lr': FAST_LR, 'smoke_grid': True}}
            for name, spec in specs.items():
                cov = rng.normal(size=(2, 3, 3, 5)).astype('float32')
                mask = np.ones((2, 3, 3), bool)
                tr = np.ones((3, 3), bool)
                tr[-1] = False
                x = np.log1p(rng.uniform(0, 3, (3, 3, 7))).astype('float32')
                target = {y: (rng.normal(size=(3, 3, 4)).astype('float32'), tr, ~tr, np.ones((3, 3), 'float32')) for y in years}
                flat = np.flatnonzero(tr.ravel())
                pool = (np.repeat(years, len(flat)), np.tile(flat, 2), np.tile(x.reshape(9, 7)[flat], (2, 1)))
                model, ema, info = train_model_ema(cov, mask, years, target, pool, tr, ~tr, [3, 2], 4,
                    {'half_life_bounds': [1., 40.], 'init_half_life': 2.}, spatial_kernel=0, epochs=2,
                    hidden_width=8, dropout=0., amp=False, metric_pairs=32, eval_kernel_pairs=32,
                    stop_at_epoch=2, eval_every=1, procrustes_diag=False, coord_conditioning=spec, return_info=True)
                from research.exp.z_field_models import is_fast
                fast = {n: p for n, p in model.named_parameters() if is_fast(n)}
                moved = {n: float(p.detach().abs().max()) for n, p in fast.items() if n != 'field_L'}
                assert fast and all(v > 0 for v in moved.values()), (name, moved)
                print('smoke', name, moved, flush=True)
        (root/'FIELD_TRAINING_SMOKE_DONE').write_text('Two CPU epochs per kind (field_output, basis_input, +change, rff) passed.\n')
        return
    if not (root/'FIELD_TRAINING_SMOKE_DONE').exists():
        raise ValueError('Missing training smoke')
    for name, h in json.loads((code/'FROZEN_SOURCE.json').read_text()).items():
        if sha(code/name) != h:
            raise ValueError('Changed source: ' + name)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a GPU')
    from src.config_utils import load_config
    cfg = load_config(code/f'config/z_field_{a.arm}.json')
    from src.community_encoder.train_DESK.desk_training import run_desk_experiment
    from src.community_encoder.build_final_z_cube import build_spacetime_cube
    out = Path(cfg['paths']['desk_output_dir'])
    if not (out/'TRAINING_DONE').exists():
        run_desk_experiment(cfg)
        (out/'TRAINING_DONE').write_text('Fixed 500 epochs; no epoch selection on historical outcomes.\n')
    build_spacetime_cube(cfg)
    (out/'CUBE_DONE').write_text('Cube exported; withheld-place scores only.\n')


if __name__ == '__main__':
    main()
