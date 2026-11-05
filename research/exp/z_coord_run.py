"""In-era interpolation DESKs: the masked control recipe retrained from scratch with the 1966-1995 summer rows at
training places added (the same rows as z_assim --summer interp), with and without space-time coordinate features.

    prepare : snapshot control_code -> coord_code, patch trainer/cube for CoordDESK, write the augmented point set
              and two configs (interp, interp_coord)
    smoke   : two CPU epochs of a tiny CoordDESK through the patched trainer
    run     : train + export one arm's cube (GPU)

Only WITHHELD-place scores are valid for these arms: known-place historical evaluation rows are among the added rows.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys

ARMS = ('interp', 'interp_coord')
COORD = {'features': 256, 'scale_km': 200., 'scale_years': 10., 'year_min': 1966, 'year_max': 2025, 'seed': 20261027}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Source context changed: ' + old[:100])
    return text.replace(old, new)


def patch_trainer(text):
    text = replace_once(text, '                    private_orth=0.0,\n',
                        '                    private_orth=0.0,\n                    coord_conditioning=None,\n')
    old = ('    model = MultiStreamAutoencoder(stream_dims, latent_dim, spatial_kernel, dropout,\n'
           '                                   hidden_width, mlp_expansion, private_dim=private_dim).to(device)')
    new = '''    if coord_conditioning:
        if int(n_spatial_tiles) != 1:
            raise ValueError("Coordinate features require full-grid training")
        from research.exp.z_coord_models import CoordDESK, grid_xy_km
        _shape = tuple(cov_window.shape[1:3])
        if coord_conditioning.get("smoke_grid"):
            _gxy = np.stack(np.meshgrid(np.arange(_shape[0]) * 27., np.arange(_shape[1]) * 27., indexing="ij"), -1).astype("float32")
        else:
            _gxy = grid_xy_km(_shape)
        model = CoordDESK(stream_dims, latent_dim, spatial_kernel, dropout,
                          hidden_width, mlp_expansion, private_dim=private_dim,
                          coord_spec=coord_conditioning, grid_xy=_gxy).to(device)
    else:
        model = MultiStreamAutoencoder(stream_dims, latent_dim, spatial_kernel, dropout,
                                       hidden_width, mlp_expansion, private_dim=private_dim).to(device)'''
    text = replace_once(text, old, new)
    text = replace_once(text, '            with autocast():\n                if _wp:',
                        '            _coord_kw = {"year": float(window_years[t])} if coord_conditioning else {}\n'
                        '            with autocast():\n                if _wp:')
    text = replace_once(text, 'checkpoint(model, xt, mk[t:t + 1], True, use_reentrant=False)',
                        'checkpoint(model, xt, mk[t:t + 1], True, use_reentrant=False, **_coord_kw)')
    text = replace_once(text, 'checkpoint(model, xt, mk[t:t + 1], use_reentrant=False)',
                        'checkpoint(model, xt, mk[t:t + 1], use_reentrant=False, **_coord_kw)')
    text = replace_once(text, '                    zt, rt = model(xt, mk[t:t + 1])',
                        '                    zt, rt = model(xt, mk[t:t + 1], **_coord_kw)')
    text = replace_once(text, '    _ck_fp =hashlib.sha256(',
                        '    if coord_conditioning:\n        _ck_key["coord_conditioning"] = coord_conditioning\n    _ck_fp =hashlib.sha256(')
    text = replace_once(text, '        private_orth=float(desk_cfg.get("private_orth", 0.0)),',
                        '        private_orth=float(desk_cfg.get("private_orth", 0.0)),\n'
                        '        coord_conditioning=desk_cfg.get("coord_conditioning"),')
    text = replace_once(text, '             latent_dim=latent_dim, label_year=label_year,',
                        '             latent_dim=latent_dim, label_year=label_year,\n'
                        '             coord_conditioning_json=json.dumps(desk_cfg.get("coord_conditioning")),')
    return text


def patch_cube(text):
    old = '    model = MultiStreamAutoencoder(stream_dims, latent_dim, spatial_kernel,'
    new = '''    _coord_spec = config.get("desk", {}).get("coord_conditioning")
    _coord_args = {}
    if _coord_spec:
        from research.exp.z_coord_models import CoordDESK, grid_xy_km
        _model_class = CoordDESK
        _coord_args = {"coord_spec": _coord_spec, "grid_xy": grid_xy_km((H, W))}
        if "coord_conditioning_json" not in dm.files or _json.loads(str(dm["coord_conditioning_json"])) != _coord_spec:
            raise ValueError("Coordinate metadata/config mismatch")
    else:
        _model_class = MultiStreamAutoencoder
    model = _model_class(stream_dims, latent_dim, spatial_kernel,'''
    text = replace_once(text, old, new)
    text = replace_once(text, 'private_dim=(int(dm["private_dim"]) if "private_dim" in dm.files else 0)).to(device)',
                        'private_dim=(int(dm["private_dim"]) if "private_dim" in dm.files else 0), **_coord_args).to(device)')
    text = replace_once(text, '            with torch.no_grad():\n                if include_private:',
                        '            _coord_kw = {"year": float(year)} if _coord_spec else {}\n'
                        '            with torch.no_grad():\n                if include_private:')
    text = replace_once(text, 'model(xg, mg, True)', 'model(xg, mg, True, **_coord_kw)')
    text = replace_once(text, 'model(xg, mg) ', 'model(xg, mg, **_coord_kw) ')
    return text


def write_points(root, task='backcast40'):
    """Training rows of the masked control plus every valid pre-1996 row outside the holdout and its buffer.

    task 'production' (FIELD54): nothing is withheld, so its training rows already hold every valid row from 1966 on;
    the interp point set IS the production training set and is returned unchanged."""
    if task == 'production':
        out = root/'data'/task/'summer_train_points'
        if not (out/'points_meta.json').exists():
            raise ValueError('Missing production training points: ' + str(out))
        return out
    import numpy as np
    from research.exp.z_repair_evaluate import matched_groups
    data = root/'data'/task
    out = root/'data'/(task + '_interp')/'summer_train_points'
    if (out/'DONE.json').exists():
        return out
    masks = dict(np.load(data/'summer_masks.npz'))
    keys = masks['keys']
    done = json.loads((root/'data/DONE.json').read_text())
    src = Path(next(p for p in done['source_sha256'] if p.endswith('X_points.npy'))).parent
    assert np.array_equal(keys, np.load(src/'point_index.npy'))
    for name in ['X_points.npy', 'point_index.npy', 'point_weights.npy', 'points_meta.json']:
        if hashlib.sha256((src/name).read_bytes()).hexdigest() != done['source_sha256'][str(src/name)]:
            raise ValueError('Source points changed: ' + name)
    extra = masks['valid'] & (keys[:, 2] < 1996) & ~masks['spatial_holdout'] & ~masks['buffer']
    assert not (extra & masks['train']).any()
    cells, spatial, _ = matched_groups(keys, masks, 2)
    held = {tuple(c) for c in cells[spatial]}
    assert not any((int(r), int(c)) in held for r, c in keys[extra, :2])
    # Identical rows to the field test (z_assim_interp_prep.py).
    prep = np.load(root/'aux'/task/'interp'/'summer_extra.npz')
    assert np.array_equal(prep['source_rows'], np.flatnonzero(extra))
    use = masks['train'] | extra
    out.mkdir(parents=True, exist_ok=True)
    X = np.load(src/'X_points.npy', mmap_mode='r')
    np.save(out/'X_points.npy', np.asarray(X[use]))
    np.save(out/'point_index.npy', keys[use])
    np.save(out/'point_weights.npy', np.load(src/'point_weights.npy')[use])
    pm = json.loads((data/'summer_train_points/points_meta.json').read_text())
    pm.update(n_rows=int(use.sum()), n_recent=int((keys[use, 2] == pm['recent_year']).sum()),
              n_hist=int((keys[use, 2] != pm['recent_year']).sum()),
              source_rows_sha256=hashlib.sha256(np.flatnonzero(use).tobytes()).hexdigest(),
              z_repair_interp='masked control training rows + all valid pre-1996 rows outside holdout/buffer')
    pm['bbs'] = {**pm['bbs'], 'n_rows': int(use.sum()), 'year_range': [int(keys[use, 2].min()), int(keys[use, 2].max())]}
    (out/'points_meta.json').write_text(json.dumps(pm, indent=2))
    np.save(out.parent/'summer_train_source_rows.npy', np.flatnonzero(use))
    (out/'DONE.json').write_text(json.dumps({'train_rows': int(masks['train'].sum()), 'added_rows': int(extra.sum()),
                                             'total': int(use.sum())}, indent=2))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'smoke', 'run'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--task', choices=['backcast40', 'backcast20', 'production'], default='backcast40')
    a = p.parse_args()
    root = a.root.resolve()
    code = root/'coord_code'
    if a.stage == 'prepare':
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        if code.exists():
            raise RuntimeError('Refuse to overwrite a source snapshot')
        points = write_points(root, a.task)
        shutil.copytree(root/'control_code', code)
        for name in ['z_coord_run.py', 'z_coord_models.py']:
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
        base = json.loads((code/'config/z_repair_control.json').read_text())
        for arm in ARMS:
            cfg = json.loads(json.dumps(base))
            cfg['target']['points_dir'] = str(points)
            if arm == 'interp_coord':
                cfg['desk']['coord_conditioning'] = COORD
            cfg['paths']['desk_output_dir'] = str(root/f'models/{a.task}/{arm}_s1')
            cfg['latent_cube']['model_path'] = str(root/f'models/{a.task}/{arm}_s1/env_model_semisup.pth')
            cfg['latent_cube']['output_dir'] = str(root/f'cubes/{a.task}/desk_{arm}_s1')
            cfg['z_repair']['arm'] = 'desk_' + arm
            cfg['z_repair']['validation_policy'] += ' Pre-1996 training-place summer rows added: score withheld places only.'
            (code/f'config/z_coord_{arm}.json').write_text(json.dumps(cfg, indent=2))
        (code/'ISOLATED_PATCHES.json').write_text(json.dumps(patches, indent=2))
        manifest = {str(q.relative_to(code)): hashlib.sha256(q.read_bytes()).hexdigest()
                    for q in sorted(code.rglob('*')) if q.is_file() and q.name != 'FROZEN_SOURCE.json'}
        (code/'FROZEN_SOURCE.json').write_text(json.dumps(manifest, indent=2))
        print(code)
        return
    if Path(__file__).resolve().parents[2] != code:
        raise ValueError('Run the frozen entry point')
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(code/'src'))
    if a.stage == 'smoke':
        import numpy as np
        import torch
        from src.community_encoder.train_DESK.desk_training import train_model_ema
        rng = np.random.default_rng(4)
        years = [2006, 2007]
        cov = rng.normal(size=(2, 3, 3, 5)).astype('float32')
        mask = np.ones((2, 3, 3), bool)
        tr = np.ones((3, 3), bool)
        tr[-1] = False
        val = ~tr
        x = np.log1p(rng.uniform(0, 3, (3, 3, 7))).astype('float32')
        target = {y: (rng.normal(size=(3, 3, 4)).astype('float32'), tr, val, np.ones((3, 3), 'float32')) for y in years}
        flat = np.flatnonzero(tr.ravel())
        pool = (np.repeat(years, len(flat)), np.tile(flat, 2), np.tile(x.reshape(9, 7)[flat], (2, 1)))
        model, ema, info = train_model_ema(cov, mask, years, target, pool, tr, val, [3, 2], 4,
            {'half_life_bounds': [1., 40.], 'init_half_life': 2.}, spatial_kernel=0, epochs=2,
            hidden_width=8, dropout=0., amp=False, metric_pairs=32, eval_kernel_pairs=32,
            stop_at_epoch=2, eval_every=1, procrustes_diag=False,
            coord_conditioning={**COORD, 'smoke_grid': True}, return_info=True)
        w = model.coord_proj.weight
        assert torch.isfinite(w).all() and w.abs().max() > 0, 'coordinate projection received no gradient'
        f0, f1 = model.features(1900), model.features(1966)
        assert torch.equal(f0, f1), 'year clamp failed'
        (root/'COORD_TRAINING_SMOKE_DONE').write_text('Two complete CPU epochs including checkpointed backward passed.\n')
        print('CoordDESK trainer CPU smoke passed')
        return
    if not (root/'COORD_TRAINING_SMOKE_DONE').exists():
        raise ValueError('Missing training smoke')
    for name, sha in json.loads((code/'FROZEN_SOURCE.json').read_text()).items():
        if hashlib.sha256((code/name).read_bytes()).hexdigest() != sha:
            raise ValueError('Changed source: ' + name)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a GPU')
    from src.config_utils import load_config
    cfg = load_config(code/f'config/z_coord_{a.arm}.json')
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
