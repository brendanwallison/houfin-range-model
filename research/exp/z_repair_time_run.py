"""Prepare an isolated time-conditioned trainer; run only after a CPU training smoke."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys


def replace_once(text, old, new):
    if text.count(old)!=1:raise ValueError('Source context changed: '+old[:100])
    return text.replace(old,new)


def patch_trainer(text):
    text=replace_once(text,'                    private_orth=0.0,\n',
                      '                    private_orth=0.0,\n                    time_conditioning=None,\n')
    old='    model = MultiStreamAutoencoder(stream_dims, latent_dim, spatial_kernel, dropout,\n                                   hidden_width, mlp_expansion, private_dim=private_dim).to(device)'
    new='''    _time_features = _time_grid = None
    if time_conditioning:
        if int(n_spatial_tiles) != 1:
            raise ValueError("Time pilot currently requires full-grid training")
        from research.exp.z_repair_models import TimeConditionedDESK, CubicTimeBasis
        _tb = CubicTimeBasis(reference=tuple(time_conditioning["reference"]),
                             knot_spacing=int(time_conditioning["knot_spacing"]))
        _time_features = torch.as_tensor(_tb(window_years), dtype=torch.float32, device=device)
        _time_grid = torch.as_tensor(_tb(np.arange(1902, 2026)), dtype=torch.float32, device=device)
        model = TimeConditionedDESK(stream_dims, latent_dim, spatial_kernel, dropout,
                    hidden_width, mlp_expansion, private_dim=private_dim,
                    temporal_width=_tb.width, time_rank=int(time_conditioning["rank"])).to(device)
    else:
        model = MultiStreamAutoencoder(stream_dims, latent_dim, spatial_kernel, dropout,
                                       hidden_width, mlp_expansion, private_dim=private_dim).to(device)'''
    text=replace_once(text,old,new)
    text=replace_once(text,'            with autocast():\n                if _wp:',
                     '            _time_kw = {} if _time_features is None else {"temporal_basis": _time_features[t]}\n            with autocast():\n                if _wp:')
    text=replace_once(text,'checkpoint(model, xt, mk[t:t + 1], True, use_reentrant=False)',
                     'checkpoint(model, xt, mk[t:t + 1], True, use_reentrant=False, **_time_kw)')
    text=replace_once(text,'checkpoint(model, xt, mk[t:t + 1], use_reentrant=False)',
                     'checkpoint(model, xt, mk[t:t + 1], use_reentrant=False, **_time_kw)')
    text=replace_once(text,'                    zt, rt = model(xt, mk[t:t + 1])',
                     '                    zt, rt = model(xt, mk[t:t + 1], **_time_kw)')
    text=replace_once(text,'            # A non-finite loss here would silently poison every weight via the optimizer step, and',
                     '''            if time_conditioning:
                _ta, _tc = model.time_modulation.penalty(_time_grid)
                loss = loss + float(time_conditioning["amplitude_penalty"]) * _ta
                loss = loss + float(time_conditioning["curvature_penalty"]) * _tc

            # A non-finite loss here would silently poison every weight via the optimizer step, and''')
    text=replace_once(text,'    _ck_fp =hashlib.sha256(',
                     '    if time_conditioning:\n        _ck_key["time_conditioning"] = time_conditioning\n    _ck_fp =hashlib.sha256(')
    text=replace_once(text,'        private_orth=float(desk_cfg.get("private_orth", 0.0)),',
                     '        private_orth=float(desk_cfg.get("private_orth", 0.0)),\n        time_conditioning=desk_cfg.get("time_conditioning"),')
    text=replace_once(text,'             latent_dim=latent_dim, label_year=label_year,',
                     '             latent_dim=latent_dim, label_year=label_year,\n             time_conditioning_json=json.dumps(desk_cfg.get("time_conditioning")),')
    return text


def patch_cube(text):
    old='    model = MultiStreamAutoencoder(stream_dims, latent_dim, spatial_kernel,'
    new='''    _time_spec = config.get("desk", {}).get("time_conditioning")
    _time_args = {}
    if _time_spec:
        from research.exp.z_repair_models import TimeConditionedDESK, CubicTimeBasis
        _time_basis = CubicTimeBasis(reference=tuple(_time_spec["reference"]),
                                     knot_spacing=int(_time_spec["knot_spacing"]))
        _model_class = TimeConditionedDESK
        _time_args = {"temporal_width": _time_basis.width, "time_rank": int(_time_spec["rank"])}
        if "time_conditioning_json" not in dm.files or _json.loads(str(dm["time_conditioning_json"])) != _time_spec:
            raise ValueError("Time-conditioning metadata/config mismatch")
    else:
        _model_class = MultiStreamAutoencoder
    model = _model_class(stream_dims, latent_dim, spatial_kernel,'''
    text=replace_once(text,old,new)
    old='private_dim=(int(dm["private_dim"]) if "private_dim" in dm.files else 0)).to(device)'
    text=replace_once(text,old,'private_dim=(int(dm["private_dim"]) if "private_dim" in dm.files else 0), **_time_args).to(device)')
    text=replace_once(text,'            with torch.no_grad():\n                if include_private:',
                     '            _time_kw = {} if not _time_spec else {"temporal_basis": torch.as_tensor(_time_basis(year), dtype=torch.float32, device=device)}\n            with torch.no_grad():\n                if include_private:')
    text=replace_once(text,'model(xg, mg, True)','model(xg, mg, True, **_time_kw)')
    text=replace_once(text,'model(xg, mg) ','model(xg, mg, **_time_kw) ')
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','smoke','run'])
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();code=root/'time_r2_code'
    if a.stage=='prepare':
        if code.exists():raise RuntimeError('Refuse to overwrite a source snapshot')
        shutil.copytree(root/'control_code',code)
        shutil.copyfile(Path(__file__),code/'research/exp/z_repair_time_run.py')
        patches={}
        for name,fn in [('src/community_encoder/train_DESK/desk_training.py',patch_trainer),
                        ('src/community_encoder/build_final_z_cube.py',patch_cube)]:
            path=code/name;before=path.read_text();after=fn(before)
            compile(after,str(path),'exec')
            path.write_text(after)
            patches[name]=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=name,tofile=name))
        cfg=json.loads((code/'config/z_repair_control.json').read_text())
        cfg['desk']['time_conditioning']={'rank':2,'knot_spacing':20,'reference':[2006,2025],
                                        'amplitude_penalty':.01,'curvature_penalty':1000.}
        cfg['paths']['desk_output_dir']=str(root/'models/backcast40/time_r2_s1')
        cfg['latent_cube']['model_path']=str(root/'models/backcast40/time_r2_s1/env_model_semisup.pth')
        cfg['latent_cube']['output_dir']=str(root/'cubes/backcast40/time_r2_s1')
        cfg['z_repair']['arm']='low_rank_time_modulation_r2'
        cfg['z_repair']['hyperparameter_status']='Initial conservative pilot choices, not a calibrated final prior.'
        (code/'config/z_repair_time_r2.json').write_text(json.dumps(cfg,indent=2))
        (code/'ISOLATED_PATCHES.json').write_text(json.dumps(patches,indent=2))
        manifest={str(q.relative_to(code)):hashlib.sha256(q.read_bytes()).hexdigest()
                  for q in sorted(code.rglob('*')) if q.is_file() and q.name!='FROZEN_SOURCE.json'}
        (code/'FROZEN_SOURCE.json').write_text(json.dumps(manifest,indent=2))
        print(code);return
    if Path(__file__).resolve().parents[2]!=code:raise ValueError('Run the frozen entry point')
    sys.path.insert(0,str(code));sys.path.insert(0,str(code/'src'))
    if a.stage=='smoke':
        import numpy as np
        import torch
        from src.community_encoder.train_DESK.desk_training import train_model_ema
        cfg=json.loads((code/'config/z_repair_time_r2.json').read_text())['desk']['time_conditioning']
        rng=np.random.default_rng(4);years=[2006,2007]
        cov=rng.normal(size=(2,3,3,5)).astype('float32');mask=np.ones((2,3,3),bool)
        tr=np.ones((3,3),bool);tr[-1]=False;val=~tr
        x=np.log1p(rng.uniform(0,3,(3,3,7))).astype('float32')
        target={y:(rng.normal(size=(3,3,4)).astype('float32'),tr,val,np.ones((3,3),'float32')) for y in years}
        flat=np.flatnonzero(tr.ravel());pool=(np.repeat(years,len(flat)),np.tile(flat,2),np.tile(x.reshape(9,7)[flat],(2,1)))
        model,ema,info=train_model_ema(cov,mask,years,target,pool,tr,val,[3,2],4,
            {'half_life_bounds':[1.,40.],'init_half_life':2.},spatial_kernel=0,epochs=2,
            hidden_width=8,dropout=0.,amp=False,metric_pairs=32,eval_kernel_pairs=32,
            stop_at_epoch=2,eval_every=1,procrustes_diag=False,time_conditioning=cfg,return_info=True)
        assert torch.isfinite(model.time_modulation.time_weights).all()
        assert model.time_modulation.time_weights.abs().max()>0
        (root/'TIME_TRAINING_SMOKE_DONE').write_text('Two complete CPU epochs including checkpointed backward passed.\n')
        print('Time-conditioned trainer complete CPU smoke passed');return
    if not (root/'TIME_TRAINING_SMOKE_DONE').exists():raise ValueError('Missing training smoke')
    for name,sha in json.loads((code/'FROZEN_SOURCE.json').read_text()).items():
        if hashlib.sha256((code/name).read_bytes()).hexdigest()!=sha:raise ValueError('Changed source: '+name)
    from src.config_utils import load_config
    cfg=load_config(code/'config/z_repair_time_r2.json')
    from src.community_encoder.train_DESK.desk_training import run_desk_experiment
    from src.community_encoder.build_final_z_cube import build_spacetime_cube
    run_desk_experiment(cfg)
    build_spacetime_cube(cfg)
    (Path(cfg['paths']['desk_output_dir'])/'TIME_PILOT_DONE').write_text('Pilot trained and cube exported; no performance claim.\n')


if __name__=='__main__':main()
