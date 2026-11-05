"""Isolated direct-change DESK continuation and matched optimizer control."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


class ChangeVectorLoss(nn.Module):
    def __init__(self,spec,years,device):
        super().__init__();self.spec=spec
        path=Path(spec['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('Changed temporal targets')
        d=dict(np.load(path));yi={int(y):i for i,y in enumerate(years)}
        if not set(map(int,d['year'].ravel()))<=set(yi):raise ValueError('Change years outside forwarded window')
        self.register_buffer('t',torch.as_tensor([[yi[int(y)] for y in row] for row in d['year']],device=device))
        self.register_buffer('r',torch.as_tensor(d['row'][:,None],device=device))
        self.register_buffer('c',torch.as_tensor(d['col'][:,None],device=device))
        self.register_buffer('target',torch.as_tensor(d['target'],device=device))
        self.register_buffer('weight',torch.as_tensor(d['weight'],device=device))
        self.balance_path=Path(spec['calibration_path']);self.balance=None
        if self.balance_path.exists():
            old=json.loads(self.balance_path.read_text())
            if old['target_sha256']!=spec['sha256']:raise ValueError('Changed loss calibration target')
            self.balance=float(old['weight'])

    def forward(self,z):
        values=z[self.t,self.r,self.c]
        n=values.shape[1]//2
        pred=values[:,n:].mean(1)-values[:,:n].mean(1)
        return (((pred-self.target)**2).mean(1)*self.weight).sum()/self.weight.sum()

    def weighted(self,z,reference_loss,parameter):
        loss=self(z)
        if self.balance is None:
            g0=torch.autograd.grad(reference_loss,parameter,retain_graph=True)[0].norm()
            g1=torch.autograd.grad(loss,parameter,retain_graph=True)[0].norm()
            if not torch.isfinite(g0+g1) or float(g1)<=0:raise ValueError('Invalid change-gradient calibration')
            # Equal initial output-head gradient norms, training data only.
            # This is an optimization heuristic, not an ecological amplitude prior.
            self.balance=float((g0/g1).clamp(.01,1000).detach())
            self.balance_path.parent.mkdir(parents=True,exist_ok=True)
            self.balance_path.write_text(json.dumps({'weight':self.balance,'reference_gradient':float(g0),
                'change_gradient':float(g1),'target_sha256':self.spec['sha256'],
                'status':'Training-only initial-gradient pilot heuristic; not validated temporal calibration.'},indent=2))
            print('[change] initial gradient calibration',self.balance_path.read_text(),flush=True)
        return self.balance*loss


def patch_trainer(text):
    from research.exp.z_repair_time_run import replace_once as replace
    text=replace(text,'                    private_orth=0.0,\n',
                 '                    private_orth=0.0,\n                    change_supervision=None, warm_start=None,\n')
    old='    model = MultiStreamAutoencoder(stream_dims, latent_dim, spatial_kernel, dropout,\n                                   hidden_width, mlp_expansion, private_dim=private_dim).to(device)'
    text=replace(text,old,old+'''
    if warm_start:
        import hashlib as _warm_hash
        if _warm_hash.sha256(open(warm_start["path"], "rb").read()).hexdigest() != warm_start["sha256"]:
            raise ValueError("Warm-start model changed")
        model.load_state_dict(torch.load(warm_start["path"], map_location=device, weights_only=True))
    _change_loss = None
    if change_supervision:
        if int(n_spatial_tiles) != 1:
            raise ValueError("Direct-change pilot requires full-grid training")
        from research.exp.z_change_neural import ChangeVectorLoss
        _change_loss = ChangeVectorLoss(change_supervision, window_years, device)
''')
    text=replace(text,'    _ck_fp =hashlib.sha256(',
        '    _ck_key["change_supervision"] = change_supervision\n    _ck_key["warm_start"] = warm_start\n    _ck_fp =hashlib.sha256(')
    text=replace(text,'            # A non-finite loss here would silently poison every weight via the optimizer step, and',
        '''            if _change_loss is not None:
                _wc = _change_loss.weighted(z_ema, weights["stabilizing"] * loss_stab, model.mixer[-1].weight)
                loss = loss + _wc
                print(f"[change] epoch {ep}: weighted vector loss {float(_wc.detach()):.6g}", flush=True)

            # A non-finite loss here would silently poison every weight via the optimizer step, and''')
    text=replace(text,'        private_orth=float(desk_cfg.get("private_orth", 0.0)),',
        '        private_orth=float(desk_cfg.get("private_orth", 0.0)),\n        change_supervision=desk_cfg.get("change_supervision"), warm_start=desk_cfg.get("warm_start"),')
    text=replace(text,'             latent_dim=latent_dim, label_year=label_year,',
        '             latent_dim=latent_dim, label_year=label_year,\n             change_supervision_json=json.dumps(desk_cfg.get("change_supervision")),\n             warm_start_json=json.dumps(desk_cfg.get("warm_start")),')
    return text


def prepare(root):
    from research.exp.z_change_cpu import build_pairs,delta_rows
    source=Path(__file__).resolve().parents[2]
    inputs=root/'change_neural_inputs';inputs.mkdir(exist_ok=True)
    data=root/'data/backcast40';m=dict(np.load(data/'summer_masks.npz'));keys=m['keys']
    cells,kind,pairs=build_pairs(keys,m['train'])
    teacher=np.full((len(keys),64),np.nan,'float32')
    rows=np.load(data/'summer_train_source_rows.npy')
    teacher[rows]=np.load(root/'aux/backcast40/full/summer_training_coords.npy')
    pidx=np.array([np.r_[*a,*b] for a,b in pairs]);assert m['train'][pidx].all()
    target=delta_rows(teacher,pairs);assert np.isfinite(target).all()
    unique,count=np.unique(cells,axis=0,return_counts=True);countmap={tuple(c):n for c,n in zip(unique,count)}
    np.savez_compressed(inputs/'summer_change.npz',year=keys[pidx,2],row=cells[:,0],col=cells[:,1],
        target=target,weight=np.array([1/countmap[tuple(c)] for c in cells],'float32'),source_rows=pidx,contrast=kind)
    original=root/'models/backcast40/control_s1/env_model_semisup.pth'
    warm=inputs/'control_500_epoch.pth'
    if not warm.exists():shutil.copyfile(original,warm)
    if original.read_bytes()!=warm.read_bytes():raise ValueError('Warm checkpoint differs')
    hl=float(np.load(root/'models/backcast40/control_s1/desk_meta.npz')['ema_half_life'])
    for arm,enabled in [('continue100',False),('change100',True)]:
        code=root/(arm+'_code')
        if code.exists():raise RuntimeError('Refuse to replace source snapshot')
        shutil.copytree(root/'control_code',code,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ['z_change_neural.py','z_change_cpu.py','z_repair_time_run.py']:
            shutil.copyfile(source/'research/exp'/name,code/'research/exp'/name)
        path=code/'src/community_encoder/train_DESK/desk_training.py'
        patched=patch_trainer(path.read_text());compile(patched,str(path),'exec');path.write_text(patched)
        cfg=json.loads((code/'config/z_repair_control.json').read_text())
        output=root/'models/backcast40'/arm
        cfg['paths']['desk_output_dir']=str(output)
        cfg['desk'].update(epochs=100,stop_at_epoch=100,lr=1e-4,checkpoint_every=5,snapshot_every=25)
        cfg['desk']['output_ema']['init_half_life']=hl
        cfg['desk']['warm_start']={'path':str(warm),'sha256':hashlib.sha256(warm.read_bytes()).hexdigest()}
        target_path=inputs/'summer_change.npz'
        cfg['desk']['change_supervision']=({'path':str(target_path),'sha256':hashlib.sha256(target_path.read_bytes()).hexdigest(),
            'calibration_path':str(output/'change_loss_calibration.json')} if enabled else None)
        cfg['latent_cube'].update(model_path=str(output/'env_model_semisup.pth'),output_dir=str(root/'cubes/backcast40'/arm))
        cfg['z_repair']['arm']=arm;cfg['z_repair']['fixed_budget']='100 continuation epochs; matched optimizer reset and warm start.'
        (code/'config/z_change_neural.json').write_text(json.dumps(cfg,indent=2))
        manifest={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*')
                  if p.is_file() and p.name!='FROZEN_SOURCE.json'}
        (code/'FROZEN_SOURCE.json').write_text(json.dumps(manifest,indent=2))
    (inputs/'PREPARED.json').write_text(json.dumps({'pairs':len(pairs),'cells':len(unique),'only_training_summer_rows':True,
        'earliest_summer_year':int(keys[pidx,2].min()),'winter_supervision':'Original masked seasonal objectives retained, including early presence.',
        'new_loss':'Primary summer coordinate vectors on the training observation EMA, not raw population cubes.'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','smoke','run']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arm',choices=['continue100','change100'],default='change100')
    a=p.parse_args();root=a.root.resolve()
    if a.stage=='prepare':prepare(root);return
    code=Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(code),str(code/'src')]
    from src.community_encoder.train_DESK.desk_training import train_model_ema,run_desk_experiment
    if a.stage=='smoke':
        import tempfile
        rng=np.random.default_rng(4);years=[2006,2007]
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);path=tmp/'change.npz'
            np.savez(path,year=np.array([[2006,2006,2007,2007]]),row=np.array([0]),col=np.array([0]),
                target=np.ones((1,4),'float32')*.1,weight=np.ones(1,'float32'))
            spec={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'calibration_path':str(tmp/'calibration.json')}
            cov=rng.normal(size=(2,3,3,5)).astype('float32');mask=np.ones((2,3,3),bool)
            tr=np.ones((3,3),bool);tr[-1]=False;val=~tr
            x=np.log1p(rng.uniform(0,3,(3,3,7))).astype('float32')
            targets={y:(rng.normal(size=(3,3,4)).astype('float32'),tr,val,np.ones((3,3),'float32')) for y in years}
            flat=np.flatnonzero(tr.ravel());pool=(np.repeat(years,len(flat)),np.tile(flat,2),np.tile(x.reshape(9,7)[flat],(2,1)))
            from src.community_encoder.train_DESK.model_arch import MultiStreamAutoencoder
            warm=tmp/'warm.pth'
            torch.save(MultiStreamAutoencoder([3,2],4,spatial_kernel=0,dropout=0,hidden_width=8).state_dict(),warm)
            warm_spec={'path':str(warm),'sha256':hashlib.sha256(warm.read_bytes()).hexdigest()}
            train_model_ema(cov,mask,years,targets,pool,tr,val,[3,2],4,
                {'half_life_bounds':[1.,40.],'init_half_life':2.},spatial_kernel=0,epochs=2,hidden_width=8,
                dropout=0.,amp=False,metric_pairs=32,eval_kernel_pairs=32,stop_at_epoch=2,eval_every=1,
                procrustes_diag=False,change_supervision=spec,warm_start=warm_spec)
            assert (tmp/'calibration.json').exists()
        (root/'CHANGE_NEURAL_SMOKE_DONE').write_text('Two full CPU training epochs with vector loss and checkpointed backward passed.\n')
        return
    if code!=root/(a.arm+'_code'):raise ValueError('Run the correct frozen snapshot')
    if not (root/'CHANGE_NEURAL_SMOKE_DONE').exists():raise ValueError('Missing full training smoke')
    for name,sha in json.loads((code/'FROZEN_SOURCE.json').read_text()).items():
        if hashlib.sha256((code/name).read_bytes()).hexdigest()!=sha:raise ValueError('Changed source '+name)
    from src.config_utils import load_config
    from src.community_encoder.build_final_z_cube import build_spacetime_cube
    cfg=load_config(code/'config/z_change_neural.json')
    run_desk_experiment(cfg);build_spacetime_cube(cfg)
    (Path(cfg['paths']['desk_output_dir'])/'CHANGE_CONTINUATION_DONE').write_text('Completed training and instantaneous export; not promoted.\n')


if __name__=='__main__':main()
