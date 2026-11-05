"""Fit a small community space-time residual to a completed mask-valid control.

Uses the established seasonal coordinate/kernel and early-list objectives, with
no reconstruction loss because the covariate encoder is frozen. No evaluation
records are loaded, and no performance/adoption decision is made here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.exp.z_repair_models import CubicTimeBasis,GaussianRegionalBasis,LowRankCommunityResidual


def load_base(cube,keys):
    z=np.empty((len(keys),128),'float32')
    for year in np.unique(keys[:,2]):
        use=np.flatnonzero(keys[:,2]==year)
        grid=np.load(cube/f'Z_latent_{year}.npy',mmap_mode='r')
        z[use]=grid[keys[use,0],keys[use,1]]
    if not np.isfinite(z).all():raise ValueError('Non-finite control features')
    return z


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--rank',type=int,choices=[2,4],default=2)
    p.add_argument('--steps',type=int,default=1000)
    a=p.parse_args();root=a.root.resolve();task='backcast40'
    torch.set_num_threads(6);torch.manual_seed(20261026)
    data=root/'data'/task;aux=root/'aux'/task/'full'
    cube=root/'cubes'/task/'control_s1'
    if not (root/'models'/task/'control_s1/CONTROL_CUBE_DONE').exists():
        raise ValueError('Require the completed masked control cube')
    out=root/'residuals'/task/f'rank{a.rank}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'FIT_DONE.json').exists():raise ValueError('Refuse to overwrite a fitted candidate')
    bkeys=np.load(data/'summer_train_points/point_index.npy')
    wkeys=np.load(data/'winter_full/count_points/point_index.npy')
    ep=dict(np.load(aux/'cbc_presence_early_w.npz'))
    ekeys=np.column_stack([ep['row'],ep['col'],ep['year']]).astype(int)
    # There is no early summer target in this fit. All winter records were filtered
    # before teacher/observation-operator construction.
    assert (bkeys[:,2]>=1996).all()
    from src.config_utils import load_data_config
    import rasterio
    allkeys=np.vstack([bkeys,wkeys,ekeys]);coords=allkeys[:,:2]
    with rasterio.open(load_data_config()['grid']['ref_raster']) as ds:
        x,y=rasterio.transform.xy(ds.transform,coords[:,0],coords[:,1])
    xy=np.column_stack([x,y])/1000.
    spatial=GaussianRegionalBasis(xy,n_centers=64,scale_km=300.)
    temporal=CubicTimeBasis(reference=(2006,2025))
    lengths=[len(bkeys),len(wkeys),len(ekeys)];bounds=np.cumsum([0]+lengths)
    S=spatial(xy).astype('float32');T=temporal(allkeys[:,2]).astype('float32')
    groups=[]
    for k,keys in enumerate([bkeys,wkeys,ekeys]):
        sl=slice(bounds[k],bounds[k+1])
        groups.append({'z':torch.from_numpy(load_base(cube,keys)),
                       's':torch.from_numpy(S[sl]),'t':torch.from_numpy(T[sl])})
    bz,wz,ez=groups
    for group,name in [(bz,'summer'),(wz,'winter')]:
        group['target']=torch.from_numpy(np.load(aux/(name+'_training_coords.npy')))
    bx=torch.from_numpy(np.load(data/'summer_train_points/X_points.npy'))
    wx=torch.from_numpy(np.load(data/'winter_full/count_points/X_points.npy'))
    bp=torch.from_numpy(np.load(data/'summer_train_points/point_weights.npy'))
    cal=dict(np.load(aux/'cbc_ruzicka.npz'))
    presence=torch.from_numpy(ep['P']);W=torch.from_numpy(ep['W']);b=torch.from_numpy(ep['b'])
    completeness=torch.from_numpy(ep['comp'])
    delta=nn.Parameter(torch.zeros(len(b)));gamma=nn.Parameter(torch.zeros(len(b)))
    model=LowRankCommunityResidual(128,a.rank,spatial.width,temporal.width)
    grid=torch.as_tensor(temporal(np.arange(1902,2026)),dtype=torch.float32)
    optimizer=torch.optim.Adam([*model.parameters(),delta,gamma],lr=.01)
    rng=np.random.default_rng(20261026)
    def draw(group,n=768):return torch.as_tensor(rng.integers(0,len(group['z']),size=n))
    def corrected(group,i):return model(group['z'][i],group['s'][i],group['t'][i])
    def ruz(X,Y):
        den=torch.maximum(X,Y).sum(-1)
        return torch.minimum(X,Y).sum(-1)/den.clamp_min(1e-8),den>1e-6
    log=[];gradient_report=None
    for step in range(1,a.steps+1):
        optimizer.zero_grad()
        bi,bj=draw(bz),draw(bz);wi,wj=draw(wz),draw(wz);ei=draw(ez)
        B,B2=corrected(bz,bi),corrected(bz,bj)
        C,C2=corrected(wz,wi),corrected(wz,wj)
        E=corrected(ez,ei)
        summer_coord=((B[:,:64]-bz['target'][bi]).square().mean(1)*bp[bi]).sum()/bp[bi].sum()
        winter_coord=(C[:,64:]-wz['target'][wi]).square().mean()
        (rb,bok),(rw,wok)=ruz(bx[bi],bx[bj]),ruz(wx[wi],wx[wj])
        bok=bok&(bi!=bj);wok=wok&(wi!=wj)
        if not bok.any() or not wok.any():raise ValueError('No valid kernel pairs')
        summer_kernel=((B[:,:64]*B2[:,:64]).sum(1)-rb).square()[bok].mean()
        winter_kernel=((C[:,64:]*C2[:,64:]).sum(1)-rw).square()[wok].mean()
        cross_kernel=((C[:,:64]*C2[:,:64]).sum(1)-(float(cal['a'])+float(cal['b'])*rw)).square()[wok].mean()
        eta=E[:,64:]@W+b+delta+completeness[ei,None]*gamma
        early=-(F.log_softmax(eta,dim=-1)*presence[ei]).sum()/presence[ei].sum()
        amp,curve=model.penalty(grid)
        parts={'summer_coordinate':64*summer_coord,'winter_coordinate':64*winter_coord,
               'summer_kernel':5*summer_kernel,'winter_kernel':5*winter_kernel,
               'cross_season_kernel':float(cal['weight'])*cross_kernel,
               'early_presence':.01*early,'amplitude':.01*amp,'curvature':1000*curve,
               'nuisance':.001*(delta.square().mean()+gamma.square().mean())}
        if step==1:
            gradient_report={}
            for name,value in parts.items():
                if name in ['amplitude','curvature','nuisance']:continue
                g=torch.autograd.grad(value,model.scores,retain_graph=True)[0]
                gradient_report[name]=float(g.norm())
            (out/'initial_weighted_gradient_norms.json').write_text(json.dumps(gradient_report,indent=2))
        loss=sum(parts.values())
        if not torch.isfinite(loss):raise RuntimeError('Non-finite residual objective')
        loss.backward();torch.nn.utils.clip_grad_norm_([*model.parameters(),delta,gamma],10.)
        optimizer.step()
        if step==1 or step%50==0:
            row={'step':step,'loss':float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()}}
            log.append(row);print(json.dumps(row),flush=True)
            torch.save({'model':model.state_dict(),'delta':delta.detach(),'gamma':gamma.detach(),
                        'step':step},out/'checkpoint.pt')
    torch.save({'model':model.state_dict(),'delta':delta.detach(),'gamma':gamma.detach()},out/'model.pt')
    np.savez_compressed(out/'basis.npz',spatial_centers_km=spatial.centers,spatial_scale_km=spatial.scale_km,
                        temporal_knots=temporal.knots,temporal_center=temporal.center,
                        temporal_reference=np.array(temporal.reference))
    (out/'training_log.json').write_text(json.dumps(log,indent=2))
    (out/'FIT_DONE.json').write_text(json.dumps({'rank':a.rank,'steps':a.steps,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'control_cube':str(cube),'data_package':str(root/'data/DONE.json'),
        'latent_dim':128,'space_time_score_process_shared_across_seasonal_loadings':True,
        'constraints':'Orthonormal latent directions; fixed spatial/temporal bases; mean-zero departure over2006–2025.',
        'scope':'Training-only pilot; no heldout score, historical validation or adoption claim.',
        'early_presence_weight':.01,'early_presence_weight_status':'Inherited matched-control weight; influence recorded, not assumed adequate.',
        'amplitude_penalty':.01,'annual_curvature_penalty':1000.,
        'initial_weighted_gradient_norms':gradient_report},indent=2))


if __name__=='__main__':main()
