"""Rebuild the existing seasonal/presence objectives from allowed records only."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_cbc_presence import winter_operator
from src.community_encoder.train_DESK.esk_kernel import project_points_to_z


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--task',choices=['backcast40','backcast20','production'],required=True)
    p.add_argument('--block-cells',type=int,default=6,
                   help='winter-operator lambda-validation block size in cells (6 = 162 km at 27 km; 3 at 54 km)')
    p.add_argument('--winter',choices=['full','sparse10','no_target'],default='full')
    p.add_argument('--reuse-presence-from',type=Path,default=None,
                   help='Reuse only after verifying identical winter coordinates and training records.')
    a=p.parse_args()
    data=a.root/'data'/a.task
    winter=data/('winter_'+a.winter)
    out=a.root/'aux'/a.task/a.winter
    out.mkdir(parents=True,exist_ok=True)
    if (out/'DONE.json').exists():raise RuntimeError('Refuse to overwrite completed auxiliaries')
    summer_basis=a.root/'teachers'/a.task/'summer'/'spacetime'
    winter_basis=a.root/'teachers'/a.task/('winter_'+a.winter)/'spacetime'
    for basis in [summer_basis,winter_basis]:
        if not (basis.parent/'TRAINING_ONLY_TEACHER.json').exists():
            raise ValueError(f'Missing audited teacher: {basis}')
    def project(points,basis,name):
        X=np.load(points/'X_points.npy')
        z=project_points_to_z(X,str(basis),64).astype('float32')
        np.save(out/(name+'_training_coords.npy'),z)
        return z,np.load(points/'point_index.npy')
    zs,ks=project(data/'summer_train_points',summer_basis,'summer')
    zw,kw=project(winter/'count_points',winter_basis,'winter')
    Xw=np.load(winter/'count_points/X_points.npy') # Already log1p standardized counts.
    summer_lookup={tuple(map(int,k)):i for i,k in enumerate(ks)}
    both=np.array([i for i,k in enumerate(kw) if tuple(map(int,k)) in summer_lookup])
    if len(both)<100:raise ValueError('Insufficient train-only cross-season calibration')
    rng=np.random.default_rng(20261026)
    i,j=rng.choice(both,size=(2,200000),replace=True)
    keep=i!=j;i,j=i[keep],j[keep]
    denom=np.maximum(Xw[i],Xw[j]).sum(1)
    ruz=np.minimum(Xw[i],Xw[j]).sum(1)/np.maximum(denom,1e-12)
    ii=np.array([summer_lookup[tuple(map(int,k))] for k in kw[i]])
    jj=np.array([summer_lookup[tuple(map(int,k))] for k in kw[j]])
    dot=(zs[ii].astype(float)*zs[jj]).sum(1)
    good=denom>1e-6
    slope,intercept=np.polyfit(ruz[good],dot[good],1)
    mse=float(np.mean((dot[good]-(intercept+slope*ruz[good]))**2))
    if not np.isfinite(mse) or mse<=0:raise ValueError('Invalid auxiliary calibration')
    raw=np.expm1(Xw).astype('float32')
    with np.load(winter/'train_lists.npz') as stream:D=dict(stream)
    species=D['species']
    np.savez_compressed(out/'cbc_ruzicka.npz',year=kw[:,2],row=kw[:,0],col=kw[:,1],
                        X=raw,species=species,a=np.float32(intercept),b=np.float32(slope),weight=np.float32(.03/mse))
    np.savez_compressed(out/'winter_coords.npz',year=kw[:,2],row=kw[:,0],col=kw[:,1],Z=zw)
    cf=dict(np.load(winter/'completeness.npz'))
    rows=D['source_rows'];bn,bh=cf['bench_n'][rows],cf['bench_hit'][rows]
    comp=np.clip(np.divide(bh,bn,out=np.full(len(rows),np.nan),where=bn>0),.02,.98)
    lc=np.log(comp/(1-comp));circle=D['circle_idx'];year=D['year_end']
    modern=(year>=1966)&np.isfinite(lc)
    ncircles=int(circle.max())+1
    n=np.bincount(circle[modern],minlength=ncircles)
    sums=np.bincount(circle[modern],weights=lc[modern],minlength=ncircles)
    mean=sums/np.maximum(n,1)
    relative=np.where(np.isfinite(lc)&(n[circle]>=3),lc-mean[circle],np.nan)
    center=modern&np.isfinite(relative)
    if not center.any():raise ValueError('No valid training completeness reference')
    relative-=np.mean(relative[center])
    P=(D['Y']>0).astype('float32')
    early=(year>=1902)&(year<=1965)&(P.sum(1)>=2)
    if a.reuse_presence_from is None:
        W,b,penalty=winter_operator(SimpleNamespace(fit_years=[1966,2024],block_cells=a.block_cells),kw,zw,
                                   {'is_train':np.ones(len(kw),bool)},D,P,year,relative)
    else:
        previous=a.reuse_presence_from
        old=dict(np.load(previous/'cbc_presence_early_w.npz'))
        if not np.array_equal(zw,np.load(previous/'winter_training_coords.npy')):
            raise ValueError('Winter coordinates differ; cannot reuse operator')
        for label,value in [('source_rows',rows[early]),('species',species),('year',year[early]),
                            ('row',D['row'][early]),('col',D['col'][early]),('circle',circle[early]),
                            ('P',P[early]),('comp',np.nan_to_num(relative[early]).astype('float32'))]:
            if not np.array_equal(value,old[label]):raise ValueError('Operator source differs: '+label)
        # Validate modern training records too, not only the early exported lists.
        if not np.array_equal(D['source_rows'],np.load(previous/'operator_training_source_rows.npy')):
            raise ValueError('Modern/operator training support differs')
        W,b=old['W'],old['b']
        penalty=json.loads((previous/'DONE.json').read_text())['operator_lambda']
        print('Reusing verified identical winter observation operator:',previous,flush=True)
    np.savez_compressed(out/'cbc_presence_early_w.npz',year=year[early],row=D['row'][early],col=D['col'][early],
                        circle=circle[early],P=P[early],W=W.astype('float32'),b=b.astype('float32'),
                        comp=np.nan_to_num(relative[early]).astype('float32'),species=species,
                        source_rows=rows[early])
    np.save(out/'operator_training_source_rows.npy',rows)
    (out/'DONE.json').write_text(json.dumps({
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'task':a.task,'winter':a.winter,'calibration_cell_years':len(both),'operator_block_cells':a.block_cells,
        'cross_season_calibration':{'a':float(intercept),'b':float(slope),'mse':mse,'weight':.03/mse},
        'operator_lambda':penalty,'early_lists':int(early.sum()),
        'reused_identical_winter_operator':str(a.reuse_presence_from) if a.reuse_presence_from else None,
        'training_only_lists':len(rows),'modern_completeness_reference':'training lists only',
        'observation_operator':'Existing E086x winter_operator and scalar relative completeness, unchanged.',
        'leave_one_species_out_completeness_used':False,
        'scope':'Matched first emulation control; no House Finch data and no evaluation records.'},indent=2))
    print('Completed masked winter auxiliaries',out,flush=True)


if __name__=='__main__':main()
