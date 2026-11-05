"""Observation-heldout reconstruction scoring using the existing ecological estimands.

Uses one modern spatial readout per species, independent modern evaluation
records, log1p of mean raw counts, matched effort, and paired spatial bootstrap.
All targets are nonsealed community/development species; no focal data are read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.exp.e069_presence_score import ridge_cv
from research.exp.z_repair_residual_fit import load_base
from research.exp.z_repair_models import LowRankCommunityResidual,CubicTimeBasis
from src.community_encoder.train_DESK.validation_core import abba_halves
from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed


def matched_groups(keys,masks,k):
    """Only reserved evaluation records; equal k years per half in both epochs."""
    groups=[];cells=[];spatial=[]
    for row,col,ho,*_ in masks['panel']:
        here=(keys[:,0]==row)&(keys[:,1]==col)
        epochs=[]
        for label in ['historical','modern']:
            available=np.flatnonzero(here&(masks[label+'_a']|masks[label+'_b']))
            if len(available)<2*k:break
            seed=int.from_bytes(hashlib.sha256(f'ZREPAIR-EVAL:{row}:{col}:{label}:{k}'.encode()).digest()[:8],'big')
            pick=np.random.default_rng(seed).choice(available,2*k,replace=False)
            a,b=abba_halves(pick,keys[pick,2])
            assert len(a)==len(b)==k
            assert not masks['train'][np.r_[a,b]].any()
            epochs.append((a,b))
        if len(epochs)==2:
            groups.append(epochs);cells.append((row,col));spatial.append(bool(ho))
    return np.asarray(cells,int),np.asarray(spatial,bool),groups


def bootstrap_components(pred,truth_a,truth_b,weights):
    """Same centered cross-half covariance estimand used by fair_rivals.grade."""
    W=np.asarray(weights,float);sw=W.sum(1)
    mp,ma,mb=W@pred,W@truth_a,W@truth_b
    vp=W@np.sum(pred*pred,axis=1)-np.sum(mp*mp,axis=1)/sw
    vt=W@np.sum(truth_a*truth_b,axis=1)-np.sum(ma*mb,axis=1)/sw
    cv=.5*(W@np.sum(pred*(truth_a+truth_b),axis=1)-np.sum(mp*(ma+mb),axis=1)/sw)
    corr=np.divide(cv,np.sqrt(np.maximum(vp*vt,0)),out=np.full(len(W),np.nan),where=(vp>0)&(vt>0))
    slope=np.divide(cv,vp,out=np.full(len(W),np.nan),where=vp>0)
    # Target-square terms cancel: a predictor with correct direction but twice
    # the true magnitude has perfect correlation but no gain over zero change.
    gain=(W@np.mean(pred*(truth_a+truth_b)-pred*pred,axis=1))/sw
    return {'corr':corr,'truth_on_prediction_slope':slope,'mse_gain_over_no_change':gain,
            'truth_centered_signal':vt/sw}


BLOCK_CELLS=6  # bootstrap block size in grid cells (6 = 162 km at 27 km); FIELD54 scoring sets 3 at 54 km


def weights_for_cells(cells,boot=500):
    blocks=cells[:,0]//BLOCK_CELLS*100000+cells[:,1]//BLOCK_CELLS
    unique,index=np.unique(blocks,return_inverse=True)
    rng=np.random.default_rng(11)
    draws=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=boot)
    return np.vstack([np.ones(len(cells)),draws[:,index]]),len(unique)


def representation(root,arm,keys,task='backcast40'):
    cube=root/'cubes'/task/('control_s1' if arm=='residual_r2' else arm+'_s1')
    z=load_base(cube,keys).astype('float64')
    if arm=='residual_r2':
        from src.config_utils import load_data_config
        import rasterio
        fit=root/'residuals'/task/'rank2'
        if not (fit/'FIT_DONE.json').exists():raise ValueError('Missing completed residual fit')
        basis=dict(np.load(fit/'basis.npz'));state=torch.load(fit/'model.pt',map_location='cpu',weights_only=True)
        temporal=CubicTimeBasis(reference=tuple(basis['temporal_reference']))
        np.testing.assert_allclose(temporal.knots,basis['temporal_knots'])
        np.testing.assert_allclose(temporal.center,basis['temporal_center'])
        centers=basis['spatial_centers_km'];scale=float(basis['spatial_scale_km'])
        model=LowRankCommunityResidual(128,2,len(centers)+1,temporal.width)
        model.load_state_dict(state['model']);model.eval()
        with rasterio.open(load_data_config()['grid']['ref_raster']) as ds:
            x,y=rasterio.transform.xy(ds.transform,keys[:,0],keys[:,1])
        xy=np.column_stack([x,y])/1000
        for start in range(0,len(keys),4096):
            sl=slice(start,start+4096)
            d2=np.sum((xy[sl,None,:]-centers)**2,axis=-1)
            S=np.column_stack([np.ones(len(d2)),np.exp(-.5*d2/scale**2)]).astype('float32')
            T=temporal(keys[sl,2]).astype('float32')
            with torch.no_grad():z[sl]=model(torch.as_tensor(z[sl],dtype=torch.float32),torch.from_numpy(S),torch.from_numpy(T)).numpy()
    return z


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--arm',choices=['control','time_r2','residual_r2'],required=True)
    p.add_argument('--boot',type=int,default=500)
    a=p.parse_args();root=a.root.resolve();task='backcast40'
    masks=dict(np.load(root/'data'/task/'summer_masks.npz'))
    keys=masks['keys'];meta=json.loads((a.base/'meta.json').read_text())
    species=meta['community']+meta['dev_species']
    if 'houfin' in species or set(species)&set(meta['sealed_species']):raise ValueError('Forbidden targets')
    if not np.array_equal(keys,np.load(a.base/'keys.npy')):raise ValueError('Cache rows differ')
    Y=np.hstack([np.load(a.base/'X_comm.npy'),np.load(a.base/'X_dev.npy')]).astype('float64')
    z=representation(root,a.arm,keys,task)
    # Spatial epoch-mean readout; no independently fitted temporal coefficients.
    training=[];fitcells=[]
    for row,col in np.unique(keys[masks['train'],:2],axis=0):
        rows=np.flatnonzero(masks['train']&(keys[:,0]==row)&(keys[:,1]==col))
        if len(rows)>=4:training.append(rows);fitcells.append((row,col))
    fitcells=np.asarray(fitcells,int)
    targets=epoch_mean_observed(Y,training).astype('float64')
    F=np.stack([z[g].mean(0) for g in training])
    groups=fitcells[:,0]//6*100000+fitcells[:,1]//6
    predict,lam=ridge_cv(F,targets,groups,[1e-4,1e-3,1e-2,1e-1,1.,10.],k=5,seed=69)
    out=root/'evaluation'/task/a.arm;out.mkdir(parents=True,exist_ok=True)
    result={'arm':a.arm,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'readout':'One spatial epoch-mean ridge coefficient vector per species, reused across time; no split readout.',
            'training_cells':len(training),'ridge_lambda':lam,'species':species,
            'target_status':'Nonsealed targets; observations held out, not wholly unseen species.',
            'development_status':'Reused ecological records with new exclusion masks; exploratory, not fresh confirmation.',
            'effort':'Exactly k independent survey years in each half of each epoch.', 'panels':{}}
    for k in [2,4]:
        cells,spatial,panel=matched_groups(keys,masks,k)
        ea,eb,ma,mb=([g[e][h] for g in panel] for e,h in [(0,0),(0,1),(1,0),(1,1)])
        E1,E2=epoch_mean_observed(Y,ea).astype(float),epoch_mean_observed(Y,eb).astype(float)
        M1,M2=epoch_mean_observed(Y,ma).astype(float),epoch_mean_observed(Y,mb).astype(float)
        early=predict(np.stack([z[np.r_[x,y]].mean(0) for x,y in zip(ea,eb)]))
        modern=predict(np.stack([z[np.r_[x,y]].mean(0) for x,y in zip(ma,mb)]))
        ta,tb=M1-E1,M2-E2;D=modern-early
        for label,keep in [('known',~spatial),('spatial',spatial)]:
            if keep.sum()<20:continue
            W,nblocks=weights_for_cells(cells[keep],a.boot)
            stats=bootstrap_components(D[keep],ta[keep],tb[keep],W)
            entry={'cells':int(keep.sum()),'blocks':nblocks}
            for metric,values in stats.items():
                finite=np.isfinite(values[1:])
                entry[metric]={'point':float(values[0]) if np.isfinite(values[0]) else None,
                               'valid_bootstraps':int(finite.sum()),
                               'ci95':np.percentile(values[1:][finite],[2.5,97.5]).tolist() if finite.sum()>=.95*a.boot else None}
            # Store paired cell risks, not just aggregate correlations. Candidate
            # comparisons can isolate historical versus modern endpoint changes.
            risks={}
            for name,pred,A,B in [('historical',early,E1,E2),('modern',modern,M1,M2)]:
                risk=.5*np.mean((pred-A)**2+(pred-B)**2,axis=1)
                risks[name+'_observed_mse']=risk
                entry[name+'_observed_mse']=float(risk[keep].mean())
            entry['mean_predicted_change']=float(D[keep].mean())
            entry['mean_observed_change']=float(.5*(ta[keep]+tb[keep]).mean())
            name=f'k{k}_{label}'
            np.savez_compressed(out/(name+'.npz'),cells=cells[keep],predicted_change=D[keep],
                                truth_a=ta[keep],truth_b=tb[keep],
                                **{n:v[keep] for n,v in risks.items()})
            result['panels'][name]=entry
            print(a.arm,name,json.dumps(entry),flush=True)
    (out/'SUMMER_DONE.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
