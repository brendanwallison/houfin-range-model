"""Masked community-change pilot using E073 histories and observed winter structure.

All temporal responses are community coordinates. No focal/sealed matrices or
species-specific temporal coefficients are fitted. This is an exploratory
current-data control, not an exact reproduction of E073's original estimand.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.exp import anchored_delta as previous
from research.exp.z_repair_evaluate import matched_groups,bootstrap_components,weights_for_cells
from research.exp.z_repair_residual_fit import load_base
from research.exp.e069_presence_score import ridge_cv
from research.exp.z_repair_compare import summary
from src.community_encoder.train_DESK.validation_core import abba_halves
from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed

EPOCHS=[(1996,2005),(2006,2015),(2016,2025)]
CONTRASTS=[(0,1),(1,2),(0,2)]
LAMS=[.01,.1,1.,10.,100.]


def build_pairs(keys,train,minimum=4):
    """Fixed equal-effort contrasts; never inspect values outside training rows."""
    keys=np.asarray(keys);train=np.asarray(train,bool)
    cells=[];groups=[];types=[]
    for row,col in np.unique(keys[train,:2],axis=0):
        here=train&(keys[:,0]==row)&(keys[:,1]==col)
        epochs={}
        for e,(lo,hi) in enumerate(EPOCHS):
            rows=np.flatnonzero(here&(keys[:,2]>=lo)&(keys[:,2]<=hi))
            if len(rows)<minimum:continue
            seed=int.from_bytes(hashlib.sha256(f'ZCHANGE:{row}:{col}:{e}'.encode()).digest()[:8],'big')
            pick=np.random.default_rng(seed).choice(rows,minimum,replace=False)
            epochs[e]=abba_halves(pick,keys[pick,2])
        for j,(a,b) in enumerate(CONTRASTS):
            if a not in epochs or b not in epochs:continue
            groups.append((epochs[a],epochs[b]));cells.append((row,col));types.append(j)
    return np.asarray(cells,int).reshape(-1,2),np.asarray(types,int),groups


def delta_rows(values,groups):
    return np.stack([values[np.r_[*b]].mean(0)-values[np.r_[*a]].mean(0) for a,b in groups])


def fold_numbers(cells):
    """Stable spatial blocks, shared by summer and winter sources."""
    blocks=np.asarray(cells)//6
    return np.array([int.from_bytes(hashlib.sha256(f'ZCHANGE-FOLD:{r}:{c}'.encode()).digest()[:8],'big')%5
                     for r,c in blocks],int)


class ChangeRidge:
    """Zero-input-zero-change ridge, block PCs fitted only on supplied fit rows."""
    def fit(self,X,Y,blocks,lam,weights=None):
        self.transforms=[];features=[]
        weight=np.ones(len(X)) if weights is None else np.asarray(weights,float)
        weight=weight/weight.mean()
        for cols,width in blocks:
            cols=np.asarray(cols,int);x=np.asarray(X[:,cols],float)
            if not np.isfinite(x).all():raise ValueError('Nonfinite predictor; audit before fitting')
            scale=np.sqrt(np.average(x*x,axis=0,weights=weight))
            scale=np.where(scale>1e-8,scale,1.)
            scaled=x/scale
            _,_,v=np.linalg.svd(scaled*np.sqrt(weight[:,None]),full_matrices=False)
            v=v[:min(width,len(v))].T
            self.transforms.append((cols,scale,v));features.append(scaled@v)
        F=np.hstack(features);sw=np.sqrt(weight[:,None])
        A=(F*sw).T@(F*sw)
        penalty=float(lam)*np.trace(A)/max(1,A.shape[0])
        self.coef=np.linalg.solve(A+max(penalty,1e-10)*np.eye(A.shape[0]),(F*sw).T@(Y*sw))
        return self

    def transform(self,X):
        return np.hstack([(np.asarray(X[:,cols],float)/scale)@v for cols,scale,v in self.transforms])

    def predict(self,X):return self.transform(X)@self.coef


def smooth_winter(query_xy,source_xy,z,reference,k=16,radius=600.):
    """Fixed spatial sharing; unsupported queries retain a fixed modern field."""
    result=np.asarray(reference,float).copy();supported=np.zeros(len(query_xy),bool)
    if not len(source_xy):return result,supported
    # Multiple years at one grid cell must not count as multiple neighboring
    # places or let frequently surveyed places dominate the smoother.
    unique,inv,n=np.unique(source_xy,axis=0,return_inverse=True,return_counts=True)
    if len(unique)!=len(source_xy):
        total=np.zeros((len(unique),z.shape[1]),float);np.add.at(total,inv,z)
        source_xy,z=unique,total/n[:,None]
    d,idx=cKDTree(source_xy).query(query_xy,k=min(k,len(source_xy)),distance_upper_bound=radius)
    if d.ndim==1:d,idx=d[:,None],idx[:,None]
    valid=np.isfinite(d);safe=np.minimum(idx,len(z)-1)
    w=np.where(valid,np.exp(-.5*(np.nan_to_num(d,posinf=1e10)/300.)**2),0.)
    den=w.sum(1);supported=valid.sum(1)>=4
    result[supported]=(w[supported,:,None]*z[safe[supported]]).sum(1)/den[supported,None]
    return result,supported


def prepare(root,base,out):
    data=root/'data/backcast40';keys=np.load(base/'keys.npy')
    masks=dict(np.load(data/'summer_masks.npz'));assert np.array_equal(keys,masks['keys'])
    cells,cidx=np.unique(keys[:,:2],axis=0,return_inverse=True)
    years=np.arange(1902,2026);tidx=keys[:,2]-1902
    print('Loading E073 covariate histories for',len(cells),'cells',flush=True)
    states,stream,names,loaded_years=previous.load_states(cells)
    assert np.array_equal(loaded_years,years)
    reference=(years>=2006)&(years<=2025)
    groups=['climate','cover','human']  # E073's no-clock physical-cover control.
    columns=[previous.group_cols(stream,names,g) for g in groups]
    manifest={'groups':{},'histories':[10,25],'reference_years':[2006,2025],
              'scope':'Raw transformed histories only; normalization/PCs refitted inside each response CV split.',
              'covariate_source_reused':'E073 anchored_delta.load_states/ema/group_cols',
              'winter_scope':'Permitted winter teacher coordinates; not focal observations or summer target labels.',
              'winter_smoother':'16 neighbors, Gaussian300km, radius600km, retrospective +/-2years. Pilot conventions, not calibrated ecological scales.'}
    selected=np.concatenate(columns);offset=0
    for g,cols in zip(groups,columns):
        manifest['groups'][g]={'columns':list(range(offset,offset+len(cols))),'pcs':previous.NPC[g],
                               'variables':list(map(str,names[cols]))};offset+=len(cols)
    for hl in [10,25]:
        sm=previous.ema(states,hl)
        rows=np.asarray(sm[tidx,cidx][:,selected],'float32')
        ref=np.asarray(sm[reference].mean(0)[:,selected],'float32')
        if not np.isfinite(rows).all() or not np.isfinite(ref).all():raise ValueError('Nonfinite history input')
        np.save(out/f'cov_h{hl}.npy',rows);np.save(out/f'cov_reference_h{hl}.npy',ref)
        print('Saved causal history',hl,rows.shape,flush=True)
        del sm
    del states
    from src.config_utils import load_data_config
    import rasterio
    winter=dict(np.load(root/'aux/backcast40/full/winter_coords.npz'))
    wk=np.column_stack([winter['row'],winter['col'],winter['year']]).astype(int)
    with rasterio.open(load_data_config()['grid']['ref_raster']) as ds:
        x,y=rasterio.transform.xy(ds.transform,cells[:,0],cells[:,1]);xy=np.column_stack([x,y])/1000.
        x,y=rasterio.transform.xy(ds.transform,wk[:,0],wk[:,1]);wxy=np.column_stack([x,y])/1000.
    z=winter['Z'];wf=fold_numbers(wk[:,:2]);coverage={}
    # Cache each CV field after removing all winter sources in the validation
    # blocks. Both fitting/validation features use that same restricted field.
    for fold in [-1,0,1,2,3,4]:
        allowed=np.ones(len(wk),bool) if fold<0 else wf!=fold
        modern=allowed&(wk[:,2]>=2006)&(wk[:,2]<=2025)
        fallback=np.tile(z[modern].mean(0),(len(cells),1))
        ref,_=smooth_winter(xy,wxy[modern],z[modern],fallback)
        byyear={};support={}
        for year in np.unique(keys[:,2]):
            use=allowed&(np.abs(wk[:,2]-year)<=2)
            byyear[int(year)],support[int(year)]=smooth_winter(xy,wxy[use],z[use],ref)
        values=np.stack([byyear[int(y)][c] for y,c in zip(keys[:,2],cidx)]).astype('float32')
        mask=np.array([support[int(y)][c] for y,c in zip(keys[:,2],cidx)])
        np.save(out/f'winter_fold{fold}.npy',values)
        # Reference is the actual mean of the same annual smoother in2006-2025.
        annual=[]
        for year in range(2006,2026):
            if year in byyear:annual.append(byyear[year])
            else:
                use=allowed&(np.abs(wk[:,2]-year)<=2)
                annual.append(smooth_winter(xy,wxy[use],z[use],ref)[0])
        np.save(out/f'winter_reference_fold{fold}.npy',np.mean(annual,axis=0).astype('float32'))
        coverage[str(fold)]={'supported_rows':int(mask.sum()),'rows':len(mask),
                              'source_cell_years':int(allowed.sum())}
        print('Winter field fold',fold,coverage[str(fold)],flush=True)
    np.savez_compressed(out/'indices.npz',cells=cells,cidx=cidx,keys=keys)
    manifest['winter_coverage']=coverage
    manifest['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'PREPARED.json').write_text(json.dumps(manifest,indent=2))


def fit(root,base,out):
    data=root/'data/backcast40';indices=dict(np.load(out/'indices.npz'))
    keys=indices['keys'];masks=dict(np.load(data/'summer_masks.npz'))
    cells,kind,pairs=build_pairs(keys,masks['train'])
    if len(pairs)<100:raise ValueError('Insufficient repeated training epochs')
    trainrows=np.load(data/'summer_train_source_rows.npy')
    teacher=np.full((len(keys),64),np.nan,'float32')
    teacher[trainrows]=np.load(root/'aux/backcast40/full/summer_training_coords.npy')
    target=delta_rows(teacher,pairs)
    assert np.isfinite(target).all()
    np.savez_compressed(out/'training_pairs.npz',cells=cells,contrast=kind,
        rows=np.array([np.r_[*a,*b] for a,b in pairs]),target=target)
    folds=fold_numbers(cells)
    countmap={tuple(c):n for c,n in zip(*np.unique(cells,axis=0,return_counts=True))}
    weights=np.array([1/countmap[tuple(c)] for c in cells])
    manifest=json.loads((out/'PREPARED.json').read_text())
    blocks=[(g['columns'],g['pcs']) for g in manifest['groups'].values()]
    raw={hl:np.load(out/f'cov_h{hl}.npy',mmap_mode='r') for hl in [10,25]}
    dx={hl:delta_rows(x,pairs) for hl,x in raw.items()}
    winter={f:np.load(out/f'winter_fold{f}.npy',mmap_mode='r') for f in [-1,0,1,2,3,4]}
    dw={f:delta_rows(v,pairs) for f,v in winter.items()}
    nraw=raw[25].shape[1]
    candidates={'cov10':(10,False,False),'cov25':(25,False,False),
                'cov25_winter4':(25,True,False),'winter4':(25,True,True)}
    def design(name,fold):
        hl,include,only=candidates[name]
        if only:return dw[fold],[(list(range(64)),4)]
        if include:return np.hstack([dx[hl],dw[fold]]),blocks+[(list(range(nraw,nraw+64)),4)]
        return dx[hl],blocks
    # Chronological/reverse checks plus a longer-span contrast, each at spatially
    # held blocks. Endpoint eras of chronological checks are absent from fit labels.
    protocols=[('forward',0,1),('reverse',1,0),('long_span',None,2)]
    reuse=(out/'FIT_DONE.json').exists()
    fits={};cv={}
    if reuse:
        import pickle
        with (out/'fits.pkl').open('rb') as f:fits=pickle.load(f)
        old=json.loads((out/'FIT_DONE.json').read_text())
        if old['training_pairs']!=len(pairs):raise ValueError('Training support changed')
        cv=old['cv']
        print('Reusing frozen fitted change maps; evaluation only',flush=True)
    for name in ([] if reuse else candidates):
        errors={lam:[] for lam in LAMS}
        for fold in range(5):
            X,b=design(name,fold)
            for label,trainkind,testkind in protocols:
                tr=(folds!=fold)&((kind!=testkind) if trainkind is None else (kind==trainkind))
                te=(folds==fold)&(kind==testkind)
                if tr.sum()<30 or te.sum()<10:continue
                # Each lambda fits its own fold-only PCs; labels/features from
                # withheld outer observations are never used to choose strength.
                for lam in LAMS:
                    m=ChangeRidge().fit(X[tr],target[tr],b,lam,weights[tr])
                    err=float(np.mean((m.predict(X[te])-target[te])**2))
                    null=float(np.mean(target[te]**2))
                    errors[lam].append({'protocol':label,'fold':fold,'mse':err,'null_mse':null,'rows':int(te.sum())})
        if not errors[LAMS[0]]:raise ValueError('No supported inner validation folds')
        selected=min(LAMS,key=lambda lam:np.mean([r['mse']/max(r['null_mse'],1e-12) for r in errors[lam]]))
        X,b=design(name,-1)
        fits[name]=ChangeRidge().fit(X,target,b,selected,weights)
        cv[name]={'selected_lambda':selected,'all_inner_scores':errors}
        print('Fitted',name,'lambda',selected,'validation relative MSE',
              np.mean([r['mse']/r['null_mse'] for r in errors[selected]]),flush=True)
    # Freeze all choices before opening evaluation responses.
    import pickle
    if not reuse:
        with (out/'fits.pkl').open('wb') as f:pickle.dump(fits,f)
    fit_info={'training_pairs':len(pairs),
        'training_cells':len(np.unique(cells,axis=0)),'contrasts':{str(i):int((kind==i).sum()) for i in range(3)},
        'cv':cv,'winter_transfer':'Shared rank4 predictor PCs; not species-specific corrections.',
        'inner_teacher_caveat':'The outer-training ESK teacher is fixed across inner folds; its basis is not refitted for each inner fold. Outer evaluation records never enter that teacher.',
        'reference':'2006-2025 mean field; summer64 dynamic component replaced, winter64 kept.'}
    if not reuse:(out/'FIT_DONE.json').write_text(json.dumps(fit_info,indent=2))
    meta=json.loads((base/'meta.json').read_text());species=meta['community']+meta['dev_species']
    if 'houfin' in species or set(species)&set(meta['sealed_species']):raise ValueError('Forbidden targets')
    Y=np.hstack([np.load(base/'X_comm.npy'),np.load(base/'X_dev.npy')]).astype(float)
    cube=root/'cubes/backcast40/control_s1';c=indices['cells'];cidx=indices['cidx']
    audit=root/'ema_audit'
    if not (audit/'DONE.json').exists():raise ValueError('Require the observation-EMA contract audit before scoring change maps')
    anchor=np.load(audit/'control_observation_reference.npy').astype(float)
    control_z=np.load(audit/'control_observation_features.npy').astype(float)
    training=[];fitcells=[]
    for row,col in np.unique(keys[masks['train'],:2],axis=0):
        rows=np.flatnonzero(masks['train']&(keys[:,0]==row)&(keys[:,1]==col))
        if len(rows)>=4:training.append(rows);fitcells.append((row,col))
    fitcells=np.asarray(fitcells);rtarget=epoch_mean_observed(Y,training)
    rg=fitcells[:,0]//6*100000+fitcells[:,1]//6
    teacher_evaluation=dict(np.load(root/'teacher_bottleneck/observed_projection.npz'))
    lookup={int(row):i for i,row in enumerate(teacher_evaluation['source_rows'])}
    result={'scope':'Initial paired-change observation-space diagnostic. Outer panels already opened; exploratory. Not an instantaneous population Z export.',
            'baseline':'Control observation EMA, using its trained half-life; no lag selected on evaluation.',
            'training_pairs':len(pairs),'arms':{}}
    for name,(hl,include,only) in candidates.items():
        centered=raw[hl]-np.load(out/f'cov_reference_h{hl}.npy')[cidx]
        dwref=winter[-1]-np.load(out/'winter_reference_fold-1.npy')[cidx]
        X=dwref if only else (np.hstack([centered,dwref]) if include else centered)
        z=control_z.copy();z[:,:64]=anchor[cidx,:64]+fits[name].predict(X)
        np.save(out/(name+'_features.npy'),z.astype('float32'))
        predict,lam=ridge_cv(np.stack([z[g].mean(0) for g in training]),rtarget,rg,[1e-4,1e-3,1e-2,1e-1,1.,10.])
        entry={'readout_lambda':lam,'panels':{}}
        for k in [2,4]:
            cc,spatial,panel=matched_groups(keys,masks,k)
            groups=([g[e][h] for g in panel] for e,h in [(0,0),(0,1),(1,0),(1,1)])
            ea,eb,ma,mb=groups
            E1,E2,M1,M2=(epoch_mean_observed(Y,g).astype('float64') for g in [ea,eb,ma,mb])
            early=predict(np.stack([z[np.r_[a,b]].mean(0) for a,b in zip(ea,eb)]))
            modern=predict(np.stack([z[np.r_[a,b]].mean(0) for a,b in zip(ma,mb)]))
            ta,tb=M1-E1,M2-E2;D=modern-early
            for label,keep in [('known',~spatial),('spatial',spatial)]:
                if keep.sum()<20:continue
                W,nblocks=weights_for_cells(cc[keep])
                s=bootstrap_components(D[keep],ta[keep],tb[keep],W)
                scored={key:summary(value) for key,value in s.items()}
                risks={f'{label}_observed_mse':.5*np.mean((pred-A)**2+(pred-B)**2,axis=1)
                       for label,pred,A,B in [('historical',early,E1,E2),('modern',modern,M1,M2)]}
                for label2,value in risks.items():scored[label2]=float(value[keep].mean())
                panelname=f'k{k}_{label}'
                control=dict(np.load(audit/'evaluation/backcast40/control'/(panelname+'.npz')))
                for key,value in [('cells',cc[keep]),('truth_a',ta[keep]),('truth_b',tb[keep])]:
                    np.testing.assert_allclose(control[key],value,rtol=0,atol=0)
                cs=bootstrap_components(control['predicted_change'],ta[keep],tb[keep],W)
                scored['paired_difference']={key:summary(s[key]-cs[key]) for key in ['corr','mse_gain_over_no_change']}
                for endpoint in ['historical','modern']:
                    diff=risks[endpoint+'_observed_mse'][keep]-control[endpoint+'_observed_mse']
                    scored['paired_difference'][endpoint+'_mse']=summary((W@diff)/W.sum(1))
                if k==2:
                    za,zb=[] ,[]
                    for a,b,m,n in zip(ea,eb,ma,mb):
                        get=lambda g:teacher_evaluation['Z'][[lookup[int(row)] for row in g]].mean(0)
                        za.append(get(m)-get(a));zb.append(get(n)-get(b))
                    dz=np.stack([z[np.r_[m,n],:64].mean(0)-z[np.r_[a,b],:64].mean(0)
                                 for a,b,m,n in zip(ea,eb,ma,mb)])
                    scored['teacher_change_corr']=summary(bootstrap_components(dz[keep],np.asarray(za)[keep],np.asarray(zb)[keep],W)['corr'])
                entry['panels'][panelname]=scored
                print(name,panelname,'corr',scored['corr']['point'],'gain',scored['mse_gain_over_no_change']['point'],flush=True)
        result['arms'][name]=entry
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    a=p.parse_args();out=a.root/'change_cpu';out.mkdir(exist_ok=True)
    if (out/'RESULTS.json').exists():raise RuntimeError('Refuse to overwrite completed pilot')
    if not (out/'PREPARED.json').exists():prepare(a.root,a.base,out)
    fit(a.root,a.base,out)


if __name__=='__main__':main()
