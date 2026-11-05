"""Development-only seasonal PSD metric calibration; fixed modern ABBA targets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.community_encoder.train_DESK.validation_core import split_half_groups
from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed

p=argparse.ArgumentParser(description=__doc__)
for name in ['base','features','out']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
meta=json.loads((a.base/'meta.json').read_text())
species=[str(s).lower() for s in meta['dev_species']]
if 'houfin' in species:raise ValueError('focal species must be excluded')
if set(species)&set(str(s).lower() for s in meta.get('sealed_species',[])):
    raise ValueError('development and sealed columns overlap')
keys=np.load(a.base/'keys.npy'); counts=np.load(a.base/'X_dev.npy',mmap_mode='r')
assert counts.shape==(len(keys),len(species))
years=np.load(a.features/'years.npy');cells=np.load(a.features/'cells.npy')
ho,bf=np.load(a.features/'holdout_cells.npy'),np.load(a.features/'buffer_cells.npy')
index={tuple(rc):i for i,rc in enumerate(cells)}
groups={}
for i,(r,c,y) in enumerate(keys):
    if ho[r,c] or bf[r,c]:continue
    epoch=0 if 1996<=y<=2005 else 1 if 2016<=y<=2025 else None
    if epoch is not None:groups.setdefault((int(r),int(c)),[[],[]])[epoch].append(i)
support=sorted(rc for rc,g in groups.items() if all(len(np.unique(keys[v,2]))>=4 for v in g))
if len(support)<100:raise ValueError('insufficient modern training support')
early=[groups[rc][0] for rc in support];late=[groups[rc][1] for rc in support]
ea,eb,ok1=split_half_groups(early,years=keys[:,2]);la,lb,ok2=split_half_groups(late,years=keys[:,2])
assert ok1.all() and ok2.all()
E1,E2=epoch_mean_observed(counts,ea),epoch_mean_observed(counts,eb)
L1,L2=epoch_mean_observed(counts,la),epoch_mean_observed(counts,lb)
del counts
target={'level':((E1+L1)/2,(E2+L2)/2),'change':(L1-E1,L2-E2)}
raw=np.load(a.features/'z_raw_all.npy',mmap_mode='r')
def mean_z(rows,rc):return np.asarray(raw[index[rc],keys[rows,2]-years[0]],dtype='float64').mean(0)
ze=np.stack([mean_z(g,rc) for g,rc in zip(early,support)])
zl=np.stack([mean_z(g,rc) for g,rc in zip(late,support)])
assert ze.shape[1]==128 and np.isfinite(ze).all() and np.isfinite(zl).all()
z={'level':(ze+zl)/2,'change':zl-ze}
rc=np.asarray(support);block=rc[:,0]//6*100000+rc[:,1]//6
unique=np.unique(block);rng=np.random.default_rng(20261026)
order=rng.permutation(unique);assignment={int(b):i%4 for i,b in enumerate(order)}
fold=np.array([assignment[int(b)] for b in block])
cal=np.array([int(hashlib.sha256(('ZCAL1:'+s).encode()).hexdigest(),16)%2==0 for s in species])
assert cal.any() and (~cal).any()
candidates={'identity':None,**{f'summer_share_{v:g}':v for v in [0.,.25,.5,.75,1.]}}

def weights(train,share):
    center=z['level'][train]-z['level'][train].mean(0)
    energy=np.square(center).mean(0)
    if share is None:return np.ones(128)/energy.sum()
    return np.r_[np.full(64,share/energy[:64].sum()),np.full(64,(1-share)/energy[64:].sum())]

scores=[]; pair_audit=[]
for f in range(4):
    train=np.flatnonzero(fold!=f);valid=np.flatnonzero(fold==f)
    pair_rng=np.random.default_rng(20261026+f)
    i,j=pair_rng.choice(valid,size=(2,20000),replace=True)
    not_same=i!=j;i,j=i[not_same],j[not_same]
    feature_products={kind:(value[i]-(value[train].mean(0) if kind=='level' else 0))*
                          (value[j]-(value[train].mean(0) if kind=='level' else 0)) for kind,value in z.items()}
    truths={}
    for set_name,mask in [('calibration',cal),('validation',~cal)]:
        A,B=(value[:,mask] for value in target['level'])
        ma,mb=A[train].mean(0),B[train].mean(0)
        spatial_scale=float(((A[train]-ma)*(B[train]-mb)).sum(1).mean())
        if spatial_scale<=0:raise ValueError('nonpositive cross-half spatial scale')
        for kind,(a0,b0) in target.items():
            A,B=a0[:,mask],b0[:,mask]
            if kind=='level':A,B=A-ma,B-mb
            T=((A[i]*B[j]).sum(1)+(B[i]*A[j]).sum(1))/(2*spatial_scale)
            truths[set_name,kind]=T
    fold_predictions={}
    for name,share in candidates.items():
        w=weights(train,share)
        for kind,products in feature_products.items():
            P=products@w
            fold_predictions[name,kind]=P
            for set_name in ['calibration','validation']:
                T=truths[set_name,kind]
                scores.append({'fold':f,'candidate':name,'species_set':set_name,'kind':kind,
                               'pairs':len(i),'loss':float(np.mean(P*P-2*P*T)),
                               'target_gram_mean_square':float(np.mean(T*T))})
    pair_audit.append((i,j,fold_predictions,truths))
    print('Completed spatial fold',f,'validation cells',len(valid),flush=True)
mean=lambda name,ss,kind,field='loss':float(np.mean([r[field] for r in scores if r['candidate']==name and r['species_set']==ss and r['kind']==kind]))
baseline=mean('identity','calibration','level')
denom=mean('identity','calibration','level','target_gram_mean_square')
eligible=[name for name in candidates if (mean(name,'calibration','level')-baseline)/max(denom,1e-12)<=.05]
selected=min(eligible,key=lambda name:mean(name,'calibration','change'))
final_weights=weights(np.arange(len(rc)),candidates[selected])
bootstrap={}
_,block_index=np.unique(block,return_inverse=True)
draws=np.random.default_rng(20261026).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=1000).astype('float64')
for kind in ['level','change']:
    loss_sum=np.zeros((len(unique),len(unique)));pair_count=np.zeros_like(loss_sum)
    for i,j,pred,truth in pair_audit:
        chosen,baseline_pred=pred[selected,kind],pred['identity',kind]
        diff=chosen**2-baseline_pred**2-2*(chosen-baseline_pred)*truth['validation',kind]
        bi,bj=block_index[i],block_index[j]
        np.add.at(loss_sum,(bi,bj),diff);np.add.at(pair_count,(bi,bj),1)
    numerator=np.einsum('ij,ij->i',draws@loss_sum,draws)
    denominator=np.einsum('ij,ij->i',draws@pair_count,draws)
    good=denominator>0;boot=numerator[good]/denominator[good]
    bootstrap[kind]={'selected_minus_identity_loss':float(loss_sum.sum()/pair_count.sum()),
                     'ci95':np.percentile(boot,[2.5,97.5]).tolist(),
                     'interpretation':'Conditional paired6x6 spatial-block bootstrap on validation species; selected metric, targets and fold normalizations held fixed. Not full fitting uncertainty.'}
result={'timestamp':datetime.now(timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'feature_manifest_sha256':hashlib.sha256((a.features/'DONE.json').read_bytes()).hexdigest(),
        'interpretation':'Exploratory development calibration; modern training cells only; no production adoption or confirmed geometry correction.',
        'support':{'cells':len(rc),'spatial_blocks':len(unique),'epochs':[[1996,2005],[2016,2025]],
                   'calibration_species':[s for s,m in zip(species,cal) if m],
                   'validation_species':[s for s,m in zip(species,cal) if not m]},
        'selection':'minimum calibration-species spatial CV change loss with spatial excess<=.05 target Gram mean-square',
        'selected':selected,'eligible':eligible,'scores':scores,'validation_bootstrap':bootstrap,
        'summary':{name:{ss:{kind:mean(name,ss,kind) for kind in ['level','change']} for ss in ['calibration','validation']} for name in candidates}}
np.savez(a.out/'exploratory_metric.npz',diagonal=final_weights,cells=rc,fold=fold)
(a.out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
(a.out/'CALIBRATION_DONE').write_text(result['timestamp'])
print('Selected exploratory metric',selected,'saved',a.out,flush=True)
