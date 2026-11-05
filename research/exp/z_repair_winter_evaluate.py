"""Reuse the established CBC conditional composition scorer on unseen circles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.exp.e069_presence_score import ridge_cv,fit_delta,list_ll
from research.exp.final_winter_score import ERAS,paired_circle_summary
from research.exp.z_repair_evaluate import representation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--cbc',type=Path,required=True)
    p.add_argument('--arm',choices=['control','time_r2','residual_r2'],required=True)
    a=p.parse_args();root=a.root.resolve();task='backcast40'
    meta=json.loads((a.base/'meta.json').read_text())
    allowed=set(meta['community'])|set(meta['dev_species'])
    D=dict(np.load(a.cbc,allow_pickle=False))
    if not set(map(str,D['species']))<=allowed:raise ValueError('Nondevelopment targets')
    masks=dict(np.load(root/'data'/task/'winter_full/record_masks.npz'))
    keys=np.column_stack([D['row'],D['col'],D['year_end']]).astype(int)
    assert np.array_equal(keys,masks['keys'])
    train=masks['train']&(keys[:,2]>=2006)&np.isfinite(D['party_hours'])&(D['party_hours']>0)
    # Eligibility uses training detections only, matching final_winter_score.
    keep=(D['Y'][train]>0).sum(0)>=10
    species=D['species'][keep];Y=D['Y'][:,keep];P=Y>0
    train&=P.sum(1)>=2
    test=masks['evaluation']&(P.sum(1)>=2)
    assert not set(D['circle_idx'][train])&set(D['circle_idx'][test])
    tr,te=np.flatnonzero(train),np.flatnonzero(test)
    rows=np.r_[tr,te]
    z=representation(root,a.arm,keys[rows],task)
    ztr,zte=z[:len(tr)],z[len(tr):]
    target=np.log1p(Y[tr]*((np.median(D['party_hours'][tr])/D['party_hours'][tr])[:,None]**.8))
    groups=keys[tr,0]//6*100000+keys[tr,1]//6
    predict,lam=ridge_cv(ztr,target,groups,[1e-4,1e-3,1e-2,1e-1,1.,10.])
    delta=fit_delta(predict(ztr),P[tr])
    cells,index=np.unique(keys[te,:2],axis=0,return_inverse=True)
    modern=np.zeros((len(cells),128),float)
    for year in range(2006,2025):
        k=np.column_stack([cells,np.full(len(cells),year)])
        modern+=representation(root,a.arm,k,task)/19
    back=list_ll(predict(zte),delta,P[te])
    fixed=list_ll(predict(modern[index]),delta,P[te])
    out=root/'evaluation'/task/a.arm;out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'winter_lists.npz',test_rows=te,species=species,
                        circle=D['circle_idx'][te],year=keys[te,2],backcast=back,modern=fixed,
                        field_counters=D['field_counters'][te])
    result={'arm':a.arm,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'training_lists':len(tr),'evaluation_lists':len(te),'species':list(map(str,species)),
            'ridge_lambda':lam,'readout':'One modern winter count ridge with modern offsets, reused across time.',
            'scope':'Conditional recorded composition at unseen circles, not latent abundance truth.',
            'evaluation_status':'Development observations; no sealed species or House Finch.', 'eras':{}}
    circles=D['circle_idx'][te];years=keys[te,2]
    modern_counts=np.bincount(circles[years>=2006],minlength=len(D['circles']))
    for era,lo,hi in ERAS:
        m=(years>=lo)&(years<=hi)
        era_counts=np.bincount(circles[m],minlength=len(D['circles']))
        stable=m&(era_counts[circles]>=3)&(modern_counts[circles]>=3)
        low_effort=m&(D['field_counters'][te]>0)&(D['field_counters'][te]<=6)
        result['eras'][era]={}
        for label,mask in [('primary',m),('same_circles',stable),('counters_1_6',low_effort)]:
            result['eras'][era][label]=paired_circle_summary(circles[mask],
                {'backcast':back[mask],'modern_placebo':fixed[mask],
                 'backcast_minus_modern':(back-fixed)[mask]})
    (out/'WINTER_DONE.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('Winter scored:',a.arm,len(te),'held-out lists',flush=True)


if __name__=='__main__':main()
