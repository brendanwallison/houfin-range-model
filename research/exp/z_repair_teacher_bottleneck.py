"""Matched capacity diagnostic: observed breeding teacher versus learned features.

Observed-community features contain the scored species, so this is an oracle
diagnostic, never an independent predictor or ecological validation. Readouts
use permitted modern training records only; evaluation halves stay separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.exp.z_repair_evaluate import matched_groups,weights_for_cells,representation
from research.exp.e069_presence_score import ridge_cv
from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
from src.community_encoder.train_DESK.esk_kernel import project_points_to_z


def score(cells,pa,pb,ta,tb,deterministic=False):
    W,blocks=weights_for_cells(cells);sw=W.sum(1)
    def cov(X,Y):
        return (W@np.sum(X*Y,axis=1)-np.sum((W@X)*(W@Y),axis=1)/sw)/sw
    vp=cov(pa,pb);vt=cov(ta,tb)
    cross=.5*(cov(pa,tb)+cov(pb,ta))
    corr=np.divide(cross,np.sqrt(np.maximum(vp*vt,0)),
                   out=np.full(len(W),np.nan),where=(vp>0)&(vt>0))
    out={'cells':len(cells),'blocks':blocks,'predicted_signal':float(vp[0]),'target_signal':float(vt[0])}
    good=np.isfinite(corr[1:])
    out['corr']={'point':float(corr[0]) if np.isfinite(corr[0]) else None,
                 'ci95':np.percentile(corr[1:][good],[2.5,97.5]).tolist() if good.mean()>=.95 else None}
    out['prediction_is_deterministic']=deterministic
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--projection-only',action='store_true',
                   help='write observed_projection.npz (what z_assim_evaluate reads) and stop; needs no model cube')
    a=p.parse_args();root=a.root;task='backcast40'
    if torch.cuda.is_available():raise RuntimeError('Run on CPU; do not compete with queued GPU training')
    torch.set_num_threads(6)
    data=root/'data'/task;out=root/'teacher_bottleneck';out.mkdir(exist_ok=True)
    if (out/'DONE.json').exists():raise RuntimeError('Refuse to overwrite completed diagnostic')
    masks=dict(np.load(data/'summer_masks.npz'));keys=masks['keys']
    meta=json.loads((a.base/'meta.json').read_text());species=meta['community']+meta['dev_species']
    if 'houfin' in species or set(species)&set(meta['sealed_species']):raise ValueError('Forbidden targets')
    assert np.array_equal(keys,np.load(a.base/'keys.npy'))
    Y=np.hstack([np.load(a.base/'X_comm.npy'),np.load(a.base/'X_dev.npy')]).astype('float64')
    pm=json.loads((data/'summer_train_points/points_meta.json').read_text())
    order=[species.index(s) for s in pm['species']]
    trainkeys=np.load(data/'summer_train_points/point_index.npy')
    source_rows=np.load(data/'summer_train_source_rows.npy')
    assert np.array_equal(keys[source_rows],trainkeys)
    np.testing.assert_allclose(np.log1p(Y[source_rows[:100]][:,order]),
        np.load(data/'summer_train_points/X_points.npy',mmap_mode='r')[:100],atol=1e-6)
    zs=np.load(root/'aux'/task/'full/summer_training_coords.npy')
    assert len(trainkeys)==len(zs)
    training=[];fitcells=[];features=[]
    for row,col in np.unique(keys[masks['train'],:2],axis=0):
        rows=np.flatnonzero(masks['train']&(keys[:,0]==row)&(keys[:,1]==col))
        if len(rows)<4:continue
        use=(trainkeys[:,0]==row)&(trainkeys[:,1]==col)
        assert use.sum()==len(rows)
        features.append(zs[use].mean(0));training.append(rows);fitcells.append((row,col))
    fitcells=np.asarray(fitcells)
    predict,lam=ridge_cv(np.asarray(features),epoch_mean_observed(Y,training),
                         fitcells[:,0]//6*100000+fitcells[:,1]//6,[1e-4,1e-3,1e-2,1e-1,1.,10.])
    cells,spatial,panel=matched_groups(keys,masks,2)
    groups=[g[e][h] for e,h in [(0,0),(0,1),(1,0),(1,1)] for g in panel]
    rows=np.unique(np.concatenate(groups))
    teacher=root/'teachers'/task/'summer/spacetime'
    print('Projecting',len(rows),'reserved survey rows into the fixed modern teacher',flush=True)
    observed=project_points_to_z(np.log1p(Y[rows][:,order]).astype('float32'),str(teacher),64,batch=1024)
    np.savez_compressed(out/'observed_projection.npz',source_rows=rows,Z=observed)
    if a.projection_only:
        (out/'PROJECTION_DONE.json').write_text(json.dumps({'rows':int(len(rows)),'teacher':str(teacher),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
        print('Projection only: wrote',out/'observed_projection.npz',flush=True)
        return
    byrow={int(row):i for i,row in enumerate(rows)}
    zobs=np.stack([observed[[byrow[int(row)] for row in g]].mean(0) for g in groups]).reshape(4,len(cells),64)
    model=representation(root,'control',keys[rows],task)[:,:64]
    zmod=np.stack([model[[byrow[int(row)] for row in g]].mean(0) for g in groups]).reshape(4,len(cells),64)
    target=np.stack([epoch_mean_observed(Y,[g])[0] for g in groups]).reshape(4,len(cells),len(species))
    # Ordering is historical-A,historical-B,modern-A,modern-B.
    za,zb=zobs[2]-zobs[0],zobs[3]-zobs[1]
    da,db=zmod[2]-zmod[0],zmod[3]-zmod[1]
    ta,tb=target[2]-target[0],target[3]-target[1]
    oa,ob=predict(zobs[2])-predict(zobs[0]),predict(zobs[3])-predict(zobs[1])
    result={'scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'readout_lambda':lam,'projection':'Mean annual projections on exactly the matched survey rows.',
            'model_scope':'Summer64 coordinate fidelity; species oracle uses observed summer64, not a new128 candidate.',
            'panels':{}}
    for label,keep in [('known',~spatial),('spatial',spatial)]:
        d=.5*(da[keep]+db[keep])
        result['panels'][label]={
            'model_to_observed_teacher_change':score(cells[keep],d,d,za[keep],zb[keep],True),
            'observed_teacher_to_species_change_oracle':score(cells[keep],oa[keep],ob[keep],ta[keep],tb[keep])}
    (out/'DONE.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
