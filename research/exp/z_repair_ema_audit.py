"""Score the learned observation EMA separately from instantaneous population Z.

The original raw results are retained. This does not change any population
export or select an EMA timescale on evaluation outcomes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.community_encoder.train_DESK.desk_training import apply_output_ema
from research.exp.z_repair_residual_fit import load_base
from research.exp import z_repair_evaluate as evaluate
from research.exp.z_repair_teacher_bottleneck import score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--model-key',default='control_s1');p.add_argument('--audit-name',default='ema_audit')
    p.add_argument('--winter',action='store_true')
    a=p.parse_args();root=a.root;out=root/a.audit_name
    out.mkdir(exist_ok=True)
    if (out/'DONE.json').exists():raise RuntimeError('Refuse to overwrite completed audit')
    keys=np.load(a.base/'keys.npy');cells,cidx=np.unique(keys[:,:2],axis=0,return_inverse=True)
    dm=dict(np.load(root/'models/backcast40'/a.model_key/'desk_meta.npz',allow_pickle=True))
    hl=float(dm['ema_half_life']);enabled=bool(dm['output_ema'])
    if not enabled:raise ValueError('Control does not have the expected training output EMA')
    start=int(dm.get('ema_warmup_start',1902))
    years=np.arange(start,2026)
    cube=root/'cubes/backcast40'/a.model_key
    if a.winter:
        initial=np.load(cube/f'Z_latent_{start}.npy',mmap_mode='r')
        land=np.isfinite(initial).all(-1)
        landcells=np.column_stack(np.where(land))
        index=np.full(land.shape,-1,int);index[land]=np.arange(len(landcells))
    else:
        landcells=cells
        shape=tuple((cells.max(0)+1).astype(int));index=np.full(shape,-1,int)
        index[cells[:,0],cells[:,1]]=np.arange(len(cells))
    raw=np.stack([load_base(cube,np.column_stack([landcells,np.full(len(landcells),y)])) for y in years])
    ema=apply_output_ema(raw,hl)
    pointidx=index[keys[:,0],keys[:,1]]
    z=ema[keys[:,2]-start,pointidx].astype('float64')
    np.save(out/'control_observation_features.npy',z.astype('float32'))
    anchor=ema[(years>=2006)&(years<=2025)].mean(0)[index[cells[:,0],cells[:,1]]]
    np.save(out/'control_observation_reference.npy',anchor)
    data=out/'data/backcast40';data.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(root/'data/backcast40/summer_masks.npz',data/'summer_masks.npz')
    def observed_features(unused_root,arm,k,task):
        if arm!='control':raise ValueError('Unexpected audit arm')
        idx=index[k[:,0],k[:,1]]
        if (idx<0).any() or (k[:,2]<start).any() or (k[:,2]>2025).any():raise ValueError('Audit query outside encoded domain')
        return ema[k[:,2]-start,idx].astype('float64')
    # Run the established scorer, changing only its feature source and isolated
    # output root. Original targets, row groups, readout/CV and bootstrap are reused.
    evaluate.representation=observed_features
    sys.argv=['ema_audit','--root',str(out),'--base',str(a.base),'--arm','control']
    evaluate.main()
    if a.winter:
        from research.exp import z_repair_winter_evaluate as winter_evaluate
        winter_data=data/'winter_full';winter_data.mkdir(exist_ok=True)
        shutil.copyfile(root/'data/backcast40/winter_full/record_masks.npz',winter_data/'record_masks.npz')
        winter_evaluate.representation=observed_features
        sys.argv=['ema_winter_audit','--root',str(out),'--base',str(a.base),'--arm','control',
                  '--cbc','/mnt/e/Datasets/cbc_raw/prepared/cbc_circles_all.npz']
        winter_evaluate.main()
    masks=dict(np.load(root/'data/backcast40/summer_masks.npz'))
    cc,spatial,panel=evaluate.matched_groups(keys,masks,2)
    observed=dict(np.load(root/'teacher_bottleneck/observed_projection.npz'))
    lookup={int(row):i for i,row in enumerate(observed['source_rows'])}
    groups=[[g[e][h] for g in panel] for e,h in [(0,0),(0,1),(1,0),(1,1)]]
    zo=np.stack([np.stack([observed['Z'][[lookup[int(r)] for r in rows]].mean(0) for rows in group]) for group in groups])
    zm=np.stack([np.stack([z[rows,:64].mean(0) for rows in group]) for group in groups])
    ta,tb=zo[2]-zo[0],zo[3]-zo[1]
    pred=.5*((zm[2]-zm[0])+(zm[3]-zm[1]))
    result={'scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'model_key':a.model_key,'winter_evaluated':a.winter,
            'training_ema_half_life':hl,'warmup_start':start,'timescale_selected_on_evaluation':False,
            'teacher_change':{label:score(cc[keep],pred[keep],pred[keep],ta[keep],tb[keep],True)
                              for label,keep in [('known',~spatial),('spatial',spatial)]}}
    (out/'DONE.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
