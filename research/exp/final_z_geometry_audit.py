"""Describe multi-decadal Z geometry without fitting to any heldout bird outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--features',type=Path,required=True)
p.add_argument('--bbs-keys',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
arms={'v1_r64':('final',slice(None)), 'v2_s64':('final_v2',slice(0,64)),
      'v2_w64':('final_v2',slice(64,128)), 'v2_r128':('final_v2',slice(None))}
windows=[((1996,2005),(2016,2025)),((1966,1975),(2006,2015)),
         ((1902,1911),(1932,1941)),((1902,1911),(2006,2015))]
bbs={tuple(rc) for rc in np.unique(np.load(a.bbs_keys)[:,:2],axis=0)}
result={'interpretation':'Feature-only geometry at BBS training cells, excluding encoder holdout/buffer. No outcome-based metric fitted. Covariance ratios describe motion, not ecological credibility or a calibrated readout amplitude.',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'arms':{}}
for arm,(family,cols) in arms.items():
    f=a.features/family
    cells,years=np.load(f/'cells.npy'),np.load(f/'years.npy')
    ho,bf=np.load(f/'holdout_cells.npy'),np.load(f/'buffer_cells.npy')
    keep=np.array([tuple(rc) in bbs and not ho[tuple(rc)] and not bf[tuple(rc)] for rc in cells])
    z=np.asarray(np.load(f/'z_raw_all.npy',mmap_mode='r')[keep,:,cols],dtype='float64')
    if not np.isfinite(z).all():raise ValueError('nonfinite geometry')
    ref=z[:,(years>=1996)&(years<=2025)].mean(1)
    centered=ref-ref.mean(0)
    between=centered.T@centered/len(ref)
    trace=float(np.trace(between))
    eig,vec=np.linalg.eigh(between)
    inv=vec@np.diag(1/np.sqrt(np.maximum(eig,1e-6*max(eig))))@vec.T
    entry={'training_cells':int(keep.sum()),'between_trace':trace,
           'between_effective_rank':trace**2/float(np.square(between).sum()),'windows':{},
           'feature_manifest_sha256':hashlib.sha256((f/'DONE.json').read_bytes()).hexdigest()}
    for early,late in windows:
        em=(years>=early[0])&(years<=early[1]);lm=(years>=late[0])&(years<=late[1])
        delta=z[:,lm].mean(1)-z[:,em].mean(1)
        mean=delta.mean(0); total=delta.T@delta/len(delta);energy=float(np.trace(total))
        whitened=inv@total@inv
        ratios=np.maximum(np.linalg.eigvalsh(whitened),0)
        entry['windows'][f'{early[0]}_{early[1]}__{late[0]}_{late[1]}']={
            'change_energy':energy,'change_to_between_rms':float(np.sqrt(energy/trace)),
            'continental_mean_share':float(mean@mean/energy) if energy else None,
            'change_effective_rank':energy**2/float(np.square(total).sum()) if energy else None,
            'largest_generalized_change_ratio':float(ratios[-1]),
            'generalized_ratio_regularization':'between eigenvalue floor=1e-6 times maximum; diagnostic only'}
    result['arms'][arm]=entry
a.out.write_text(json.dumps(result,indent=2,allow_nan=False))
print('Saved feature-only geometry',a.out)
