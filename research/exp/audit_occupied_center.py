"""Host-only occupied-habitat centering audit using original/synthetic outcomes."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np


def centers(input_dir,synthetic):
    with (input_dir/'metadata.pkl').open('rb') as stream: meta=pickle.load(stream)
    shape=(len(meta['years']),len(meta['land_rows']),int(meta['M']))
    path=input_dir/meta['z_gathered_path']
    if path.stat().st_size!=np.prod(shape)*np.dtype('float32').itemsize:
        raise ValueError('unexpected Z memmap layout')
    z=np.memmap(path,dtype='float32',mode='r',shape=shape)
    lookup=np.full((meta['Ny'],meta['Nx']),-1,dtype=int)
    lookup[meta['land_rows'],meta['land_cols']]=np.arange(shape[1])
    cell=lookup[meta['obs_rows'],meta['obs_cols']]
    with np.load(synthetic,allow_pickle=False) as source: syn=np.asarray(source['y'])
    if syn.shape!=np.shape(meta['observed_results']): raise ValueError('synthetic design mismatch')
    out={}
    for name,y in [('original',np.asarray(meta['observed_results'])),('synthetic',syn)]:
        keep=(y>0)&(cell>=0)
        pairs=np.unique(np.column_stack((meta['obs_time_indices'][keep],cell[keep])),axis=0)
        c=np.asarray(z[pairs[:,0],pairs[:,1]],dtype=float).mean(0)
        out[name]={'positive_cell_years':len(pairs),'vector':c.tolist()}
    delta=np.asarray(out['synthetic']['vector'])-out['original']['vector']
    out.update(status='complete',delta_l2=float(np.linalg.norm(delta)),delta_max_abs=float(abs(delta).max()),
               note='This quantifies a data-dependent reparameterization. Prediction matching is required before assigning causal importance.')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--synthetic',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=centers(a.input_dir,a.synthetic);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+'\n');print('centering shift',out['delta_l2'],out['delta_max_abs'])
