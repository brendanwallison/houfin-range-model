"""Attribute heldout community change to regional means and local departures.

This is a supplementary diagnostic on the already fixed evaluation panels,
not an additional fit or a model-selection score. Regions are the existing
6-by-6 grid blocks. Results weight surveyed cells, not geographic area.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.exp.z_repair_evaluate import bootstrap_components,weights_for_cells
from research.exp.z_repair_compare import summary


def partition_change(cells,*arrays,block_cells=6,min_cells=2):
    """Exact mean/departure decomposition, restricted to replicated regions."""
    cells=np.asarray(cells)
    blocks=cells//block_cells
    _,index,counts=np.unique(blocks,axis=0,return_inverse=True,return_counts=True)
    keep=counts[index]>=min_cells
    if not keep.any():return keep,[]
    _,idx,n=np.unique(blocks[keep],axis=0,return_inverse=True,return_counts=True)
    result=[]
    for a in arrays:
        a=np.asarray(a,float)[keep]
        total=np.zeros((len(n),a.shape[1]),float)
        np.add.at(total,idx,a)
        regional=(total/n[:,None])[idx]
        result.append((regional,a-regional))
    return keep,result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arms',nargs='+',choices=['time_r2','residual_r2'],default=['time_r2','residual_r2'])
    p.add_argument('--output-stem',default='REGIONAL_CHANGE')
    a=p.parse_args();root=a.root/'evaluation/backcast40'
    control=json.loads((root/'control/SUMMER_DONE.json').read_text())
    result={'scope':'Supplementary development diagnostic; no fitting or adoption.',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'regions':'Existing 6x6 grid blocks; at least two evaluation cells per block.',
            'weighting':'Surveyed cells; regional means repeated at their cells. Not area-weighted.',
            'interpretation':'Within-region departures include ecological heterogeneity and observation effects; regional correlation is not a causal or spatial-decay estimate.',
            'panels':{}}
    for panel in control['panels']:
        arrays={arm:dict(np.load(root/arm/(panel+'.npz')))
                for arm in ['control',*a.arms]}
        c=arrays['control']
        for d in arrays.values():
            for name in ['cells','truth_a','truth_b']:
                if not np.array_equal(c[name],d[name]):raise ValueError('Unpaired '+panel+' '+name)
        keep,truth=partition_change(c['cells'],c['truth_a'],c['truth_b'])
        if keep.sum()<20:continue
        W,nblocks=weights_for_cells(c['cells'][keep])
        entry={'eligible_cells':int(keep.sum()),'excluded_singleton_cells':int((~keep).sum()),
               'blocks':nblocks,'components':{}}
        for component,i in [('regional',0),('local_departure',1)]:
            scores={}
            for arm,d in arrays.items():
                _,pred=partition_change(d['cells'],d['predicted_change'])
                scores[arm]=bootstrap_components(pred[0][i],truth[0][i],truth[1][i],W)
            entry['components'][component]={
                'arms':{arm:{k:summary(v) for k,v in s.items()} for arm,s in scores.items()},
                'treatment_minus_control':{
                    arm:{k:summary(scores[arm][k]-scores['control'][k])
                         for k in ['corr','mse_gain_over_no_change']}
                    for arm in a.arms}}
        result['panels'][panel]=entry
    (root/(a.output_stem+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False))
    lines=['# Regional and local community-change diagnostic','',result['regions'],
           result['weighting'],'',
           'All components use the same replicated-region subset. Signal is centered cross-half covariance per cell, summed over species; it can be negative with insufficient support.',
           '', '| Panel | Component | Cells | Observed signal |'+'|'.join(arm+' correlation' for arm in ['control',*a.arms])+'|',
           '|---|---|---:|---:|'+'---:|'*(1+len(a.arms))]
    def fmt(d):
        if d['point'] is None:return 'unsupported'
        ci=d['ci95']
        return f"{d['point']:.4f}"+(f" [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ' [unsupported interval]')
    for panel,e in result['panels'].items():
        for name,component in e['components'].items():
            s=component['arms']
            values=[fmt(s['control']['truth_centered_signal'])]+[fmt(s[arm]['corr']) for arm in ['control',*a.arms]]
            lines.append(f"|{panel}|{name}|{e['eligible_cells']}|"+'|'.join(values)+'|')
    lines+=['',f'Paired gains, slopes and bootstrap support are in {a.output_stem}.json.',
            result['interpretation'],'This diagnostic does not validate pre1966 summer reconstruction.']
    (root/(a.output_stem+'.md')).write_text('\n'.join(lines)+'\n')
    print(root/(a.output_stem+'.md'))


if __name__=='__main__':main()
